import logging
from fastapi import APIRouter, HTTPException, Depends, status, Query, BackgroundTasks
from pydantic import BaseModel, Field
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Set
import pymysql
import pymysql.cursors
from api.database import get_database
from api.routes.auth import get_current_user, get_company_db_connection
from api.core.redis_cache import generate_cache_key, get_cached_data, set_cached_data
import json
import statistics
from collections import defaultdict, Counter
import uuid
import os
import traceback
from io import BytesIO

# Conditional imports for optional dependencies
try:
    import paramiko
    PARAMIKO_AVAILABLE = True
except ImportError:
    PARAMIKO_AVAILABLE = False
    paramiko = None

try:
    from google.cloud import storage
    from google.oauth2 import service_account
    GCS_AVAILABLE = True
except ImportError:
    GCS_AVAILABLE = False
    storage = None
    service_account = None

try:
    import assemblyai as aai
    ASSEMBLYAI_AVAILABLE = True
except ImportError:
    ASSEMBLYAI_AVAILABLE = False
    aai = None

try:
    from motor.motor_asyncio import AsyncIOMotorClient
    import asyncio
    MONGODB_AVAILABLE = True
except ImportError:
    MONGODB_AVAILABLE = False
    AsyncIOMotorClient = None

try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    openai = None

try:
    from langfuse.openai import openai as langfuse_openai
    from langfuse import Langfuse
    LANGFUSE_AVAILABLE = True
except ImportError:
    LANGFUSE_AVAILABLE = False
    langfuse_openai = None
    Langfuse = None

# Import LLM class for business intelligence analysis
try:
    from utils.llm import LLM
    LLM_AVAILABLE = True
except ImportError:
    LLM_AVAILABLE = False
    LLM = None

# Set up logging
logger = logging.getLogger(__name__)

# Initialize router
router = APIRouter(prefix="/ai-insights", tags=["ai_insights"])

# Global storage for background task results (similar to audio_files.py)
audio_task_results: Dict[str, Dict[str, Any]] = {}

# Global storage for failed file patterns to skip in future uploads
failed_file_patterns: Set[str] = set()
failed_file_extensions: Set[str] = set()

def check_audio_dependencies():
    """Check if all required dependencies for audio processing are available"""
    missing_deps = []
    
    if not PARAMIKO_AVAILABLE:
        missing_deps.append("paramiko")
    
    if not GCS_AVAILABLE:
        missing_deps.append("google-cloud-storage")
    
    if not ASSEMBLYAI_AVAILABLE:
        missing_deps.append("assemblyai")
    
    if not MONGODB_AVAILABLE:
        missing_deps.append("motor")
    
    if not OPENAI_AVAILABLE:
        missing_deps.append("openai")
    
    # Langfuse is optional for tracing - warn if not available but don't fail
    if not LANGFUSE_AVAILABLE:
        logger.warning("⚠️ Langfuse not available - OpenAI API calls will not be traced")
    
    return missing_deps

def should_skip_file(file_path: str) -> bool:
    """Check if a file should be skipped based on previous failures"""
    filename = os.path.basename(file_path)
    file_extension = os.path.splitext(filename)[1].lower()
    
    # Skip files with consistently failing extensions
    if file_extension in failed_file_extensions:
        logger.info(f"⏭️ Skipping file with failing extension {file_extension}: {filename}")
        return True
    
    # Skip files that match failed patterns
    for pattern in failed_file_patterns:
        if pattern in filename:
            logger.info(f"⏭️ Skipping file matching failed pattern '{pattern}': {filename}")
            return True
    
    # Skip files that are too small (likely corrupted)
    if 'empty' in filename.lower() or 'null' in filename.lower():
        logger.info(f"⏭️ Skipping likely empty/null file: {filename}")
        return True
    
    return False

def update_failed_patterns(file_path: str, error: str):
    """Update failed patterns based on the file that failed"""
    filename = os.path.basename(file_path)
    file_extension = os.path.splitext(filename)[1].lower()
    
    # Track failing extensions
    if file_extension and file_extension not in ['.wav', '.mp3', '.m4a', '.flac', '.ogg']:
        failed_file_extensions.add(file_extension)
        logger.info(f"📝 Added failing extension to skip list: {file_extension}")
    
    # Track specific error patterns
    if any(keyword in error.lower() for keyword in ['not found', 'no such file', 'permission denied']):
        # Extract common patterns from filename
        if '-' in filename:
            pattern = filename.split('-')[0]
            failed_file_patterns.add(pattern)
            logger.info(f"📝 Added failing pattern to skip list: {pattern}")
    
    # Limit the size of patterns to prevent memory issues
    if len(failed_file_patterns) > 100:
        failed_file_patterns.clear()
    if len(failed_file_extensions) > 20:
        failed_file_extensions.clear()

# Response Models
class TimePeriod(BaseModel):
    """Time period representation for AI insights"""
    start_date: str
    end_date: str
    total_days: int

class AIInsight(BaseModel):
    """Individual AI insight model"""
    type: str = Field(..., description="Type of insight (trend, anomaly, pattern, recommendation)")
    title: str = Field(..., description="Brief title of the insight")
    description: str = Field(..., description="Detailed description of the insight")
    confidence: float = Field(..., description="Confidence level (0-100)")
    priority: str = Field(..., description="Priority level (low, medium, high, critical)")
    category: str = Field(..., description="Category (performance, quality, efficiency, cost)")
    data_points: List[Dict[str, Any]] = Field(default_factory=list, description="Supporting data")
    recommendation: Optional[str] = Field(None, description="Actionable recommendation")

class CallPatternInsight(BaseModel):
    """Call pattern analysis insight"""
    pattern_type: str
    description: str
    frequency: int
    impact_score: float
    examples: List[Dict[str, Any]]

class TrendAnalysis(BaseModel):
    """Trend analysis model"""
    metric: str
    direction: str  # "increasing", "decreasing", "stable", "volatile"
    percentage_change: float
    significance: str  # "significant", "moderate", "minor"
    time_period: str
    forecast: Optional[Dict[str, Any]] = None

class AnomalyDetection(BaseModel):
    """Anomaly detection model"""
    anomaly_type: str
    description: str
    severity: str  # "low", "medium", "high", "critical"
    affected_period: str
    deviation_score: float
    baseline_value: float
    actual_value: float

class AIInsightsResponse(BaseModel):
    """Complete AI insights response"""
    generated_at: datetime
    time_period: TimePeriod
    insights: List[AIInsight]
    trends: List[TrendAnalysis]
    anomalies: List[AnomalyDetection]
    patterns: List[CallPatternInsight]
    summary: Dict[str, Any]
    recommendations: List[str]

# Helper Functions
def determine_call_direction(src: str, dst: str) -> str:
    """Determine call direction based on source and destination"""
    if not src or not dst:
        return "unknown"
    
    src_is_extension = len(str(src).strip()) <= 5 and str(src).strip().isdigit()
    dst_is_extension = len(str(dst).strip()) <= 5 and str(dst).strip().isdigit()
    
    if src_is_extension and dst_is_extension:
        return "internal"
    elif src_is_extension and not dst_is_extension:
        return "outbound"
    elif not src_is_extension and dst_is_extension:
        return "inbound"
    else:
        return "unknown"

def analyze_call_trends(daily_data: List[Dict[str, Any]]) -> List[TrendAnalysis]:
    """Analyze trends in call data"""
    trends = []
    
    if len(daily_data) < 3:
        return trends
    
    # Extract metrics for trend analysis
    call_counts = [day['total_calls'] for day in daily_data]
    answer_rates = [day['answer_rate'] for day in daily_data if day['answer_rate'] is not None]
    avg_durations = [day['avg_duration'] for day in daily_data if day['avg_duration'] is not None]
    
    # Analyze call volume trend
    if len(call_counts) >= 3:
        first_half_avg = statistics.mean(call_counts[:len(call_counts)//2])
        second_half_avg = statistics.mean(call_counts[len(call_counts)//2:])
        
        if first_half_avg > 0:
            change_percent = ((second_half_avg - first_half_avg) / first_half_avg) * 100
            
            if abs(change_percent) > 20:
                significance = "significant"
            elif abs(change_percent) > 10:
                significance = "moderate"
            else:
                significance = "minor"
            
            direction = "increasing" if change_percent > 5 else "decreasing" if change_percent < -5 else "stable"
            
            trends.append(TrendAnalysis(
                metric="call_volume",
                direction=direction,
                percentage_change=round(change_percent, 2),
                significance=significance,
                time_period=f"{len(daily_data)} days"
            ))
    
    # Analyze answer rate trend
    if len(answer_rates) >= 3:
        first_half_avg = statistics.mean(answer_rates[:len(answer_rates)//2])
        second_half_avg = statistics.mean(answer_rates[len(answer_rates)//2:])
        
        change_percent = second_half_avg - first_half_avg
        
        if abs(change_percent) > 10:
            significance = "significant"
        elif abs(change_percent) > 5:
            significance = "moderate"
        else:
            significance = "minor"
        
        direction = "increasing" if change_percent > 2 else "decreasing" if change_percent < -2 else "stable"
        
        trends.append(TrendAnalysis(
            metric="answer_rate",
            direction=direction,
            percentage_change=round(change_percent, 2),
            significance=significance,
            time_period=f"{len(daily_data)} days"
        ))
    
    return trends

def detect_anomalies(daily_data: List[Dict[str, Any]]) -> List[AnomalyDetection]:
    """Detect anomalies in call data"""
    anomalies = []
    
    if len(daily_data) < 7:
        return anomalies
    
    # Analyze call volume anomalies
    call_counts = [day['total_calls'] for day in daily_data]
    mean_calls = statistics.mean(call_counts)
    stdev_calls = statistics.stdev(call_counts) if len(call_counts) > 1 else 0
    
    for i, day in enumerate(daily_data):
        if stdev_calls > 0:
            z_score = abs((day['total_calls'] - mean_calls) / stdev_calls)
            
            if z_score > 2.5:  # Significant deviation
                severity = "high" if z_score > 3 else "medium"
                
                anomalies.append(AnomalyDetection(
                    anomaly_type="call_volume_spike" if day['total_calls'] > mean_calls else "call_volume_drop",
                    description=f"Unusual call volume on {day['date']}: {day['total_calls']} calls vs average {round(mean_calls, 1)}",
                    severity=severity,
                    affected_period=day['date'],
                    deviation_score=round(z_score, 2),
                    baseline_value=round(mean_calls, 1),
                    actual_value=day['total_calls']
                ))
    
    # Analyze answer rate anomalies
    answer_rates = [day['answer_rate'] for day in daily_data if day['answer_rate'] is not None]
    if len(answer_rates) > 1:
        mean_answer_rate = statistics.mean(answer_rates)
        stdev_answer_rate = statistics.stdev(answer_rates)
        
        for day in daily_data:
            if day['answer_rate'] is not None and stdev_answer_rate > 0:
                z_score = abs((day['answer_rate'] - mean_answer_rate) / stdev_answer_rate)
                
                if z_score > 2 and day['answer_rate'] < mean_answer_rate - 10:  # Significant drop
                    severity = "critical" if day['answer_rate'] < 50 else "high"
                    
                    anomalies.append(AnomalyDetection(
                        anomaly_type="answer_rate_drop",
                        description=f"Poor answer rate on {day['date']}: {day['answer_rate']:.1f}% vs average {mean_answer_rate:.1f}%",
                        severity=severity,
                        affected_period=day['date'],
                        deviation_score=round(z_score, 2),
                        baseline_value=round(mean_answer_rate, 1),
                        actual_value=day['answer_rate']
                    ))
    
    return anomalies

def identify_call_patterns(records: List[Dict[str, Any]]) -> List[CallPatternInsight]:
    """Identify patterns in call data"""
    patterns = []
    
    if not records:
        return patterns
    
    # Analyze hourly patterns
    hourly_counts = defaultdict(int)
    for record in records:
        if record.get('calldate'):
            hour = record['calldate'].hour
            hourly_counts[hour] += 1
    
    if hourly_counts:
        peak_hour = max(hourly_counts.items(), key=lambda x: x[1])
        total_calls = sum(hourly_counts.values())
        peak_percentage = (peak_hour[1] / total_calls) * 100
        
        if peak_percentage > 15:  # Significant concentration
            patterns.append(CallPatternInsight(
                pattern_type="peak_hour_concentration",
                description=f"High call concentration at {peak_hour[0]:02d}:00 ({peak_percentage:.1f}% of all calls)",
                frequency=peak_hour[1],
                impact_score=peak_percentage,
                examples=[{"hour": peak_hour[0], "calls": peak_hour[1], "percentage": peak_percentage}]
            ))
    
    # Analyze repeat callers
    caller_counts = Counter(record.get('src') for record in records if record.get('src'))
    frequent_callers = [(caller, count) for caller, count in caller_counts.items() if count >= 5]
    
    if frequent_callers:
        patterns.append(CallPatternInsight(
            pattern_type="frequent_callers",
            description=f"Identified {len(frequent_callers)} frequent callers with 5+ calls",
            frequency=len(frequent_callers),
            impact_score=sum(count for _, count in frequent_callers) / len(records) * 100,
            examples=[{"caller": caller, "calls": count} for caller, count in frequent_callers[:5]]
        ))
    
    # Analyze short duration calls
    short_calls = [r for r in records if r.get('duration', 0) < 30 and r.get('disposition') == 'ANSWERED']
    if short_calls:
        short_call_percentage = (len(short_calls) / len(records)) * 100
        if short_call_percentage > 20:
            patterns.append(CallPatternInsight(
                pattern_type="short_duration_calls",
                description=f"High percentage of short calls (<30s): {short_call_percentage:.1f}%",
                frequency=len(short_calls),
                impact_score=short_call_percentage,
                examples=[{"duration": r.get('duration', 0), "disposition": r.get('disposition')} for r in short_calls[:5]]
            ))
    
    return patterns

def generate_insights(daily_data: List[Dict[str, Any]], records: List[Dict[str, Any]], 
                     trends: List[TrendAnalysis], anomalies: List[AnomalyDetection], 
                     patterns: List[CallPatternInsight]) -> List[AIInsight]:
    """Generate AI insights based on analyzed data"""
    insights = []
    
    # Performance insights
    if daily_data:
        avg_answer_rate = statistics.mean([d['answer_rate'] for d in daily_data if d['answer_rate'] is not None])
        
        if avg_answer_rate < 70:
            insights.append(AIInsight(
                type="performance",
                title="Low Answer Rate Detected",
                description=f"Average answer rate is {avg_answer_rate:.1f}%, below recommended 80%",
                confidence=85.0,
                priority="high",
                category="performance",
                recommendation="Review staffing levels during peak hours and implement call routing optimization"
            ))
        elif avg_answer_rate > 90:
            insights.append(AIInsight(
                type="performance",
                title="Excellent Answer Rate",
                description=f"Outstanding answer rate of {avg_answer_rate:.1f}% indicates efficient call handling",
                confidence=90.0,
                priority="low",
                category="performance",
                recommendation="Maintain current staffing and processes"
            ))
    
    # Trend-based insights
    for trend in trends:
        if trend.metric == "call_volume" and trend.direction == "increasing" and trend.significance == "significant":
            insights.append(AIInsight(
                type="trend",
                title="Significant Call Volume Increase",
                description=f"Call volume has increased by {trend.percentage_change:.1f}% over {trend.time_period}",
                confidence=80.0,
                priority="medium",
                category="efficiency",
                recommendation="Consider increasing staffing or optimizing call routing to handle growing demand"
            ))
        elif trend.metric == "answer_rate" and trend.direction == "decreasing" and trend.significance != "minor":
            insights.append(AIInsight(
                type="trend",
                title="Declining Answer Rate",
                description=f"Answer rate has decreased by {abs(trend.percentage_change):.1f}% over {trend.time_period}",
                confidence=75.0,
                priority="high",
                category="quality",
                recommendation="Investigate staffing issues or technical problems affecting call handling"
            ))
    
    # Anomaly-based insights
    for anomaly in anomalies:
        if anomaly.severity in ["high", "critical"]:
            insights.append(AIInsight(
                type="anomaly",
                title=f"Anomaly Detected: {anomaly.anomaly_type.replace('_', ' ').title()}",
                description=anomaly.description,
                confidence=70.0,
                priority="high" if anomaly.severity == "critical" else "medium",
                category="quality",
                recommendation="Investigate root cause and implement corrective measures"
            ))
    
    # Pattern-based insights
    for pattern in patterns:
        if pattern.pattern_type == "peak_hour_concentration" and pattern.impact_score > 20:
            insights.append(AIInsight(
                type="pattern",
                title="Call Volume Concentration Identified",
                description=pattern.description,
                confidence=85.0,
                priority="medium",
                category="efficiency",
                recommendation="Optimize staffing schedule to match call volume patterns"
            ))
    
    return insights

def generate_recommendations(insights: List[AIInsight], trends: List[TrendAnalysis], 
                           anomalies: List[AnomalyDetection]) -> List[str]:
    """Generate actionable recommendations"""
    recommendations = []
    
    # High-priority recommendations based on insights
    high_priority_insights = [i for i in insights if i.priority in ["high", "critical"]]
    
    if high_priority_insights:
        recommendations.append("Address high-priority performance issues identified in the analysis")
    
    # Staffing recommendations
    volume_trends = [t for t in trends if t.metric == "call_volume"]
    if any(t.direction == "increasing" and t.significance != "minor" for t in volume_trends):
        recommendations.append("Consider scaling up staff or optimizing call routing due to increasing call volume")
    
    # Quality recommendations
    critical_anomalies = [a for a in anomalies if a.severity == "critical"]
    if critical_anomalies:
        recommendations.append("Immediate investigation required for critical performance anomalies")
    
    # Process improvements
    answer_rate_trends = [t for t in trends if t.metric == "answer_rate"]
    if any(t.direction == "decreasing" for t in answer_rate_trends):
        recommendations.append("Review and optimize call handling processes to improve answer rates")
    
    # Technology recommendations
    if len(anomalies) > 3:
        recommendations.append("Consider implementing real-time monitoring and alerting systems")
    
    return recommendations

# Route Implementation
@router.get("", response_model=AIInsightsResponse)
async def get_ai_insights(
    start_date: str = Query(..., description="Start date in YYYY-MM-DD format"),
    end_date: str = Query(..., description="End date in YYYY-MM-DD format"),
    use_cache: bool = Query(True, description="Whether to use cached data if available"),
    current_user: dict = Depends(get_current_user),
    db = Depends(get_database)
):
    """
    Get comprehensive AI-powered insights about call patterns, trends, and performance.
    
    This endpoint analyzes call data to provide:
    - Trend analysis (volume, answer rates, duration patterns)
    - Anomaly detection (unusual spikes, drops, or patterns)
    - Pattern identification (peak hours, frequent callers, call types)
    - Performance insights and recommendations
    - Predictive analytics where applicable
    """
    try:
        # Validate date format
        try:
            start_dt = datetime.strptime(start_date, "%Y-%m-%d")
            end_dt = datetime.strptime(end_date, "%Y-%m-%d")
        except ValueError:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid date format. Use YYYY-MM-DD"
            )
        
        # Check cache first
        cache_key = generate_cache_key("ai_insights", {
            "start_date": start_date,
            "end_date": end_date,
            "company_id": current_user.get("company_id")
        })
        
        if use_cache:
            cached_data = get_cached_data(cache_key)
            if cached_data:
                logger.info(f"Returning cached AI insights for {start_date} to {end_date}")
                return AIInsightsResponse(**cached_data)
        
        # Get the company_id from the current user
        company_id = current_user.get("company_id")
        if not company_id or company_id == "None":
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="User is not associated with any company"
            )
        
        # Get database connection details
        connection_details = await get_company_db_connection(company_id, current_user, db)
        
        # Connect to the database
        conn = None
        try:
            conn = pymysql.connect(
                host=connection_details["host"],
                user=connection_details["user"],
                password=connection_details["password"],
                database=connection_details["database"],
                port=connection_details["port"],
                connect_timeout=10,
                cursorclass=pymysql.cursors.DictCursor
            )
            
            with conn.cursor() as cursor:
                # Get daily aggregated data for trend analysis
                daily_query = """
                    SELECT 
                        DATE(calldate) as date,
                        COUNT(*) as total_calls,
                        SUM(CASE WHEN disposition = 'ANSWERED' THEN 1 ELSE 0 END) as answered_calls,
                        AVG(duration) as avg_duration,
                        AVG(billsec) as avg_billsec,
                        (SUM(CASE WHEN disposition = 'ANSWERED' THEN 1 ELSE 0 END) / COUNT(*)) * 100 as answer_rate
                    FROM cdr 
                    WHERE calldate >= %s AND calldate <= %s
                    GROUP BY DATE(calldate)
                    ORDER BY date
                """
                
                cursor.execute(daily_query, (start_date, end_date + " 23:59:59"))
                daily_data = cursor.fetchall()
                
                # Convert date objects to strings and handle None values
                for day in daily_data:
                    day['date'] = day['date'].strftime('%Y-%m-%d')
                    day['avg_duration'] = float(day['avg_duration']) if day['avg_duration'] else 0
                    day['avg_billsec'] = float(day['avg_billsec']) if day['avg_billsec'] else 0
                    day['answer_rate'] = float(day['answer_rate']) if day['answer_rate'] else 0
                
                # Get sample of call records for pattern analysis
                records_query = """
                    SELECT calldate, src, dst, duration, billsec, disposition, 
                           CASE 
                               WHEN LENGTH(TRIM(dst)) <= 5 AND TRIM(dst) REGEXP '^[0-9]+$' THEN 'inbound'
                               WHEN LENGTH(TRIM(src)) <= 5 AND TRIM(src) REGEXP '^[0-9]+$' 
                                    AND LENGTH(TRIM(dst)) > 5 THEN 'outbound'
                               WHEN LENGTH(TRIM(src)) <= 5 AND LENGTH(TRIM(dst)) <= 5 
                                    AND TRIM(src) REGEXP '^[0-9]+$' AND TRIM(dst) REGEXP '^[0-9]+$' THEN 'internal'
                               ELSE 'unknown'
                           END as direction
                    FROM cdr 
                    WHERE calldate >= %s AND calldate <= %s
                    ORDER BY calldate DESC
                    LIMIT 5000
                """
                
                cursor.execute(records_query, (start_date, end_date + " 23:59:59"))
                records = cursor.fetchall()
                
        finally:
            if conn:
                conn.close()
        
        # Perform AI analysis
        trends = analyze_call_trends(daily_data)
        anomalies = detect_anomalies(daily_data)
        patterns = identify_call_patterns(records)
        insights = generate_insights(daily_data, records, trends, anomalies, patterns)
        recommendations = generate_recommendations(insights, trends, anomalies)
        
        # Calculate summary statistics
        total_calls = sum(day['total_calls'] for day in daily_data)
        total_answered = sum(day['answered_calls'] for day in daily_data)
        avg_answer_rate = (total_answered / total_calls * 100) if total_calls > 0 else 0
        
        summary = {
            "total_calls_analyzed": total_calls,
            "average_answer_rate": round(avg_answer_rate, 2),
            "insights_count": len(insights),
            "trends_identified": len(trends),
            "anomalies_detected": len(anomalies),
            "patterns_found": len(patterns),
            "high_priority_issues": len([i for i in insights if i.priority in ["high", "critical"]]),
            "analysis_confidence": round(statistics.mean([i.confidence for i in insights]) if insights else 0, 1)
        }
        
        # Prepare response
        response_data = {
            "generated_at": datetime.utcnow(),
            "time_period": TimePeriod(
                start_date=start_date,
                end_date=end_date,
                total_days=(end_dt - start_dt).days + 1
            ),
            "insights": insights,
            "trends": trends,
            "anomalies": anomalies,
            "patterns": patterns,
            "summary": summary,
            "recommendations": recommendations
        }
        
        result = AIInsightsResponse(**response_data)
        
        # Cache the results for 30 minutes
        if use_cache:
            set_cached_data(cache_key, result.dict(), expiry_minutes=30)
        
        logger.info(f"Generated AI insights for {start_date} to {end_date}: {len(insights)} insights, {len(trends)} trends, {len(anomalies)} anomalies")
        
        return result
        
    except pymysql.MySQLError as e:
        logger.error(f"MySQL Error in AI insights: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Database error: {str(e)}"
        )
    except Exception as e:
        logger.error(f"Error generating AI insights: {str(e)}")
        if isinstance(e, HTTPException):
            raise
        else:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to generate AI insights: {str(e)}"
            )

@router.get("/summary", response_model=Dict[str, Any])
async def get_ai_insights_summary(
    start_date: str = Query(..., description="Start date in YYYY-MM-DD format"),
    end_date: str = Query(..., description="End date in YYYY-MM-DD format"),
    current_user: dict = Depends(get_current_user),
    db = Depends(get_database)
):
    """
    Get a quick summary of AI insights without detailed analysis.
    Useful for dashboards and overview displays.
    """
    try:
        # Get full insights
        full_insights = await get_ai_insights(start_date, end_date, True, current_user, db)
        
        # Return summarized version
        return {
            "time_period": full_insights.time_period,
            "summary": full_insights.summary,
            "top_insights": full_insights.insights[:3],  # Top 3 insights
            "critical_anomalies": [a for a in full_insights.anomalies if a.severity == "critical"],
            "key_recommendations": full_insights.recommendations[:3],  # Top 3 recommendations
            "generated_at": full_insights.generated_at
        }
        
    except Exception as e:
        logger.error(f"Error generating AI insights summary: {str(e)}")
        if isinstance(e, HTTPException):
            raise
        else:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to generate AI insights summary: {str(e)}"
            )

def get_openai_client_with_tracing():
    """Get OpenAI client with optional Langfuse tracing"""
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        raise Exception("OPENAI_API_KEY environment variable not set")
    
    if LANGFUSE_AVAILABLE:
        # Check if Langfuse environment variables are set
        langfuse_public_key = os.getenv("LANGFUSE_PUBLIC_KEY")
        langfuse_secret_key = os.getenv("LANGFUSE_SECRET_KEY") 
        langfuse_host = os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")
        
        if langfuse_public_key and langfuse_secret_key:
            # Use Langfuse OpenAI integration - environment variables are automatically picked up
            logger.info("🔍 Langfuse tracing enabled for OpenAI API calls")
            return langfuse_openai.OpenAI(api_key=openai_api_key)
        else:
            logger.warning("⚠️ Langfuse credentials not found - using standard OpenAI client")
    
    # Fallback to standard OpenAI client
    return openai.OpenAI(api_key=openai_api_key)

async def check_existing_files_in_mongodb(file_paths: List[str], user_id: str) -> Dict[str, Any]:
    """Check which audio files already exist in MongoDB to avoid reprocessing"""
    try:
        # MongoDB Configuration
        mongo_uri = os.getenv("MONGODB_URI", "mongodb://admin:delaphone%4001@102.22.15.141:37037/")
        mongo_client = AsyncIOMotorClient(mongo_uri)
        mongo_db = mongo_client["delaphone_transcriptions"]
        transcriptions_collection = mongo_db["transcriptions"]
        
        logger.info(f"🔍 Checking {len(file_paths)} files against MongoDB for user {user_id}")
        
        # Query MongoDB for existing files
        existing_docs = await transcriptions_collection.find({
            "user_id": user_id,
            "remote_path": {"$in": file_paths}
        }).to_list(length=None)
        
        # Close MongoDB connection
        mongo_client.close()
        
        # Organize results
        existing_files = {}
        for doc in existing_docs:
            remote_path = doc.get('remote_path')
            lemur_analysis = doc.get('lemur_analysis', {})
            existing_files[remote_path] = {
                'mongodb_id': str(doc.get('_id')),
                'transcript_id': doc.get('transcript_id'),
                'gcs_url': doc.get('gcs_url'),
                'public_url': doc.get('public_url'),
                'original_filename': doc.get('original_filename'),
                'remote_path': doc.get('remote_path'),
                'text': doc.get('text'),
                'speakers_count': doc.get('speakers_count', 0),
                'words_count': doc.get('words_count', 0),
                'audio_duration': doc.get('audio_duration'),
                'created_at': doc.get('created_at'),
                'lemur_analysis': lemur_analysis,  # Include complete lemur_analysis data
                'custom_topic': lemur_analysis.get('custom_topic'),
                'two_party_sentiment': lemur_analysis.get('sentiment_analysis'),
                'call_completion': lemur_analysis.get('call_completion'),
                'agent_performance': lemur_analysis.get('agent_performance'),
                'topic_detection_summary': doc.get('topic_detection', {}).get('summary', {}),  # Will be empty for new files (IAB disabled)
                'utterances': doc.get('utterances', []),
                'sentiment_analysis': doc.get('sentiment_analysis', []),
                'status': 'existing'
            }
        
        # Identify new files that need processing
        new_files = [fp for fp in file_paths if fp not in existing_files]
        
        logger.info(f"📊 MongoDB Check Results:")
        logger.info(f"   ✅ Already processed: {len(existing_files)} files")
        logger.info(f"   🔄 Need processing: {len(new_files)} files")
        logger.info(f"   💾 Resource savings: {len(existing_files)}/{len(file_paths)} ({(len(existing_files)/len(file_paths)*100):.1f}%)")
        
        return {
            'existing_files': existing_files,
            'new_files': new_files,
            'total_files': len(file_paths),
            'existing_count': len(existing_files),
            'new_count': len(new_files),
            'savings_percentage': (len(existing_files)/len(file_paths)*100) if file_paths else 0
        }
        
    except Exception as e:
        logger.error(f"❌ Error checking existing files in MongoDB: {e}")
        # If check fails, treat all files as new to be safe
        return {
            'existing_files': {},
            'new_files': file_paths,
            'total_files': len(file_paths),
            'existing_count': 0,
            'new_count': len(file_paths),
            'savings_percentage': 0,
            'check_error': str(e)
        }

async def process_audio_files_background_task(
    mongodb_check_result: Dict[str, Any], 
    user_id: str, 
    task_id: str,
    original_response: Dict[str, Any],
    skip_failed_uploads: bool = True
):
    """Background task for processing audio files from the logs endpoint - only processes new files"""
    try:
        # Extract the new files that need processing
        file_paths = mongodb_check_result['new_files']
        existing_files = mongodb_check_result['existing_files']
        
        logger.info(f"🚀 Starting background audio processing task {task_id} for user {user_id}")
        logger.info(f"📊 Processing {len(file_paths)} new files (skipping {len(existing_files)} existing)")
        
        # Check dependencies first
        missing_deps = check_audio_dependencies()
        if missing_deps:
            error_msg = f"❌ Missing required dependencies for audio processing: {', '.join(missing_deps)}. Please install with: pip install {' '.join(missing_deps)}"
            logger.error(error_msg)
            audio_task_results[task_id] = {
                "status": "failed",
                "user_id": user_id,
                "error": error_msg,
                "end_time": datetime.now().isoformat(),
                "message": error_msg
            }
            return
        
        # Update task status
        audio_task_results[task_id] = {
            "status": "processing",
            "user_id": user_id,
            "total_files": mongodb_check_result['total_files'],
            "existing_files": len(existing_files),
            "new_files": len(file_paths),
            "savings_percentage": mongodb_check_result['savings_percentage'],
            "start_time": datetime.now().isoformat(),
            "progress": 0,
            "processed_files": 0,
            "successful_uploads": 0,
            "failed_uploads": 0,
            "transcriptions_processed": 0,
            "transcriptions_saved": 0,
            "current_phase": "uploading",
            "message": f"Processing {len(file_paths)} new files (skipped {len(existing_files)} existing)...",
            "original_response": original_response
        }
        
        # Initialize services
        try:
            # SFTP Configuration
            sftp_host = os.getenv("SFTP_HOST")
            sftp_port = int(os.getenv("SFTP_PORT", "22"))
            sftp_username = os.getenv("SFTP_USERNAME")
            sftp_password = os.getenv("SFTP_PASSWORD")
            
            # GCS Configuration
            service_account_path = "/Users/redeemer/Desktop/Projectx/delaphone-data/src/dashboard-461709-99d6d4dbc31b.json"
            bucket_name = os.getenv("GCS_BUCKET_NAME", "delaphone-audio-files")
            
            # AssemblyAI Configuration
            assembly_api_key = os.getenv("ASSEMBLY_API_KEY")
            if not assembly_api_key:
                raise Exception("ASSEMBLY_API_KEY environment variable not set")
            aai.settings.api_key = assembly_api_key
            
            # MongoDB Configuration
            mongo_uri = os.getenv("MONGODB_URI", "mongodb://admin:delaphone%4001@102.22.15.141:37037/")
            mongo_client = AsyncIOMotorClient(mongo_uri)
            mongo_db = mongo_client["delaphone_transcriptions"]
            transcriptions_collection = mongo_db["transcriptions"]
            
            # Initialize GCS client
            credentials = service_account.Credentials.from_service_account_file(
                service_account_path,
                scopes=["https://www.googleapis.com/auth/cloud-platform"]
            )
            gcs_client = storage.Client(credentials=credentials)
            logger.info("☁️ All services initialized for background task")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize services: {e}")
            audio_task_results[task_id] = {
                "status": "failed",
                "user_id": user_id,
                "error": f"Failed to initialize services: {str(e)}",
                "end_time": datetime.now().isoformat(),
                "message": f"❌ Initialization failed: {str(e)}"
            }
            return
        
        upload_results = {}
        transcription_results = {}
        
        # Phase 1: Upload files to GCS
        audio_task_results[task_id]["current_phase"] = "uploading"
        logger.info(f"📤 Phase 1: Uploading {len(file_paths)} files to GCS")
        
        skipped_files = 0
        for i, file_path in enumerate(file_paths, 1):
            try:
                # Check if file should be skipped based on previous failures (only if enabled)
                if skip_failed_uploads and should_skip_file(file_path):
                    upload_results[file_path] = {
                        'success': False,
                        'error': 'Skipped due to previous failure patterns',
                        'gcs_url': None,
                        'skipped': True
                    }
                    skipped_files += 1
                    audio_task_results[task_id]["failed_uploads"] += 1
                    continue
                
                logger.info(f"📥 Processing file {i-skipped_files}/{len(file_paths)-skipped_files}: {file_path}")
                
                # Update progress (50% for uploads)
                upload_progress = (i / len(file_paths)) * 50
                audio_task_results[task_id]["progress"] = upload_progress
                audio_task_results[task_id]["processed_files"] = i
                audio_task_results[task_id]["message"] = f"Uploading file {i}/{len(file_paths)}: {file_path.split('/')[-1]}"
                
                # Process single audio file
                result = await process_single_audio_file(
                    file_path, user_id, sftp_host, sftp_port, 
                    sftp_username, sftp_password, gcs_client, bucket_name
                )
                upload_results[file_path] = result
                
                if result['success']:
                    audio_task_results[task_id]["successful_uploads"] += 1
                    logger.info(f"✅ Successfully uploaded file {i}/{len(file_paths)}: {file_path}")
                else:
                    audio_task_results[task_id]["failed_uploads"] += 1
                    error_msg = result.get('error', 'Unknown error')
                    logger.error(f"❌ Failed to upload file {i}/{len(file_paths)}: {file_path} - {error_msg}")
                    
                    # Update failed patterns for future skipping
                    update_failed_patterns(file_path, error_msg)
                    
            except Exception as e:
                logger.error(f"❌ Error processing {file_path}: {e}")
                upload_results[file_path] = {
                    'success': False,
                    'error': str(e),
                    'gcs_url': None
                }
                audio_task_results[task_id]["failed_uploads"] += 1
                
                # Update failed patterns for future skipping
                update_failed_patterns(file_path, str(e))
        
        # Phase 2: Process transcriptions for successful uploads
        successful_uploads = [result for result in upload_results.values() if result['success'] and result.get('public_url')]
        
        if successful_uploads:
            audio_task_results[task_id]["current_phase"] = "transcribing"
            logger.info(f"🎙️ Phase 2: Processing {len(successful_uploads)} transcriptions")
            
            # Process transcriptions in batches of 5 to avoid rate limits
            batch_size = 5
            total_batches = (len(successful_uploads) + batch_size - 1) // batch_size
            
            for batch_idx in range(0, len(successful_uploads), batch_size):
                batch_uploads = successful_uploads[batch_idx:batch_idx + batch_size]
                batch_num = (batch_idx // batch_size) + 1
                
                logger.info(f"🔄 Processing batch {batch_num}/{total_batches} ({len(batch_uploads)} files)")
                
                # Process batch concurrently
                batch_tasks = []
                for upload_result in batch_uploads:
                    # Use public URL for transcription, but track by GCS URL
                    public_url = upload_result['public_url']
                    gcs_url = upload_result['gcs_url']
                    original_filename = upload_result['original_filename']
                    remote_path = upload_result['remote_path']
                    batch_tasks.append(process_transcription(public_url, gcs_url, original_filename, remote_path, user_id, transcriptions_collection))
                
                # Wait for batch to complete
                batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)
                
                # Update results
                for upload_result, result in zip(batch_uploads, batch_results):
                    gcs_url = upload_result['gcs_url']
                    if isinstance(result, Exception):
                        logger.error(f"❌ Transcription failed for {gcs_url}: {result}")
                        transcription_results[gcs_url] = {
                            'success': False,
                            'error': str(result),
                            'transcript_id': None
                        }
                    else:
                        transcription_results[gcs_url] = result
                        if result['success']:
                            audio_task_results[task_id]["transcriptions_processed"] += 1
                            if result.get('saved_to_db'):
                                audio_task_results[task_id]["transcriptions_saved"] += 1
                
                # Update progress (50% base + 50% for transcriptions)
                transcription_progress = 50 + ((batch_idx + len(batch_uploads)) / len(successful_uploads)) * 50
                audio_task_results[task_id]["progress"] = min(transcription_progress, 99)
                audio_task_results[task_id]["message"] = f"Processed batch {batch_num}/{total_batches} - {audio_task_results[task_id]['transcriptions_processed']}/{len(successful_uploads)} transcriptions completed"
                
                # Small delay between batches to be respectful of API limits
                if batch_num < total_batches:
                    await asyncio.sleep(2)
        
        # Update original response records with results
        updated_records = []
        if "records" in original_response:
            for record in original_response["records"]:
                recording_file = record.get('recordingfile')
                
                # Check if this file already existed in MongoDB
                if recording_file and recording_file in existing_files:
                    existing_data = existing_files[recording_file]
                    record.update({
                        'status': 'existing',
                        'gcs_url': existing_data.get('gcs_url'),
                        'public_url': existing_data.get('public_url'),
                        'original_filename': existing_data.get('original_filename'),
                        'remote_path': existing_data.get('remote_path'),
                        'upload_status': 'existing',
                        'transcription_id': existing_data.get('transcript_id'),
                        'transcription_status': 'existing',
                        'mongodb_id': existing_data.get('mongodb_id'),
                        'speakers_count': existing_data.get('speakers_count', 0),
                        'words_count': existing_data.get('words_count', 0),
                        'audio_duration': existing_data.get('audio_duration'),
                        'lemur_analysis': existing_data.get('lemur_analysis', {}),  # Complete lemur_analysis data
                        'custom_topic': existing_data.get('custom_topic'),
                        'two_party_sentiment': existing_data.get('two_party_sentiment'),
                        'call_completion': existing_data.get('call_completion'),
                        'agent_performance': existing_data.get('agent_performance'),
                        'topic_detection_summary': existing_data.get('topic_detection_summary'),
                        'transcription_text': existing_data.get('text'),
                        'utterances': existing_data.get('utterances', []),
                        'sentiment_analysis': existing_data.get('sentiment_analysis', []),
                        'created_at': existing_data.get('created_at')
                    })
                    
                # Check if this file was newly processed
                elif recording_file and recording_file in upload_results:
                    upload_result = upload_results[recording_file]
                    record.update({
                        'status': 'newly_processed',
                        'gcs_url': upload_result.get('gcs_url'),
                        'public_url': upload_result.get('public_url'),
                        'original_filename': upload_result.get('original_filename'),
                        'remote_path': upload_result.get('remote_path'),
                        'upload_status': 'success' if upload_result['success'] else 'failed'
                    })
                    
                    # Add transcription results (keyed by GCS URL)
                    if upload_result.get('gcs_url') and upload_result.get('gcs_url') in transcription_results:
                        trans_result = transcription_results[upload_result['gcs_url']]
                        record.update({
                            'transcription_id': trans_result.get('transcript_id'),
                            'transcription_status': 'success' if trans_result['success'] else 'failed',
                            'mongodb_id': trans_result.get('mongodb_id'),
                            'speakers_count': trans_result.get('speakers_count', 0),
                            'words_count': trans_result.get('words_count', 0),
                            'custom_topic': trans_result.get('custom_topic'),
                            'two_party_sentiment': trans_result.get('two_party_sentiment'),
                            'call_completion': trans_result.get('call_completion')
                        })
                        if not trans_result['success']:
                            record['transcription_error'] = trans_result.get('error')
                    
                    if not upload_result['success']:
                        record['upload_error'] = upload_result.get('error')
                
                # File not found in either existing or processed
                else:
                    record.update({
                        'status': 'not_processed',
                        'gcs_url': None,
                        'upload_status': None,
                        'transcription_status': None
                    })
                        
                updated_records.append(record)
        
        # Update original response with comprehensive results
        updated_response = original_response.copy()
        if upload_results or existing_files:
            successful_uploads = len([r for r in upload_results.values() if r['success']]) if upload_results else 0
            failed_uploads = len([r for r in upload_results.values() if not r['success']]) if upload_results else 0
            successful_transcriptions = len([r for r in transcription_results.values() if r.get('success')]) if transcription_results else 0
            
            total_files_with_data = len(existing_files) + successful_uploads
            total_transcriptions_available = len(existing_files) + successful_transcriptions
            
            # Update summary metrics
            if "summary" in updated_response:
                updated_response["summary"].update({
                    "files_processed": mongodb_check_result['total_files'],
                    "existing_files": len(existing_files),
                    "new_files_processed": len(file_paths),
                    "successful_uploads": successful_uploads,
                    "failed_uploads": failed_uploads,
                    "upload_success_rate": (successful_uploads / len(file_paths) * 100) if file_paths else 0,
                    "transcriptions_processed": len(transcription_results),
                    "successful_transcriptions": successful_transcriptions,
                    "transcription_success_rate": (successful_transcriptions / len(transcription_results) * 100) if transcription_results else 0,
                    "total_files_with_data": total_files_with_data,
                    "total_transcriptions_available": total_transcriptions_available,
                    "resource_savings_percentage": mongodb_check_result['savings_percentage']
                })
            
            # Add comprehensive summary
            updated_response["processing_summary"] = {
                "total_files_requested": mongodb_check_result['total_files'],
                "existing_files_found": len(existing_files),
                "new_files_processed": len(file_paths),
                "successful_uploads": successful_uploads,
                "failed_uploads": failed_uploads,
                "transcriptions_processed": len(transcription_results),
                "successful_transcriptions": successful_transcriptions,
                "transcriptions_saved_to_db": audio_task_results[task_id]["transcriptions_saved"],
                "total_files_with_data": total_files_with_data,
                "total_transcriptions_available": total_transcriptions_available,
                "resource_savings": f"{mongodb_check_result['savings_percentage']:.1f}% of files were already processed",
                "user_folder": f"users/{user_id}/audio_files/"
            }
        
        # Update records with all results
        updated_response["records"] = updated_records
        
        # Update task with final results
        skipped_count = len([r for r in upload_results.values() if r.get('skipped', False)])
        audio_task_results[task_id] = {
            "status": "completed",
            "user_id": user_id,
            "total_files": mongodb_check_result['total_files'],
            "existing_files": len(existing_files),
            "new_files": len(file_paths),
            "skipped_files": skipped_count,
            "resource_savings_percentage": mongodb_check_result['savings_percentage'],
            "start_time": audio_task_results[task_id]["start_time"],
            "end_time": datetime.now().isoformat(),
            "processed_files": len(file_paths),
            "successful_uploads": audio_task_results[task_id]["successful_uploads"],
            "failed_uploads": audio_task_results[task_id]["failed_uploads"],
            "transcriptions_processed": audio_task_results[task_id]["transcriptions_processed"],
            "transcriptions_saved": audio_task_results[task_id]["transcriptions_saved"],
            "progress": 100,
            "current_phase": "completed",
            "message": f"✅ Processing completed! {len(existing_files)} existing + {audio_task_results[task_id]['successful_uploads']}/{len(file_paths)} new uploads ({skipped_count} skipped), {audio_task_results[task_id]['transcriptions_processed']} transcriptions",
            "results": updated_response,
            "upload_results": upload_results,
            "transcription_results": transcription_results,
            "existing_files_data": existing_files,
            "skip_settings": {
                "skip_failed_uploads_enabled": skip_failed_uploads,
                "failed_patterns": list(failed_file_patterns),
                "failed_extensions": list(failed_file_extensions)
            }
        }
        
        # Close MongoDB connection
        mongo_client.close()
        
        # Log final summary
        logger.info(f"🎯 Background audio processing completed for task {task_id}!")
        logger.info(f"📊 Total files requested: {mongodb_check_result['total_files']}")
        logger.info(f"♻️  Files already processed: {len(existing_files)} ({mongodb_check_result['savings_percentage']:.1f}% savings)")
        logger.info(f"🔄 New files processed: {len(file_paths)}")
        logger.info(f"✅ Successful uploads: {audio_task_results[task_id]['successful_uploads']}")
        logger.info(f"❌ Failed uploads: {audio_task_results[task_id]['failed_uploads']}")
        if skipped_count > 0:
            logger.info(f"⏭️ Skipped uploads: {skipped_count} (matching failure patterns)")
        logger.info(f"🎙️ Transcriptions processed: {audio_task_results[task_id]['transcriptions_processed']}")
        logger.info(f"💾 Transcriptions saved to DB: {audio_task_results[task_id]['transcriptions_saved']}")
        logger.info(f"💰 Resource savings: {mongodb_check_result['savings_percentage']:.1f}% of processing avoided!")
        
    except Exception as e:
        logger.error(f"❌ Background audio processing task {task_id} failed: {e}")
        audio_task_results[task_id] = {
            "status": "failed",
            "user_id": user_id,
            "error": str(e),
            "end_time": datetime.now().isoformat(),
            "message": f"❌ Task failed: {str(e)}"
        }

async def process_transcription(public_url: str, gcs_url: str, original_filename: str, remote_path: str, user_id: str, collection) -> Dict[str, Any]:
    """Process a single audio file transcription with AssemblyAI using public URL"""
    try:
        logger.info(f"🎙️ Starting transcription for: {public_url}")
        logger.info(f"📍 GCS location: {gcs_url}")

        # Configure AssemblyAI with speaker labels
        config = aai.TranscriptionConfig(
            speaker_labels=True,
            # iab_categories=True,
            sentiment_analysis=True,
        )
        
        # Create transcriber and submit job using public URL
        transcriber = aai.Transcriber()
        transcript = transcriber.transcribe(public_url, config)
        
        if transcript.status == aai.TranscriptStatus.error:
            logger.error(f"❌ Transcription failed: {transcript.error}")
            return {
                'success': False,
                'error': transcript.error,
                'transcript_id': None,
                'saved_to_db': False
            }
        
        # Process LeMUR-style analysis using OpenAI instead
        openai_results = {}
        try:
            # Initialize OpenAI client with Langfuse tracing
            client = get_openai_client_with_tracing()
            
            # Define custom tag categories
            tag_list = {
                'Account Issues': 'Problems related to user accounts, such as login difficulties or account access.',
                'Technical Support': 'Inquiries regarding software or hardware functionality and troubleshooting.',
                'Billing and Payments': 'Questions or problems about invoices, payments, or subscription plans.',
                'Product Inquiry': 'Requests for information about product features, capabilities, or availability.',
                'Service Disruption': 'Reports of outages or interruptions in service performance or availability.'
            }
            
            # Check if we have multiple speakers for sentiment analysis
            speakers_in_transcript = set()
            if transcript.utterances:
                speakers_in_transcript = set(utterance.speaker for utterance in transcript.utterances)
            
            # Build enhanced prompt based on speaker count
            if len(speakers_in_transcript) >= 2:
                # Multi-party conversation - include sentiment analysis
                enhanced_prompt = f"""
You are a helpful assistant designed to analyze call center conversations with focus on agent performance, customer experience, and call completion quality.

TRANSCRIPT TO ANALYZE:
{transcript.text}

TASK 1 - TOPIC CLASSIFICATION:
I will give you a list of topics and definitions. Select the most relevant topic from the list.

{tag_list}

TASK 2 - AGENT & CUSTOMER IDENTIFICATION + PERFORMANCE ANALYSIS:
Since this is a call center conversation, identify who is the CALL CENTER AGENT and who is the CUSTOMER:
- Analyze the conversation flow and speaking patterns
- The agent typically greets first, asks how they can help, follows procedures
- The customer typically has a problem, question, or request

Then evaluate the AGENT'S PERFORMANCE using the HEAT model:
- H (HALT): Did the agent listen actively and acknowledge the customer's concerns?
- E (EMPATHY): Did the agent show understanding and empathy toward the customer?
- A (APOLOGIZE): Did the agent apologize when appropriate for issues or inconvenience?
- T (TAKE ACTION): Did the agent take concrete steps to resolve the customer's issue or was the agent's response helpful?

Rate each HEAT component with both:
- Categorical: EXCELLENT, GOOD, FAIR, or POOR
- Numeric: Score from 1-10 (1=Very Poor, 3=Poor, 5=Fair, 7=Good, 9=Excellent, 10=Outstanding)

Rate the agent's overall performance as: EXCELLENT, GOOD, FAIR, or POOR
Calculate overall numeric score as average of all HEAT scores (1-10 scale)

TASK 3 - SENTIMENT ANALYSIS:
Analyze the sentiment of both parties:
- CUSTOMER SENTIMENT: How did the customer feel throughout the conversation?
- AGENT SENTIMENT: How professional and positive was the agent's demeanor?
- Rate each as: POSITIVE, NEUTRAL, or NEGATIVE
- Provide explanations for each sentiment rating

TASK 4 - CALL COMPLETION ANALYSIS:
Analyze if this call ended naturally or was truncated/incomplete:
- COMPLETE: Natural conversation flow with proper greeting, issue resolution, and closure
- TRUNCATED: Conversation ended abruptly, technical issues, unresolved customer issue, sudden disconnection
- Look for signs like: repeated "hello?", "can you hear me?", mid-sentence cutoffs, no proper closure, unresolved customer concerns

Return your response in this exact JSON format:
{{
    "topic": "selected_topic_name",
    "agent_performance": {{
        "agent_identified": "Speaker A/Speaker B/Unknown",
        "customer_identified": "Speaker A/Speaker B/Unknown", 
        "heat_model_analysis": {{
            "halt_score": "EXCELLENT/GOOD/FAIR/POOR",
            "halt_numeric": "1-10 numeric score",
            "empathy_score": "EXCELLENT/GOOD/FAIR/POOR",
            "empathy_numeric": "1-10 numeric score",
            "apologize_score": "EXCELLENT/GOOD/FAIR/POOR or N/A",
            "apologize_numeric": "1-10 numeric score or N/A",
            "take_action_score": "EXCELLENT/GOOD/FAIR/POOR",
            "take_action_numeric": "1-10 numeric score"
        }},
        "overall_performance": "EXCELLENT/GOOD/FAIR/POOR",
        "overall_numeric_score": "1-10 average of all HEAT scores",
        "performance_explanation": "detailed explanation of agent performance"
    }},
    "sentiment_analysis": {{
        "customer_sentiment": "POSITIVE/NEUTRAL/NEGATIVE",
        "customer_explanation": "brief explanation of customer's emotional state",
        "agent_sentiment": "POSITIVE/NEUTRAL/NEGATIVE",
        "agent_explanation": "brief explanation of agent's professional demeanor"
    }},
    "call_completion": {{
        "status": "COMPLETE/TRUNCATED",
        "explanation": "brief explanation of why the call is considered complete or truncated"
    }}
}}
"""
            else:
                # Single speaker - only topic classification and call completion
                enhanced_prompt = f"""
You are a helpful assistant designed to analyze audio content with topic tags and call completion quality.

TRANSCRIPT TO ANALYZE:
{transcript.text}

TASK 1 - TOPIC CLASSIFICATION:
I will give you a list of topics and definitions. Select the most relevant topic from the list.

{tag_list}

TASK 2 - CALL COMPLETION ANALYSIS:
Analyze if this call ended naturally or was truncated/incomplete:
- COMPLETE: Natural flow with proper beginning and ending
- TRUNCATED: Abrupt ending, technical issues, incomplete message, sudden disconnection
- Look for signs like: mid-sentence cutoffs, incomplete thoughts, no proper closure

Return your response in this exact JSON format:
{{
    "topic": "selected_topic_name",
    "sentiment_analysis": null,
    "call_completion": {{
        "status": "COMPLETE/TRUNCATED",
        "explanation": "brief explanation of why the call is considered complete or truncated"
    }}
}}
"""
            
            # Prepare metadata for Langfuse tracing
            trace_metadata = {
                "session_name": "ai-analysis",
                "user_id": user_id,
                "transcript_id": transcript.id,
                "audio_duration": transcript.audio_duration,
                "speakers_count": len(speakers_in_transcript),
                "words_count": len(transcript.text.split()) if transcript.text else 0,
                "original_filename": original_filename,
                "analysis_type": "multi_party" if len(speakers_in_transcript) >= 2 else "single_party"
            }
            
            # Call OpenAI API with Langfuse tracing
            if LANGFUSE_AVAILABLE:
                try:
                    # Create Langfuse client for tracing
                    langfuse_client = Langfuse()
                    
                    # Create Langfuse trace for this analysis session
                    trace = langfuse_client.trace(
                        name="ai-analysis",
                        metadata=trace_metadata,
                        tags=["audio-transcription", "sentiment-analysis", "topic-classification", "call-completion"]
                    )
                    
                    # Create generation within the trace
                    generation = trace.generation(
                        name="openai-gpt-analysis",
                        model="gpt-3.5-turbo",
                        input=[
                            {"role": "system", "content": "You are a helpful assistant that analyzes transcripts and returns JSON responses."},
                            {"role": "user", "content": enhanced_prompt}
                        ],
                        metadata={
                            "temperature": 0.1,
                            "max_tokens": 500,
                            "prompt_type": "multi_party" if len(speakers_in_transcript) >= 2 else "single_party",
                            "transcript_length": len(transcript.text) if transcript.text else 0
                        }
                    )
                    
                    response = client.chat.completions.create(
                        model="gpt-3.5-turbo",
                        messages=[
                            {"role": "system", "content": "You are a helpful assistant that analyzes transcripts and returns JSON responses."},
                            {"role": "user", "content": enhanced_prompt}
                        ],
                        temperature=0.1,
                        max_tokens=500
                    )
                    
                    # Update generation with response
                    generation.end(
                        output=response.choices[0].message.content,
                        usage={
                            "input": response.usage.prompt_tokens,
                            "output": response.usage.completion_tokens,
                            "total": response.usage.total_tokens
                        }
                    )
                    
                    logger.info(f"📊 Langfuse trace created for session 'ai-analysis' - Trace ID: {trace.id}")
                    
                except Exception as langfuse_error:
                    logger.error(f"⚠️ Langfuse tracing failed: {langfuse_error} - continuing with standard OpenAI call")
                    # Fallback to standard OpenAI call if Langfuse fails
                    response = client.chat.completions.create(
                        model="gpt-3.5-turbo",
                        messages=[
                            {"role": "system", "content": "You are a helpful assistant that analyzes transcripts and returns JSON responses."},
                            {"role": "user", "content": enhanced_prompt}
                        ],
                        temperature=0.1,
                        max_tokens=500
                    )
            
            else:
                # Standard OpenAI call without tracing
                response = client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant that analyzes transcripts and returns JSON responses."},
                        {"role": "user", "content": enhanced_prompt}
                    ],
                    temperature=0.1,
                    max_tokens=500
                )
            
            openai_response_text = response.choices[0].message.content.strip()
            
            # Parse OpenAI response
            try:
                import json
                openai_results = json.loads(openai_response_text)
                logger.info(f"🤖 OpenAI Analysis - Topic: {openai_results.get('topic', 'Unknown')}")
                if openai_results.get('agent_performance'):
                    agent_perf = openai_results['agent_performance']
                    heat_analysis = agent_perf.get('heat_model_analysis', {})
                    logger.info(f"👨‍💼 Agent Performance: {agent_perf.get('overall_performance', 'Unknown')} (Score: {agent_perf.get('overall_numeric_score', 'N/A')}/10)")
                    logger.info(f"🔥 HEAT Model Scores: H:{heat_analysis.get('halt_score', 'N/A')}({heat_analysis.get('halt_numeric', 'N/A')}) E:{heat_analysis.get('empathy_score', 'N/A')}({heat_analysis.get('empathy_numeric', 'N/A')}) A:{heat_analysis.get('apologize_score', 'N/A')}({heat_analysis.get('apologize_numeric', 'N/A')}) T:{heat_analysis.get('take_action_score', 'N/A')}({heat_analysis.get('take_action_numeric', 'N/A')})")
                if openai_results.get('sentiment_analysis'):
                    sentiment = openai_results['sentiment_analysis']
                    logger.info(f"😊 Customer Sentiment: {sentiment.get('customer_sentiment', 'N/A')}")
                    logger.info(f"🎯 Agent Sentiment: {sentiment.get('agent_sentiment', 'N/A')}")
                if openai_results.get('call_completion'):
                    logger.info(f"📞 Call Completion: {openai_results['call_completion'].get('status', 'Unknown')} - {openai_results['call_completion'].get('explanation', 'No explanation')}")
                if LANGFUSE_AVAILABLE:
                    logger.info(f"🔍 Analysis traced in Langfuse session: ai-analysis")
            except json.JSONDecodeError:
                logger.warning(f"⚠️ Could not parse OpenAI response as JSON: {openai_response_text}")
                openai_results = {
                    'topic': openai_response_text,
                    'agent_performance': None,
                    'sentiment_analysis': None,
                    'call_completion': None,
                    'raw_response': openai_response_text
                }
                
        except Exception as e:
            logger.error(f"❌ OpenAI analysis failed: {e}")
            openai_results = {
                'error': str(e),
                'topic': None,
                'agent_performance': None,
                'sentiment_analysis': None,
                'call_completion': None
            }
        
        # Process utterances
        utterances = []
        if transcript.utterances:
            for utterance in transcript.utterances:
                utterances.append({
                    'speaker': utterance.speaker,
                    'text': utterance.text,
                    'start': utterance.start,
                    'end': utterance.end,
                    'confidence': getattr(utterance, 'confidence', None)
                })
                logger.info(f"Speaker {utterance.speaker}: {utterance.text}")
        
        # Process sentiment analysis results
        sentiment_results = []
        if hasattr(transcript, 'sentiment_analysis') and transcript.sentiment_analysis:
            for sentiment_result in transcript.sentiment_analysis:
                sentiment_results.append({
                    'text': sentiment_result.text,
                    'start': sentiment_result.start,
                    'end': sentiment_result.end,
                    'sentiment': sentiment_result.sentiment,  # POSITIVE, NEUTRAL, or NEGATIVE
                    'confidence': sentiment_result.confidence,
                    'speaker': getattr(sentiment_result, 'speaker', None)
                })
        
        # Process topic detection results
        topic_results = []
        topic_summary = {}

        # Prepare document for MongoDB
        transcription_doc = {
            'transcript_id': transcript.id,
            'user_id': user_id,
            'gcs_url': gcs_url,  # Store GCS URL for reference
            'public_url': public_url,  # Store public URL used for transcription
            'original_filename': original_filename,  # Original filename from SFTP
            'remote_path': remote_path,  # Full remote path on SFTP server
            'status': str(transcript.status),
            'text': transcript.text,
            'utterances': utterances,  # Speaker utterances with timing
            'sentiment_analysis': sentiment_results,  # AssemblyAI sentiment analysis results
            'topic_detection': {
                'results': topic_results,  # Empty when IAB categories disabled
                'summary': topic_summary,  # Empty when IAB categories disabled
                'iab_categories_enabled': False  # Flag to indicate IAB categories are disabled
            },
            'lemur_analysis': {  # Custom LeMUR analysis
                'custom_topic': openai_results.get('topic'),
                'agent_performance': openai_results.get('agent_performance'),
                'sentiment_analysis': openai_results.get('sentiment_analysis'),  # Now contains customer_sentiment and agent_sentiment
                'call_completion': openai_results.get('call_completion'),
                'error': openai_results.get('error'),
                'raw_response': openai_results.get('raw_response')
            },
            'audio_duration': transcript.audio_duration,
            'language_model': getattr(transcript, 'language_model', None),
            'acoustic_model': getattr(transcript, 'acoustic_model', None),
            'created_at': datetime.now(),
            'confidence': getattr(transcript, 'confidence', None),
            'words_count': len(transcript.text.split()) if transcript.text else 0,
            'speakers_count': len(set(u['speaker'] for u in utterances)) if utterances else 0,
            'sentiments_count': len(sentiment_results),
            'topics_detected': len(topic_results)  # Will be 0 when IAB categories disabled
        }
        
        # Save to MongoDB
        result = await collection.insert_one(transcription_doc)
        
        logger.info(f"✅ Transcription completed and saved: {transcript.id}")
        logger.info(f"💾 MongoDB document ID: {result.inserted_id}")
        logger.info(f"🎤 Speakers detected: {transcription_doc['speakers_count']}")
        logger.info(f"📝 Words transcribed: {transcription_doc['words_count']}")
        logger.info(f"🏷️ IAB Categories: Disabled (using OpenAI custom topics instead)")
        
        return {
            'success': True,
            'transcript_id': transcript.id,
            'mongodb_id': str(result.inserted_id),
            'saved_to_db': True,
            'speakers_count': transcription_doc['speakers_count'],
            'words_count': transcription_doc['words_count'],
            'audio_duration': transcript.audio_duration,
            'public_url': public_url,
            'gcs_url': gcs_url,
            'original_filename': original_filename,
            'remote_path': remote_path,
            'custom_topic': openai_results.get('topic'),
            'two_party_sentiment': openai_results.get('sentiment_analysis'),
            'call_completion': openai_results.get('call_completion')
        }
        
    except Exception as e:
        logger.error(f"❌ Error processing transcription for {public_url}: {e}")
        return {
            'success': False,
            'error': str(e),
            'transcript_id': None,
            'saved_to_db': False
        }

async def process_single_audio_file(
    remote_path: str, 
    user_id: str, 
    sftp_host: str, 
    sftp_port: int, 
    sftp_username: str, 
    sftp_password: str, 
    gcs_client, 
    bucket_name: str
) -> Dict[str, Any]:
    """Process a single audio file: download from SFTP and upload to GCS"""
    start_time = datetime.now()
    result = {
        "remote_path": remote_path,
        "original_filename": os.path.basename(remote_path),
        "user_id": user_id,
        "success": False,
        "gcs_url": None,
        "public_url": None,
        "error": None,
        "processing_time": None,
        "file_size": 0
    }
    
    sftp = None
    ssh = None
    
    try:
        logger.info(f"🚀 Processing audio file for user {user_id}: {remote_path}")
        
        # Create SFTP connection
        ssh = paramiko.SSHClient()
        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        
        ssh.connect(
            sftp_host,
            port=sftp_port,
            username=sftp_username,
            password=sftp_password,
            timeout=30,
            banner_timeout=30
        )
        
        sftp = ssh.open_sftp()
        logger.info("🔗 SFTP connection established")
        
        # Download file from SFTP
        buffer = await download_file_from_sftp(sftp, remote_path)
        if not buffer:
            result["error"] = "Failed to download from SFTP"
            return result
        
        result["file_size"] = len(buffer.getvalue())
        
        # Extract filename
        filename = os.path.basename(remote_path)
        
        # Upload to GCS
        gcs_result = await upload_to_gcs(buffer, user_id, filename, gcs_client, bucket_name)
        if not gcs_result:
            result["error"] = "Failed to upload to GCS"
            return result
        
        result["success"] = True
        result["gcs_url"] = gcs_result["gcs_url"]
        result["public_url"] = gcs_result["public_url"]
        
        processing_time = (datetime.now() - start_time).total_seconds()
        result["processing_time"] = processing_time
        
        logger.info(f"🎉 Successfully processed {filename} for user {user_id} in {processing_time:.2f}s")
        
    except Exception as e:
        result["error"] = str(e)
        logger.error(f"❌ Error processing {remote_path}: {e}")
        
    finally:
        # Clean up connections
        if sftp:
            sftp.close()
        if ssh:
            ssh.close()
    
    return result

async def download_file_from_sftp(sftp, remote_path: str) -> Optional[BytesIO]:
    """Download file from SFTP to memory buffer"""
    try:
        logger.info(f"📥 Downloading file: {remote_path}")
        
        # Check if file exists
        try:
            sftp.stat(remote_path)
        except FileNotFoundError:
            logger.error(f"❌ File not found: {remote_path}")
            return None
        
        # Download to memory buffer
        buffer = BytesIO()
        sftp.getfo(remote_path, buffer)
        buffer.seek(0)
        
        file_size = len(buffer.getvalue())
        logger.info(f"✅ Downloaded {remote_path} ({file_size:,} bytes)")
        
        return buffer
        
    except Exception as e:
        logger.error(f"❌ Error downloading {remote_path}: {e}")
        return None

async def upload_to_gcs(
    buffer: BytesIO, 
    user_id: str, 
    filename: str, 
    gcs_client, 
    bucket_name: str
) -> Optional[Dict[str, str]]:
    """Upload file buffer to Google Cloud Storage organized by user_id"""
    try:
        # Get or create bucket
        bucket = gcs_client.bucket(bucket_name)
        
        # Check if bucket exists, create if it doesn't
        try:
            if not bucket.exists():
                logger.info(f"📦 Creating bucket: {bucket_name}")
                bucket = gcs_client.create_bucket(bucket_name)
                logger.info(f"✅ Created bucket: {bucket_name}")
            else:
                logger.info(f"📦 Using existing bucket: {bucket_name}")
        except Exception as e:
            logger.error(f"❌ Error with bucket operations: {e}")
            return None
        
        # Create user-specific path
        blob_name = f"users/{user_id}/audio_files/{filename}"
        blob = bucket.blob(blob_name)
        
        # Upload the file
        buffer.seek(0)
        blob.upload_from_file(buffer, content_type='audio/wav')
        
        # Generate both GCS URI and public HTTP URL
        gcs_url = f"gs://{bucket_name}/{blob_name}"
        public_url = f"https://storage.googleapis.com/{bucket_name}/{blob_name}"
        
        logger.info(f"☁️ Uploaded to GCS: {gcs_url}")
        logger.info(f"🌐 Public URL: {public_url}")
        
        return {
            "gcs_url": gcs_url,
            "public_url": public_url
        }
        
    except Exception as e:
        logger.error(f"❌ Error uploading to GCS: {e}")
        return None

@router.get("/logs", response_model=Dict[str, Any])
async def get_call_logs(
    start_date: str = Query(..., description="Start date in YYYY-MM-DD format"),
    end_date: str = Query(..., description="End date in YYYY-MM-DD format"),
    src: Optional[str] = Query(None, description="Filter by source phone number"),
    dst: Optional[str] = Query(None, description="Filter by destination phone number"),
    disposition: Optional[str] = Query(None, description="Filter by call disposition (ANSWERED, NO ANSWER, BUSY, FAILED)"),
    direction: Optional[str] = Query(None, description="Filter by call direction (inbound, outbound, internal)"),
    min_duration: Optional[int] = Query(None, description="Minimum call duration in seconds"),
    max_duration: Optional[int] = Query(None, description="Maximum call duration in seconds"),
    has_recording: Optional[bool] = Query(None, description="Filter for calls with recordings"),
    did: Optional[str] = Query(None, description="Filter by DID number"),
    extension: Optional[str] = Query(None, description="Filter by extension (destination ≤ 5 digits)"),
    cnam: Optional[str] = Query(None, description="Filter by caller name"),
    queue: Optional[str] = Query(None, description="Filter by queue number (e.g., 11111, 11112)"),
    unique_callers_only: bool = Query(False, description="Return only unique callers (distinct src)"),
    include_details: bool = Query(True, description="Include detailed call records in response"),
    upload_to_gcs: bool = Query(False, description="🎵 Download audio files and upload to Google Cloud Storage (background task)"),
    skip_failed_uploads: bool = Query(True, description="⏭️ Skip files that match previous failure patterns to avoid repeated failures"),
    limit: int = Query(1000000, description="Maximum number of records to return"),
    offset: int = Query(0, description="Number of records to skip"),
    sort_by: str = Query("calldate", description="Field to sort by (calldate, duration, billsec)"),
    sort_order: str = Query("desc", description="Sort order (asc or desc)"),
    background_tasks: BackgroundTasks = None,
    current_user: dict = Depends(get_current_user),
    db = Depends(get_database)
):
    """
    Get detailed call logs with extensive filtering options and explicit recording file information.
    
    🎵 NEW: Now supports automatic audio file processing to Google Cloud Storage via background tasks!
    Set upload_to_gcs=true to start a background task that downloads audio files from SFTP and uploads them to GCS.
    
    When upload_to_gcs=true:
    - Returns immediately with a task_id
    - Audio files are processed in the background
    - Use the /audio-task-status/{task_id} endpoint to check progress
    - Final results include GCS URLs for all processed files
    
    This endpoint provides comprehensive call records with multiple filter options.
    Users can only access records for their own company.
    Recording files are explicitly included in the response.
    """
    try:
        # 🛑 TEMPORARY DISABLE: Audio processing temporarily disabled
        if upload_to_gcs:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="🛑 Audio processing is temporarily disabled for maintenance"
            )
        
        # Parse dates
        try:
            start_dt = datetime.strptime(start_date, "%Y-%m-%d")
            
            # If end_date is provided as just a date, set time to end of day
            end_dt = datetime.strptime(end_date, "%Y-%m-%d")
            end_dt = end_dt.replace(hour=23, minute=59, second=59)
            
            # Validate date range
            date_diff = (end_dt - start_dt).days
            if date_diff < 0:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="End date must be after start date"
                )
                
        except ValueError:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid date format. Use YYYY-MM-DD"
            )
        
        # Validate sort parameters
        valid_sort_fields = ["calldate", "duration", "billsec", "src", "dst"]
        if sort_by not in valid_sort_fields:
            sort_by = "calldate"  # Default to calldate if invalid
        
        sort_order = sort_order.lower()
        if sort_order not in ["asc", "desc"]:
            sort_order = "desc"  # Default to descending if invalid
        
        # Normalize direction parameter for case-insensitive comparison
        direction_filter = None
        if direction:
            direction_lower = direction.lower()
            # Map to valid direction constants
            if direction_lower == "inbound":
                direction_filter = "inbound"
            elif direction_lower == "outbound":
                direction_filter = "outbound"
            elif direction_lower == "internal":
                direction_filter = "internal"
            elif direction_lower == "unknown":
                direction_filter = "unknown"
        
        # Get the company_id from the current user
        company_id = current_user.get("company_id")
        if not company_id or company_id == "None":
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="User is not associated with any company"
            )
        
        # Get user_id for GCS upload organization
        user_id = current_user.get("_id") or current_user.get("id")
        
        # Get database connection details
        connection_details = await get_company_db_connection(company_id, current_user, db)
        
        # Connect to the database
        conn = None
        try:
            conn = pymysql.connect(
                host=connection_details["host"],
                user=connection_details["user"],
                password=connection_details["password"],
                database=connection_details["database"],
                port=connection_details["port"],
                connect_timeout=10,
                cursorclass=pymysql.cursors.DictCursor
            )
            
            with conn.cursor() as cursor:
                # Build the base WHERE clause
                base_where = """
                    WHERE calldate BETWEEN %s AND %s
                    AND (dstchannel IS NOT NULL AND dstchannel != '' AND dstchannel != 'N/A')
                    AND (dst IS NOT NULL AND dst != '' AND dst != 'N/A')
                """
                params = [start_dt, end_dt]
                
                # Add optional filters to WHERE clause
                additional_filters = ""
                
                if disposition:
                    additional_filters += " AND disposition = %s"
                    params.append(disposition)
                    
                if src:
                    additional_filters += " AND src LIKE %s"
                    params.append(f"%{src}%")
                    
                if dst:
                    additional_filters += " AND dst LIKE %s"
                    params.append(f"%{dst}%")
                
                if has_recording is not None:
                    if has_recording:
                        additional_filters += " AND recordingfile != '' AND recordingfile IS NOT NULL"
                    else:
                        additional_filters += " AND (recordingfile = '' OR recordingfile IS NULL)"
                
                if min_duration is not None:
                    additional_filters += " AND duration >= %s"
                    params.append(min_duration)
                
                if max_duration is not None:
                    additional_filters += " AND duration <= %s"
                    params.append(max_duration)
                
                if did:
                    additional_filters += " AND did LIKE %s"
                    params.append(f"%{did}%")
                
                if extension:
                    additional_filters += " AND dst = %s AND LENGTH(dst) <= 5"
                    params.append(extension)
                
                if cnam:
                    additional_filters += " AND (cnam LIKE %s OR cnum LIKE %s)"
                    params.append(f"%{cnam}%")
                    params.append(f"%{cnam}%")
                
                if queue:
                    additional_filters += " AND dst = %s AND dcontext = 'ext-queues'"
                    params.append(queue)
                
                # Complete WHERE clause
                complete_where = base_where + additional_filters
                
                # Build count query
                count_query = f"SELECT COUNT(*) as total FROM cdr {complete_where}"
                
                # Build main query with explicit field selection
                main_query = f"""
                    SELECT recordingfile
                    FROM cdr {complete_where}
                """
                
                # Handle unique callers request
                if unique_callers_only:
                    main_query = f"""
                        SELECT DISTINCT recordingfile
                        FROM cdr {complete_where}
                        GROUP BY recordingfile
                    """
                
                # Add sorting and pagination to the main query
                main_query += f" ORDER BY {sort_by} {sort_order} LIMIT %s OFFSET %s"
                main_params = params + [limit, offset]
                
                # Execute count query first
                try:
                    cursor.execute(count_query, params)
                    count_result = cursor.fetchone()
                    total_count = count_result.get("total", 0) if count_result else 0
                except Exception as e:
                    logger.error(f"Count query error: {str(e)}")
                    total_count = 0
                
                # Execute main query
                cursor.execute(main_query, main_params)
                records = cursor.fetchall()
                
                # Process records and collect audio files
                processed_records = []
                audio_files_to_process = []
                
                for record in records:
                    if isinstance(record, dict):  # Ensure record is a dictionary
                        # Process recordingfile - ensure it's properly formatted
                        recording_file = record.get('recordingfile', '')
                        
                        if recording_file and recording_file.strip():
                            processed_record = {
                                'recordingfile': recording_file,
                                'has_recording': True,
                                'gcs_url': None,  # Will be populated by background task
                                'upload_status': 'pending' if upload_to_gcs else None
                            }
                            
                            # Collect audio files for processing if requested
                            if upload_to_gcs:
                                audio_files_to_process.append(recording_file)
                                
                        else:
                            processed_record = {
                                'recordingfile': None,
                                'has_recording': False,
                                'gcs_url': None,
                                'upload_status': None
                            }
                        
                        processed_records.append(processed_record)
                
                # Get summary metrics
                summary_metrics = {}
                
                # Only calculate summary if we have records
                if processed_records:
                    # Basic counts
                    total_records = len(processed_records)
                    
                    # Recording metrics
                    has_recording_count = sum(1 for r in processed_records if r.get('has_recording', False))
                    recording_percentage = (has_recording_count / total_records * 100) if total_records > 0 else 0
                    
                    summary_metrics = {
                        "total_records": total_records,
                        "recording_count": has_recording_count,
                        "recording_percentage": round(recording_percentage, 2),
                        "no_recording_count": total_records - has_recording_count,
                    }
                
                # Prepare time period info
                time_period = TimePeriod(
                    start_date=start_date,
                    end_date=end_date,
                    total_days=date_diff + 1
                )
                
                # Prepare response
                response = {
                    "time_period": time_period.dict(),
                    "summary": summary_metrics,
                    "total_count": total_count,
                    "filtered_count": len(processed_records),
                    "gcs_upload_enabled": upload_to_gcs,
                }
                
                # Only include records if requested
                if include_details:
                    response["records"] = processed_records
                
                # 🎵 Start background task for audio processing if requested
                if upload_to_gcs and audio_files_to_process and background_tasks:
                    # Check dependencies first
                    missing_deps = check_audio_dependencies()
                    if missing_deps:
                        error_msg = f"❌ Missing required dependencies for audio processing: {', '.join(missing_deps)}. Please install with: pip install {' '.join(missing_deps)}"
                        logger.error(error_msg)
                        response["audio_processing"] = {
                            "status": "error",
                            "message": error_msg,
                            "total_files": len(audio_files_to_process),
                            "required_dependencies": missing_deps
                        }
                        return response
                    
                    # Check MongoDB for existing files first
                    logger.info(f"🔍 Checking MongoDB for existing files before processing...")
                    mongodb_check = await check_existing_files_in_mongodb(audio_files_to_process, user_id)
                    
                    # Update response with existing files data immediately
                    if mongodb_check['existing_files']:
                        logger.info(f"⚡ Found {len(mongodb_check['existing_files'])} existing files - returning data immediately")
                        
                        # Update records with existing data
                        for record in response.get("records", []):
                            recording_file = record.get('recordingfile')
                            if recording_file and recording_file in mongodb_check['existing_files']:
                                existing_data = mongodb_check['existing_files'][recording_file]
                                record.update({
                                    'status': 'existing',
                                    'gcs_url': existing_data.get('gcs_url'),
                                    'public_url': existing_data.get('public_url'),
                                    'original_filename': existing_data.get('original_filename'),
                                    'remote_path': existing_data.get('remote_path'),
                                    'upload_status': 'existing',
                                    'transcription_id': existing_data.get('transcript_id'),
                                    'transcription_status': 'existing',
                                    'mongodb_id': existing_data.get('mongodb_id'),
                                    'speakers_count': existing_data.get('speakers_count', 0),
                                    'words_count': existing_data.get('words_count', 0),
                                    'audio_duration': existing_data.get('audio_duration'),
                                    'lemur_analysis': existing_data.get('lemur_analysis', {}),  # Complete lemur_analysis data
                                    'custom_topic': existing_data.get('custom_topic'),
                                    'two_party_sentiment': existing_data.get('two_party_sentiment'),
                                    'call_completion': existing_data.get('call_completion'),
                                    'agent_performance': existing_data.get('agent_performance'),
                                    'topic_detection_summary': existing_data.get('topic_detection_summary'),
                                    'transcription_text': existing_data.get('text'),
                                    'utterances': existing_data.get('utterances', []),
                                    'sentiment_analysis': existing_data.get('sentiment_analysis', []),
                                    'created_at': existing_data.get('created_at')
                                })
                        
                        # Update summary with immediate results
                        if "summary" in response:
                            response["summary"].update({
                                "existing_files_found": len(mongodb_check['existing_files']),
                                "new_files_to_process": len(mongodb_check['new_files']),
                                "resource_savings_percentage": mongodb_check['savings_percentage'],
                                "immediate_results_available": len(mongodb_check['existing_files'])
                            })
                    
                    # Only start background task if there are new files to process
                    if mongodb_check['new_files']:
                        task_id = str(uuid.uuid4())
                        
                        logger.info(f"🎵 Starting background processing for {len(mongodb_check['new_files'])} new files")
                        logger.info(f"👤 User ID: {user_id}")
                        logger.info(f"🆔 Task ID: {task_id}")
                        
                        # Start background task with MongoDB check results
                        background_tasks.add_task(
                            process_audio_files_background_task,
                            mongodb_check,
                            user_id,
                            task_id,
                            response,
                            skip_failed_uploads
                        )
                        
                        # Add task info to response
                        response["audio_processing"] = {
                            "task_id": task_id,
                            "status": "processing",
                            "total_files": len(audio_files_to_process),
                            "existing_files": len(mongodb_check['existing_files']),
                            "new_files": len(mongodb_check['new_files']),
                            "resource_savings": f"{mongodb_check['savings_percentage']:.1f}%",
                            "message": f"🎵 Processing {len(mongodb_check['new_files'])} new files (skipped {len(mongodb_check['existing_files'])} existing)",
                            "check_status_url": f"/ai-insights/audio-task-status/{task_id}"
                        }
                    else:
                        # All files already exist - no background task needed
                        logger.info(f"🎉 All {len(audio_files_to_process)} files already processed - no background task needed!")
                        response["audio_processing"] = {
                            "status": "all_existing",
                            "total_files": len(audio_files_to_process),
                            "existing_files": len(mongodb_check['existing_files']),
                            "new_files": 0,
                            "resource_savings": "100.0%",
                            "message": f"🎉 All {len(audio_files_to_process)} files already processed - data available immediately!"
                        }
                
                elif upload_to_gcs and audio_files_to_process:
                    # Check dependencies first
                    missing_deps = check_audio_dependencies()
                    if missing_deps:
                        error_msg = f"❌ Missing required dependencies for audio processing: {', '.join(missing_deps)}. Please install with: pip install {' '.join(missing_deps)}"
                        response["audio_processing"] = {
                            "status": "error",
                            "message": error_msg,
                            "total_files": len(audio_files_to_process),
                            "required_dependencies": missing_deps
                        }
                    else:
                        # Check MongoDB for existing files even without background tasks
                        logger.info(f"🔍 Checking MongoDB for existing files...")
                        mongodb_check = await check_existing_files_in_mongodb(audio_files_to_process, user_id)
                        
                        # Update response with existing files data
                        if mongodb_check['existing_files']:
                            # Update records with existing data
                            for record in response.get("records", []):
                                recording_file = record.get('recordingfile')
                                if recording_file and recording_file in mongodb_check['existing_files']:
                                    existing_data = mongodb_check['existing_files'][recording_file]
                                    record.update({
                                        'status': 'existing',
                                        'gcs_url': existing_data.get('gcs_url'),
                                        'public_url': existing_data.get('public_url'),
                                        'original_filename': existing_data.get('original_filename'),
                                        'remote_path': existing_data.get('remote_path'),
                                        'upload_status': 'existing',
                                        'transcription_id': existing_data.get('transcript_id'),
                                        'transcription_status': 'existing',
                                        'mongodb_id': existing_data.get('mongodb_id'),
                                        'speakers_count': existing_data.get('speakers_count', 0),
                                        'words_count': existing_data.get('words_count', 0),
                                        'audio_duration': existing_data.get('audio_duration'),
                                        'lemur_analysis': existing_data.get('lemur_analysis', {}),  # Complete lemur_analysis data
                                        'custom_topic': existing_data.get('custom_topic'),
                                        'two_party_sentiment': existing_data.get('two_party_sentiment'),
                                        'call_completion': existing_data.get('call_completion'),
                                        'agent_performance': existing_data.get('agent_performance'),
                                        'topic_detection_summary': existing_data.get('topic_detection_summary'),
                                        'transcription_text': existing_data.get('text'),
                                        'utterances': existing_data.get('utterances', []),
                                        'sentiment_analysis': existing_data.get('sentiment_analysis', []),
                                        'created_at': existing_data.get('created_at')
                                    })
                            
                            # Update summary with existing files
                            if "summary" in response:
                                response["summary"].update({
                                    "existing_files_found": len(mongodb_check['existing_files']),
                                    "new_files_to_process": len(mongodb_check['new_files']),
                                    "resource_savings_percentage": mongodb_check['savings_percentage']
                                })
                        
                        # No background tasks available, inform user
                        response["audio_processing"] = {
                            "status": "unavailable",
                            "message": f"❌ Background task processing not available. Found {len(mongodb_check['existing_files'])} existing files, {len(mongodb_check['new_files'])} would need processing",
                            "total_files": len(audio_files_to_process),
                            "existing_files": len(mongodb_check['existing_files']),
                            "new_files": len(mongodb_check['new_files']),
                            "resource_savings": f"{mongodb_check['savings_percentage']:.1f}%"
                        }
                
                return response
                
        finally:
            if conn:
                conn.close()
                
    except pymysql.MySQLError as e:
        logger.error(f"MySQL Error in AI insights call logs: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Database error: {str(e)}"
        )
    except Exception as e:
        logger.error(f"Error getting call logs: {str(e)}")
        if isinstance(e, HTTPException):
            raise
        else:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to retrieve call logs: {str(e)}"
            )

@router.get("/audio-task-status/{task_id}", response_model=Dict[str, Any])
async def get_audio_task_status(
    task_id: str,
    current_user: dict = Depends(get_current_user)
):
    """
    🎵 Get the status of a background audio processing task from the logs endpoint
    """
    if task_id not in audio_task_results:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="❌ Audio processing task not found"
        )
    
    task_info = audio_task_results[task_id]
    
    # Basic security check - ensure user can only see their own tasks
    user_id = current_user.get("_id") or current_user.get("id")
    if task_info.get("user_id") != user_id and not current_user.get("is_admin", False):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="❌ Access denied"
        )
    
    return {
        "task_id": task_id,
        **task_info
    }

@router.delete("/audio-task-cleanup")
async def cleanup_audio_tasks(
    current_user: dict = Depends(get_current_user)
):
    """
    🧹 Clean up completed audio processing tasks (admin only)
    """
    if not current_user.get("is_admin", False):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="❌ Admin access required"
        )
    
    completed_tasks = [
        task_id for task_id, task_info in audio_task_results.items()
        if task_info.get("status") in ["completed", "failed"]
    ]
    
    for task_id in completed_tasks:
        del audio_task_results[task_id]
    
    logger.info(f"🧹 Cleaned up {len(completed_tasks)} completed audio processing tasks")
    
    return {
        "message": f"🧹 Cleaned up {len(completed_tasks)} completed audio processing tasks",
        "cleaned_tasks": len(completed_tasks),
        "remaining_tasks": len(audio_task_results)
    }

@router.get("/failed-patterns", response_model=Dict[str, Any])
async def get_failed_patterns(
    current_user: dict = Depends(get_current_user)
):
    """
    🚫 Get current failed file patterns and extensions that are being skipped
    """
    return {
        "failed_patterns": list(failed_file_patterns),
        "failed_extensions": list(failed_file_extensions),
        "total_patterns": len(failed_file_patterns),
        "total_extensions": len(failed_file_extensions),
        "description": "Files matching these patterns will be skipped when skip_failed_uploads=true"
    }

@router.delete("/failed-patterns", response_model=Dict[str, Any])
async def clear_failed_patterns(
    current_user: dict = Depends(get_current_user)
):
    """
    🧹 Clear all failed file patterns and extensions (admin only)
    This will reset the skip list, allowing previously failed files to be retried
    """
    if not current_user.get("is_admin", False):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="❌ Admin access required to clear failed patterns"
        )
    
    patterns_count = len(failed_file_patterns)
    extensions_count = len(failed_file_extensions)
    
    failed_file_patterns.clear()
    failed_file_extensions.clear()
    
    logger.info(f"🧹 Admin '{current_user['username']}' cleared {patterns_count} failed patterns and {extensions_count} failed extensions")
    
    return {
        "message": f"🧹 Cleared {patterns_count} failed patterns and {extensions_count} failed extensions",
        "cleared_patterns": patterns_count,
        "cleared_extensions": extensions_count,
        "status": "success"
    }

async def get_transcriptions_for_BI_analysis(recording_files: List[str], user_id: str, company_code: str) -> Dict[str, Dict]:
    """
    Get transcriptions for recording files from MongoDB for BI analysis
    
    Args:
        recording_files: List of recording file paths from CDR
        user_id: User ID for filtering transcripts
        company_code: Company code to use as collection name
    
    Returns:
        Dictionary mapping recording_file to transcript data
    """
    transcripts = {}
    
    if not recording_files or not company_code:
        return transcripts
    
    try:
        mongo_uri = os.getenv("MONGODB_URI", "mongodb://admin:delaphone%4001@102.22.15.141:37037/")
        mongo_client = AsyncIOMotorClient(mongo_uri)
        
        try:
            # Database: delaphone_transcriptions, Collection: company_code
            mongo_db = mongo_client["delaphone_transcriptions"]
            transcriptions_collection = mongo_db[company_code]
            
            logger.info(f"🔍 Searching delaphone_transcriptions.{company_code} for {len(recording_files)} files")
            
            # Extract filenames from paths for matching
            filenames_to_paths = {}
            for recording_file in recording_files:
                if recording_file:
                    filename = os.path.basename(recording_file)
                    if filename:
                        filenames_to_paths[filename] = recording_file
            
            # Query by filename fields
            cursor = transcriptions_collection.find({
                "$or": [
                    {"file_name": {"$in": list(filenames_to_paths.keys())}},
                    {"filename": {"$in": list(filenames_to_paths.keys())}},
                    {"original_filename": {"$in": list(filenames_to_paths.keys())}},
                    {"remote_path": {"$in": recording_files}}
                ]
            })
            
            async for transcript_doc in cursor:
                # Try different filename fields
                filename = (transcript_doc.get("file_name") or 
                          transcript_doc.get("filename") or 
                          transcript_doc.get("original_filename"))
                
                recording_file = None
                if filename and filename in filenames_to_paths:
                    recording_file = filenames_to_paths[filename]
                elif transcript_doc.get("remote_path") in recording_files:
                    recording_file = transcript_doc.get("remote_path")
                
                if recording_file:
                    lemur_analysis = transcript_doc.get('lemur_analysis', {})
                    transcripts[recording_file] = {
                        'text': transcript_doc.get('text'),
                        'transcript_id': transcript_doc.get('transcript_id'),
                        'speakers_count': transcript_doc.get('speakers_count', 0),
                        'words_count': transcript_doc.get('words_count', 0),
                        'audio_duration': transcript_doc.get('audio_duration'),
                        'custom_topic': lemur_analysis.get('custom_topic'),
                        'two_party_sentiment': lemur_analysis.get('sentiment_analysis'),
                        'call_completion': lemur_analysis.get('call_completion'),
                        'agent_performance': lemur_analysis.get('agent_performance'),
                        'utterances': transcript_doc.get('utterances', []),
                        'sentiment_analysis': transcript_doc.get('sentiment_analysis', []),
                        'created_at': transcript_doc.get('created_at')
                    }
                    logger.debug(f"Found transcript for {filename}")
            
            logger.info(f"📊 Found {len(transcripts)} transcripts out of {len(recording_files)} recording files")
            
        finally:
            mongo_client.close()
            
    except Exception as e:
        logger.error(f"❌ Error fetching transcriptions for BI analysis: {str(e)}")
        logger.error(traceback.format_exc())
    
    return transcripts

@router.get("/business-intelligence-analysis", response_model=Dict[str, Any])
async def get_business_intelligence_analysis(
    start_date: str = Query(..., description="Start date in YYYY-MM-DD format"),
    end_date: str = Query(..., description="End date in YYYY-MM-DD format"),
    src: Optional[str] = Query(None, description="Filter by source phone number"),
    dst: Optional[str] = Query(None, description="Filter by destination phone number"),
    disposition: Optional[str] = Query(None, description="Filter by call disposition (ANSWERED, NO ANSWER, BUSY, FAILED)"),
    direction: Optional[str] = Query(None, description="Filter by call direction (inbound, outbound, internal)"),
    min_duration: Optional[int] = Query(None, description="Minimum call duration in seconds"),
    max_duration: Optional[int] = Query(None, description="Maximum call duration in seconds"),
    has_recording: Optional[bool] = Query(None, description="Filter for calls with recordings"),
    did: Optional[str] = Query(None, description="Filter by DID number"),
    extension: Optional[str] = Query(None, description="Filter by extension (destination ≤ 5 digits)"),
    cnam: Optional[str] = Query(None, description="Filter by caller name"),
    queue: Optional[str] = Query(None, description="Filter by queue number (e.g., 11111, 11112)"),
    unique_callers_only: bool = Query(False, description="Return only unique callers (distinct src)"),
    limit: int = Query(1000000, description="Maximum number of records to return"),
    offset: int = Query(0, description="Number of records to skip"),
    sort_by: str = Query("calldate", description="Field to sort by (calldate, duration, billsec)"),
    sort_order: str = Query("desc", description="Sort order (asc or desc)"),
    model: str = Query("gpt-4-turbo", description="LLM model to use for analysis (gpt-4-turbo, gpt-4, gemini-2.0-flash, gemini-1.5-pro)"),
    company_name: str = Query("Shell Club", description="Company name for the analysis"),
    industry: str = Query("Petroleum", description="Industry type for the analysis"),
    current_user: dict = Depends(get_current_user),
    db = Depends(get_database)
):
    """
    Get comprehensive business intelligence analysis of call center data.
    
    This endpoint:
    1. Retrieves call logs data with the same filtering options as the /logs endpoint
    2. Passes the structured call data to LLM for business intelligence analysis
    3. Returns comprehensive BI insights including executive summary, customer experience, 
       agent performance, operational efficiency, risk assessment, and recommendations
    
    The analysis uses advanced AI to generate strategic insights, identify trends,
    detect anomalies, and provide actionable recommendations for call center optimization.
    """
    try:
        # Check if LLM is available
        if not LLM_AVAILABLE:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="❌ LLM analysis service is not available. Please ensure the utils.llm module is properly installed."
            )
        
        # Parse dates
        try:
            start_dt = datetime.strptime(start_date, "%Y-%m-%d")
            
            # If end_date is provided as just a date, set time to end of day
            end_dt = datetime.strptime(end_date, "%Y-%m-%d")
            end_dt = end_dt.replace(hour=23, minute=59, second=59)
            
            # Validate date range
            date_diff = (end_dt - start_dt).days
            if date_diff < 0:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="End date must be after start date"
                )
                
        except ValueError:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid date format. Use YYYY-MM-DD"
            )
        
        # Validate sort parameters
        valid_sort_fields = ["calldate", "duration", "billsec", "src", "dst"]
        if sort_by not in valid_sort_fields:
            sort_by = "calldate"  # Default to calldate if invalid
        
        sort_order = sort_order.lower()
        if sort_order not in ["asc", "desc"]:
            sort_order = "desc"  # Default to descending if invalid
        
        # Normalize direction parameter for case-insensitive comparison
        direction_filter = None
        if direction:
            direction_lower = direction.lower()
            # Map to valid direction constants
            if direction_lower == "inbound":
                direction_filter = "inbound"
            elif direction_lower == "outbound":
                direction_filter = "outbound"
            elif direction_lower == "internal":
                direction_filter = "internal"
            elif direction_lower == "unknown":
                direction_filter = "unknown"
        
        # Get the company_id from the current user
        company_id = current_user.get("company_id")
        if not company_id or company_id == "None":
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="User is not associated with any company"
            )
        
        # Get user_id for tracing
        user_id = current_user.get("_id") or current_user.get("id")
        
        # Get database connection details
        connection_details = await get_company_db_connection(company_id, current_user, db)
        
        # Connect to the database and retrieve call data
        conn = None
        try:
            conn = pymysql.connect(
                host=connection_details["host"],
                user=connection_details["user"],
                password=connection_details["password"],
                database=connection_details["database"],
                port=connection_details["port"],
                connect_timeout=10,
                cursorclass=pymysql.cursors.DictCursor
            )
            
            with conn.cursor() as cursor:
                # Build the base WHERE clause
                base_where = """
                    WHERE calldate BETWEEN %s AND %s
                    AND (dstchannel IS NOT NULL AND dstchannel != '' AND dstchannel != 'N/A')
                    AND (dst IS NOT NULL AND dst != '' AND dst != 'N/A')
                """
                params = [start_dt, end_dt]
                
                # Add optional filters to WHERE clause
                additional_filters = ""
                
                if disposition:
                    additional_filters += " AND disposition = %s"
                    params.append(disposition)
                    
                if src:
                    additional_filters += " AND src LIKE %s"
                    params.append(f"%{src}%")
                    
                if dst:
                    additional_filters += " AND dst LIKE %s"
                    params.append(f"%{dst}%")
                
                if has_recording is not None:
                    if has_recording:
                        additional_filters += " AND recordingfile != '' AND recordingfile IS NOT NULL"
                    else:
                        additional_filters += " AND (recordingfile = '' OR recordingfile IS NULL)"
                
                if min_duration is not None:
                    additional_filters += " AND duration >= %s"
                    params.append(min_duration)
                
                if max_duration is not None:
                    additional_filters += " AND duration <= %s"
                    params.append(max_duration)
                
                if did:
                    additional_filters += " AND did LIKE %s"
                    params.append(f"%{did}%")
                
                if extension:
                    additional_filters += " AND dst = %s AND LENGTH(dst) <= 5"
                    params.append(extension)
                
                if cnam:
                    additional_filters += " AND (cnam LIKE %s OR cnum LIKE %s)"
                    params.append(f"%{cnam}%")
                    params.append(f"%{cnam}%")
                
                if queue:
                    additional_filters += " AND dst = %s AND dcontext = 'ext-queues'"
                    params.append(queue)
                
                # Complete WHERE clause
                complete_where = base_where + additional_filters
                
                # Build count query
                count_query = f"SELECT COUNT(*) as total FROM cdr {complete_where}"
                
                # Build main query with comprehensive data selection for BI analysis
                main_query = f"""
                    SELECT 
                        calldate, src, dst, dcontext, clid, channel, dstchannel,
                        lastapp, lastdata, duration, billsec, disposition, amaflags,
                        accountcode, uniqueid, userfield, did, recordingfile, cnum, cnam,
                        outbound_cnum, outbound_cnam, dst_cnam
                    FROM cdr {complete_where}
                """
                
                # Handle unique callers request
                if unique_callers_only:
                    main_query = f"""
                        SELECT 
                            calldate, src, dst, dcontext, clid, channel, dstchannel,
                            lastapp, lastdata, duration, billsec, disposition, amaflags,
                            accountcode, uniqueid, userfield, did, recordingfile, cnum, cnam,
                            outbound_cnum, outbound_cnam, dst_cnam
                        FROM cdr {complete_where}
                        GROUP BY src
                    """
                
                # Add sorting and pagination to the main query
                main_query += f" ORDER BY {sort_by} {sort_order} LIMIT %s OFFSET %s"
                main_params = params + [limit, offset]
                
                # Execute count query first
                try:
                    cursor.execute(count_query, params)
                    count_result = cursor.fetchone()
                    total_count = count_result.get("total", 0) if count_result else 0
                except Exception as e:
                    logger.error(f"Count query error: {str(e)}")
                    total_count = 0
                
                # Execute main query
                cursor.execute(main_query, main_params)
                records = cursor.fetchall()
                
                # Get recording files for transcript lookup
                recording_files = []
                for record in records:
                    recording_file = record.get('recordingfile', '')
                    if recording_file and recording_file.strip():
                        recording_files.append(recording_file)
                
                logger.info(f"🔍 Found {len(recording_files)} recordings to check for transcripts")
                
                # Check MongoDB for existing transcription data
                mongodb_transcripts = {}
                if recording_files:
                    try:
                        # Get company code from current user for transcript collection lookup
                        company_code = current_user.get("company_code")
                        if not company_code:
                            logger.warning("⚠️ No company_code found for user, cannot fetch transcripts")
                        else:
                            mongodb_transcripts = await get_transcriptions_for_BI_analysis(recording_files, user_id, company_code)
                            logger.info(f"📝 Found {len(mongodb_transcripts)} existing transcripts in MongoDB from collection: {company_code}")
                    except Exception as e:
                        logger.warning(f"⚠️ Could not fetch transcripts from MongoDB: {e}")
                
                # Process records and build structured data for BI analysis
                processed_records = []
                call_directions = []
                hourly_distribution = defaultdict(int)
                daily_distribution = defaultdict(int)
                disposition_counts = defaultdict(int)
                duration_stats = []
                
                # Enhanced data for transcript analysis
                transcript_data = []
                topic_counts = defaultdict(int)
                sentiment_distribution = defaultdict(int)
                agent_performance_scores = []
                call_completion_stats = {"complete": 0, "truncated": 0}
                
                for record in records:
                    if isinstance(record, dict):  # Ensure record is a dictionary
                        # Convert datetime to string for JSON serialization
                        if record.get('calldate'):
                            calldate = record['calldate']
                            if isinstance(calldate, datetime):
                                record['calldate'] = calldate.isoformat()
                                
                                # Extract metrics for BI analysis
                                hour = calldate.hour
                                date_str = calldate.strftime('%Y-%m-%d')
                                hourly_distribution[hour] += 1
                                daily_distribution[date_str] += 1
                        
                        # Determine call direction
                        direction = determine_call_direction(record.get('src', ''), record.get('dst', ''))
                        record['direction'] = direction
                        call_directions.append(direction)
                        
                        # Track disposition
                        disposition = record.get('disposition', 'UNKNOWN')
                        disposition_counts[disposition] += 1
                        
                        # Track duration for statistics
                        duration = record.get('duration', 0)
                        if duration and isinstance(duration, (int, float)):
                            duration_stats.append(duration)
                        
                        # Add recording status
                        recording_file = record.get('recordingfile', '')
                        record['has_recording'] = bool(recording_file and recording_file.strip())
                        
                        # Add transcript data if available
                        if recording_file and recording_file in mongodb_transcripts:
                            transcript_info = mongodb_transcripts[recording_file]
                            record['transcript'] = {
                                'text': transcript_info.get('text'),
                                'transcript_id': transcript_info.get('transcript_id'),
                                'speakers_count': transcript_info.get('speakers_count', 0),
                                'words_count': transcript_info.get('words_count', 0),
                                'audio_duration': transcript_info.get('audio_duration'),
                                'custom_topic': transcript_info.get('custom_topic'),
                                'sentiment_analysis': transcript_info.get('two_party_sentiment'),
                                'call_completion': transcript_info.get('call_completion'),
                                'agent_performance': transcript_info.get('agent_performance'),
                                'utterances': transcript_info.get('utterances', [])
                            }
                            
                            # Collect transcript data for analysis
                            if transcript_info.get('text'):
                                transcript_data.append({
                                    'call_id': record.get('uniqueid'),
                                    'text': transcript_info.get('text'),
                                    'topic': transcript_info.get('custom_topic'),
                                    'sentiment': transcript_info.get('two_party_sentiment'),
                                    'completion': transcript_info.get('call_completion'),
                                    'agent_performance': transcript_info.get('agent_performance'),
                                    'public_url': transcript_info.get('public_url'),
                                    'speakers_count': transcript_info.get('speakers_count', 0),
                                    'dst': record.get('dst'),
                                    'src': record.get('src'),
                                    'calldate': record.get('calldate')
                                })
                                
                                # Analyze topics
                                topic = transcript_info.get('custom_topic')
                                if topic:
                                    topic_counts[topic] += 1
                                
                                # Analyze sentiment
                                sentiment = transcript_info.get('two_party_sentiment')
                                if sentiment and isinstance(sentiment, dict):
                                    customer_sentiment = sentiment.get('customer_sentiment')
                                    if customer_sentiment:
                                        sentiment_distribution[customer_sentiment] += 1
                                
                                # Analyze agent performance
                                agent_perf = transcript_info.get('agent_performance')
                                if agent_perf and isinstance(agent_perf, dict):
                                    overall_score = agent_perf.get('overall_score')
                                    if overall_score and isinstance(overall_score, (int, float)):
                                        agent_performance_scores.append(overall_score)
                                
                                # Analyze call completion
                                completion = transcript_info.get('call_completion')
                                if completion and isinstance(completion, dict):
                                    status = completion.get('status', '').upper()
                                    if status in ['COMPLETE', 'TRUNCATED']:
                                        call_completion_stats[status.lower()] += 1
                        
                        processed_records.append(record)
                
                # Calculate summary metrics for BI analysis
                total_records = len(processed_records)
                recording_count = sum(1 for r in processed_records if r.get('has_recording', False))
                recording_percentage = (recording_count / total_records * 100) if total_records > 0 else 0
                
                # Direction distribution
                direction_counts = Counter(call_directions)
                
                # Duration statistics
                duration_summary = {}
                if duration_stats:
                    duration_summary = {
                        "average_duration": statistics.mean(duration_stats),
                        "median_duration": statistics.median(duration_stats),
                        "min_duration": min(duration_stats),
                        "max_duration": max(duration_stats),
                        "total_duration": sum(duration_stats)
                    }
                
                # Calculate transcript-based metrics
                transcript_coverage = len(mongodb_transcripts)
                transcript_percentage = (transcript_coverage / total_records * 100) if total_records > 0 else 0
                
                # Agent performance statistics
                agent_performance_summary = {}
                if agent_performance_scores:
                    agent_performance_summary = {
                        "average_score": round(statistics.mean(agent_performance_scores), 2),
                        "median_score": round(statistics.median(agent_performance_scores), 2),
                        "min_score": min(agent_performance_scores),
                        "max_score": max(agent_performance_scores),
                        "total_analyzed": len(agent_performance_scores)
                    }
                
                # Prepare structured call data for BI analysis
                call_data = {
                    "time_period": {
                        "start_date": start_date,
                        "end_date": end_date,
                        "total_days": date_diff + 1
                    },
                    "summary": {
                        "total_calls": total_records,
                        "unique_callers": len(set(r.get('src', '') for r in processed_records if r.get('src'))),
                        "recording_count": recording_count,
                        "recording_percentage": round(recording_percentage, 2),
                        "transcript_count": transcript_coverage,
                        "transcript_percentage": round(transcript_percentage, 2),
                        "answered_calls": disposition_counts.get('ANSWERED', 0),
                        "answered_percentage": round((disposition_counts.get('ANSWERED', 0) / total_records * 100) if total_records > 0 else 0, 2),
                        "average_call_duration": round(duration_summary.get("average_duration", 0), 2),
                        "total_talk_time": duration_summary.get("total_duration", 0)
                    },
                    "call_distribution": {
                        "by_direction": dict(direction_counts),
                        "by_disposition": dict(disposition_counts),
                        "by_hour": dict(hourly_distribution),
                        "by_day": dict(daily_distribution)
                    },
                    "performance_metrics": {
                        "duration_statistics": duration_summary,
                        "call_success_rate": round((disposition_counts.get('ANSWERED', 0) / total_records * 100) if total_records > 0 else 0, 2),
                        "recording_coverage": round(recording_percentage, 2),
                        "transcript_coverage": round(transcript_percentage, 2),
                        "agent_performance": agent_performance_summary
                    },
                    "transcript_analysis": {
                        "total_transcripts": len(transcript_data),
                        "topic_distribution": dict(topic_counts),
                        "sentiment_distribution": dict(sentiment_distribution),
                        "call_completion_stats": call_completion_stats,
                        "most_common_topics": sorted(topic_counts.items(), key=lambda x: x[1], reverse=True)[:10],
                        "agent_performance_stats": agent_performance_summary
                    },
                    "product_service_requests": {
                        "by_extension": {},
                        "by_topic": dict(topic_counts),
                        "by_queue": {},
                        "peak_request_times": {}
                    },
                    "call_records": processed_records[:100] if len(processed_records) > 100 else processed_records,  # Limit records for LLM analysis
                    "transcripts": transcript_data[:50] if len(transcript_data) > 50 else transcript_data,  # Include transcript samples for LLM analysis
                    "data_quality": {
                        "total_retrieved": total_count,
                        "filtered_count": len(processed_records),
                        "data_completeness": round((len(processed_records) / total_count * 100) if total_count > 0 else 0, 2),
                        "transcript_completeness": round(transcript_percentage, 2)
                    }
                }
                
                # Analyze product/service requests by extension and topic
                extension_topics = defaultdict(list)
                extension_counts = defaultdict(int)
                queue_topics = defaultdict(list)
                
                for transcript_item in transcript_data:
                    dst = transcript_item.get('dst')
                    topic = transcript_item.get('topic')
                    
                    if dst:
                        extension_counts[dst] += 1
                        if topic:
                            extension_topics[dst].append(topic)
                    
                    # Analyze queue patterns (extensions starting with 1111)
                    if dst and dst.startswith('1111'):
                        if topic:
                            queue_topics[dst].append(topic)
                
                # Update product service requests with actual data
                call_data["product_service_requests"] = {
                    "by_extension": {
                        ext: {
                            "call_count": count,
                            "percentage": round((count / len(transcript_data) * 100) if transcript_data else 0, 2),
                            "common_topics": list(set(extension_topics[ext]))[:5],
                            "top_topic": max(set(extension_topics[ext]), key=extension_topics[ext].count) if extension_topics[ext] else None
                        }
                        for ext, count in sorted(extension_counts.items(), key=lambda x: x[1], reverse=True)[:10]
                    },
                    "by_topic": {
                        topic: {
                            "call_count": count,
                            "percentage": round((count / len(transcript_data) * 100) if transcript_data else 0, 2),
                            "sample_extensions": list(set([
                                t.get('dst') for t in transcript_data 
                                if t.get('topic') == topic and t.get('dst')
                            ]))[:5]
                        }
                        for topic, count in sorted(topic_counts.items(), key=lambda x: x[1], reverse=True)[:10]
                    },
                    "by_queue": {
                        queue: {
                            "call_count": len(topics),
                            "common_topics": list(set(topics))[:5],
                            "top_topic": max(set(topics), key=topics.count) if topics else None
                        }
                        for queue, topics in queue_topics.items()
                    }
                }
                
                # Initialize LLM and perform business intelligence analysis
                logger.info(f"🧠 Starting BI analysis with model: {model}")
                logger.info(f"📊 Analyzing {total_records} call records for {company_name} ({industry})")
                
                try:
                    llm = LLM(model=model)
                    
                    # Prepare user info for tracing
                    user_info = {
                        "user_id": user_id,
                        "username": current_user.get("username", "unknown"),
                        "company_id": company_id
                    }
                    
                    # Perform BI analysis
                    bi_result = llm.analyze_business_intelligence(
                        call_data=call_data,
                        company_name=company_name,
                        industry=industry,
                        user_info=user_info
                    )
                    
                    logger.info(f"✅ BI analysis completed successfully: {bi_result.get('status', 'unknown')}")
                    
                    # Prepare final response
                    response = {
                        "business_intelligence": bi_result,
                        "data_summary": {
                            "time_period": call_data["time_period"],
                            "total_calls_analyzed": total_records,
                            "company_context": {
                                "name": company_name,
                                "industry": industry
                            },
                            "analysis_model": model,
                            "filters_applied": {
                                "start_date": start_date,
                                "end_date": end_date,
                                "disposition": disposition,
                                "direction": direction,
                                "has_recording": has_recording,
                                "min_duration": min_duration,
                                "max_duration": max_duration
                            }
                        },
                        "quick_metrics": call_data["summary"]
                    }
                    
                    return response
                    
                except Exception as llm_error:
                    logger.error(f"❌ LLM analysis failed: {str(llm_error)}")
                    
                    # Return call data with error info if LLM fails
                    return {
                        "business_intelligence": {
                            "status": "error",
                            "error": str(llm_error),
                            "message": "BI analysis failed, but call data was retrieved successfully"
                        },
                        "data_summary": {
                            "time_period": call_data["time_period"],
                            "total_calls_analyzed": total_records,
                            "company_context": {
                                "name": company_name,
                                "industry": industry
                            },
                            "analysis_model": model
                        },
                        "call_data": call_data,  # Include raw data for debugging
                        "quick_metrics": call_data["summary"]
                    }
                    
        finally:
            if conn:
                conn.close()
                
    except pymysql.MySQLError as e:
        logger.error(f"MySQL Error in BI analysis: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Database error: {str(e)}"
        )
    except Exception as e:
        logger.error(f"Error in business intelligence analysis: {str(e)}")
        if isinstance(e, HTTPException):
            raise
        else:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to perform business intelligence analysis: {str(e)}"
            ) 