import logging
from fastapi import APIRouter, HTTPException, Depends, status, Query, Path
from pydantic import BaseModel, validator, Field
from datetime import datetime, date, timedelta
from typing import List, Dict, Any, Optional
import pymysql
import pymysql.cursors
from api.database import get_database
from api.routes.auth import get_current_user, get_company_db_connection
from api.core.redis_cache import generate_cache_key, get_cached_data, set_cached_data, invalidate_cache, get_cache_stats
from motor.motor_asyncio import AsyncIOMotorClient
import os
from pathlib import Path as PathlibPath

# Set up logging
logger = logging.getLogger(__name__)

# Initialize router
router = APIRouter(prefix="/call-records", tags=["call_records"])

# MongoDB configuration for transcriptions
MONGODB_HOST = os.getenv("MONGODB_HOST")
MONGODB_PORT = int(os.getenv("MONGODB_PORT", 37037))
TRANSCRIPTIONS_DB_NAME = "delaphone_transcriptions"

# Constants for call direction
DIRECTION_INBOUND = "inbound"
DIRECTION_OUTBOUND = "outbound"
DIRECTION_INTERNAL = "internal"
DIRECTION_UNKNOWN = "unknown"

# Constants for call disposition
DISPOSITION_ANSWERED = "ANSWERED"
DISPOSITION_NO_ANSWER = "NO ANSWER"
DISPOSITION_BUSY = "BUSY"
DISPOSITION_FAILED = "FAILED"

# Helper functions
async def get_company_code(company_id: str, db) -> Optional[str]:
    """
    Get company_code from company_id
    """
    try:
        company = await db["companies"].find_one({"_id": company_id})
        if company:
            company_code = company.get("company_code")
            logger.info(f"Found company_code '{company_code}' for company_id '{company_id}'")
            return company_code
        else:
            logger.warning(f"No company found for company_id '{company_id}'")
        return None
    except Exception as e:
        logger.error(f"Error getting company code: {str(e)}")
        return None

async def get_transcriptions_for_recordings(company_code: str, recording_files: List[str]) -> Dict[str, Dict]:
    """
    Get transcriptions for recording files from MongoDB
    
    Args:
        company_code: Company code to use as collection name
        recording_files: List of recording file paths
    
    Returns:
        Dictionary mapping filename to transcript data
    """
    transcripts = {}
    
    if not company_code or not recording_files:
        return transcripts
    
    try:
        # Extract filenames from full paths
        filenames = []
        for recording_file in recording_files:
            if recording_file:
                # Extract filename from path like /var/spool/asterisk/monitor/2025/06/09/q-11111-+233598663529-20250609-182328-1749493397.2727.WAV
                filename = PathlibPath(recording_file).name
                if filename:
                    filenames.append(filename)
        
        if not filenames:
            return transcripts
        
        # Connect to MongoDB transcriptions database
        mongodb_uri = f"mongodb://{MONGODB_HOST}:{MONGODB_PORT}/"
        logger.info(f"Connecting to MongoDB transcriptions at {mongodb_uri}")
        client = AsyncIOMotorClient(mongodb_uri)
        
        try:
            # Test connection
            await client.admin.command('ping')
            
            db = client[TRANSCRIPTIONS_DB_NAME]
            collection = db[company_code]  # Use company_code as collection name
            
            logger.info(f"Querying collection '{company_code}' for {len(filenames)} filenames")
            
            # Query for transcriptions matching the filenames
            cursor = collection.find({
                "filename": {"$in": filenames}
            })
            
            async for transcript_doc in cursor:
                filename = transcript_doc.get("filename")
                if filename:
                    # Clean up the transcript data for response
                    transcript_data = {
                        "transcript_id": transcript_doc.get("transcript_id"),
                        "text": transcript_doc.get("text"),
                        "confidence": transcript_doc.get("confidence"),
                        "audio_duration": transcript_doc.get("audio_duration"),
                        "words_count": transcript_doc.get("words_count"),
                        "speakers_count": transcript_doc.get("speakers_count"),
                        "sentiments_count": transcript_doc.get("sentiments_count"),
                        "topics_detected": transcript_doc.get("topics_detected"),
                        "status": transcript_doc.get("status"),
                        "processed_at": transcript_doc.get("processed_at"),
                        "utterances": transcript_doc.get("utterances", []),
                        "sentiment_analysis": transcript_doc.get("sentiment_analysis", []),
                        "lemur_analysis": transcript_doc.get("lemur_analysis", {}),
                        "public_url": transcript_doc.get("public_url")
                    }
                    transcripts[filename] = transcript_data
                    logger.debug(f"Found transcript for {filename}")
            
            logger.info(f"Found {len(transcripts)} transcripts for {len(filenames)} recordings")
            
        except Exception as e:
            logger.error(f"Error querying MongoDB transcriptions: {str(e)}")
            raise
        finally:
            client.close()
            
    except Exception as e:
        logger.error(f"Error fetching transcriptions: {str(e)}")
    
    return transcripts

def determine_call_direction(src: str, dst: str) -> str:
    """
    Determine call direction based on the following rules:
    - Inbound: dst is ≤ 5 digits (internal extension)
    - Outbound: src is ≤ 5 digits (extension), dst is > 5 digits (external number)
    - Internal: src and dst are both ≤ 5 digits
    - Unknown: any other case
    """
    if not src or not dst:
        return DIRECTION_UNKNOWN
    
    src_is_extension = len(str(src).strip()) <= 5 and str(src).strip().isdigit()
    dst_is_extension = len(str(dst).strip()) <= 5 and str(dst).strip().isdigit()
    
    if src_is_extension and dst_is_extension:
        return DIRECTION_INTERNAL
    elif src_is_extension and not dst_is_extension:
        return DIRECTION_OUTBOUND
    elif not src_is_extension and dst_is_extension:
        return DIRECTION_INBOUND
    else:
        return DIRECTION_UNKNOWN

async def get_mysql_connection(current_user: dict, db):
    """
    Establishes a connection to the company's MySQL database.
    
    Args:
        current_user: The authenticated user object
        db: Database dependency
        
    Returns:
        A tuple containing (connection, close_connection) where:
        - connection is the MySQL connection object
        - close_connection is an async function to properly close the connection
        
    Raises:
        HTTPException: If connection cannot be established
    """
    try:
        # Get the company_id from the current user
        company_id = current_user.get("company_id")
        if not company_id or company_id == "None":
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="User is not associated with any company"
            )
        
        # Get connection details for the user's company
        connection_details = await get_company_db_connection(company_id, current_user, db)
        print(f"Connection details: {connection_details["password"]}")
        # Connect to the MySQL database
        connection = pymysql.connect(
            host=connection_details["host"],
            user=connection_details["user"],
            password=connection_details["password"],
            database=connection_details["database"],
            port=connection_details["port"],
            connect_timeout=10,
            cursorclass=pymysql.cursors.DictCursor
        )
        
        # Create a function to close the connection
        async def close_connection():
            if connection and connection.open:
                connection.close()
                logger.debug(f"MySQL connection closed for company {company_id}")
        
        return connection, close_connection
    
    except pymysql.MySQLError as e:
        logger.error(f"MySQL connection error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Database connection error: {str(e)}"
        )
    except Exception as e:
        logger.error(f"Error establishing MySQL connection: {str(e)}")
        if isinstance(e, HTTPException):
            raise
        else:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to connect to database: {str(e)}"
            )

# Models for responses
class CallRecord(BaseModel):
    calldate: datetime
    clid: Optional[str] = None
    src: Optional[str] = None
    dst: Optional[str] = None
    dcontext: Optional[str] = None
    channel: Optional[str] = None
    dstchannel: Optional[str] = None
    lastapp: Optional[str] = None
    lastdata: Optional[str] = None
    duration: Optional[int] = None
    billsec: Optional[int] = None
    disposition: Optional[str] = None
    amaflags: Optional[int] = None
    accountcode: Optional[str] = None
    uniqueid: Optional[str] = None
    userfield: Optional[str] = None
    recordingfile: Optional[str] = None
    cnum: Optional[str] = None
    cnam: Optional[str] = None
    outbound_cnum: Optional[str] = None
    outbound_cnam: Optional[str] = None
    dst_cnam: Optional[str] = None
    did: Optional[str] = None
    direction: Optional[str] = None
    # Transcript fields
    transcript_id: Optional[str] = None
    transcript_text: Optional[str] = None
    transcript_confidence: Optional[float] = None
    transcript_audio_duration: Optional[int] = None
    transcript_words_count: Optional[int] = None
    transcript_speakers_count: Optional[int] = None
    transcript_sentiments_count: Optional[int] = None
    transcript_topics_detected: Optional[int] = None
    transcript_status: Optional[str] = None
    transcript_processed_at: Optional[datetime] = None
    transcript_utterances: Optional[List[Dict[str, Any]]] = None
    transcript_sentiment_analysis: Optional[List[Dict[str, Any]]] = None
    transcript_lemur_analysis: Optional[Dict[str, Any]] = None
    transcript_summary: Optional[str] = None
    transcript_public_url: Optional[str] = None

class TimePeriod(BaseModel):
    """Time period representation for consistent responses"""
    start_date: str
    end_date: str
    total_days: int

class CallSummary(BaseModel):
    """Summary metrics for call data"""
    total_calls: int
    answered_calls: int
    no_answer_calls: int
    busy_calls: int
    failed_calls: int
    avg_duration: float
    avg_billsec: float
    answer_rate: float
    total_inbound: int
    total_outbound: int
    total_internal: int
    recording_percentage: float

class CallRecordsResponse(BaseModel):
    """Response model for call records listing"""
    time_period: TimePeriod
    summary: CallSummary
    records: List[CallRecord]
    total_count: int
    filtered_count: int
    
class CallMetricsResponse(BaseModel):
    """Response model for call metrics"""
    time_period: TimePeriod
    basic_metrics: Dict[str, Any]
    daily_distribution: List[Dict[str, Any]]
    hourly_distribution: List[Dict[str, Any]]
    disposition_distribution: List[Dict[str, Any]]
    direction_distribution: Dict[str, Any]  # New field for call direction metrics

# Helper function to calculate metrics from call records
def calculate_metrics(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not records:
        return {
            "total_calls": 0,
            "avg_duration": 0,
            "total_duration": 0,
            "avg_billsec": 0,
            "total_billsec": 0,
            "answered_calls": 0,
            "no_answer_calls": 0,
            "busy_calls": 0,
            "failed_calls": 0,
            "answer_percentage": 0,
            "unique_numbers": 0,
            "inbound_calls": 0,
            "outbound_calls": 0,
            "internal_calls": 0,
            "unknown_direction_calls": 0,
            "has_recording": 0,
            "recording_percentage": 0
        }
    
    # Add direction to each record
    for record in records:
        record['direction'] = determine_call_direction(record.get('src', ''), record.get('dst', ''))
    
    # Basic counts
    total_calls = len(records)
    answered_calls = sum(1 for r in records if r.get('disposition') == DISPOSITION_ANSWERED)
    no_answer_calls = sum(1 for r in records if r.get('disposition') == DISPOSITION_NO_ANSWER)
    busy_calls = sum(1 for r in records if r.get('disposition') == DISPOSITION_BUSY)
    failed_calls = sum(1 for r in records if r.get('disposition') == DISPOSITION_FAILED)
    
    # Direction counts
    inbound_calls = sum(1 for r in records if r.get('direction') == DIRECTION_INBOUND)
    outbound_calls = sum(1 for r in records if r.get('direction') == DIRECTION_OUTBOUND)
    internal_calls = sum(1 for r in records if r.get('direction') == DIRECTION_INTERNAL)
    unknown_direction_calls = sum(1 for r in records if r.get('direction') == DIRECTION_UNKNOWN)
    
    # Duration metrics
    total_duration = sum(r.get('duration', 0) for r in records)
    avg_duration = total_duration / total_calls if total_calls > 0 else 0
    
    # Billing metrics
    total_billsec = sum(r.get('billsec', 0) for r in records)
    avg_billsec = total_billsec / total_calls if total_calls > 0 else 0
    
    # Unique numbers
    unique_sources = set(r.get('src', '') for r in records if r.get('src'))
    
    # Recording metrics
    has_recording = sum(1 for r in records if r.get('recordingfile', ''))
    recording_percentage = (has_recording / total_calls * 100) if total_calls > 0 else 0
    
    # Direction-specific metrics
    inbound_answered = sum(1 for r in records if r.get('direction') == DIRECTION_INBOUND and r.get('disposition') == DISPOSITION_ANSWERED)
    outbound_answered = sum(1 for r in records if r.get('direction') == DIRECTION_OUTBOUND and r.get('disposition') == DISPOSITION_ANSWERED)
    
    inbound_answer_rate = (inbound_answered / inbound_calls * 100) if inbound_calls > 0 else 0
    outbound_answer_rate = (outbound_answered / outbound_calls * 100) if outbound_calls > 0 else 0
    
    inbound_avg_duration = sum(r.get('duration', 0) for r in records if r.get('direction') == DIRECTION_INBOUND) / inbound_calls if inbound_calls > 0 else 0
    outbound_avg_duration = sum(r.get('duration', 0) for r in records if r.get('direction') == DIRECTION_OUTBOUND) / outbound_calls if outbound_calls > 0 else 0
    internal_avg_duration = sum(r.get('duration', 0) for r in records if r.get('direction') == DIRECTION_INTERNAL) / internal_calls if internal_calls > 0 else 0
    
    return {
        "total_calls": total_calls,
        "avg_duration": round(avg_duration, 2),
        "total_duration": total_duration,
        "avg_billsec": round(avg_billsec, 2),
        "total_billsec": total_billsec,
        "answered_calls": answered_calls,
        "no_answer_calls": no_answer_calls,
        "busy_calls": busy_calls,
        "failed_calls": failed_calls,
        "answer_percentage": round((answered_calls / total_calls * 100), 2) if total_calls > 0 else 0,
        "unique_numbers": len(unique_sources),
        "inbound_calls": inbound_calls,
        "outbound_calls": outbound_calls,
        "internal_calls": internal_calls,
        "unknown_direction_calls": unknown_direction_calls,
        "inbound_answer_rate": round(inbound_answer_rate, 2),
        "outbound_answer_rate": round(outbound_answer_rate, 2),
        "inbound_avg_duration": round(inbound_avg_duration, 2),
        "outbound_avg_duration": round(outbound_avg_duration, 2),
        "internal_avg_duration": round(internal_avg_duration, 2),
        "has_recording": has_recording,
        "recording_percentage": round(recording_percentage, 2)
    }

@router.get("", response_model=CallRecordsResponse)
async def get_call_records(
    start_date: str = Query(..., description="Start date in YYYY-MM-DD format"),
    end_date: str = Query(..., description="End date in YYYY-MM-DD format"),
    limit: int = Query(100, description="Maximum number of records to return"),
    offset: int = Query(0, description="Number of records to skip"),
    disposition: Optional[str] = Query(None, description="Filter by call disposition (e.g., ANSWERED, NO ANSWER)"),
    direction: Optional[str] = Query(None, description="Filter by call direction (inbound, outbound, internal)"),
    src: Optional[str] = Query(None, description="Filter by source phone number"),
    dst: Optional[str] = Query(None, description="Filter by destination phone number"),
    has_recording: Optional[bool] = Query(None, description="Filter for calls with recordings"),
    min_duration: Optional[int] = Query(None, description="Minimum call duration in seconds"),
    max_duration: Optional[int] = Query(None, description="Maximum call duration in seconds"),
    sort_by: str = Query("calldate", description="Field to sort by"),
    sort_order: str = Query("desc", description="Sort order (asc or desc)"),
    current_user: dict = Depends(get_current_user),
    db = Depends(get_database)
):
    """
    Get call records from the company's ISSABEL database.
    
    This endpoint retrieves call records between the specified dates.
    Users can only access records for their own company.
    """
    try:
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
        
        # Get the company_id from the current user
        company_id = current_user.get("company_id")
        if not company_id or company_id == "None":
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="User is not associated with any company"
            )
        
        # Get database connection details for the user's company
        try:
            connection_details = await get_company_db_connection(company_id, current_user, db)
        except HTTPException as e:
            # Re-raise with more specific context
            if e.status_code == status.HTTP_400_BAD_REQUEST:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Your company does not have ISSABEL database configured. Please contact an administrator."
                )
            raise
        
        # Connect to the ISSABEL database
        conn = None
        try:
            conn = pymysql.connect(
                host=connection_details["host"],
                user=connection_details["user"],
                password=connection_details["password"],
                database=connection_details["database"],
                port=connection_details["port"],
                connect_timeout=10,
                cursorclass=pymysql.cursors.DictCursor  # Return results as dictionaries
            )
            
            # Build the SQL query with proper date formatting and parameterization
            query = """
                SELECT * FROM cdr 
                WHERE calldate BETWEEN %s AND %s
                AND (dstchannel IS NOT NULL AND dstchannel != '' AND dstchannel != 'N/A')
            """
            params = [start_dt, end_dt]
            
            # Add optional filters
            if disposition:
                query += " AND disposition = %s"
                params.append(disposition)
                
            if src:
                query += " AND src LIKE %s"
                params.append(f"%{src}%")
                
            if dst:
                query += " AND dst LIKE %s"
                params.append(f"%{dst}%")
            
            if has_recording is not None:
                if has_recording:
                    query += " AND recordingfile != ''"
                else:
                    query += " AND (recordingfile = '' OR recordingfile IS NULL)"
            
            if min_duration is not None:
                query += " AND duration >= %s"
                params.append(min_duration)
            
            if max_duration is not None:
                query += " AND duration <= %s"
                params.append(max_duration)
            
            # Add count query to get total records
            count_query = query.replace("SELECT *", "SELECT COUNT(*) as total")
            
            # Add sorting and pagination to the main query
            query += f" ORDER BY {sort_by} {sort_order} LIMIT %s OFFSET %s"
            params.extend([limit, offset])
            
            with conn.cursor() as cursor:
                # Get total count
                cursor.execute(count_query, params[:-2])  # Exclude limit and offset params
                count_result = cursor.fetchone()
                total_count = count_result["total"] if count_result else 0
                
                # Get the actual records
                cursor.execute(query, params)
                records = cursor.fetchall()
                
                # Add direction to each record
                for record in records:
                    record['direction'] = determine_call_direction(record.get('src', ''), record.get('dst', ''))
                
                # Filter by direction if specified
                if direction:
                    records = [r for r in records if r.get('direction') == direction]
                
                # Calculate metrics
                metrics = calculate_metrics(records)
                
                # Get summary for all matching records (not just the page)
                summary_query = """
                    SELECT 
                        COUNT(*) AS total_calls,
                        SUM(CASE WHEN disposition = 'ANSWERED' THEN 1 ELSE 0 END) AS answered_calls,
                        SUM(CASE WHEN disposition = 'NO ANSWER' THEN 1 ELSE 0 END) AS no_answer_calls,
                        SUM(CASE WHEN disposition = 'BUSY' THEN 1 ELSE 0 END) AS busy_calls,
                        SUM(CASE WHEN disposition = 'FAILED' THEN 1 ELSE 0 END) AS failed_calls,
                        AVG(duration) AS avg_duration,
                        AVG(billsec) AS avg_billsec,
                        SUM(CASE WHEN recordingfile != '' THEN 1 ELSE 0 END) AS has_recording
                    FROM cdr
                    WHERE calldate BETWEEN %s AND %s
                    AND (dstchannel IS NOT NULL AND dstchannel != '' AND dstchannel != 'N/A')
                """
                cursor.execute(summary_query, [start_dt, end_dt])
                summary = cursor.fetchone()
                
                if summary:
                    # Process numeric values
                    for key in ['avg_duration', 'avg_billsec']:
                        if summary[key] is not None:
                            summary[key] = round(float(summary[key]), 1)
                    
                    # Calculate answer rate
                    if summary["total_calls"] > 0:
                        summary["answer_rate"] = round(summary["answered_calls"] / summary["total_calls"] * 100, 1)
                        summary["recording_percentage"] = round(summary["has_recording"] / summary["total_calls"] * 100, 1)
                    else:
                        summary["answer_rate"] = 0
                        summary["recording_percentage"] = 0
                    
                    # Get direction counts for summary
                    direction_query = """
                        SELECT 
                            src, dst
                        FROM cdr
                        WHERE calldate BETWEEN %s AND %s
                        AND (dstchannel IS NOT NULL AND dstchannel != '' AND dstchannel != 'N/A')
                    """
                    cursor.execute(direction_query, [start_dt, end_dt])
                    direction_records = cursor.fetchall()
                    
                    inbound_count = 0
                    outbound_count = 0
                    internal_count = 0
                    unknown_count = 0
                    
                    for record in direction_records:
                        call_direction = determine_call_direction(record.get('src', ''), record.get('dst', ''))
                        if call_direction == DIRECTION_INBOUND:
                            inbound_count += 1
                        elif call_direction == DIRECTION_OUTBOUND:
                            outbound_count += 1
                        elif call_direction == DIRECTION_INTERNAL:
                            internal_count += 1
                        else:
                            unknown_count += 1
                    
                    summary["total_inbound"] = inbound_count
                    summary["total_outbound"] = outbound_count
                    summary["total_internal"] = internal_count
                    summary["total_unknown"] = unknown_count  # Add unknown count
                    summary["inbound_percentage"] = round(inbound_count / summary["total_calls"] * 100, 1) if summary["total_calls"] > 0 else 0
                    summary["outbound_percentage"] = round(outbound_count / summary["total_calls"] * 100, 1) if summary["total_calls"] > 0 else 0
                    summary["internal_percentage"] = round(internal_count / summary["total_calls"] * 100, 1) if summary["total_calls"] > 0 else 0
                    summary["unknown_percentage"] = round(unknown_count / summary["total_calls"] * 100, 1) if summary["total_calls"] > 0 else 0
                else:
                    summary = {
                        "total_calls": 0,
                        "answered_calls": 0,
                        "no_answer_calls": 0,
                        "busy_calls": 0,
                        "failed_calls": 0,
                        "avg_duration": 0,
                        "total_duration": 0,
                        "avg_billsec": 0,
                        "total_billsec": 0,
                        "with_recording": 0,
                        "answer_rate": 0,
                        "recording_percentage": 0,
                        "total_inbound": 0,
                        "total_outbound": 0, 
                        "total_internal": 0,
                        "total_unknown": 0,  # Add unknown count
                        "inbound_percentage": 0,
                        "outbound_percentage": 0,
                        "internal_percentage": 0,
                        "unknown_percentage": 0  # Add unknown percentage
                    }
                
                # Prepare time period info
                time_period = {
                    "start_date": start_date,
                    "end_date": end_date,
                    "total_days": date_diff + 1
                }
                
                # Prepare response
                return {
                    "time_period": time_period,
                    "summary": summary,
                    "records": records,
                    "total_count": total_count,
                    "filtered_count": len(records)
                }
                
        except pymysql.MySQLError as e:
            logger.error(f"MySQL Error: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Database error: {str(e)}"
            )
        finally:
            if conn:
                conn.close()
                
    except Exception as e:
        logger.error(f"Error getting call records: {str(e)}")
        if isinstance(e, HTTPException):
            raise
        else:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to retrieve call records: {str(e)}"
            )

@router.get("/metrics", response_model=Dict[str, Any])
async def get_call_metrics(
    start_date: str = Query(..., description="Start date in YYYY-MM-DD format"),
    end_date: str = Query(..., description="End date in YYYY-MM-DD format"),
    disposition: Optional[str] = Query(None, description="Filter by call disposition (e.g., ANSWERED, NO ANSWER)"),
    current_user: dict = Depends(get_current_user),
    db = Depends(get_database)
):
    """
    Get aggregated call metrics from the company's ISSABEL database.
    
    This endpoint provides analytics based on call records between the specified dates.
    Users can only access metrics for their own company.
    """
    try:
        # Parse dates (same as in get_call_records)
        try:
            start_dt = datetime.strptime(start_date, "%Y-%m-%d")
            end_dt = datetime.strptime(end_date, "%Y-%m-%d").replace(hour=23, minute=59, second=59)
            
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
        
        # Get the company_id from the current user
        company_id = current_user.get("company_id")
        if not company_id or company_id == "None":
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="User is not associated with any company"
            )
        
        # Get database connection details
        connection_details = await get_company_db_connection(company_id, current_user, db)
        
        # Connect to the ISSABEL database
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
                # Basic metrics query
                basic_query = """
                    SELECT 
                        COUNT(*) AS total_calls,
                        SUM(CASE WHEN disposition = 'ANSWERED' THEN 1 ELSE 0 END) AS answered_calls,
                        AVG(duration) AS avg_duration,
                        SUM(duration) AS total_duration,
                        AVG(billsec) AS avg_billsec,
                        SUM(billsec) AS total_billsec
                    FROM cdr
                    WHERE calldate BETWEEN %s AND %s
                """
                
                params = [start_dt, end_dt]
                
                if disposition:
                    basic_query += " AND disposition = %s"
                    params.append(disposition)
                
                cursor.execute(basic_query, params)
                basic_metrics = cursor.fetchone()
                
                # Daily distribution query
                daily_query = """
                    SELECT 
                        DATE(calldate) AS call_date,
                        COUNT(*) AS total_calls,
                        src, dst
                    FROM cdr
                    WHERE calldate BETWEEN %s AND %s
                    AND (dstchannel IS NOT NULL AND dstchannel != '' AND dstchannel != 'N/A')
                    GROUP BY DATE(calldate), src, dst
                    ORDER BY call_date
                """
                
                params = [start_dt, end_dt]
                
                if disposition:
                    daily_query += " AND disposition = %s"
                    params.append(disposition)
                    
                cursor.execute(daily_query, params)
                daily_distribution = cursor.fetchall()
                
                # Hourly distribution (time of day)
                hourly_query = """
                    SELECT 
                        HOUR(calldate) AS hour_of_day,
                        COUNT(*) AS num_calls
                    FROM cdr
                    WHERE calldate BETWEEN %s AND %s
                """
                
                params = [start_dt, end_dt]
                
                if disposition:
                    hourly_query += " AND disposition = %s"
                    params.append(disposition)
                    
                hourly_query += " GROUP BY HOUR(calldate) ORDER BY hour_of_day"
                
                cursor.execute(hourly_query, params)
                hourly_distribution = cursor.fetchall()
                
                # Top callers
                top_callers_query = """
                    SELECT 
                        src,
                        COUNT(*) AS call_count,
                        SUM(duration) AS total_duration
                    FROM cdr
                    WHERE calldate BETWEEN %s AND %s
                """
                
                params = [start_dt, end_dt]
                
                if disposition:
                    top_callers_query += " AND disposition = %s"
                    params.append(disposition)
                    
                top_callers_query += " GROUP BY src ORDER BY call_count DESC LIMIT 10"
                
                cursor.execute(top_callers_query, params)
                top_callers = cursor.fetchall()
                
                # Disposition distribution
                disposition_query = """
                    SELECT 
                        disposition,
                        COUNT(*) AS count,
                        AVG(duration) AS avg_duration
                    FROM cdr
                    WHERE calldate BETWEEN %s AND %s
                    AND (dstchannel IS NOT NULL AND dstchannel != '' AND dstchannel != 'N/A')
                    GROUP BY disposition
                    ORDER BY count DESC
                """
                
                cursor.execute(disposition_query, [start_dt, end_dt])
                disposition_distribution = cursor.fetchall()
                
                # Format the metrics for better readability
                if basic_metrics:
                    for key in ['avg_duration', 'avg_billsec']:
                        if basic_metrics[key] is not None:
                            basic_metrics[key] = round(float(basic_metrics[key]), 2)
                
                # Format the daily distribution dates to strings
                for day in daily_distribution:
                    if 'call_date' in day and day['call_date'] is not None:
                        day['call_date'] = day['call_date'].strftime('%Y-%m-%d')
                    if 'avg_duration' in day and day['avg_duration'] is not None:
                        day['avg_duration'] = round(float(day['avg_duration']), 2)
                
                return {
                    "time_period": {
                        "start_date": start_date,
                        "end_date": end_date,
                        "days": (end_dt - start_dt).days + 1
                    },
                    "basic_metrics": basic_metrics,
                    "daily_distribution": daily_distribution,
                    "hourly_distribution": hourly_distribution,
                    "top_callers": top_callers,
                    "disposition_distribution": disposition_distribution
                }
                
        except pymysql.MySQLError as e:
            logger.error(f"MySQL Error: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Database error: {str(e)}"
            )
        finally:
            if conn:
                conn.close()
                
    except Exception as e:
        logger.error(f"Error getting call metrics: {str(e)}")
        if isinstance(e, HTTPException):
            raise
        else:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to retrieve call metrics: {str(e)}"
            )

@router.get("/dashboard", response_model=Dict[str, Any])
async def get_call_dashboard(
    start_date: str = Query(..., description="Start date in YYYY-MM-DD format"),
    end_date: str = Query(..., description="End date in YYYY-MM-DD format"),
    current_user: dict = Depends(get_current_user),
    db = Depends(get_database),
    use_cache: bool = Query(True, description="Whether to use cached data if available")
):
    """
    Get comprehensive dashboard data for call center visualizations.
    
    This endpoint provides multiple metrics suitable for dashboard visualizations,
    including direction-based metrics (inbound, outbound, internal).
    """
    try:
        # Parse dates
        try:
            start_dt = datetime.strptime(start_date, "%Y-%m-%d")
            end_dt = datetime.strptime(end_date, "%Y-%m-%d").replace(hour=23, minute=59, second=59)
            
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
        
        # Get the company_id from the current user
        company_id = current_user.get("company_id")
        if not company_id or company_id == "None":
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="User is not associated with any company"
            )
        
        # Check cache first if caching is enabled
        if use_cache:
            cache_params = {
                "company_id": company_id,
                "start_date": start_date,
                "end_date": end_date
            }
            cache_key = generate_cache_key("dashboard", cache_params)
            cached_data = get_cached_data(cache_key)
            if cached_data:
                logger.info(f"Returning cached dashboard data for company {company_id}")
                return cached_data
        
        # Get database connection details
        connection_details = await get_company_db_connection(company_id, current_user, db)
        
        # Connect to the ISSABEL database
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
                # Get all calls in the date range for direction analysis
                all_calls_query = """
                    SELECT 
                        calldate, src, dst, duration, billsec, disposition, recordingfile
                    FROM cdr
                    WHERE calldate BETWEEN %s AND %s
                    AND (dstchannel IS NOT NULL AND dstchannel != '' AND dstchannel != 'N/A')
                """
                cursor.execute(all_calls_query, [start_dt, end_dt])
                all_calls = cursor.fetchall()
                
                # Process calls to add direction
                for call in all_calls:
                    call['direction'] = determine_call_direction(call.get('src', ''), call.get('dst', ''))
                
                # Count by direction
                inbound_count = sum(1 for c in all_calls if c['direction'] == DIRECTION_INBOUND)
                outbound_count = sum(1 for c in all_calls if c['direction'] == DIRECTION_OUTBOUND)
                internal_count = sum(1 for c in all_calls if c['direction'] == DIRECTION_INTERNAL)
                unknown_count = sum(1 for c in all_calls if c['direction'] == DIRECTION_UNKNOWN)
                
                # Summary metrics
                summary_query = """
                    SELECT 
                        COUNT(*) AS total_calls,
                        SUM(CASE WHEN disposition = 'ANSWERED' THEN 1 ELSE 0 END) AS answered_calls,
                        SUM(CASE WHEN disposition = 'NO ANSWER' THEN 1 ELSE 0 END) AS no_answer_calls,
                        SUM(CASE WHEN disposition = 'BUSY' THEN 1 ELSE 0 END) AS busy_calls,
                        SUM(CASE WHEN disposition = 'FAILED' THEN 1 ELSE 0 END) AS failed_calls,
                        AVG(duration) AS avg_duration,
                        SUM(duration) AS total_duration,
                        AVG(billsec) AS avg_billsec,
                        SUM(billsec) AS total_billsec,
                        SUM(CASE WHEN recordingfile != '' THEN 1 ELSE 0 END) AS with_recording
                    FROM cdr
                    WHERE calldate BETWEEN %s AND %s
                    AND (dstchannel IS NOT NULL AND dstchannel != '' AND dstchannel != 'N/A')
                """
                cursor.execute(summary_query, [start_dt, end_dt])
                summary = cursor.fetchone()
                
                # Calculate percentages
                total_calls = summary["total_calls"] if summary else 0
                if total_calls > 0:
                    summary["answer_rate"] = round(summary["answered_calls"] / total_calls * 100, 1)
                    summary["recording_percentage"] = round(summary["with_recording"] / total_calls * 100, 1)
                    
                    # Add direction counts and percentages
                    summary["total_inbound"] = inbound_count
                    summary["total_outbound"] = outbound_count
                    summary["total_internal"] = internal_count
                    summary["total_unknown"] = unknown_count  # Add unknown count
                    summary["inbound_percentage"] = round(inbound_count / summary["total_calls"] * 100, 1) if summary["total_calls"] > 0 else 0
                    summary["outbound_percentage"] = round(outbound_count / summary["total_calls"] * 100, 1) if summary["total_calls"] > 0 else 0
                    summary["internal_percentage"] = round(internal_count / summary["total_calls"] * 100, 1) if summary["total_calls"] > 0 else 0
                    summary["unknown_percentage"] = round(unknown_count / summary["total_calls"] * 100, 1) if summary["total_calls"] > 0 else 0
                else:
                    summary = {
                        "total_calls": 0,
                        "answered_calls": 0,
                        "no_answer_calls": 0,
                        "busy_calls": 0,
                        "failed_calls": 0,
                        "avg_duration": 0,
                        "total_duration": 0,
                        "avg_billsec": 0,
                        "total_billsec": 0,
                        "with_recording": 0,
                        "answer_rate": 0,
                        "recording_percentage": 0,
                        "total_inbound": 0,
                        "total_outbound": 0, 
                        "total_internal": 0,
                        "total_unknown": 0,  # Add unknown count
                        "inbound_percentage": 0,
                        "outbound_percentage": 0,
                        "internal_percentage": 0,
                        "unknown_percentage": 0  # Add unknown percentage
                    }
                
                # Round numeric values
                for key in ['avg_duration', 'avg_billsec']:
                    if summary[key] is not None:
                        summary[key] = round(float(summary[key]), 1)
                
                # Daily data with direction breakdown
                daily_data = {}
                
                for call in all_calls:
                    date_str = call['calldate'].strftime('%Y-%m-%d')
                    direction = call['direction']
                    disposition = call['disposition']
                    
                    if date_str not in daily_data:
                        daily_data[date_str] = {
                            "date": date_str,
                            "total": 0,
                            "answered": 0,
                            "inbound": 0,
                            "outbound": 0,
                            "internal": 0,
                            "unknown": 0,  # Add unknown field for consistency
                            "duration": 0,
                            "billsec": 0
                        }
                    
                    daily_data[date_str]["total"] += 1
                    
                    if disposition == DISPOSITION_ANSWERED:
                        daily_data[date_str]["answered"] += 1
                    
                    if direction == DIRECTION_INBOUND:
                        daily_data[date_str]["inbound"] += 1
                    elif direction == DIRECTION_OUTBOUND:
                        daily_data[date_str]["outbound"] += 1
                    elif direction == DIRECTION_INTERNAL:
                        daily_data[date_str]["internal"] += 1
                    else:
                        daily_data[date_str]["unknown"] += 1  # Handle unknown direction
                    
                    daily_data[date_str]["duration"] += call.get('duration', 0)
                    daily_data[date_str]["billsec"] += call.get('billsec', 0)
                
                # Calculate averages
                for date_str in daily_data:
                    if daily_data[date_str]["total"] > 0:
                        daily_data[date_str]["avg_duration"] = round(daily_data[date_str]["duration"] / daily_data[date_str]["total"], 1)
                        daily_data[date_str]["avg_billsec"] = round(daily_data[date_str]["billsec"] / daily_data[date_str]["total"], 1)
                    else:
                        daily_data[date_str]["avg_duration"] = 0
                        daily_data[date_str]["avg_billsec"] = 0
                
                # Convert to sorted list
                daily_data_list = sorted(daily_data.values(), key=lambda x: x["date"])
                
                # Hourly distribution with direction breakdown
                hourly_data = {}
                
                for call in all_calls:
                    hour = call['calldate'].hour
                    direction = call['direction']
                    
                    if hour not in hourly_data:
                        hourly_data[hour] = {
                            "hour": hour,
                            "total": 0,
                            "inbound": 0,
                            "outbound": 0,
                            "internal": 0,
                            "unknown": 0,  # Add unknown field to initialization
                            "answered": 0
                        }
                    
                    hourly_data[hour]["total"] += 1
                    
                    if direction == DIRECTION_INBOUND:
                        hourly_data[hour]["inbound"] += 1
                    elif direction == DIRECTION_OUTBOUND:
                        hourly_data[hour]["outbound"] += 1
                    elif direction == DIRECTION_INTERNAL:
                        hourly_data[hour]["internal"] += 1
                    else:  # Add this to track unknown calls
                        hourly_data[hour]["unknown"] += 1
                    
                    if call['disposition'] == DISPOSITION_ANSWERED:
                        hourly_data[hour]["answered"] += 1
                
                # Convert to sorted list
                hourly_data_list = [hourly_data[hour] for hour in sorted(hourly_data.keys())]
                
                # Get disposition breakdown
                disposition_query = """
                    SELECT 
                        disposition,
                        COUNT(*) AS count,
                        AVG(duration) AS avg_duration
                    FROM cdr
                    WHERE calldate BETWEEN %s AND %s
                    AND (dstchannel IS NOT NULL AND dstchannel != '' AND dstchannel != 'N/A')
                    GROUP BY disposition
                    ORDER BY count DESC
                """
                cursor.execute(disposition_query, [start_dt, end_dt])
                disposition_data = cursor.fetchall()
                
                # Format disposition data
                for item in disposition_data:
                    if 'avg_duration' in item and item['avg_duration'] is not None:
                        item['avg_duration'] = round(float(item['avg_duration']), 1)
                
                # Get top 10 source numbers with direction
                top_sources = []
                source_counts = {}
                
                for call in all_calls:
                    source = call.get('src', '')
                    if source:
                        if source not in source_counts:
                            source_counts[source] = {
                                "src": source,
                                "calls": 0,
                                "inbound": 0,
                                "outbound": 0,
                                "internal": 0,
                                "unknown": 0,  # Add unknown field to avoid KeyError
                                "duration": 0
                            }
                        
                        direction = call['direction']
                        source_counts[source]["calls"] += 1
                        
                        if direction == DIRECTION_INBOUND:
                            source_counts[source]["inbound"] += 1
                        elif direction == DIRECTION_OUTBOUND:
                            source_counts[source]["outbound"] += 1
                        elif direction == DIRECTION_INTERNAL:
                            source_counts[source]["internal"] += 1
                        else:
                            source_counts[source]["unknown"] += 1  # Handle unknown direction
                        
                        source_counts[source]["duration"] += call.get('duration', 0)
                
                # Calculate average duration and sort
                for source in source_counts:
                    if source_counts[source]["calls"] > 0:
                        source_counts[source]["avg_duration"] = round(source_counts[source]["duration"] / source_counts[source]["calls"], 1)
                    else:
                        source_counts[source]["avg_duration"] = 0
                
                # Get top 10
                top_sources = sorted(source_counts.values(), key=lambda x: x["calls"], reverse=True)[:10]
                
                # Get duration distribution
                duration_ranges = {
                    '0-15s': 0,
                    '16-30s': 0,
                    '31-60s': 0,
                    '1-3m': 0,
                    '3-5m': 0,
                    '>5m': 0
                }
                
                for call in all_calls:
                    duration = call.get('duration', 0)
                    
                    if duration <= 15:
                        duration_ranges['0-15s'] += 1
                    elif duration <= 30:
                        duration_ranges['16-30s'] += 1
                    elif duration <= 60:
                        duration_ranges['31-60s'] += 1
                    elif duration <= 180:
                        duration_ranges['1-3m'] += 1
                    elif duration <= 300:
                        duration_ranges['3-5m'] += 1
                    else:
                        duration_ranges['>5m'] += 1
                
                # Format as list
                duration_distribution = [
                    {"duration_range": range_name, "count": count}
                    for range_name, count in duration_ranges.items()
                ]
                
                # Add distribution by direction
                direction_duration = {
                    DIRECTION_INBOUND: {
                        '0-15s': 0, '16-30s': 0, '31-60s': 0, 
                        '1-3m': 0, '3-5m': 0, '>5m': 0
                    },
                    DIRECTION_OUTBOUND: {
                        '0-15s': 0, '16-30s': 0, '31-60s': 0, 
                        '1-3m': 0, '3-5m': 0, '>5m': 0
                    },
                    DIRECTION_INTERNAL: {
                        '0-15s': 0, '16-30s': 0, '31-60s': 0, 
                        '1-3m': 0, '3-5m': 0, '>5m': 0
                    },
                    DIRECTION_UNKNOWN: {  # Add unknown direction
                        '0-15s': 0, '16-30s': 0, '31-60s': 0, 
                        '1-3m': 0, '3-5m': 0, '>5m': 0
                    }
                }
                
                for call in all_calls:
                    direction = call['direction']
                    if direction in direction_duration:  # This will now include DIRECTION_UNKNOWN
                        duration = call.get('duration', 0)
                        
                        if duration <= 15:
                            direction_duration[direction]['0-15s'] += 1
                        elif duration <= 30:
                            direction_duration[direction]['16-30s'] += 1
                        elif duration <= 60:
                            direction_duration[direction]['31-60s'] += 1
                        elif duration <= 180:
                            direction_duration[direction]['1-3m'] += 1
                        elif duration <= 300:
                            direction_duration[direction]['3-5m'] += 1
                        else:
                            direction_duration[direction]['>5m'] += 1
                
                # Get recording statistics
                recordings_query = """
                    SELECT
                        COUNT(*) AS total,
                        SUM(CASE WHEN recordingfile != '' THEN 1 ELSE 0 END) AS with_recording
                    FROM cdr
                    WHERE calldate BETWEEN %s AND %s
                    AND (dstchannel IS NOT NULL AND dstchannel != '' AND dstchannel != 'N/A')
                """
                cursor.execute(recordings_query, [start_dt, end_dt])
                recordings_data = cursor.fetchone()
                
                # Calculate recording percentage
                if recordings_data and recordings_data["total"] > 0:
                    recordings_data["recording_percentage"] = round(recordings_data["with_recording"] / recordings_data["total"] * 100, 1)
                else:
                    recordings_data = {"total": 0, "with_recording": 0, "recording_percentage": 0}
                
                # Add direction-specific recording stats
                recording_by_direction = {
                    DIRECTION_INBOUND: 0,
                    DIRECTION_OUTBOUND: 0,
                    DIRECTION_INTERNAL: 0,
                    DIRECTION_UNKNOWN: 0  # Add unknown direction
                }
                
                for call in all_calls:
                    if call.get('recordingfile', ''):
                        direction = call['direction']
                        if direction in recording_by_direction:
                            recording_by_direction[direction] += 1
                
                recordings_data["by_direction"] = recording_by_direction
                
                # Prepare the final response
                response = {
                    "time_period": {
                        "start_date": start_date,
                        "end_date": end_date,
                        "total_days": date_diff + 1
                    },
                    "summary": summary,
                    "direction_breakdown": {
                        "inbound": inbound_count,
                        "outbound": outbound_count,
                        "internal": internal_count,
                        "unknown": unknown_count
                    },
                    "daily_data": daily_data_list,
                    "hourly_distribution": hourly_data_list,
                    "disposition_data": disposition_data,
                    "top_sources": top_sources,
                    "duration_distribution": duration_distribution,
                    "duration_by_direction": direction_duration,
                    "recordings_data": recordings_data,
                    "cache_info": {
                        "source": "database",
                        "generated_at": datetime.now().isoformat()
                    }
                }
                
                # Cache the response for future requests if there's data
                if total_calls > 0:
                    cache_key = generate_cache_key("dashboard", cache_params)
                    set_cached_data(cache_key, response, expiry_type="medium")
                
                return response
                
        except pymysql.MySQLError as e:
            logger.error(f"MySQL Error: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, 
                detail=f"Database error: {str(e)}"
            )
        finally:
            if conn:
                conn.close()
                
    except Exception as e:
        logger.error(f"Error getting dashboard data: {str(e)}")
        if isinstance(e, HTTPException):
            raise
        else:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to retrieve dashboard data: {str(e)}"
            )

@router.get("/caller-insights", response_model=Dict[str, Any])
async def get_caller_insights(
    start_date: str = Query(..., description="Start date in YYYY-MM-DD format"),
    end_date: str = Query(..., description="End date in YYYY-MM-DD format"),
    min_calls: int = Query(1, description="Minimum number of calls to be considered a frequent caller"),
    current_user: dict = Depends(get_current_user),
    db = Depends(get_database)
):
    """
    Get insights about callers and their behavior patterns.
    
    This endpoint analyzes call patterns by source number.
    """
    try:
        # Parse dates
        try:
            start_dt = datetime.strptime(start_date, "%Y-%m-%d")
            end_dt = datetime.strptime(end_date, "%Y-%m-%d").replace(hour=23, minute=59, second=59)
            
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
        
        # Get the company_id from the current user
        company_id = current_user.get("company_id")
        if not company_id or company_id == "None":
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="User is not associated with any company"
            )
        
        # Get database connection details
        connection_details = await get_company_db_connection(company_id, current_user, db)
        
        # Connect to the ISSABEL database
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
                # Top callers with detailed metrics
                top_callers_query = """
                    SELECT 
                        src,
                        COUNT(*) AS call_count,
                        SUM(duration) AS total_duration,
                        AVG(duration) AS avg_duration,
                        SUM(CASE WHEN disposition = 'ANSWERED' THEN 1 ELSE 0 END) AS answered_calls,
                        MIN(calldate) AS first_call,
                        MAX(calldate) AS last_call
                    FROM cdr
                    WHERE calldate BETWEEN %s AND %s
                      AND src != ''
                    GROUP BY src
                    HAVING COUNT(*) >= %s
                    ORDER BY call_count DESC
                    LIMIT 20
                """
                cursor.execute(top_callers_query, [start_dt, end_dt, min_calls])
                top_callers = cursor.fetchall()
                
                # Format caller data
                for caller in top_callers:
                    if 'avg_duration' in caller and caller['avg_duration'] is not None:
                        caller['avg_duration'] = round(float(caller['avg_duration']), 1)
                    
                    if 'first_call' in caller and caller['first_call'] is not None:
                        caller['first_call'] = caller['first_call'].strftime('%Y-%m-%d %H:%M:%S')
                    
                    if 'last_call' in caller and caller['last_call'] is not None:
                        caller['last_call'] = caller['last_call'].strftime('%Y-%m-%d %H:%M:%S')
                    
                    # Calculate answer rate
                    if caller['call_count'] > 0:
                        caller['answer_rate'] = round(caller['answered_calls'] / caller['call_count'] * 100, 1)
                
                # Caller frequency distribution
                frequency_query = """
                    SELECT
                        CASE
                            WHEN calls = 1 THEN 'one_time'
                            WHEN calls BETWEEN 2 AND 5 THEN '2_5_calls'
                            WHEN calls BETWEEN 6 AND 10 THEN '6_10_calls'
                            ELSE 'more_than_10'
                        END AS frequency,
                        COUNT(*) AS count
                    FROM (
                        SELECT 
                            src,
                            COUNT(*) AS calls
                        FROM cdr
                        WHERE calldate BETWEEN %s AND %s
                          AND src != ''
                        GROUP BY src
                    ) AS caller_counts
                    GROUP BY frequency
                    ORDER BY
                        CASE frequency
                            WHEN 'one_time' THEN 1
                            WHEN '2_5_calls' THEN 2
                            WHEN '6_10_calls' THEN 3
                            ELSE 4
                        END
                """
                cursor.execute(frequency_query, [start_dt, end_dt])
                caller_frequency = cursor.fetchall()
                
                # Time of day distribution for calls
                time_of_day_query = """
                    SELECT
                        src,
                        SUM(CASE WHEN HOUR(calldate) BETWEEN 5 AND 11 THEN 1 ELSE 0 END) AS morning_calls,
                        SUM(CASE WHEN HOUR(calldate) BETWEEN 12 AND 17 THEN 1 ELSE 0 END) AS afternoon_calls,
                        SUM(CASE WHEN HOUR(calldate) BETWEEN 18 AND 23 OR HOUR(calldate) BETWEEN 0 AND 4 THEN 1 ELSE 0 END) AS evening_calls,
                        COUNT(*) AS total_calls
                    FROM cdr
                    WHERE calldate BETWEEN %s AND %s
                      AND src != ''
                    GROUP BY src
                    HAVING COUNT(*) >= %s
                    ORDER BY total_calls DESC
                    LIMIT 50
                """
                cursor.execute(time_of_day_query, [start_dt, end_dt, min_calls])
                time_of_day_data = cursor.fetchall()
                
                # Identify caller patterns
                caller_patterns = {
                    "morning_callers": [],
                    "afternoon_callers": [],
                    "evening_callers": [],
                    "no_answer_callers": []
                }
                
                for caller in time_of_day_data:
                    # Determine primary calling time
                    max_time = max(caller['morning_calls'], caller['afternoon_calls'], caller['evening_calls'])
                    total = caller['total_calls']
                    
                    if max_time == caller['morning_calls'] and caller['morning_calls'] > total * 0.5:
                        caller_patterns["morning_callers"].append(caller['src'])
                    elif max_time == caller['afternoon_calls'] and caller['afternoon_calls'] > total * 0.5:
                        caller_patterns["afternoon_callers"].append(caller['src'])
                    elif max_time == caller['evening_calls'] and caller['evening_calls'] > total * 0.5:
                        caller_patterns["evening_callers"].append(caller['src'])
                
                # Identify callers with high rate of no answers
                no_answer_query = """
                    SELECT
                        src,
                        COUNT(*) AS total_calls,
                        SUM(CASE WHEN disposition = 'NO ANSWER' THEN 1 ELSE 0 END) AS no_answer_calls
                    FROM cdr
                    WHERE calldate BETWEEN %s AND %s
                      AND src != ''
                    GROUP BY src
                    HAVING COUNT(*) >= %s AND SUM(CASE WHEN disposition = 'NO ANSWER' THEN 1 ELSE 0 END) / COUNT(*) > 0.5
                    ORDER BY SUM(CASE WHEN disposition = 'NO ANSWER' THEN 1 ELSE 0 END) / COUNT(*) DESC
                    LIMIT 20
                """
                cursor.execute(no_answer_query, [start_dt, end_dt, min_calls])
                no_answer_callers = cursor.fetchall()
                
                for caller in no_answer_callers:
                    caller_patterns["no_answer_callers"].append(caller['src'])
                
                # Get unique count of sources/callers
                unique_callers_query = """
                    SELECT COUNT(DISTINCT src) AS unique_callers
                    FROM cdr
                    WHERE calldate BETWEEN %s AND %s
                      AND src != ''
                """
                cursor.execute(unique_callers_query, [start_dt, end_dt])
                unique_callers_result = cursor.fetchone()
                unique_callers = unique_callers_result['unique_callers'] if unique_callers_result else 0
                
                return {
                    "time_period": {
                        "start_date": start_date,
                        "end_date": end_date,
                        "total_days": date_diff + 1
                    },
                    "summary": {
                        "unique_callers": unique_callers,
                        "repeat_callers": unique_callers - (next((item for item in caller_frequency if item['frequency'] == 'one_time'), {'count': 0}))['count']
                    },
                    "top_callers": top_callers,
                    "caller_frequency": caller_frequency,
                    "time_of_day_data": time_of_day_data,
                    "caller_patterns": caller_patterns
                }
                
        except pymysql.MySQLError as e:
            logger.error(f"MySQL Error: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, 
                detail=f"Database error: {str(e)}"
            )
        finally:
            if conn:
                conn.close()
                
    except Exception as e:
        logger.error(f"Error getting caller insights: {str(e)}")
        if isinstance(e, HTTPException):
            raise
        else:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to retrieve caller insights: {str(e)}"
            )

@router.get("/direction-analysis", response_model=Dict[str, Any])
async def get_direction_analysis(
    start_date: str = Query(..., description="Start date in YYYY-MM-DD format"),
    end_date: str = Query(..., description="End date in YYYY-MM-DD format"),
    current_user: dict = Depends(get_current_user),
    db = Depends(get_database),
    use_cache: bool = Query(True, description="Whether to use cached data if available")
):
    """
    Get detailed analysis of call directions (inbound, outbound, internal).
    
    This endpoint provides metrics and patterns related to call directions.
    """
    try:
        # Parse dates
        try:
            start_dt = datetime.strptime(start_date, "%Y-%m-%d")
            end_dt = datetime.strptime(end_date, "%Y-%m-%d").replace(hour=23, minute=59, second=59)
            
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
        
        # Get the company_id from the current user
        company_id = current_user.get("company_id")
        if not company_id or company_id == "None":
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="User is not associated with any company"
            )
        
        # Check cache first if caching is enabled
        if use_cache:
            cache_params = {
                "company_id": company_id,
                "start_date": start_date,
                "end_date": end_date
            }
            cache_key = generate_cache_key("direction_analysis", cache_params)
            cached_data = get_cached_data(cache_key)
            if cached_data:
                logger.info(f"Returning cached direction analysis data for company {company_id}")
                return cached_data
        
        # Get database connection details
        connection_details = await get_company_db_connection(company_id, current_user, db)
        
        # Connect to the ISSABEL database
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
                # Get all calls in the date range
                call_query = """
                    SELECT 
                        calldate, src, dst, duration, billsec, disposition, recordingfile
                    FROM cdr
                    WHERE calldate BETWEEN %s AND %s
                    AND (dstchannel IS NOT NULL AND dstchannel != '' AND dstchannel != 'N/A')
                """
                cursor.execute(call_query, [start_dt, end_dt])
                calls = cursor.fetchall()
                
                # Process calls to add direction
                inbound_calls = []
                outbound_calls = []
                internal_calls = []
                unknown_calls = []
                
                for call in calls:
                    direction = determine_call_direction(call.get('src', ''), call.get('dst', ''))
                    call['direction'] = direction
                    
                    if direction == DIRECTION_INBOUND:
                        inbound_calls.append(call)
                    elif direction == DIRECTION_OUTBOUND:
                        outbound_calls.append(call)
                    elif direction == DIRECTION_INTERNAL:
                        internal_calls.append(call)
                    else:
                        unknown_calls.append(call)
                
                # Calculate basic metrics
                total_calls = len(calls)
                inbound_count = len(inbound_calls)
                outbound_count = len(outbound_calls)
                internal_count = len(internal_calls)
                unknown_count = len(unknown_calls)
                
                # Calculate percentages
                inbound_pct = round((inbound_count / total_calls * 100), 1) if total_calls > 0 else 0
                outbound_pct = round((outbound_count / total_calls * 100), 1) if total_calls > 0 else 0
                internal_pct = round((internal_count / total_calls * 100), 1) if total_calls > 0 else 0
                unknown_pct = round((unknown_count / total_calls * 100), 1) if total_calls > 0 else 0
                
                # Calculate disposition metrics by direction
                direction_metrics = {}
                
                for direction_name, direction_calls in [
                    (DIRECTION_INBOUND, inbound_calls),
                    (DIRECTION_OUTBOUND, outbound_calls),
                    (DIRECTION_INTERNAL, internal_calls)
                ]:
                    if not direction_calls:
                        direction_metrics[direction_name] = {
                            "total": 0,
                            "answered": 0,
                            "no_answer": 0,
                            "busy": 0,
                            "failed": 0,
                            "avg_duration": 0,
                            "avg_billsec": 0,
                            "answer_rate": 0,
                            "recording_count": 0,
                            "recording_rate": 0
                        }
                        continue
                    
                    answered = sum(1 for c in direction_calls if c.get('disposition') == DISPOSITION_ANSWERED)
                    no_answer = sum(1 for c in direction_calls if c.get('disposition') == DISPOSITION_NO_ANSWER)
                    busy = sum(1 for c in direction_calls if c.get('disposition') == DISPOSITION_BUSY)
                    failed = sum(1 for c in direction_calls if c.get('disposition') == DISPOSITION_FAILED)
                    
                    avg_duration = sum(c.get('duration', 0) for c in direction_calls) / len(direction_calls)
                    avg_billsec = sum(c.get('billsec', 0) for c in direction_calls) / len(direction_calls)
                    
                    recordings = sum(1 for c in direction_calls if c.get('recordingfile', ''))
                    
                    direction_metrics[direction_name] = {
                        "total": len(direction_calls),
                        "answered": answered,
                        "no_answer": no_answer,
                        "busy": busy,
                        "failed": failed,
                        "avg_duration": round(avg_duration, 1),
                        "avg_billsec": round(avg_billsec, 1),
                        "answer_rate": round((answered / len(direction_calls) * 100), 1),
                        "recording_count": recordings,
                        "recording_rate": round((recordings / len(direction_calls) * 100), 1) if len(direction_calls) > 0 else 0
                    }
                
                # Daily distribution by direction
                daily_query = """
                    SELECT 
                        DATE(calldate) AS call_date,
                        COUNT(*) AS total_calls,
                        src, dst
                    FROM cdr
                    WHERE calldate BETWEEN %s AND %s
                    AND (dstchannel IS NOT NULL AND dstchannel != '' AND dstchannel != 'N/A')
                    GROUP BY DATE(calldate), src, dst
                    ORDER BY call_date
                """
                cursor.execute(daily_query, [start_dt, end_dt])
                daily_results = cursor.fetchall()
                
                # Build daily direction distribution
                daily_direction = {}
                
                for result in daily_results:
                    date_str = result['call_date'].strftime('%Y-%m-%d')
                    direction = determine_call_direction(result.get('src', ''), result.get('dst', ''))
                    
                    if date_str not in daily_direction:
                        daily_direction[date_str] = {
                            "date": date_str,
                            "total": 0,
                            DIRECTION_INBOUND: 0,
                            DIRECTION_OUTBOUND: 0,
                            DIRECTION_INTERNAL: 0,
                            DIRECTION_UNKNOWN: 0
                        }
                    
                    daily_direction[date_str]["total"] += result['total_calls']
                    daily_direction[date_str][direction] += result['total_calls']
                
                # Hourly distribution by direction
                hourly_direction = {}
                
                for call in calls:
                    hour = call['calldate'].hour
                    direction = call['direction']
                    
                    if hour not in hourly_direction:
                        hourly_direction[hour] = {
                            "hour": hour,
                            "total": 0,
                            DIRECTION_INBOUND: 0,
                            DIRECTION_OUTBOUND: 0,
                            DIRECTION_INTERNAL: 0,
                            DIRECTION_UNKNOWN: 0
                        }
                    
                    hourly_direction[hour]["total"] += 1
                    hourly_direction[hour][direction] += 1
                
                # Top inbound sources
                inbound_sources = {}
                for call in inbound_calls:
                    source = call.get('src', '')
                    if source:
                        if source not in inbound_sources:
                            inbound_sources[source] = 0
                        inbound_sources[source] += 1
                
                # Top outbound destinations
                outbound_destinations = {}
                for call in outbound_calls:
                    dest = call.get('dst', '')
                    if dest:
                        if dest not in outbound_destinations:
                            outbound_destinations[dest] = 0
                        outbound_destinations[dest] += 1
                
                # Convert to sorted lists
                top_inbound_sources = [
                    {"source": source, "calls": count}
                    for source, count in sorted(inbound_sources.items(), key=lambda x: x[1], reverse=True)[:10]
                ]
                
                top_outbound_destinations = [
                    {"destination": dest, "calls": count}
                    for dest, count in sorted(outbound_destinations.items(), key=lambda x: x[1], reverse=True)[:10]
                ]
                
                # Format daily_direction as a list
                daily_direction_list = list(daily_direction.values())
                # Format hourly_direction as a list
                hourly_direction_list = [hourly_direction[hour] for hour in sorted(hourly_direction.keys())]
                
                # Prepare the final response
                response = {
                    "time_period": {
                        "start_date": start_date,
                        "end_date": end_date,
                        "total_days": date_diff + 1
                    },
                    "summary": {
                        "total_calls": total_calls,
                        "direction_distribution": {
                            DIRECTION_INBOUND: {
                                "count": inbound_count,
                                "percentage": inbound_pct
                            },
                            DIRECTION_OUTBOUND: {
                                "count": outbound_count,
                                "percentage": outbound_pct
                            },
                            DIRECTION_INTERNAL: {
                                "count": internal_count,
                                "percentage": internal_pct
                            },
                            DIRECTION_UNKNOWN: {
                                "count": unknown_count,
                                "percentage": unknown_pct
                            }
                        }
                    },
                    "metrics_by_direction": direction_metrics,
                    "daily_direction": daily_direction_list,
                    "hourly_direction": hourly_direction_list,
                    "top_inbound_sources": top_inbound_sources,
                    "top_outbound_destinations": top_outbound_destinations,
                    "cache_info": {
                        "source": "database",
                        "generated_at": datetime.now().isoformat()
                    }
                }
                
                # Cache the response for future requests if there's data
                if total_calls > 0:
                    cache_key = generate_cache_key("direction_analysis", cache_params)
                    set_cached_data(cache_key, response, expiry_type="medium")
                
                return response
                
        except pymysql.MySQLError as e:
            logger.error(f"MySQL Error: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, 
                detail=f"Database error: {str(e)}"
            )
        finally:
            if conn:
                conn.close()
                
    except Exception as e:
        logger.error(f"Error analyzing call directions: {str(e)}")
        if isinstance(e, HTTPException):
            raise
        else:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to analyze call directions: {str(e)}"
            )

@router.get("/cache/stats", response_model=Dict[str, Any])
async def get_cache_statistics(current_user: dict = Depends(get_current_user)):
    """
    Get statistics about the Redis cache for call records.
    
    Administrators can use this endpoint to monitor cache performance and usage.
    """
    # Check if user is admin
    is_admin = current_user.get("is_admin", False)
    if not is_admin:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Only administrators can access cache statistics"
        )
    
    try:
        # Get cache statistics
        stats = get_cache_stats()
        
        return {
            "timestamp": datetime.now().isoformat(),
            "cache_stats": stats,
            "cache_prefixes": ["dashboard", "direction_analysis", "caller_insights", "metrics"],
            "expiry_times": {
                "short": "5 minutes",
                "medium": "30 minutes",
                "long": "24 hours"
            }
        }
    except Exception as e:
        logger.error(f"Error retrieving cache statistics: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve cache statistics: {str(e)}"
        )

@router.delete("/cache/{prefix}", response_model=Dict[str, Any])
async def clear_cache(
    prefix: str = Path(..., description="Cache prefix to clear (e.g., 'dashboard', 'all')"),
    current_user: dict = Depends(get_current_user)
):
    """
    Clear the Redis cache for a specific prefix or all call record data.
    
    This is useful when data has been updated in the database and the cache needs to be refreshed.
    Only administrators can clear the cache.
    """
    # Check if user is admin
    is_admin = current_user.get("is_admin", False)
    if not is_admin:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Only administrators can clear the cache"
        )
    
    try:
        if prefix.lower() == "all":
            # Clear all call records related caches
            prefixes = ["dashboard", "direction_analysis", "caller_insights", "metrics"]
            total_cleared = 0
            
            for p in prefixes:
                cleared = invalidate_cache(p)
                total_cleared += cleared
                
            return {
                "success": True,
                "timestamp": datetime.now().isoformat(),
                "message": f"Cleared {total_cleared} cache entries across all prefixes",
                "prefixes_cleared": prefixes,
                "entries_cleared": total_cleared
            }
        else:
            # Clear a specific prefix
            cleared = invalidate_cache(prefix)
            
            return {
                "success": True,
                "timestamp": datetime.now().isoformat(),
                "message": f"Cleared {cleared} cache entries with prefix '{prefix}'",
                "prefix_cleared": prefix,
                "entries_cleared": cleared
            }
    except Exception as e:
        logger.error(f"Error clearing cache: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to clear cache: {str(e)}"
        )

@router.get("/volume-heatmap", response_model=Dict[str, Any])
async def get_call_volume_heatmap(
    start_date: str = Query(..., description="Start date in YYYY-MM-DD format"),
    end_date: str = Query(..., description="End date in YYYY-MM-DD format"),
    direction: Optional[str] = Query(None, description="Filter by call direction (inbound, outbound, internal)"),
    current_user: dict = Depends(get_current_user),
    db = Depends(get_database)
):
    """
    Get call volume data organized by day of week and hour for heatmap visualization.
    """
    try:
        # Parse dates
        start_dt = datetime.strptime(start_date, "%Y-%m-%d")
        end_dt = datetime.strptime(end_date, "%Y-%m-%d").replace(hour=23, minute=59, second=59)
        
        # Get database connection
        company_id = current_user.get("company_id")
        connection_details = await get_company_db_connection(company_id, current_user, db)
        
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
                # Query for day of week and hour heatmap
                query = """
                    SELECT 
                        DAYOFWEEK(calldate) AS day_of_week,
                        HOUR(calldate) AS hour_of_day,
                        COUNT(*) AS call_count,
                        src, dst
                    FROM cdr
                    WHERE calldate BETWEEN %s AND %s
                    GROUP BY DAYOFWEEK(calldate), HOUR(calldate), src, dst
                """
                cursor.execute(query, [start_dt, end_dt])
                results = cursor.fetchall()
                
                # Initialize heatmap data (7 days x 24 hours)
                heatmap_data = []
                for day in range(1, 8):  # 1=Sunday, 2=Monday, ..., 7=Saturday
                    for hour in range(24):
                        heatmap_data.append({
                            "day_of_week": day,
                            "hour_of_day": hour,
                            "total": 0,
                            "inbound": 0,
                            "outbound": 0, 
                            "internal": 0
                        })
                
                # Populate heatmap data
                for result in results:
                    day = result["day_of_week"]
                    hour = result["hour_of_day"]
                    call_direction = determine_call_direction(result.get('src', ''), result.get('dst', ''))
                    
                    # Find the corresponding entry in heatmap_data
                    idx = (day - 1) * 24 + hour
                    heatmap_data[idx]["total"] += result["call_count"]
                    
                    if call_direction == DIRECTION_INBOUND:
                        heatmap_data[idx]["inbound"] += result["call_count"]
                    elif call_direction == DIRECTION_OUTBOUND:
                        heatmap_data[idx]["outbound"] += result["call_count"]
                    elif call_direction == DIRECTION_INTERNAL:
                        heatmap_data[idx]["internal"] += result["call_count"]
                
                # Filter by direction if specified
                if direction:
                    heatmap_data = [{**item, "total": item[direction]} for item in heatmap_data]
                
                day_names = ["Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"]
                
                # Add day names for better readability
                for item in heatmap_data:
                    item["day_name"] = day_names[item["day_of_week"] - 1]
                
                return {
                    "time_period": {
                        "start_date": start_date,
                        "end_date": end_date
                    },
                    "heatmap_data": heatmap_data
                }
                
        finally:
            if conn:
                conn.close()
                
    except Exception as e:
        logger.error(f"Error generating heatmap data: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to generate heatmap data: {str(e)}"
        )

@router.get("/hourly-distribution", response_model=Dict[str, Any])
async def get_hourly_distribution(
    start_date: str = Query(..., description="Start date in YYYY-MM-DD format"),
    end_date: str = Query(..., description="End date in YYYY-MM-DD format"),
    direction: Optional[str] = Query(None, description="Filter by call direction (inbound, outbound, internal)"),
    group_by_day: bool = Query(False, description="Group data by day of week instead of all days combined"),
    current_user: dict = Depends(get_current_user),
    db = Depends(get_database)
):
    """
    Get hourly call distribution data formatted for bar chart visualization.
    
    This endpoint provides data about call volume distribution across hours of the day,
    which is ideal for bar chart visualization to identify peak calling hours.
    """
    try:
        # Parse dates
        start_dt = datetime.strptime(start_date, "%Y-%m-%d")
        end_dt = datetime.strptime(end_date, "%Y-%m-%d").replace(hour=23, minute=59, second=59)
        
        # Validate date range
        date_diff = (end_dt - start_dt).days
        if date_diff < 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="End date must be after start date"
            )
        
        # Get MySQL connection using our dedicated function
        conn, close_conn = await get_mysql_connection(current_user, db)
        
        try:
            with conn.cursor() as cursor:
                if group_by_day:
                    # Query for hourly distribution grouped by day of week
                    query = """
                        SELECT 
                            DAYOFWEEK(calldate) AS day_of_week,
                            HOUR(calldate) AS hour,
                            COUNT(*) AS call_count,
                            src, dst
                        FROM cdr
                        WHERE calldate BETWEEN %s AND %s
                        GROUP BY DAYOFWEEK(calldate), HOUR(calldate), src, dst
                    """
                else:
                    # Query for overall hourly distribution
                    query = """
                        SELECT 
                            HOUR(calldate) AS hour,
                            COUNT(*) AS call_count,
                            src, dst
                        FROM cdr
                        WHERE calldate BETWEEN %s AND %s
                        GROUP BY HOUR(calldate), src, dst
                    """
                    
                cursor.execute(query, [start_dt, end_dt])
                results = cursor.fetchall()
                
                if group_by_day:
                    # Initialize data structure for day-of-week grouping
                    hourly_data = {}
                    day_names = ["Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"]
                    
                    for day_num in range(1, 8):  # 1=Sunday, 2=Monday, ..., 7=Saturday
                        day_name = day_names[day_num-1]
                        hourly_data[day_name] = {}
                        
                        for hour in range(24):
                            hourly_data[day_name][hour] = {
                                "hour": hour,
                                "total": 0,
                                "inbound": 0,
                                "outbound": 0,
                                "internal": 0,
                                "unknown": 0  # Add this line to initialize the unknown key
                            }
                    
                    # Populate data
                    for result in results:
                        day_num = result["day_of_week"]
                        hour = result["hour"]
                        call_direction = determine_call_direction(result.get('src', ''), result.get('dst', ''))
                        day_name = day_names[day_num-1]
                        
                        hourly_data[day_name][hour]["total"] += result["call_count"]
                        
                        if call_direction == DIRECTION_INBOUND:
                            hourly_data[day_name][hour]["inbound"] += result["call_count"]
                        elif call_direction == DIRECTION_OUTBOUND:
                            hourly_data[day_name][hour]["outbound"] += result["call_count"]
                        elif call_direction == DIRECTION_INTERNAL:
                            hourly_data[day_name][hour]["internal"] += result["call_count"]
                        else:  # Add this to track unknown calls
                            hourly_data[day_name][hour]["unknown"] += result["call_count"]
                    
                    # Convert to list format for each day
                    formatted_data = {}
                    for day_name in hourly_data:
                        formatted_data[day_name] = [hourly_data[day_name][hour] for hour in range(24)]
                    
                else:
                    # Initialize data for overall hourly distribution
                    hourly_data = {}
                    for hour in range(24):
                        hourly_data[hour] = {
                            "hour": hour,
                            "total": 0,
                            "inbound": 0,
                            "outbound": 0,
                            "internal": 0,
                            "unknown": 0  # Add this line to initialize the unknown key
                        }
                    
                    # Populate data
                    for result in results:
                        hour = result["hour"]
                        call_direction = determine_call_direction(result.get('src', ''), result.get('dst', ''))
                        
                        hourly_data[hour]["total"] += result["call_count"]
                        
                        if call_direction == DIRECTION_INBOUND:
                            hourly_data[hour]["inbound"] += result["call_count"]
                        elif call_direction == DIRECTION_OUTBOUND:
                            hourly_data[hour]["outbound"] += result["call_count"]
                        elif call_direction == DIRECTION_INTERNAL:
                            hourly_data[hour]["internal"] += result["call_count"]
                        else:  # Add this to track unknown calls
                            hourly_data[hour]["unknown"] += result["call_count"]
                    
                    # Convert to list format sorted by hour
                    formatted_data = [hourly_data[hour] for hour in range(24)]
                
                # Filter by direction if specified
                if direction and not group_by_day:
                    # Replace all direction data with only the specified direction
                    for item in formatted_data:
                        # Keep only total and the specified direction
                        direction_count = item[direction]
                        # Create a new dictionary with only relevant fields
                        new_item = {
                            "hour": item["hour"],
                            "total": direction_count,  # Set total to match filtered direction count
                            direction: direction_count,
                            "hour_formatted": f"{item['hour']}:00"  # Add formatting here
                        }
                        # Replace the original item with filtered one
                        formatted_data[formatted_data.index(item)] = new_item
                elif direction and group_by_day:
                    # Do the same for day-of-week grouping
                    for day in formatted_data:
                        for i, item in enumerate(formatted_data[day]):
                            direction_count = item[direction]
                            # Create a new dictionary with only relevant fields
                            formatted_data[day][i] = {
                                "hour": item["hour"],
                                "total": direction_count,  # Set total to match filtered direction count
                                direction: direction_count,
                                "hour_formatted": f"{item['hour']}:00"  # Add formatting here
                            }
                
                # Calculate peak hours
                if group_by_day:
                    peak_hours = {}
                    for day in formatted_data:
                        if direction:
                            peak_hour = max(formatted_data[day], key=lambda x: x[direction])
                            peak_hours[day] = {
                                "hour": peak_hour["hour"],
                                "call_count": peak_hour[direction]
                            }
                        else:
                            peak_hour = max(formatted_data[day], key=lambda x: x["total"])
                            peak_hours[day] = {
                                "hour": peak_hour["hour"],
                                "call_count": peak_hour["total"]
                            }
                else:
                    if direction:
                        peak_hour = max(formatted_data, key=lambda x: x[direction])
                        peak_hour_data = {
                            "hour": peak_hour["hour"],
                            "call_count": peak_hour[direction]
                        }
                    else:
                        peak_hour = max(formatted_data, key=lambda x: x["total"])
                        peak_hour_data = {
                            "hour": peak_hour["hour"],
                            "call_count": peak_hour["total"]
                        }
                
                # Add human-readable hour format
                if group_by_day:
                    for day in formatted_data:
                        for item in formatted_data[day]:
                            item["hour_formatted"] = f"{item['hour']}:00"
                else:
                    for item in formatted_data:
                        item["hour_formatted"] = f"{item['hour']}:00"
                
                return {
                    "time_period": {
                        "start_date": start_date,
                        "end_date": end_date,
                        "total_days": date_diff + 1
                    },
                    "hourly_distribution": formatted_data,
                    "peak_hours": peak_hours if group_by_day else peak_hour_data,
                    "direction_filter": direction if direction else "total"
                }
                
        except pymysql.MySQLError as e:
            logger.error(f"MySQL Error: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Database error: {str(e)}"
            )
        finally:
            # Use the close_conn function to properly close the connection
            await close_conn()
                
    except Exception as e:
        logger.error(f"Error getting hourly distribution: {str(e)}")
        if isinstance(e, HTTPException):
            raise
        else:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to get hourly distribution: {str(e)}"
            )

@router.get("/performance-metrics", response_model=Dict[str, Any])
async def get_performance_metrics(
    start_date: str = Query(..., description="Start date in YYYY-MM-DD format"),
    end_date: str = Query(..., description="End date in YYYY-MM-DD format"),
    disposition: Optional[str] = Query(None, description="Filter by call disposition (e.g., ANSWERED, NO ANSWER)"),
    dst_extension: Optional[str] = Query(None, description="Filter by destination extension"),
    time_resolution: str = Query("day", description="Time resolution for trends (day, hour, week, month)"),
    current_user: dict = Depends(get_current_user),
    db = Depends(get_database)
):
    """
    Get call performance metrics for visualization.
    
    This endpoint provides data for:
    - Average call duration
    - Call answer rates
    - Call duration trends over time
    - Duration by disposition
    
    It supports filtering by disposition, destination extension, and date range.
    """
    try:
        # Parse dates
        start_dt = datetime.strptime(start_date, "%Y-%m-%d")
        end_dt = datetime.strptime(end_date, "%Y-%m-%d").replace(hour=23, minute=59, second=59)
        
        # Validate date range
        date_diff = (end_dt - start_dt).days
        if date_diff < 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="End date must be after start date"
            )
            
        # Get database connection
        company_id = current_user.get("company_id")
        if not company_id or company_id == "None":
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="User is not associated with any company"
            )
            
        connection_details = await get_company_db_connection(company_id, current_user, db)
        
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
                # Build base query conditions
                where_conditions = ["calldate BETWEEN %s AND %s"]
                params = [start_dt, end_dt]
                
                if disposition and disposition.lower() != 'all':
                    where_conditions.append("disposition = %s")
                    params.append(disposition)
                
                where_clause = " AND ".join(where_conditions)
                
                # 1. Overall metrics query
                overall_query = f"""
                    SELECT 
                        COUNT(*) AS total_calls,
                        SUM(CASE WHEN disposition = 'ANSWERED' THEN 1 ELSE 0 END) AS answered_calls,
                        SUM(CASE WHEN disposition = 'NO ANSWER' THEN 1 ELSE 0 END) AS no_answer_calls,
                        SUM(CASE WHEN disposition = 'BUSY' THEN 1 ELSE 0 END) AS busy_calls,
                        SUM(CASE WHEN disposition = 'FAILED' THEN 1 ELSE 0 END) AS failed_calls,
                        AVG(duration) AS avg_duration,
                        AVG(billsec) AS avg_billsec,
                        MAX(duration) AS max_duration,
                        MIN(CASE WHEN duration > 0 THEN duration ELSE NULL END) AS min_duration
                    FROM cdr
                    WHERE {where_clause}
                    AND (dstchannel IS NOT NULL AND dstchannel != '' AND dstchannel != 'N/A')
                """
                cursor.execute(overall_query, params)
                overall_metrics = cursor.fetchone()
                
                # Calculate answer rate
                if overall_metrics and overall_metrics["total_calls"] > 0:
                    overall_metrics["answer_rate"] = round(
                        (overall_metrics["answered_calls"] / overall_metrics["total_calls"]) * 100, 2
                    )
                    # Round duration metrics
                    overall_metrics["avg_duration"] = round(float(overall_metrics["avg_duration"]), 2) if overall_metrics["avg_duration"] else 0
                    overall_metrics["avg_billsec"] = round(float(overall_metrics["avg_billsec"]), 2) if overall_metrics["avg_billsec"] else 0
                else:
                    overall_metrics = {
                        "total_calls": 0,
                        "answered_calls": 0,
                        "no_answer_calls": 0,
                        "busy_calls": 0,
                        "failed_calls": 0,
                        "avg_duration": 0,
                        "avg_billsec": 0,
                        "max_duration": 0,
                        "min_duration": 0,
                        "answer_rate": 0
                    }
                
                # 2. Duration by disposition
                disposition_query = f"""
                    SELECT 
                        disposition,
                        COUNT(*) AS call_count,
                        AVG(duration) AS avg_duration,
                        AVG(billsec) AS avg_billsec
                    FROM cdr
                    WHERE {where_clause}
                    AND (dstchannel IS NOT NULL AND dstchannel != '' AND dstchannel != 'N/A')
                    GROUP BY disposition
                    ORDER BY call_count DESC
                """
                cursor.execute(disposition_query, params)
                disposition_metrics = cursor.fetchall()
                
                # Process disposition metrics
                for item in disposition_metrics:
                    item["avg_duration"] = round(float(item["avg_duration"]), 2) if item["avg_duration"] else 0
                    item["avg_billsec"] = round(float(item["avg_billsec"]), 2) if item["avg_billsec"] else 0
                
                # 3. Call duration trends over time
                time_grouping = ""
                date_format = ""
                
                if time_resolution == "hour":
                    time_grouping = "DATE_FORMAT(calldate, '%Y-%m-%d %H:00')"
                    date_format = "%Y-%m-%d %H:00"
                elif time_resolution == "day":
                    time_grouping = "DATE(calldate)"
                    date_format = "%Y-%m-%d"
                elif time_resolution == "week":
                    time_grouping = "DATE(DATE_SUB(calldate, INTERVAL WEEKDAY(calldate) DAY))"
                    date_format = "%Y-%m-%d" # Week beginning
                elif time_resolution == "month":
                    time_grouping = "DATE_FORMAT(calldate, '%Y-%m-01')"
                    date_format = "%Y-%m-%d" # Month beginning
                else:
                    # Default to day if invalid
                    time_grouping = "DATE(calldate)"
                    date_format = "%Y-%m-%d"
                
                trends_query = f"""
                    SELECT 
                        {time_grouping} AS time_period,
                        COUNT(*) AS call_count,
                        SUM(CASE WHEN disposition = 'ANSWERED' THEN 1 ELSE 0 END) AS answered_calls,
                        AVG(duration) AS avg_duration,
                        AVG(billsec) AS avg_billsec,
                        SUM(duration) AS total_duration
                    FROM cdr
                    WHERE {where_clause}
                    AND (dstchannel IS NOT NULL AND dstchannel != '' AND dstchannel != 'N/A')
                    GROUP BY time_period
                    ORDER BY time_period
                """
                cursor.execute(trends_query, params)
                trends_data = cursor.fetchall()
                
                # Process trends data
                for item in trends_data:
                    if isinstance(item["time_period"], datetime):
                        item["time_period"] = item["time_period"].strftime(date_format)
                    
                    item["avg_duration"] = round(float(item["avg_duration"]), 2) if item["avg_duration"] else 0
                    item["avg_billsec"] = round(float(item["avg_billsec"]), 2) if item["avg_billsec"] else 0
                    
                    if item["call_count"] > 0:
                        item["answer_rate"] = round((item["answered_calls"] / item["call_count"]) * 100, 2)
                    else:
                        item["answer_rate"] = 0
                
                # 4. Duration by hour of day
                hourly_duration_query = f"""
                    SELECT 
                        HOUR(calldate) AS hour_of_day,
                        COUNT(*) AS call_count,
                        AVG(CASE WHEN disposition = 'ANSWERED' THEN duration ELSE NULL END) AS avg_answered_duration,
                        AVG(duration) AS avg_duration,
                        SUM(CASE WHEN disposition = 'ANSWERED' THEN 1 ELSE 0 END) AS answered_calls
                    FROM cdr
                    WHERE {where_clause}
                    AND (dstchannel IS NOT NULL AND dstchannel != '' AND dstchannel != 'N/A')
                    GROUP BY hour_of_day
                    ORDER BY hour_of_day
                """
                cursor.execute(hourly_duration_query, params)
                hourly_duration = cursor.fetchall()
                
                # Process hourly duration data
                for item in hourly_duration:
                    item["avg_duration"] = round(float(item["avg_duration"]), 2) if item["avg_duration"] else 0
                    item["avg_answered_duration"] = round(float(item["avg_answered_duration"]), 2) if item["avg_answered_duration"] else 0
                    item["hour_formatted"] = f"{item['hour_of_day']}:00"
                    
                    if item["call_count"] > 0:
                        item["answer_rate"] = round((item["answered_calls"] / item["call_count"]) * 100, 2)
                    else:
                        item["answer_rate"] = 0
                
                # 5. Duration distribution (binned)
                duration_bins = [
                    {"name": "0-15s", "min": 0, "max": 15},
                    {"name": "16-30s", "min": 16, "max": 30},
                    {"name": "31-60s", "min": 31, "max": 60},
                    {"name": "1-3m", "min": 61, "max": 180},
                    {"name": "3-5m", "min": 181, "max": 300},
                    {"name": ">5m", "min": 301, "max": 99999}
                ]
                
                duration_dist_query = f"""
                    SELECT 
                        duration
                    FROM cdr
                    WHERE {where_clause} AND disposition = 'ANSWERED'
                    AND (dstchannel IS NOT NULL AND dstchannel != '' AND dstchannel != 'N/A')
                """
                cursor.execute(duration_dist_query, params)
                duration_results = cursor.fetchall()
                
                # Build duration distribution
                duration_distribution = []
                for bin_def in duration_bins:
                    bin_count = sum(1 for r in duration_results if bin_def["min"] <= r["duration"] <= bin_def["max"])
                    duration_distribution.append({
                        "range": bin_def["name"],
                        "count": bin_count,
                        "percentage": round((bin_count / len(duration_results)) * 100, 2) if duration_results else 0
                    })
                
                return {
                    "time_period": {
                        "start_date": start_date,
                        "end_date": end_date,
                        "total_days": date_diff + 1
                    },
                    "filters": {
                        "disposition": disposition,
                        "dst_extension": dst_extension
                    },
                    "overall_metrics": overall_metrics,
                    "disposition_metrics": disposition_metrics,
                    "trends_data": trends_data,
                    "hourly_duration": hourly_duration,
                    "duration_distribution": duration_distribution
                }
                
        except pymysql.MySQLError as e:
            logger.error(f"MySQL Error: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Database error: {str(e)}"
            )
        finally:
            if conn:
                conn.close()
                
    except Exception as e:
        logger.error(f"Error getting performance metrics: {str(e)}")
        if isinstance(e, HTTPException):
            raise
        else:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to get performance metrics: {str(e)}"
            )

@router.get("/caller-analysis", response_model=Dict[str, Any])
async def get_caller_analysis(
    start_date: str = Query(..., description="Start date in YYYY-MM-DD format"),
    end_date: str = Query(..., description="End date in YYYY-MM-DD format"),
    direction: Optional[str] = Query(None, description="Filter by call direction (inbound, outbound, internal)"),
    limit: int = Query(20, description="Number of top callers to return"),
    min_calls: int = Query(1, description="Minimum number of calls to be included"),
    disposition: Optional[str] = Query(None, description="Filter by call disposition"),
    sort_by: str = Query("count", description="Sort by: count, duration, avg_duration"),
    current_user: dict = Depends(get_current_user),
    db = Depends(get_database)
):
    """
    Get caller analysis data for visualization.
    
    This endpoint provides data for:
    - Top callers by frequency
    - Callers with longest call durations
    - Detailed metrics for each caller
    
    It supports various filtering options and sorting criteria.
    """
    try:
        # Parse dates
        start_dt = datetime.strptime(start_date, "%Y-%m-%d")
        end_dt = datetime.strptime(end_date, "%Y-%m-%d").replace(hour=23, minute=59, second=59)
        
        # Validate date range
        date_diff = (end_dt - start_dt).days
        if date_diff < 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="End date must be after start date"
            )
            
        # Get database connection
        company_id = current_user.get("company_id")
        if not company_id or company_id == "None":
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="User is not associated with any company"
            )
            
        connection_details = await get_company_db_connection(company_id, current_user, db)
        
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
                # Build base query conditions
                where_conditions = ["calldate BETWEEN %s AND %s"]
                params = [start_dt, end_dt]
                
                if disposition and disposition.lower() != 'all':
                    where_conditions.append("disposition = %s")
                    params.append(disposition)
                
                where_clause = " AND ".join(where_conditions)
                
                # Get all calls to process direction filtering
                query = f"""
                    SELECT
                        src, dst, duration, billsec, disposition, calldate,
                        recordingfile
                    FROM cdr
                    WHERE {where_clause}
                    AND (dstchannel IS NOT NULL AND dstchannel != '' AND dstchannel != 'N/A')
                    AND LENGTH(src) > 5
                """
                cursor.execute(query, params)
                all_calls = cursor.fetchall()
                
                # Process calls by direction and caller
                inbound_callers = {}
                outbound_callers = {}
                internal_callers = {}
                
                for call in all_calls:
                    # Determine direction
                    call_direction = determine_call_direction(call.get('src', ''), call.get('dst', ''))
                    
                    # Skip if direction filter doesn't match (unless direction is 'all')
                    if direction and direction.lower() != 'all' and call_direction != direction:
                        continue
                    
                    # Process based on direction
                    if call_direction == DIRECTION_INBOUND:
                        caller = call.get('src', '')
                        if not caller:
                            continue
                            
                        if caller not in inbound_callers:
                            inbound_callers[caller] = {
                                "number": caller,
                                "call_count": 0,
                                "total_duration": 0,
                                "answered_calls": 0,
                                "first_call": call['calldate'],
                                "last_call": call['calldate'],
                                "has_recording": 0,
                                "direction": DIRECTION_INBOUND
                            }
                        
                        caller_data = inbound_callers[caller]
                        caller_data["call_count"] += 1
                        caller_data["total_duration"] += call.get('duration', 0)
                        
                        if call.get('disposition') == DISPOSITION_ANSWERED:
                            caller_data["answered_calls"] += 1
                            
                        if call.get('recordingfile', ''):
                            caller_data["has_recording"] += 1
                            
                        # Update first/last call
                        if call['calldate'] < caller_data["first_call"]:
                            caller_data["first_call"] = call['calldate']
                        if call['calldate'] > caller_data["last_call"]:
                            caller_data["last_call"] = call['calldate']
                    
                    elif call_direction == DIRECTION_OUTBOUND:
                        caller = call.get('src', '')
                        if not caller:
                            continue
                            
                        if caller not in outbound_callers:
                            outbound_callers[caller] = {
                                "number": caller,
                                "call_count": 0,
                                "total_duration": 0,
                                "answered_calls": 0,
                                "first_call": call['calldate'],
                                "last_call": call['calldate'],
                                "has_recording": 0,
                                "direction": DIRECTION_OUTBOUND
                            }
                        
                        caller_data = outbound_callers[caller]
                        caller_data["call_count"] += 1
                        caller_data["total_duration"] += call.get('duration', 0)
                        
                        if call.get('disposition') == DISPOSITION_ANSWERED:
                            caller_data["answered_calls"] += 1
                            
                        if call.get('recordingfile', ''):
                            caller_data["has_recording"] += 1
                            
                        # Update first/last call
                        if call['calldate'] < caller_data["first_call"]:
                            caller_data["first_call"] = call['calldate']
                        if call['calldate'] > caller_data["last_call"]:
                            caller_data["last_call"] = call['calldate']
                    
                    elif call_direction == DIRECTION_INTERNAL:
                        caller = call.get('src', '')
                        if not caller:
                            continue
                            
                        if caller not in internal_callers:
                            internal_callers[caller] = {
                                "number": caller,
                                "call_count": 0,
                                "total_duration": 0,
                                "answered_calls": 0,
                                "first_call": call['calldate'],
                                "last_call": call['calldate'],
                                "has_recording": 0,
                                "direction": DIRECTION_INTERNAL
                            }
                        
                        caller_data = internal_callers[caller]
                        caller_data["call_count"] += 1
                        caller_data["total_duration"] += call.get('duration', 0)
                        
                        if call.get('disposition') == DISPOSITION_ANSWERED:
                            caller_data["answered_calls"] += 1
                            
                        if call.get('recordingfile', ''):
                            caller_data["has_recording"] += 1
                            
                        # Update first/last call
                        if call['calldate'] < caller_data["first_call"]:
                            caller_data["first_call"] = call['calldate']
                        if call['calldate'] > caller_data["last_call"]:
                            caller_data["last_call"] = call['calldate']
                
                # Combine caller data based on direction filter
                all_callers = []
                
                if direction == DIRECTION_INBOUND:
                    all_callers = list(inbound_callers.values())
                elif direction == DIRECTION_OUTBOUND:
                    all_callers = list(outbound_callers.values())
                elif direction == DIRECTION_INTERNAL:
                    all_callers = list(internal_callers.values())
                else:
                    # Combine all directions (for 'all' or unspecified direction)
                    all_callers = list(inbound_callers.values()) + list(outbound_callers.values()) + list(internal_callers.values())
                
                # Calculate derived metrics
                for caller in all_callers:
                    # Average duration
                    if caller["call_count"] > 0:
                        caller["avg_duration"] = round(caller["total_duration"] / caller["call_count"], 2)
                        caller["answer_rate"] = round((caller["answered_calls"] / caller["call_count"]) * 100, 2)
                        caller["recording_rate"] = round((caller["has_recording"] / caller["call_count"]) * 100, 2)
                    else:
                        caller["avg_duration"] = 0
                        caller["answer_rate"] = 0
                        caller["recording_rate"] = 0
                    
                    # Format dates
                    caller["first_call"] = caller["first_call"].strftime('%Y-%m-%d %H:%M:%S')
                    caller["last_call"] = caller["last_call"].strftime('%Y-%m-%d %H:%M:%S')
                
                # Filter by minimum calls
                if min_calls > 1:
                    all_callers = [c for c in all_callers if c["call_count"] >= min_calls]
                
                # Sort by selected criterion
                if sort_by == "count":
                    all_callers.sort(key=lambda x: x["call_count"], reverse=True)
                elif sort_by == "duration":
                    all_callers.sort(key=lambda x: x["total_duration"], reverse=True)
                elif sort_by == "avg_duration":
                    all_callers.sort(key=lambda x: x["avg_duration"], reverse=True)
                else:
                    # Default to call count
                    all_callers.sort(key=lambda x: x["call_count"], reverse=True)
                
                # Limit results
                top_callers = all_callers[:limit] if len(all_callers) > limit else all_callers
                
                # Get the top 10 callers by frequency
                top_10_by_frequency = sorted(all_callers, key=lambda x: x["call_count"], reverse=True)[:10]
                
                # Get the top 10 callers by duration
                top_10_by_duration = sorted(all_callers, key=lambda x: x["total_duration"], reverse=True)[:10]
                
                # Get the top 10 callers by average duration (min 3 calls)
                callers_with_min_calls = [c for c in all_callers if c["call_count"] >= 3]
                top_10_by_avg_duration = sorted(callers_with_min_calls, key=lambda x: x["avg_duration"], reverse=True)[:10]
                
                return {
                    "time_period": {
                        "start_date": start_date,
                        "end_date": end_date,
                        "total_days": date_diff + 1
                    },
                    "filters": {
                        "direction": direction,
                        "min_calls": min_calls,
                        "disposition": disposition
                    },
                    "summary": {
                        "total_callers": len(all_callers),
                        "total_calls": sum(c["call_count"] for c in all_callers),
                        "total_duration": sum(c["total_duration"] for c in all_callers),
                        "avg_calls_per_caller": round(sum(c["call_count"] for c in all_callers) / len(all_callers), 2) if all_callers else 0
                    },
                    "top_callers": top_callers,
                    "top_callers_by_frequency": top_10_by_frequency,
                    "top_callers_by_duration": top_10_by_duration,
                    "top_callers_by_avg_duration": top_10_by_avg_duration
                }
                
        except pymysql.MySQLError as e:
            logger.error(f"MySQL Error: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Database error: {str(e)}"
            )
        finally:
            if conn:
                conn.close()
                
    except Exception as e:
        logger.error(f"Error getting caller analysis: {str(e)}")
        if isinstance(e, HTTPException):
            raise
        else:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to get caller analysis: {str(e)}"
            )

@router.get("/extension-performance", response_model=Dict[str, Any])
async def get_extension_performance(
    start_date: str = Query(..., description="Start date in YYYY-MM-DD format"),
    end_date: str = Query(..., description="End date in YYYY-MM-DD format"),
    extension: Optional[str] = Query(None, description="Filter for a specific extension"),
    disposition: Optional[str] = Query(None, description="Filter by call disposition"),
    min_calls: int = Query(5, description="Minimum number of calls to include extension in results"),
    limit: int = Query(20, description="Number of extensions to return"),
    sort_by: str = Query("call_count", description="Sort by: call_count, answer_rate, avg_duration"),
    current_user: dict = Depends(get_current_user),
    db = Depends(get_database)
):
    """
    Get performance metrics for extensions/agents.
    
    This endpoint provides data for:
    - Call volumes per extension
    - Answer rates per extension
    - Average handling time per extension
    - Call disposition breakdown by extension
    
    It supports filtering by specific extension, date range, and call disposition.
    """
    try:
        # Parse dates
        start_dt = datetime.strptime(start_date, "%Y-%m-%d")
        end_dt = datetime.strptime(end_date, "%Y-%m-%d").replace(hour=23, minute=59, second=59)
        
        # Validate date range
        date_diff = (end_dt - start_dt).days
        if date_diff < 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="End date must be after start date"
            )
            
        # Get MySQL connection using the dedicated function
        conn, close_conn = await get_mysql_connection(current_user, db)
        
        try:
            with conn.cursor() as cursor:
                # Build base query conditions
                where_conditions = ["calldate BETWEEN %s AND %s"]
                params = [start_dt, end_dt]
                
                # Filter for internal extensions only (≤ 5 digits)
                where_conditions.append("LENGTH(dst) <= 5 AND dst REGEXP '^[0-9]+$'")
                
                if extension:
                    where_conditions.append("dst = %s")
                    params.append(extension)
                
                if disposition:
                    where_conditions.append("disposition = %s")
                    params.append(disposition)
                
                where_clause = " AND ".join(where_conditions)
                
                # Query for all relevant calls to extensions
                query = f"""
                    SELECT 
                        dst AS extension,
                        disposition,
                        duration,
                        billsec,
                        recordingfile,
                        src,
                        dstchannel
                    FROM cdr
                    WHERE {where_clause}
                    AND (dstchannel IS NOT NULL AND dstchannel != '' AND dstchannel != 'N/A')
                    ORDER BY calldate
                """
                
                cursor.execute(query, params)
                all_calls = cursor.fetchall()
                
                # Process calls by extension
                extension_metrics = {}
                
                for call in all_calls:
                    ext = call.get('extension', '')
                    if not ext:
                        continue
                    
                    # Initialize extension data if not exists
                    if ext not in extension_metrics:
                        extension_metrics[ext] = {
                            "extension": ext,
                            "call_count": 0,
                            "answered_calls": 0,
                            "no_answer_calls": 0,
                            "busy_calls": 0,
                            "failed_calls": 0,
                            "total_duration": 0,
                            "total_billsec": 0,
                            "inbound_calls": 0,
                            "recording_count": 0,
                            "unique_callers": set()
                        }
                    
                    # Update metrics
                    metrics = extension_metrics[ext]
                    metrics["call_count"] += 1
                    
                    # Update disposition counts
                    disposition = call.get('disposition', '')
                    if disposition == DISPOSITION_ANSWERED:
                        metrics["answered_calls"] += 1
                        metrics["total_duration"] += call.get('duration', 0)
                        metrics["total_billsec"] += call.get('billsec', 0)
                    elif disposition == DISPOSITION_NO_ANSWER:
                        metrics["no_answer_calls"] += 1
                    elif disposition == DISPOSITION_BUSY:
                        metrics["busy_calls"] += 1
                    elif disposition == DISPOSITION_FAILED:
                        metrics["failed_calls"] += 1
                    
                    # Check if inbound call
                    src = call.get('src', '')
                    if src and len(src.strip()) > 5:  # If src is external number
                        metrics["inbound_calls"] += 1
                    
                    # Check if has recording
                    if call.get('recordingfile', ''):
                        metrics["recording_count"] += 1
                    
                    # Add to unique callers
                    metrics["unique_callers"].add(src)
                    
                # Calculate derived metrics and process sets
                extension_list = []
                for ext, metrics in extension_metrics.items():
                    # Convert unique callers set to count
                    metrics["unique_caller_count"] = len(metrics["unique_callers"])
                    del metrics["unique_callers"]  # Remove the set for serialization
                    
                    # Calculate averages and rates
                    if metrics["call_count"] > 0:
                        metrics["answer_rate"] = round((metrics["answered_calls"] / metrics["call_count"]) * 100, 2)
                        metrics["recording_rate"] = round((metrics["recording_count"] / metrics["call_count"]) * 100, 2)
                    else:
                        metrics["answer_rate"] = 0
                        metrics["recording_rate"] = 0
                    
                    if metrics["answered_calls"] > 0:
                        metrics["avg_duration"] = round(metrics["total_duration"] / metrics["answered_calls"], 2)
                        metrics["avg_billsec"] = round(metrics["total_billsec"] / metrics["answered_calls"], 2)
                    else:
                        metrics["avg_duration"] = 0
                        metrics["avg_billsec"] = 0
                    
                    # Add disposition breakdown percentages
                    metrics["answered_percentage"] = round((metrics["answered_calls"] / metrics["call_count"]) * 100, 2) if metrics["call_count"] > 0 else 0
                    metrics["no_answer_percentage"] = round((metrics["no_answer_calls"] / metrics["call_count"]) * 100, 2) if metrics["call_count"] > 0 else 0
                    metrics["busy_percentage"] = round((metrics["busy_calls"] / metrics["call_count"]) * 100, 2) if metrics["call_count"] > 0 else 0
                    metrics["failed_percentage"] = round((metrics["failed_calls"] / metrics["call_count"]) * 100, 2) if metrics["call_count"] > 0 else 0
                    
                    # Add to list if meets minimum call threshold
                    if metrics["call_count"] >= min_calls:
                        extension_list.append(metrics)
                
                # Sort by selected criterion
                if sort_by == "call_count":
                    extension_list.sort(key=lambda x: x["call_count"], reverse=True)
                elif sort_by == "answer_rate":
                    extension_list.sort(key=lambda x: x["answer_rate"], reverse=True)
                elif sort_by == "avg_duration":
                    extension_list.sort(key=lambda x: x["avg_duration"], reverse=True)
                else:
                    # Default to call count
                    extension_list.sort(key=lambda x: x["call_count"], reverse=True)
                
                # Limit results
                top_extensions = extension_list[:limit] if len(extension_list) > limit else extension_list
                
                # Get disposition breakdown for all extensions
                disposition_summary = {
                    "ANSWERED": sum(e["answered_calls"] for e in extension_list),
                    "NO ANSWER": sum(e["no_answer_calls"] for e in extension_list),
                    "BUSY": sum(e["busy_calls"] for e in extension_list),
                    "FAILED": sum(e["failed_calls"] for e in extension_list)
                }
                
                # Calculate total calls and percentages
                total_calls = sum(disposition_summary.values())
                
                if total_calls > 0:
                    disposition_percentages = {
                        disp: round((count / total_calls) * 100, 2)
                        for disp, count in disposition_summary.items()
                    }
                else:
                    disposition_percentages = {disp: 0 for disp in disposition_summary}
                
                # Prepare disposition data for stacked bar chart
                disposition_data_for_chart = []
                for ext in top_extensions[:10]:  # Limit to top 10 for chart clarity
                    disposition_data_for_chart.append({
                        "extension": ext["extension"],
                        "ANSWERED": ext["answered_calls"],
                        "NO_ANSWER": ext["no_answer_calls"],
                        "BUSY": ext["busy_calls"],
                        "FAILED": ext["failed_calls"],
                        "total": ext["call_count"]
                    })
                
                return {
                    "time_period": {
                        "start_date": start_date,
                        "end_date": end_date,
                        "total_days": date_diff + 1
                    },
                    "filters": {
                        "extension": extension,
                        "disposition": disposition,
                        "min_calls": min_calls
                    },
                    "summary": {
                        "total_extensions": len(extension_list),
                        "total_calls": total_calls,
                        "disposition_summary": disposition_summary,
                        "disposition_percentages": disposition_percentages,
                        "avg_answer_rate": round(sum(e["answer_rate"] * e["call_count"] for e in extension_list) / total_calls, 2) if total_calls > 0 else 0,
                        "avg_handling_time": round(sum(e["total_duration"] for e in extension_list) / sum(e["answered_calls"] for e in extension_list), 2) if sum(e["answered_calls"] for e in extension_list) > 0 else 0
                    },
                    "extensions": top_extensions,
                    "disposition_data": disposition_data_for_chart
                }
                
        except pymysql.MySQLError as e:
            logger.error(f"MySQL Error: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Database error: {str(e)}"
            )
        finally:
            # Close the connection using our async function
            await close_conn()
                
    except Exception as e:
        logger.error(f"Error getting extension performance: {str(e)}")
        if isinstance(e, HTTPException):
            raise
        else:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to get extension performance: {str(e)}"
        )

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
    include_transcripts: bool = Query(False, description="Include transcript data for recordings (may impact performance)"),
    limit: int = Query(1000000, description="Maximum number of records to return"),
    offset: int = Query(0, description="Number of records to skip"),
    sort_by: str = Query("calldate", description="Field to sort by (calldate, duration, billsec)"),
    sort_order: str = Query("desc", description="Sort order (asc or desc)"),
    current_user: dict = Depends(get_current_user),
    db = Depends(get_database)
):
    """
    Get detailed call logs with extensive filtering options.
    
    This endpoint provides comprehensive call records with multiple filter options.
    Users can only access records for their own company.
    
    Optionally includes transcript data for recorded calls when include_transcripts=True.
    Transcripts are fetched from the MongoDB transcriptions database using the company's code.
    """
    try:
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
                direction_filter = DIRECTION_INBOUND
            elif direction_lower == "outbound":
                direction_filter = DIRECTION_OUTBOUND
            elif direction_lower == "internal":
                direction_filter = DIRECTION_INTERNAL
            elif direction_lower == "unknown":
                direction_filter = DIRECTION_UNKNOWN
        
        # Get MySQL connection using the dedicated function
        conn, close_conn = await get_mysql_connection(current_user, db)
        
        try:
            with conn.cursor() as cursor:
                # Build the SQL query with proper date formatting and parameterization
                query = """
                    SELECT * FROM cdr 
                    WHERE calldate BETWEEN %s AND %s
                    AND (dstchannel IS NOT NULL AND dstchannel != '' AND dstchannel != 'N/A')
                """
                params = [start_dt, end_dt]
                
                # Add optional filters
                if disposition:
                    query += " AND disposition = %s"
                    params.append(disposition)
                    
                if src:
                    query += " AND src LIKE %s"
                    params.append(f"%{src}%")
                    
                if dst:
                    query += " AND dst LIKE %s"
                    params.append(f"%{dst}%")
                
                if has_recording is not None:
                    if has_recording:
                        query += " AND recordingfile != ''"
                    else:
                        query += " AND (recordingfile = '' OR recordingfile IS NULL)"
                
                if min_duration is not None:
                    query += " AND duration >= %s"
                    params.append(min_duration)
                
                if max_duration is not None:
                    query += " AND duration <= %s"
                    params.append(max_duration)
                
                if did:
                    query += " AND did LIKE %s"
                    params.append(f"%{did}%")
                
                if extension:
                    query += " AND dst = %s AND LENGTH(dst) <= 5"
                    params.append(extension)
                
                if cnam:
                    query += " AND (cnam LIKE %s OR cnum LIKE %s)"
                    params.append(f"%{cnam}%")
                    params.append(f"%{cnam}%")
                
                if queue:
                    query += " AND dst = %s AND dcontext = 'ext-queues'"
                    params.append(queue)
                
                # Add count query to get total records
                count_query = query.replace("SELECT *", "SELECT COUNT(*) as total")
                
                # Handle unique callers request
                if unique_callers_only:
                    query = query.replace("SELECT *", "SELECT DISTINCT src, MIN(calldate) as calldate")
                    query += " GROUP BY src"
                
                # Add sorting and pagination to the main query
                query += f" ORDER BY {sort_by} {sort_order} LIMIT %s OFFSET %s"
                params.extend([limit, offset])
                
                # Execute count query
                cursor.execute(count_query, params[:-2])  # Exclude limit and offset params
                count_result = cursor.fetchone()
                total_count = count_result["total"] if count_result else 0
                
                # Execute main query
                cursor.execute(query, params)
                records = cursor.fetchall()
                
                # Get transcripts for recordings if requested
                transcripts = {}
                if include_transcripts:
                    # Get company_code for transcript lookup
                    company_id = current_user.get("company_id")
                    company_code = None
                    if company_id:
                        company_code = await get_company_code(company_id, db)
                    
                    if company_code:
                        # Extract recording files for transcript lookup
                        recording_files = []
                        for record in records:
                            recording_file = record.get('recordingfile')
                            if recording_file:
                                recording_files.append(recording_file)
                        
                        # Get transcripts for all recording files
                        if recording_files:
                            transcripts = await get_transcriptions_for_recordings(company_code, recording_files)
                            logger.info(f"Retrieved {len(transcripts)} transcripts for company {company_code}")
                    else:
                        logger.warning("Could not retrieve company_code for transcript lookup")
                
                # Process records (add direction, transcripts, etc.)
                processed_records = []
                
                for record in records:
                    # Add direction to the record
                    if isinstance(record, dict):  # Ensure record is a dictionary
                        src = record.get('src', '')
                        dst = record.get('dst', '')
                        call_direction = determine_call_direction(src, dst)
                        record['direction'] = call_direction
                        
                        # Add transcript data if available
                        recording_file = record.get('recordingfile')
                        if recording_file and transcripts:
                            filename = PathlibPath(recording_file).name
                            transcript_data = transcripts.get(filename)
                            
                            if transcript_data:
                                # Add transcript fields to the record
                                record['transcript_id'] = transcript_data.get('transcript_id')
                                record['transcript_text'] = transcript_data.get('text')
                                record['transcript_confidence'] = transcript_data.get('confidence')
                                record['transcript_audio_duration'] = transcript_data.get('audio_duration')
                                record['transcript_words_count'] = transcript_data.get('words_count')
                                record['transcript_speakers_count'] = transcript_data.get('speakers_count')
                                record['transcript_sentiments_count'] = transcript_data.get('sentiments_count')
                                record['transcript_topics_detected'] = transcript_data.get('topics_detected')
                                record['transcript_status'] = transcript_data.get('status')
                                record['transcript_processed_at'] = transcript_data.get('processed_at')
                                record['transcript_utterances'] = transcript_data.get('utterances')
                                record['transcript_sentiment_analysis'] = transcript_data.get('sentiment_analysis')
                                record['transcript_lemur_analysis'] = transcript_data.get('lemur_analysis')
                                # Extract summary from lemur_analysis if available
                                lemur_analysis = transcript_data.get('lemur_analysis', {})
                                record['transcript_summary'] = lemur_analysis.get('summary') if lemur_analysis else None
                                record['transcript_public_url'] = transcript_data.get('public_url')
                        
                        # Only add records that match direction filter
                        if direction_filter is None or call_direction == direction_filter:
                            processed_records.append(record)
                
                # Get summary metrics
                summary_metrics = {}
                
                # Only calculate summary if we have records
                if processed_records:
                    # Basic counts
                    total_calls = len(processed_records)
                    answered_calls = sum(1 for r in processed_records if r.get('disposition') == DISPOSITION_ANSWERED)
                    no_answer_calls = sum(1 for r in processed_records if r.get('disposition') == DISPOSITION_NO_ANSWER)
                    busy_calls = sum(1 for r in processed_records if r.get('disposition') == DISPOSITION_BUSY)
                    failed_calls = sum(1 for r in processed_records if r.get('disposition') == DISPOSITION_FAILED)
                    
                    # Direction counts
                    inbound_calls = sum(1 for r in processed_records if r.get('direction') == DIRECTION_INBOUND)
                    outbound_calls = sum(1 for r in processed_records if r.get('direction') == DIRECTION_OUTBOUND)
                    internal_calls = sum(1 for r in processed_records if r.get('direction') == DIRECTION_INTERNAL)
                    unknown_direction_calls = sum(1 for r in processed_records if r.get('direction') == DIRECTION_UNKNOWN)
                    
                    # Duration metrics
                    total_duration = sum(r.get('duration', 0) for r in processed_records)
                    avg_duration = total_duration / total_calls if total_calls > 0 else 0
                    
                    # Billing metrics
                    total_billsec = sum(r.get('billsec', 0) for r in processed_records)
                    avg_billsec = total_billsec / total_calls if total_calls > 0 else 0
                    
                    # Recording metrics
                    has_recording_count = sum(1 for r in processed_records if r.get('recordingfile', ''))
                    recording_percentage = (has_recording_count / total_calls * 100) if total_calls > 0 else 0
                    
                    # Unique sources (for unique caller count)
                    unique_sources = set(r.get('src', '') for r in processed_records if r.get('src'))
                    unique_callers_count = len(unique_sources)
                    
                    # Transcript metrics
                    transcribed_calls = sum(1 for r in processed_records if r.get('transcript_text'))
                    transcribed_percentage = (transcribed_calls / total_calls * 100) if total_calls > 0 else 0
                    
                    summary_metrics = {
                        "total_calls": total_calls,
                        "answered_calls": answered_calls,
                        "no_answer_calls": no_answer_calls,
                        "busy_calls": busy_calls,
                        "failed_calls": failed_calls,
                        "avg_duration": round(avg_duration, 2),
                        "total_duration": total_duration,
                        "avg_billsec": round(avg_billsec, 2),
                        "total_billsec": total_billsec,
                        "answer_rate": round((answered_calls / total_calls * 100), 2) if total_calls > 0 else 0,
                        "total_inbound": inbound_calls,
                        "total_outbound": outbound_calls,
                        "total_internal": internal_calls,
                        "total_unknown": unknown_direction_calls,
                        "recording_count": has_recording_count,
                        "recording_percentage": round(recording_percentage, 2),
                        "unique_callers": unique_callers_count,
                        "transcribed_calls": transcribed_calls,
                        "transcribed_percentage": round(transcribed_percentage, 2)
                    }
                
                # Prepare time period info
                time_period = {
                    "start_date": start_date,
                    "end_date": end_date,
                    "total_days": date_diff + 1
                }
                
                # Prepare filters used
                filters_applied = {
                    "src": src,
                    "dst": dst,
                    "disposition": disposition,
                    "direction": direction,
                    "min_duration": min_duration,
                    "max_duration": max_duration,
                    "has_recording": has_recording,
                    "did": did,
                    "extension": extension,
                    "cnam": cnam,
                    "queue": queue,
                    "unique_callers_only": unique_callers_only,
                    "include_transcripts": include_transcripts
                }
                
                # Prepare response
                response = {
                    "time_period": time_period,
                    "filters": filters_applied,
                    "summary": summary_metrics,
                    "total_count": total_count,
                    "filtered_count": len(processed_records)
                }
                
                # Only include records if requested
                if include_details:
                    response["records"] = processed_records
                
                return response
                
        except pymysql.MySQLError as e:
            logger.error(f"MySQL Error: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Database error: {str(e)}"
            )
        finally:
            await close_conn()
                
    except Exception as e:
        logger.error(f"Error getting call logs: {str(e)}")
        if isinstance(e, HTTPException):
            raise
        else:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to retrieve call logs: {str(e)}"
            )

@router.get("/agent-performance", response_model=Dict[str, Any])
async def get_agent_performance(
    start_date: str = Query(..., description="Start date in YYYY-MM-DD format"),
    end_date: str = Query(..., description="End date in YYYY-MM-DD format"),
    agent: Optional[str] = Query(None, description="Filter for a specific agent"),
    disposition: Optional[str] = Query("ALL", description="Filter by call disposition (ANSWERED, NO ANSWER, BUSY, FAILED, ALL)"),
    direction: Optional[str] = Query("all", description="Filter by call direction (inbound, outbound, internal, all)"),
    min_calls: int = Query(5, description="Minimum number of calls to include agent in results"),
    limit: int = Query(20, description="Number of agents to return"),
    sort_by: str = Query("call_count", description="Sort by: call_count, answer_rate, avg_duration"),
    current_user: dict = Depends(get_current_user),
    db = Depends(get_database)
):
    """
    Get performance metrics for agents.
    
    This endpoint provides data for:
    - Call volumes per agent
    - Answer rates per agent
    - Average handling time per agent
    - Call disposition breakdown by agent
    
    It supports filtering by specific agent, date range, call disposition, and call direction.
    """
    try:
        # Parse dates
        start_dt = datetime.strptime(start_date, "%Y-%m-%d")
        end_dt = datetime.strptime(end_date, "%Y-%m-%d").replace(hour=23, minute=59, second=59)
        
        # Validate date range
        date_diff = (end_dt - start_dt).days
        if date_diff < 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="End date must be after start date"
            )
        
        # Normalize disposition and direction values for consistent handling
        print(f"[DEBUG] Original parameters: disposition={disposition}, direction={direction}")
        
        # Standardize "all" values case-insensitively
        disposition_filter = None
        if disposition and disposition.upper() != "ALL":
            disposition_filter = disposition
        
        direction_filter = None
        if direction and direction.lower() != "all":
            direction_filter = direction
            
        print(f"[DEBUG] Normalized filters: disposition_filter={disposition_filter}, direction_filter={direction_filter}")
            
        # Get MySQL connection using the dedicated function
        conn, close_conn = await get_mysql_connection(current_user, db)
        
        try:
            with conn.cursor() as cursor:
                # First, get team-wide data for accurate averages (regardless of agent filter)
                print(f"[DEBUG] Getting team-wide data for accurate team averages")
                
                # Build base query conditions for team data (without agent filter)
                team_where_conditions = ["calldate BETWEEN %s AND %s"]
                team_params = [start_dt, end_dt]
                
                # Filter for agents only (≤ 5 digits)
                team_where_conditions.append("LENGTH(src) <= 5 AND src REGEXP '^[0-9]+$'")
                
                # Apply disposition filter if specified
                if disposition_filter:
                    team_where_conditions.append("disposition = %s")
                    team_params.append(disposition_filter)
                
                team_where_clause = " AND ".join(team_where_conditions)
                
                # Query for team-wide data
                team_query = f"""
                    SELECT 
                        src AS agent,
                        COUNT(*) AS call_count,
                        SUM(CASE WHEN disposition = 'ANSWERED' THEN 1 ELSE 0 END) AS answered_calls,
                        SUM(CASE WHEN disposition = 'NO ANSWER' THEN 1 ELSE 0 END) AS no_answer_calls,
                        SUM(CASE WHEN disposition = 'BUSY' THEN 1 ELSE 0 END) AS busy_calls,
                        SUM(CASE WHEN disposition = 'FAILED' THEN 1 ELSE 0 END) AS failed_calls,
                        SUM(duration) AS total_duration,
                        SUM(billsec) AS total_billsec,
                        SUM(CASE WHEN recordingfile != '' THEN 1 ELSE 0 END) AS recording_count
                    FROM cdr
                    WHERE {team_where_clause}
                    AND (dstchannel IS NOT NULL AND dstchannel != '' AND dstchannel != 'N/A')
                    GROUP BY src
                """
                
                print(f"[DEBUG] Team-wide SQL query: {team_query}")
                
                cursor.execute(team_query, team_params)
                team_stats = cursor.fetchall()
                
                print(f"[DEBUG] Total agents in team stats: {len(team_stats)}")
                
                # Now run the main query which may be filtered by agent
                # Build base query conditions
                where_conditions = ["calldate BETWEEN %s AND %s"]
                params = [start_dt, end_dt]
                
                # Filter for agents only (≤ 5 digits)
                where_conditions.append("LENGTH(src) <= 5 AND src REGEXP '^[0-9]+$'")
                
                if agent:
                    where_conditions.append("src = %s")
                    params.append(agent)
                
                if disposition_filter:
                    where_conditions.append("disposition = %s")
                    params.append(disposition_filter)
                
                where_clause = " AND ".join(where_conditions)
                
                # Query for all relevant calls from agents
                query = f"""
                    SELECT 
                        src AS agent,
                        dst,
                        disposition,
                        duration,
                        billsec,
                        recordingfile,
                        dstchannel,
                        cnam
                    FROM cdr
                    WHERE {where_clause}
                    AND (dstchannel IS NOT NULL AND dstchannel != '' AND dstchannel != 'N/A')
                    ORDER BY calldate
                """
                
                print(f"[DEBUG] Main SQL query: {query}")
                print(f"[DEBUG] Main query params: {params}")
                
                cursor.execute(query, params)
                all_calls = cursor.fetchall()
                
                print(f"[DEBUG] Total calls fetched: {len(all_calls)}")
                
                # Process calls by agent
                agent_metrics = {}
                
                for call in all_calls:
                    # Determine direction for each call
                    call_direction = determine_call_direction(call.get('agent', ''), call.get('dst', ''))
                    call['direction'] = call_direction
                    
                    # Skip if direction filter doesn't match (unless direction is 'all')
                    if direction_filter and call_direction != direction_filter:
                        continue
                    
                    agent_id = call.get('agent', '')
                    if not agent_id:
                        continue
                    
                    # Initialize agent data if not exists
                    if agent_id not in agent_metrics:
                        agent_metrics[agent_id] = {
                            "agent": agent_id,
                            "agent_cnam": call.get('cnam', ''),  # Initial cnam value
                            "call_count": 0,
                            "answered_calls": 0,
                            "no_answer_calls": 0,
                            "busy_calls": 0,
                            "failed_calls": 0,
                            "total_duration": 0,
                            "total_billsec": 0,
                            "outbound_calls": 0,
                            "inbound_calls": 0,
                            "internal_calls": 0,
                            "recording_count": 0,
                            "unique_destinations": set()
                        }
                    
                    # Update metrics
                    metrics = agent_metrics[agent_id]
                    
                    # Update cnam if empty and current call has a cnam
                    if not metrics["agent_cnam"] and call.get('cnam'):
                        metrics["agent_cnam"] = call.get('cnam')
                        
                    metrics["call_count"] += 1
                    
                    # Update disposition counts
                    disposition = call.get('disposition', '')
                    if disposition == DISPOSITION_ANSWERED:
                        metrics["answered_calls"] += 1
                        metrics["total_duration"] += call.get('duration', 0)
                        metrics["total_billsec"] += call.get('billsec', 0)
                    elif disposition == DISPOSITION_NO_ANSWER:
                        metrics["no_answer_calls"] += 1
                    elif disposition == DISPOSITION_BUSY:
                        metrics["busy_calls"] += 1
                    elif disposition == DISPOSITION_FAILED:
                        metrics["failed_calls"] += 1
                    
                    # Update direction counts
                    if call_direction == DIRECTION_OUTBOUND:
                        metrics["outbound_calls"] += 1
                    elif call_direction == DIRECTION_INBOUND:
                        metrics["inbound_calls"] += 1
                    elif call_direction == DIRECTION_INTERNAL:
                        metrics["internal_calls"] += 1
                    
                    # Check if has recording
                    if call.get('recordingfile', ''):
                        metrics["recording_count"] += 1
                    
                    # Add to unique destinations
                    metrics["unique_destinations"].add(call.get('dst', ''))
                
                print(f"[DEBUG] Total agents processed: {len(agent_metrics)}")
                    
                # Calculate derived metrics and process sets
                agent_list = []
                
                # Calculate team averages for gauge comparisons using team_stats
                team_avg_answer_rate = 0
                team_avg_call_duration = 0
                team_avg_recording_rate = 0
                team_avg_calls_per_day = 0
                total_valid_agents = 0
                
                print(f"[DEBUG] Starting team averages calculation with min_calls={min_calls}")
                
                # Calculate team averages from team-wide stats
                for team_agent in team_stats:
                    if team_agent["call_count"] >= min_calls:
                        total_valid_agents += 1
                        
                        # Convert Decimal values to float to avoid type errors
                        call_count = float(team_agent["call_count"])
                        answered_calls = float(team_agent["answered_calls"])
                        total_duration = float(team_agent["total_duration"])
                        recording_count = float(team_agent["recording_count"])
                        
                        # Calculate individual agent metrics
                        answer_rate = (answered_calls / call_count * 100) if call_count > 0 else 0
                        avg_duration = (total_duration / answered_calls) if answered_calls > 0 else 0
                        recording_rate = (recording_count / call_count * 100) if call_count > 0 else 0
                        calls_per_day = call_count / date_diff if date_diff > 0 else call_count
                        
                        print(f"[DEBUG] Team Agent {team_agent['agent']} metrics: calls={call_count}, answer_rate={answer_rate:.2f}%, avg_duration={avg_duration:.2f}s, recording_rate={recording_rate:.2f}%, calls_per_day={calls_per_day:.2f}")
                        
                        # Add to team totals
                        team_avg_answer_rate += answer_rate
                        team_avg_call_duration += avg_duration
                        team_avg_recording_rate += recording_rate
                        team_avg_calls_per_day += calls_per_day
                
                print(f"[DEBUG] Raw team totals (before division): answer_rate={team_avg_answer_rate:.2f}, avg_duration={team_avg_call_duration:.2f}, recording_rate={team_avg_recording_rate:.2f}, calls_per_day={team_avg_calls_per_day:.2f}")
                print(f"[DEBUG] Valid agents in team-wide data (meeting min_calls threshold): {total_valid_agents}")
                
                # Calculate team averages
                if total_valid_agents > 0:
                    team_avg_answer_rate /= total_valid_agents
                    team_avg_call_duration /= total_valid_agents
                    team_avg_recording_rate /= total_valid_agents
                    team_avg_calls_per_day /= total_valid_agents
                
                print(f"[DEBUG] Final team averages (from all agents): answer_rate={team_avg_answer_rate:.2f}%, avg_duration={team_avg_call_duration:.2f}s, recording_rate={team_avg_recording_rate:.2f}%, calls_per_day={team_avg_calls_per_day:.2f}")
                
                # Second pass to add gauge metrics
                for agent_id, metrics in agent_metrics.items():
                    # Convert unique destinations set to count
                    metrics["unique_destination_count"] = len(metrics["unique_destinations"])
                    del metrics["unique_destinations"]  # Remove the set for serialization
                    
                    # Calculate averages and rates
                    if metrics["call_count"] > 0:
                        metrics["answer_rate"] = round((metrics["answered_calls"] / metrics["call_count"]) * 100, 2)
                        metrics["recording_rate"] = round((metrics["recording_count"] / metrics["call_count"]) * 100, 2)
                    else:
                        metrics["answer_rate"] = 0
                        metrics["recording_rate"] = 0
                    
                    if metrics["answered_calls"] > 0:
                        metrics["avg_duration"] = round(metrics["total_duration"] / metrics["answered_calls"], 2)
                        metrics["avg_billsec"] = round(metrics["total_billsec"] / metrics["answered_calls"], 2)
                    else:
                        metrics["avg_duration"] = 0
                        metrics["avg_billsec"] = 0
                    
                    # Add disposition breakdown percentages
                    metrics["answered_percentage"] = round((metrics["answered_calls"] / metrics["call_count"]) * 100, 2) if metrics["call_count"] > 0 else 0
                    metrics["no_answer_percentage"] = round((metrics["no_answer_calls"] / metrics["call_count"]) * 100, 2) if metrics["call_count"] > 0 else 0
                    metrics["busy_percentage"] = round((metrics["busy_calls"] / metrics["call_count"]) * 100, 2) if metrics["call_count"] > 0 else 0
                    metrics["failed_percentage"] = round((metrics["failed_calls"] / metrics["call_count"]) * 100, 2) if metrics["call_count"] > 0 else 0
                    
                    # Add direction breakdown percentages
                    metrics["outbound_percentage"] = round((metrics["outbound_calls"] / metrics["call_count"]) * 100, 2) if metrics["call_count"] > 0 else 0
                    metrics["inbound_percentage"] = round((metrics["inbound_calls"] / metrics["call_count"]) * 100, 2) if metrics["call_count"] > 0 else 0
                    metrics["internal_percentage"] = round((metrics["internal_calls"] / metrics["call_count"]) * 100, 2) if metrics["call_count"] > 0 else 0
                    
                    # Add gauge metrics
                    # 1. Performance index (compared to team average)
                    calls_per_day = metrics["call_count"] / date_diff if date_diff > 0 else metrics["call_count"]
                    metrics["calls_per_day"] = round(calls_per_day, 2)
                    
                    # 2. Performance relative to team (percentage of team average)
                    answer_rate_vs_team = round((metrics["answer_rate"] / team_avg_answer_rate * 100) if team_avg_answer_rate > 0 else 0, 2)
                    duration_vs_team = round((metrics["avg_duration"] / team_avg_call_duration * 100) if team_avg_call_duration > 0 else 0, 2)
                    calls_vs_team = round((calls_per_day / team_avg_calls_per_day * 100) if team_avg_calls_per_day > 0 else 0, 2)
                    
                    metrics["answer_rate_vs_team"] = answer_rate_vs_team
                    metrics["duration_vs_team"] = duration_vs_team
                    metrics["calls_vs_team"] = calls_vs_team
                    
                    if metrics["call_count"] >= min_calls:
                        print(f"[DEBUG] Agent {agent_id} comparisons: answer_rate_vs_team={answer_rate_vs_team:.2f}%, duration_vs_team={duration_vs_team:.2f}%, calls_vs_team={calls_vs_team:.2f}%")
                    
                    # 3. Efficiency score (composite metric for gauge)
                    efficiency_factors = [
                        metrics["answer_rate"] / 100,  # Higher is better
                        min(1, 180 / metrics["avg_duration"]) if metrics["avg_duration"] > 0 else 0,  # Closer to target duration (180s) is better
                        metrics["recording_rate"] / 100  # Higher is better
                    ]
                    metrics["efficiency_score"] = round(sum(efficiency_factors) / len(efficiency_factors) * 100, 2)
                    
                    # 4. Quality indicator (for gauge chart)
                    if metrics["efficiency_score"] >= 85:
                        metrics["performance_level"] = "excellent"
                    elif metrics["efficiency_score"] >= 70:
                        metrics["performance_level"] = "good"
                    elif metrics["efficiency_score"] >= 55:
                        metrics["performance_level"] = "average"
                    elif metrics["efficiency_score"] >= 40:
                        metrics["performance_level"] = "below_average"
                    else:
                        metrics["performance_level"] = "needs_improvement"
                    
                    # Add to list if meets minimum call threshold
                    if metrics["call_count"] >= min_calls:
                        agent_list.append(metrics)
                
                print(f"[DEBUG] Total agents meeting min_calls threshold: {len(agent_list)}")
                
                # Sort by selected criterion
                if sort_by == "call_count":
                    agent_list.sort(key=lambda x: x["call_count"], reverse=True)
                elif sort_by == "answer_rate":
                    agent_list.sort(key=lambda x: x["answer_rate"], reverse=True)
                elif sort_by == "avg_duration":
                    agent_list.sort(key=lambda x: x["avg_duration"], reverse=True)
                else:
                    # Default to call count
                    agent_list.sort(key=lambda x: x["call_count"], reverse=True)
                
                # Limit results
                top_agents = agent_list[:limit] if len(agent_list) > limit else agent_list
                
                # Get disposition breakdown for all agents
                disposition_summary = {
                    "ANSWERED": sum(a["answered_calls"] for a in agent_list),
                    "NO ANSWER": sum(a["no_answer_calls"] for a in agent_list),
                    "BUSY": sum(a["busy_calls"] for a in agent_list),
                    "FAILED": sum(a["failed_calls"] for a in agent_list)
                }
                
                # Get direction breakdown for all agents
                direction_summary = {
                    "OUTBOUND": sum(a["outbound_calls"] for a in agent_list),
                    "INBOUND": sum(a["inbound_calls"] for a in agent_list),
                    "INTERNAL": sum(a["internal_calls"] for a in agent_list)
                }
                
                # Calculate total calls and percentages
                total_calls = sum(disposition_summary.values())
                
                if total_calls > 0:
                    disposition_percentages = {
                        disp: round((count / total_calls) * 100, 2)
                        for disp, count in disposition_summary.items()
                    }
                    
                    direction_percentages = {
                        dir_type: round((count / total_calls) * 100, 2)
                        for dir_type, count in direction_summary.items()
                    }
                else:
                    disposition_percentages = {disp: 0 for disp in disposition_summary}
                    direction_percentages = {dir_type: 0 for dir_type in direction_summary}
                
                # Prepare disposition data for stacked bar chart
                disposition_data_for_chart = []
                for agt in top_agents[:10]:  # Limit to top 10 for chart clarity
                    disposition_data_for_chart.append({
                        "agent": agt["agent"],
                        "agent_cnam": agt["agent_cnam"],
                        "ANSWERED": agt["answered_calls"],
                        "NO_ANSWER": agt["no_answer_calls"],
                        "BUSY": agt["busy_calls"],
                        "FAILED": agt["failed_calls"],
                        "total": agt["call_count"]
                    })
                
                # Prepare direction data for stacked bar chart
                direction_data_for_chart = []
                for agt in top_agents[:10]:  # Limit to top 10 for chart clarity
                    direction_data_for_chart.append({
                        "agent": agt["agent"],
                        "agent_cnam": agt["agent_cnam"],
                        "OUTBOUND": agt["outbound_calls"],
                        "INBOUND": agt["inbound_calls"],
                        "INTERNAL": agt["internal_calls"],
                        "total": agt["call_count"]
                    })
                
                return {
                    "time_period": {
                        "start_date": start_date,
                        "end_date": end_date,
                        "total_days": date_diff + 1
                    },
                    "filters": {
                        "agent": agent,
                        "disposition": "ALL" if not disposition_filter else disposition_filter,
                        "direction": "all" if not direction_filter else direction_filter,
                        "min_calls": min_calls
                    },
                    "summary": {
                        "total_agents": len(agent_list),
                        "total_calls": total_calls,
                        "disposition_summary": disposition_summary,
                        "disposition_percentages": disposition_percentages,
                        "direction_summary": direction_summary,
                        "direction_percentages": direction_percentages,
                        "avg_answer_rate": round(sum(a["answer_rate"] * a["call_count"] for a in agent_list) / total_calls, 2) if total_calls > 0 else 0,
                        "avg_handling_time": round(sum(a["total_duration"] for a in agent_list) / sum(a["answered_calls"] for a in agent_list), 2) if sum(a["answered_calls"] for a in agent_list) > 0 else 0,
                        "team_averages": {
                            "answer_rate": round(team_avg_answer_rate, 2),
                            "call_duration": round(team_avg_call_duration, 2),
                            "recording_rate": round(team_avg_recording_rate, 2),
                            "calls_per_day": round(team_avg_calls_per_day, 2)
                        }
                    },
                    "agents": top_agents,
                    "disposition_data": disposition_data_for_chart,
                    "direction_data": direction_data_for_chart,
                    "gauge_metrics": {
                        "answer_rate": [
                            {"agent": a["agent"], "agent_cnam": a.get("agent_cnam", ""), "value": a["answer_rate"], "vs_team": a["answer_rate_vs_team"]} 
                            for a in top_agents[:10]
                        ],
                        "efficiency_score": [
                            {"agent": a["agent"], "agent_cnam": a.get("agent_cnam", ""), "value": a["efficiency_score"], "level": a["performance_level"]} 
                            for a in top_agents[:10]
                        ],
                        "calls_per_day": [
                            {"agent": a["agent"], "agent_cnam": a.get("agent_cnam", ""), "value": a["calls_per_day"], "vs_team": a["calls_vs_team"]} 
                            for a in top_agents[:10]
                        ]
                    }
                }
                
        except pymysql.MySQLError as e:
            logger.error(f"MySQL Error: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Database error: {str(e)}"
            )
        finally:
            # Close the connection using our async function
            await close_conn()
                
    except Exception as e:
        logger.error(f"Error getting agent performance: {str(e)}")
        if isinstance(e, HTTPException):
            raise
        else:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to get agent performance: {str(e)}"
            )