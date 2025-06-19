import httpx
import json
import os
from datetime import datetime
from typing import Dict, Any, Optional, Annotated, List
import logging
from langchain_core.tools import tool
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, START, END
from langgraph.graph import MessagesState
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import MemorySaver
from langfuse import Langfuse
from langfuse.decorators import observe, langfuse_context
import tiktoken
import hashlib
import traceback
import asyncio
import re
import time

# Import the new Redis memory configuration
from .redis_memory_config import get_redis_checkpoint_saver, get_agent_memory_stats

logger = logging.getLogger(__name__)

# Initialize Redis-based memory saver for conversation state with fallback
redis_memory_saver = get_redis_checkpoint_saver(
    ttl_hours=24,  # 24 hour TTL for conversations
    fallback_to_memory=True  # Enable fallback to in-memory storage
)

# Constants for chunking
MAX_TOKENS_PER_RESPONSE = 8000  # Conservative limit for tool responses
MAX_CHUNK_SIZE = 3000  # Size of each chunk for processing
OVERLAP_SIZE = 200  # Overlap between chunks to maintain context

# Initialize tiktoken encoder for token counting
try:
    encoding = tiktoken.encoding_for_model("gpt-4")
except Exception:
    encoding = tiktoken.get_encoding("cl100k_base")

def count_tokens(text: str) -> int:
    """Count tokens in a text string."""
    try:
        return len(encoding.encode(text))
    except Exception:
        # Fallback to approximate token count
        return len(text.split()) * 1.3

def chunk_text(text: str, chunk_size: int = MAX_CHUNK_SIZE, overlap: int = OVERLAP_SIZE) -> List[str]:
    """
    Split text into overlapping chunks that fit within token limits.
    
    Args:
        text: Text to chunk
        chunk_size: Maximum tokens per chunk
        overlap: Number of tokens to overlap between chunks
    
    Returns:
        List of text chunks
    """
    tokens = encoding.encode(text)
    chunks = []
    
    start = 0
    while start < len(tokens):
        end = min(start + chunk_size, len(tokens))
        chunk_tokens = tokens[start:end]
        chunk_text = encoding.decode(chunk_tokens)
        chunks.append(chunk_text)
        
        if end == len(tokens):
            break
            
        start = end - overlap
    
    return chunks

async def summarize_chunk(llm: ChatOpenAI, chunk: str, data_type: str) -> str:
    """
    Summarize a chunk of call center data.
    
    Args:
        llm: The language model to use for summarization
        chunk: Text chunk to summarize
        data_type: Type of data (dashboard, metrics, etc.)
    
    Returns:
        Summarized text
    """
    try:
        summary_prompt = f"""
        Summarize the following {data_type} call center data, focusing on key metrics, trends, and insights:
        
        Data:
        {chunk}
        
        Please provide a concise summary that captures:
        - Key performance indicators and numbers
        - Important trends or patterns
        - Notable insights or anomalies
        - Actionable information for decision making
        
        Keep the summary under 500 words while preserving critical information.
        """
        
        response = await llm.ainvoke([HumanMessage(content=summary_prompt)])
        return response.content.strip()
        
    except Exception as e:
        logger.error(f"Error summarizing chunk: {e}")
        return f"Error summarizing {data_type} data: {str(e)}"

async def process_large_response(llm: ChatOpenAI, response_data: str, data_type: str) -> str:
    """
    Process potentially large response data with chunking and summarization.
    
    Args:
        llm: The language model for summarization
        response_data: Raw response data (JSON string)
        data_type: Type of data for context
    
    Returns:
        Processed response (original or summarized)
    """
    token_count = count_tokens(response_data)
    
    # If response is within limits, return as-is
    if token_count <= MAX_TOKENS_PER_RESPONSE:
        logger.info(f"{data_type} response size: {token_count} tokens (within limits)")
        return response_data
    
    logger.info(f"{data_type} response size: {token_count} tokens (exceeds limit, chunking...)")
    
    try:
        # Parse JSON to extract key sections for intelligent chunking
        data = json.loads(response_data)
        
        # Create structured summary based on data type
        if data_type == "dashboard":
            summary = await create_dashboard_summary(llm, data)
        elif data_type == "metrics":
            summary = await create_metrics_summary(llm, data)
        elif data_type == "direction_analysis":
            summary = await create_direction_summary(llm, data)
        else:
            # Generic chunking and summarization
            summary = await create_generic_summary(llm, response_data, data_type)
        
        logger.info(f"Summarized {data_type} from {token_count} to {count_tokens(summary)} tokens")
        return summary
        
    except json.JSONDecodeError:
        # If not JSON, use generic text chunking
        return await create_generic_summary(llm, response_data, data_type)

async def create_dashboard_summary(llm: ChatOpenAI, data: Dict[str, Any]) -> str:
    """Create intelligent summary of dashboard data."""
    try:
        summary_parts = []
        
        # Time period
        time_period = data.get("time_period", {})
        summary_parts.append(f"**Analysis Period**: {time_period.get('start_date')} to {time_period.get('end_date')} ({time_period.get('total_days')} days)")
        
        # Key metrics from summary
        summary = data.get("summary", {})
        if summary:
            summary_parts.append(f"""
**Key Performance Indicators**:
- Total Calls: {summary.get('total_calls', 'N/A')}
- Answer Rate: {summary.get('answer_rate', 'N/A')}%
- Average Duration: {summary.get('avg_duration', 'N/A')} seconds
- Recording Coverage: {summary.get('recording_percentage', 'N/A')}%
- Call Distribution: {summary.get('total_inbound', 0)} inbound, {summary.get('total_outbound', 0)} outbound, {summary.get('total_internal', 0)} internal
            """)
        
        # Summarize daily trends if available
        daily_data = data.get("daily_data", [])
        if daily_data and len(daily_data) > 0:
            # Get trend information
            first_day = daily_data[0].get("total", 0)
            last_day = daily_data[-1].get("total", 0)
            max_day = max(day.get("total", 0) for day in daily_data)
            
            summary_parts.append(f"""
**Daily Trends**:
- Call volume range: {min(day.get('total', 0) for day in daily_data)} to {max_day} calls per day
- Trend: {"Increasing" if last_day > first_day else "Decreasing" if last_day < first_day else "Stable"}
- Peak day had {max_day} calls
            """)
        
        # Summarize hourly patterns
        hourly_data = data.get("hourly_distribution", [])
        if hourly_data:
            peak_hour = max(hourly_data, key=lambda h: h.get("total", 0))
            summary_parts.append(f"**Peak Time**: {peak_hour.get('hour')}:00 with {peak_hour.get('total')} calls")
        
        return "\n".join(summary_parts)
        
    except Exception as e:
        logger.error(f"Error creating dashboard summary: {e}")
        return await create_generic_summary(llm, json.dumps(data), "dashboard")

async def create_metrics_summary(llm: ChatOpenAI, data: Dict[str, Any]) -> str:
    """Create intelligent summary of metrics data."""
    try:
        summary_parts = []
        
        # Basic metrics
        basic_metrics = data.get("basic_metrics", {})
        if basic_metrics:
            summary_parts.append(f"""
**Call Metrics Summary**:
- Total Calls: {basic_metrics.get('total_calls', 'N/A')}
- Answered: {basic_metrics.get('answered_calls', 'N/A')} calls
- Average Duration: {basic_metrics.get('avg_duration', 'N/A')} seconds
- Total Talk Time: {basic_metrics.get('total_duration', 'N/A')} seconds
            """)
        
        # Top callers
        top_callers = data.get("top_callers", [])
        if top_callers:
            summary_parts.append(f"**Top Caller**: {top_callers[0].get('src', 'N/A')} with {top_callers[0].get('call_count', 'N/A')} calls")
        
        return "\n".join(summary_parts)
        
    except Exception as e:
        logger.error(f"Error creating metrics summary: {e}")
        return await create_generic_summary(llm, json.dumps(data), "metrics")

async def create_direction_summary(llm: ChatOpenAI, data: Dict[str, Any]) -> str:
    """Create intelligent summary of direction analysis data."""
    try:
        summary_parts = []
        
        # Direction distribution
        summary = data.get("summary", {})
        direction_dist = summary.get("direction_distribution", {})
        
        if direction_dist:
            summary_parts.append(f"""
**Call Direction Analysis**:
- Inbound: {direction_dist.get('inbound', {}).get('count', 0)} calls ({direction_dist.get('inbound', {}).get('percentage', 0)}%)
- Outbound: {direction_dist.get('outbound', {}).get('count', 0)} calls ({direction_dist.get('outbound', {}).get('percentage', 0)}%)
- Internal: {direction_dist.get('internal', {}).get('count', 0)} calls ({direction_dist.get('internal', {}).get('percentage', 0)}%)
            """)
        
        # Top sources/destinations
        top_inbound = data.get("top_inbound_sources", [])[:3]
        top_outbound = data.get("top_outbound_destinations", [])[:3]
        
        if top_inbound:
            summary_parts.append(f"**Top Inbound Sources**: {', '.join([f'{s.get('source', '')} ({s.get('calls', 0)} calls)' for s in top_inbound])}")
        
        if top_outbound:
            summary_parts.append(f"**Top Outbound Destinations**: {', '.join([f'{d.get('destination', '')} ({d.get('calls', 0)} calls)' for d in top_outbound])}")
        
        return "\n".join(summary_parts)
        
    except Exception as e:
        logger.error(f"Error creating direction summary: {e}")
        return await create_generic_summary(llm, json.dumps(data), "direction_analysis")

async def create_generic_summary(llm: ChatOpenAI, response_data: str, data_type: str) -> str:
    """Create generic summary using chunking for any data type."""
    try:
        chunks = chunk_text(response_data)
        summaries = []
        
        for i, chunk in enumerate(chunks):
            chunk_summary = await summarize_chunk(llm, chunk, data_type)
            summaries.append(f"**Part {i+1}**: {chunk_summary}")
        
        final_summary = "\n\n".join(summaries)
        
        # If still too large, summarize the summaries
        if count_tokens(final_summary) > MAX_TOKENS_PER_RESPONSE:
            final_summary = await summarize_chunk(llm, final_summary, f"{data_type}_combined")
        
        return final_summary
        
    except Exception as e:
        logger.error(f"Error in generic summary: {e}")
        return f"Error processing {data_type} data: {str(e)}"

# Initialize Langfuse
def get_langfuse_client():
    """Initialize Langfuse client with environment credentials."""
    try:
        secret_key = os.getenv("LANGFUSE_SECRET_KEY")
        public_key = os.getenv("LANGFUSE_PUBLIC_KEY")
        host = os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")
        
        if not secret_key or not public_key:
            logger.warning("Langfuse credentials not found in environment variables")
            logger.warning(f"LANGFUSE_SECRET_KEY present: {bool(secret_key)}")
            logger.warning(f"LANGFUSE_PUBLIC_KEY present: {bool(public_key)}")
            logger.warning(f"LANGFUSE_HOST: {host}")
            return None
        
        langfuse = Langfuse(
            secret_key=secret_key,
            public_key=public_key,
            host=host
        )
        
        # Test the connection
        try:
            # This will test if we can connect to Langfuse
            langfuse.flush()
            logger.info(f"Langfuse client initialized successfully with host: {host}")
            logger.info(f"Public key: {public_key[:8]}...")
            return langfuse
        except Exception as e:
            logger.error(f"Failed to connect to Langfuse: {e}")
            return None
            
    except Exception as e:
        logger.error(f"Failed to initialize Langfuse: {e}")
        return None

# Global Langfuse client
langfuse_client = get_langfuse_client()

# Custom state schema that includes auth information
class CallCenterState(MessagesState):
    """Custom state that includes authentication and date information."""
    auth_token: Annotated[str, "Authentication token for API calls"]
    start_date: Annotated[str, "Start date for analysis"]
    end_date: Annotated[str, "End date for analysis"]
    session_id: Annotated[str, "Session ID for grouping traces"]
    original_question: Annotated[str, "Original user question for final validation"]
    recursion_count: Annotated[int, "Number of times the flow has restarted"]
    final_model_used: Annotated[str, "Model that generated the final response (gpt-4o or gemini-2.5-flash)"]

class CallCenterAgent:
    def __init__(self, openai_api_key: str, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.llm = ChatOpenAI(model="gpt-4o", api_key=openai_api_key, temperature=0.3)
        
        # Initialize Gemini for final formatting
        google_api_key = os.getenv("GOOGLE_API_KEY")
        if not google_api_key:
            logger.warning("GOOGLE_API_KEY not found in environment variables. Gemini formatting will be disabled.")
            self.gemini_llm = None
        else:
            self.gemini_llm = ChatGoogleGenerativeAI(
                model="gemini-1.5-flash",
                google_api_key=google_api_key,
                temperature=0.1
            )
            logger.info("Gemini LLM initialized successfully")
        
        self.current_state = {}  # Store current state for tools to access
        self.tools = self._create_tools()
        self.llm_with_tools = self.llm.bind_tools(self.tools)
        self.graph = self._create_graph()
    
    def _create_tools(self):
        """Create all the endpoint tools for the agent."""
        
        # Capture base_url and agent instance in closure
        base_url = self.base_url
        agent_instance = self
        
        @tool
        @observe(name="get_dashboard_data_tool")
        async def get_dashboard_data(start_date: str = None, end_date: str = None, auth_token: str = None) -> str:
            """
            Get comprehensive call center dashboard data including metrics, trends, and visualizations.
            
            Args:
                start_date: Start date in YYYY-MM-DD format (optional, will use from state if not provided)
                end_date: End date in YYYY-MM-DD format (optional, will use from state if not provided)
                auth_token: Authentication token for API calls (optional, will use from state if not provided)
            
            Returns:
                JSON string with comprehensive dashboard data (chunked and summarized if needed)
            """
            # Get values from state if not provided - ensure we get the right values
            agent_state = agent_instance.current_state
            start_date = start_date or agent_state.get("start_date")
            end_date = end_date or agent_state.get("end_date")
            auth_token = auth_token or agent_state.get("auth_token")
            
            # CRITICAL: Validate that auth_token is not session_id
            if auth_token and auth_token.startswith("agent_session_"):
                print(f"[CRITICAL ERROR] auth_token is actually session_id: {auth_token}")
                auth_token = agent_state.get("auth_token")  # Force re-fetch from state
                print(f"[CRITICAL FIX] Using correct auth_token from state: {auth_token[:30] if auth_token else 'None'}...")
            
            # Debug logging with validation
            print(f"[DEBUG] get_dashboard_data called with:")
            print(f"  start_date: {start_date}")
            print(f"  end_date: {end_date}")
            print(f"  auth_token: {auth_token[:30] if auth_token else 'None/Empty'}...")
            print(f"  agent_state keys: {list(agent_state.keys()) if agent_state else 'None'}")
            
            # Log to Langfuse
            if langfuse_client:
                # Get session_id from agent state
                current_session_id = agent_state.get("session_id", "")
                
                trace = langfuse_client.trace(
                    name="dashboard_data_tool",
                    input={
                        "start_date": start_date,
                        "end_date": end_date,
                        "tool": "get_dashboard_data"
                    },
                    tags=["call_center_agent", "dashboard", "tool_call"],
                    session_id=current_session_id  # Add session_id to group with main session
                )
            
            # Validate required parameters
            if not auth_token or auth_token == "None" or len(str(auth_token).strip()) == 0:
                error_msg = "AUTHENTICATION ERROR: auth_token is missing or empty! Cannot access call center data without valid authentication."
                print(f"[DEBUG] {error_msg}")
                
                # Log error to Langfuse
                if langfuse_client and 'trace' in locals():
                    trace.update(output={"error": error_msg, "error_type": "authentication"}, level="ERROR")
                
                return error_msg
                
            if not start_date or not end_date:
                error_msg = "PARAMETER ERROR: start_date or end_date is missing! Cannot fetch data without valid date range."
                print(f"[DEBUG] {error_msg}")
                
                # Log error to Langfuse
                if langfuse_client and 'trace' in locals():
                    trace.update(output={"error": error_msg, "error_type": "missing_parameters"}, level="ERROR")
                
                return error_msg
            
            try:
                async with httpx.AsyncClient() as client:
                    response = await client.get(
                        f"{base_url}/call-records/dashboard",
                        params={"start_date": start_date, "end_date": end_date},
                        headers={"Authorization": f"Bearer {auth_token}"},
                        timeout=30.0
                    )
                    print(f"[DEBUG] HTTP response status: {response.status_code}")
                    
                    # Handle specific HTTP errors
                    if response.status_code == 401:
                        error_msg = "AUTHENTICATION_TERMINAL_ERROR: The provided token is invalid or expired. Please check your authentication credentials and try again with a valid token."
                        print(f"[DEBUG] {error_msg}")
                        if langfuse_client and 'trace' in locals():
                            trace.update(output={"error": error_msg, "error_type": "invalid_token", "http_status": 401, "terminal": True}, level="ERROR")
                        return error_msg
                    elif response.status_code == 403:
                        error_msg = "ACCESS FORBIDDEN: Your token does not have permission to access this data. Please check your access rights."
                        print(f"[DEBUG] {error_msg}")
                        if langfuse_client and 'trace' in locals():
                            trace.update(output={"error": error_msg, "error_type": "forbidden", "http_status": 403}, level="ERROR")
                        return error_msg
                    elif response.status_code >= 400:
                        error_msg = f"HTTP ERROR {response.status_code}: {response.text}"
                        print(f"[DEBUG] {error_msg}")
                        if langfuse_client and 'trace' in locals():
                            trace.update(output={"error": error_msg, "error_type": "http_error", "http_status": response.status_code}, level="ERROR")
                        return error_msg
                    
                    response.raise_for_status()
                    result = response.json()
                    raw_response = json.dumps(result)
                    
                    print(f"[DEBUG] Raw dashboard data size: {len(raw_response)} characters")
                    
                    # Process response with chunking and summarization if needed
                    processed_response = await process_large_response(
                        agent_instance.llm, 
                        raw_response, 
                        "dashboard"
                    )
                    
                    print(f"[DEBUG] Processed dashboard data size: {len(processed_response)} characters")
                    
                    # Log success to Langfuse
                    if langfuse_client and 'trace' in locals():
                        trace.update(
                            output={
                                "status": "success",
                                "raw_data_size": len(raw_response),
                                "processed_data_size": len(processed_response),
                                "http_status": response.status_code,
                                "was_chunked": len(raw_response) != len(processed_response),
                                "summary": {
                                    "total_calls": result.get("summary", {}).get("total_calls"),
                                    "answer_rate": result.get("summary", {}).get("answer_rate"),
                                    "time_period": result.get("time_period")
                                }
                            },
                            level="DEFAULT"
                        )
                    
                    return processed_response
                    
            except httpx.HTTPStatusError as e:
                error_msg = f"HTTP STATUS ERROR {e.response.status_code}: Failed to fetch dashboard data - {str(e)}"
                print(f"[DEBUG] {error_msg}")
                if langfuse_client and 'trace' in locals():
                    trace.update(output={"error": error_msg, "error_type": "http_status_error"}, level="ERROR")
                return error_msg
            except httpx.TimeoutException:
                error_msg = "TIMEOUT ERROR: Request to dashboard endpoint timed out. The server may be overloaded."
                print(f"[DEBUG] {error_msg}")
                if langfuse_client and 'trace' in locals():
                    trace.update(output={"error": error_msg, "error_type": "timeout"}, level="ERROR")
                return error_msg
            except Exception as e:
                error_msg = f"UNEXPECTED ERROR fetching dashboard data: {str(e)}"
                print(f"[DEBUG] {error_msg}")
                
                # Log error to Langfuse
                if langfuse_client and 'trace' in locals():
                    trace.update(output={"error": error_msg, "error_type": "unexpected"}, level="ERROR")
                
                return error_msg

        @tool
        @observe(name="get_call_metrics_tool")
        async def get_call_metrics(start_date: str = None, end_date: str = None, auth_token: str = None, disposition: str = None) -> str:
            """
            Get aggregated call metrics and analytics.
            
            Args:
                start_date: Start date in YYYY-MM-DD format (optional, will use from state if not provided)
                end_date: End date in YYYY-MM-DD format (optional, will use from state if not provided)
                auth_token: Authentication token (optional, will use from state if not provided)
                disposition: Optional filter by call disposition (ANSWERED, NO ANSWER, etc.)
            
            Returns:
                JSON string with call metrics (chunked and summarized if needed)
            """
            # Get values from state if not provided - ensure we get the right values
            agent_state = agent_instance.current_state
            start_date = start_date or agent_state.get("start_date")
            end_date = end_date or agent_state.get("end_date")
            auth_token = auth_token or agent_state.get("auth_token")
            
            # CRITICAL: Validate that auth_token is not session_id
            if auth_token and auth_token.startswith("agent_session_"):
                print(f"[CRITICAL ERROR] auth_token is actually session_id: {auth_token}")
                auth_token = agent_state.get("auth_token")  # Force re-fetch from state
                print(f"[CRITICAL FIX] Using correct auth_token from state: {auth_token[:30] if auth_token else 'None'}...")
            
            print(f"[DEBUG] get_call_metrics called with:")
            print(f"  start_date: {start_date}")
            print(f"  end_date: {end_date}")
            print(f"  auth_token: {auth_token[:30] if auth_token else 'None/Empty'}...")
            print(f"  disposition: {disposition}")
            print(f"  agent_state keys: {list(agent_state.keys()) if agent_state else 'None'}")
            
            # Log to Langfuse
            if langfuse_client:
                # Get session_id from agent state
                current_session_id = agent_state.get("session_id", "")
                
                trace = langfuse_client.trace(
                    name="call_metrics_tool",
                    input={
                        "start_date": start_date,
                        "end_date": end_date,
                        "disposition": disposition,
                        "tool": "get_call_metrics"
                    },
                    tags=["call_center_agent", "metrics", "tool_call"],
                    session_id=current_session_id  # Add session_id to group with main session
                )
            
            # Validate required parameters
            if not auth_token or len(str(auth_token).strip()) == 0:
                error_msg = "AUTHENTICATION ERROR: auth_token is missing or empty! Cannot access call center data without valid authentication."
                print(f"[DEBUG] {error_msg}")
                
                # Log error to Langfuse
                if langfuse_client and 'trace' in locals():
                    trace.update(output={"error": error_msg, "error_type": "authentication"}, level="ERROR")
                
                return error_msg
                
            if not start_date or not end_date:
                error_msg = "PARAMETER ERROR: start_date or end_date is missing! Cannot fetch data without valid date range."
                print(f"[DEBUG] {error_msg}")
                
                # Log error to Langfuse
                if langfuse_client and 'trace' in locals():
                    trace.update(output={"error": error_msg, "error_type": "missing_parameters"}, level="ERROR")
                
                return error_msg
            
            try:
                params = {"start_date": start_date, "end_date": end_date}
                if disposition:
                    params["disposition"] = disposition
                    
                async with httpx.AsyncClient() as client:
                    response = await client.get(
                        f"{base_url}/call-records/metrics",
                        params=params,
                        headers={"Authorization": f"Bearer {auth_token}"},
                        timeout=30.0
                    )
                    print(f"[DEBUG] get_call_metrics HTTP response status: {response.status_code}")
                    
                    # Handle specific HTTP errors
                    if response.status_code == 401:
                        error_msg = "AUTHENTICATION_TERMINAL_ERROR: The provided token is invalid or expired. Please check your authentication credentials and try again with a valid token."
                        print(f"[DEBUG] {error_msg}")
                        if langfuse_client and 'trace' in locals():
                            trace.update(output={"error": error_msg, "error_type": "invalid_token", "http_status": 401, "terminal": True}, level="ERROR")
                        return error_msg
                    elif response.status_code == 403:
                        error_msg = "ACCESS FORBIDDEN: Your token does not have permission to access this data. Please check your access rights."
                        print(f"[DEBUG] {error_msg}")
                        if langfuse_client and 'trace' in locals():
                            trace.update(output={"error": error_msg, "error_type": "forbidden", "http_status": 403}, level="ERROR")
                        return error_msg
                    elif response.status_code >= 400:
                        error_msg = f"HTTP ERROR {response.status_code}: {response.text}"
                        print(f"[DEBUG] {error_msg}")
                        if langfuse_client and 'trace' in locals():
                            trace.update(output={"error": error_msg, "error_type": "http_error", "http_status": response.status_code}, level="ERROR")
                        return error_msg
                    
                    response.raise_for_status()
                    result = response.json()
                    raw_response = json.dumps(result)
                    
                    print(f"[DEBUG] Raw metrics data size: {len(raw_response)} characters")
                    
                    # Process response with chunking and summarization if needed
                    processed_response = await process_large_response(
                        agent_instance.llm, 
                        raw_response, 
                        "metrics"
                    )
                    
                    print(f"[DEBUG] Processed metrics data size: {len(processed_response)} characters")
                    
                    # Log success to Langfuse
                    if langfuse_client and 'trace' in locals():
                        trace.update(
                            output={
                                "status": "success",
                                "raw_data_size": len(raw_response),
                                "processed_data_size": len(processed_response),
                                "http_status": response.status_code,
                                "was_chunked": len(raw_response) != len(processed_response),
                                "basic_metrics": result.get("basic_metrics"),
                                "time_period": result.get("time_period")
                            },
                            level="DEFAULT"
                        )
                    
                    return processed_response
                    
            except httpx.HTTPStatusError as e:
                error_msg = f"HTTP STATUS ERROR {e.response.status_code}: Failed to fetch call metrics - {str(e)}"
                print(f"[DEBUG] {error_msg}")
                if langfuse_client and 'trace' in locals():
                    trace.update(output={"error": error_msg, "error_type": "http_status_error"}, level="ERROR")
                return error_msg
            except httpx.TimeoutException:
                error_msg = "TIMEOUT ERROR: Request to metrics endpoint timed out. The server may be overloaded."
                print(f"[DEBUG] {error_msg}")
                if langfuse_client and 'trace' in locals():
                    trace.update(output={"error": error_msg, "error_type": "timeout"}, level="ERROR")
                return error_msg
            except Exception as e:
                error_msg = f"UNEXPECTED ERROR fetching call metrics: {str(e)}"
                print(f"[DEBUG] {error_msg}")
                
                # Log error to Langfuse
                if langfuse_client and 'trace' in locals():
                    trace.update(output={"error": error_msg, "error_type": "unexpected"}, level="ERROR")
                
                return error_msg

        @tool
        async def get_direction_analysis(start_date: str = None, end_date: str = None, auth_token: str = None) -> str:
            """
            Get detailed analysis of call directions (inbound, outbound, internal).
            
            Args:
                start_date: Start date in YYYY-MM-DD format (optional, will use from state if not provided)
                end_date: End date in YYYY-MM-DD format (optional, will use from state if not provided)
                auth_token: Authentication token (optional, will use from state if not provided)
            
            Returns:
                JSON string with direction analysis data
            """
            # Get values from state if not provided - ensure we get the right values
            agent_state = agent_instance.current_state
            start_date = start_date or agent_state.get("start_date")
            end_date = end_date or agent_state.get("end_date")
            auth_token = auth_token or agent_state.get("auth_token")
            
            # CRITICAL: Validate that auth_token is not session_id
            if auth_token and auth_token.startswith("agent_session_"):
                print(f"[CRITICAL ERROR] get_direction_analysis auth_token is actually session_id: {auth_token}")
                auth_token = agent_state.get("auth_token")  # Force re-fetch from state
                print(f"[CRITICAL FIX] Using correct auth_token from state: {auth_token[:30] if auth_token else 'None'}...")
            
            try:
                async with httpx.AsyncClient() as client:
                    response = await client.get(
                        f"{base_url}/call-records/direction-analysis",
                        params={"start_date": start_date, "end_date": end_date},
                        headers={"Authorization": f"Bearer {auth_token}"},
                        timeout=30.0
                    )
                    response.raise_for_status()
                    return json.dumps(response.json())
            except Exception as e:
                return f"Error fetching direction analysis: {str(e)}"

        @tool
        async def get_caller_analysis(start_date: str = None, end_date: str = None, auth_token: str = None, 
                                    direction: str = None, limit: int = 20) -> str:
            """
            Get caller behavior analysis and top caller insights.
            
            Args:
                start_date: Start date in YYYY-MM-DD format (optional, will use from state if not provided)
                end_date: End date in YYYY-MM-DD format (optional, will use from state if not provided)
                auth_token: Authentication token (optional, will use from state if not provided)
                direction: Optional filter by direction (inbound, outbound, internal)
                limit: Number of top callers to return
            
            Returns:
                JSON string with caller analysis
            """
            # Get values from state if not provided - ensure we get the right values
            agent_state = agent_instance.current_state
            start_date = start_date or agent_state.get("start_date")
            end_date = end_date or agent_state.get("end_date")
            auth_token = auth_token or agent_state.get("auth_token")
            
            # CRITICAL: Validate that auth_token is not session_id
            if auth_token and auth_token.startswith("agent_session_"):
                print(f"[CRITICAL ERROR] get_caller_analysis auth_token is actually session_id: {auth_token}")
                auth_token = agent_state.get("auth_token")  # Force re-fetch from state
                print(f"[CRITICAL FIX] Using correct auth_token from state: {auth_token[:30] if auth_token else 'None'}...")
            
            try:
                params = {"start_date": start_date, "end_date": end_date, "limit": limit}
                if direction:
                    params["direction"] = direction
                    
                async with httpx.AsyncClient() as client:
                    response = await client.get(
                        f"{base_url}/call-records/caller-analysis",
                        params=params,
                        headers={"Authorization": f"Bearer {auth_token}"},
                        timeout=30.0
                    )
                    response.raise_for_status()
                    return json.dumps(response.json())
            except Exception as e:
                return f"Error fetching caller analysis: {str(e)}"

        @tool
        async def get_performance_metrics(start_date: str = None, end_date: str = None, auth_token: str = None,
                                        time_resolution: str = "day") -> str:
            """
            Get call performance metrics including duration trends and answer rates.
            
            Args:
                start_date: Start date in YYYY-MM-DD format (optional, will use from state if not provided)
                end_date: End date in YYYY-MM-DD format (optional, will use from state if not provided)
                auth_token: Authentication token (optional, will use from state if not provided)
                time_resolution: Time resolution (day, hour, week, month)
            
            Returns:
                JSON string with performance metrics
            """
            # Get values from state if not provided - ensure we get the right values
            agent_state = agent_instance.current_state
            start_date = start_date or agent_state.get("start_date")
            end_date = end_date or agent_state.get("end_date")
            auth_token = auth_token or agent_state.get("auth_token")
            
            # CRITICAL: Validate that auth_token is not session_id
            if auth_token and auth_token.startswith("agent_session_"):
                print(f"[CRITICAL ERROR] get_performance_metrics auth_token is actually session_id: {auth_token}")
                auth_token = agent_state.get("auth_token")  # Force re-fetch from state
                print(f"[CRITICAL FIX] Using correct auth_token from state: {auth_token[:30] if auth_token else 'None'}...")
            
            try:
                async with httpx.AsyncClient() as client:
                    response = await client.get(
                        f"{base_url}/call-records/performance-metrics",
                        params={
                            "start_date": start_date, 
                            "end_date": end_date,
                            "time_resolution": time_resolution
                        },
                        headers={"Authorization": f"Bearer {auth_token}"},
                        timeout=30.0
                    )
                    response.raise_for_status()
                    return json.dumps(response.json())
            except Exception as e:
                return f"Error fetching performance metrics: {str(e)}"

        @tool
        async def get_agent_performance(start_date: str = None, end_date: str = None, auth_token: str = None,
                                      agent: str = None, limit: int = 20) -> str:
            """
            Get agent performance metrics and comparisons.
            
            Args:
                start_date: Start date in YYYY-MM-DD format (optional, will use from state if not provided)
                end_date: End date in YYYY-MM-DD format (optional, will use from state if not provided)
                auth_token: Authentication token (optional, will use from state if not provided)
                agent: Optional specific agent to analyze
                limit: Number of top agents to return
            
            Returns:
                JSON string with agent performance data
            """
            # Get values from state if not provided - ensure we get the right values
            agent_state = agent_instance.current_state
            start_date = start_date or agent_state.get("start_date")
            end_date = end_date or agent_state.get("end_date")
            auth_token = auth_token or agent_state.get("auth_token")
            
            # CRITICAL: Validate that auth_token is not session_id
            if auth_token and auth_token.startswith("agent_session_"):
                print(f"[CRITICAL ERROR] get_agent_performance auth_token is actually session_id: {auth_token}")
                auth_token = agent_state.get("auth_token")  # Force re-fetch from state
                print(f"[CRITICAL FIX] Using correct auth_token from state: {auth_token[:30] if auth_token else 'None'}...")
            
            try:
                params = {"start_date": start_date, "end_date": end_date, "limit": limit}
                if agent:
                    params["agent"] = agent
                    
                async with httpx.AsyncClient() as client:
                    response = await client.get(
                        f"{base_url}/call-records/agent-performance",
                        params=params,
                        headers={"Authorization": f"Bearer {auth_token}"},
                        timeout=30.0
                    )
                    response.raise_for_status()
                    return json.dumps(response.json())
            except Exception as e:
                return f"Error fetching agent performance: {str(e)}"

        @tool
        async def get_extension_performance(start_date: str = None, end_date: str = None, auth_token: str = None,
                                          extension: str = None, limit: int = 20) -> str:
            """
            Get extension performance metrics and call handling statistics.
            
            Args:
                start_date: Start date in YYYY-MM-DD format (optional, will use from state if not provided)
                end_date: End date in YYYY-MM-DD format (optional, will use from state if not provided)
                auth_token: Authentication token (optional, will use from state if not provided)
                extension: Optional specific extension to analyze
                limit: Number of top extensions to return
            
            Returns:
                JSON string with extension performance data
            """
            # Get values from state if not provided - ensure we get the right values
            agent_state = agent_instance.current_state
            start_date = start_date or agent_state.get("start_date")
            end_date = end_date or agent_state.get("end_date")
            auth_token = auth_token or agent_state.get("auth_token")
            
            # CRITICAL: Validate that auth_token is not session_id
            if auth_token and auth_token.startswith("agent_session_"):
                print(f"[CRITICAL ERROR] get_extension_performance auth_token is actually session_id: {auth_token}")
                auth_token = agent_state.get("auth_token")  # Force re-fetch from state
                print(f"[CRITICAL FIX] Using correct auth_token from state: {auth_token[:30] if auth_token else 'None'}...")
            
            try:
                params = {"start_date": start_date, "end_date": end_date, "limit": limit}
                if extension:
                    params["extension"] = extension
                    
                async with httpx.AsyncClient() as client:
                    response = await client.get(
                        f"{base_url}/call-records/extension-performance",
                        params=params,
                        headers={"Authorization": f"Bearer {auth_token}"},
                        timeout=30.0
                    )
                    response.raise_for_status()
                    return json.dumps(response.json())
            except Exception as e:
                return f"Error fetching extension performance: {str(e)}"

        @tool
        async def get_hourly_distribution(start_date: str = None, end_date: str = None, auth_token: str = None,
                                        direction: str = None, group_by_day: bool = False) -> str:
            """
            Get hourly call distribution for identifying peak times.
            
            Args:
                start_date: Start date in YYYY-MM-DD format (optional, will use from state if not provided)
                end_date: End date in YYYY-MM-DD format (optional, will use from state if not provided)
                auth_token: Authentication token (optional, will use from state if not provided)
                direction: Optional filter by direction
                group_by_day: Whether to group by day of week
            
            Returns:
                JSON string with hourly distribution data
            """
            # Get values from state if not provided - ensure we get the right values
            agent_state = agent_instance.current_state
            start_date = start_date or agent_state.get("start_date")
            end_date = end_date or agent_state.get("end_date")
            auth_token = auth_token or agent_state.get("auth_token")
            
            # CRITICAL: Validate that auth_token is not session_id
            if auth_token and auth_token.startswith("agent_session_"):
                print(f"[CRITICAL ERROR] get_hourly_distribution auth_token is actually session_id: {auth_token}")
                auth_token = agent_state.get("auth_token")  # Force re-fetch from state
                print(f"[CRITICAL FIX] Using correct auth_token from state: {auth_token[:30] if auth_token else 'None'}...")
            
            try:
                params = {
                    "start_date": start_date, 
                    "end_date": end_date,
                    "group_by_day": group_by_day
                }
                if direction:
                    params["direction"] = direction
                    
                async with httpx.AsyncClient() as client:
                    response = await client.get(
                        f"{base_url}/call-records/hourly-distribution",
                        params=params,
                        headers={"Authorization": f"Bearer {auth_token}"},
                        timeout=30.0
                    )
                    response.raise_for_status()
                    return json.dumps(response.json())
            except Exception as e:
                return f"Error fetching hourly distribution: {str(e)}"

        @tool
        async def get_call_logs(start_date: str = None, end_date: str = None, auth_token: str = None,
                              disposition: str = None, direction: str = None, 
                              limit: int = 100, include_details: bool = False) -> str:
            """
            Get detailed call logs with filtering options.
            
            Args:
                start_date: Start date in YYYY-MM-DD format (optional, will use from state if not provided)
                end_date: End date in YYYY-MM-DD format (optional, will use from state if not provided)
                auth_token: Authentication token (optional, will use from state if not provided)
                disposition: Optional filter by disposition
                direction: Optional filter by direction
                limit: Maximum number of records
                include_details: Whether to include detailed call records
            
            Returns:
                JSON string with call logs and summary
            """
            # Get values from state if not provided - ensure we get the right values
            agent_state = agent_instance.current_state
            start_date = start_date or agent_state.get("start_date")
            end_date = end_date or agent_state.get("end_date")
            auth_token = auth_token or agent_state.get("auth_token")
            
            # CRITICAL: Validate that auth_token is not session_id
            if auth_token and auth_token.startswith("agent_session_"):
                print(f"[CRITICAL ERROR] get_call_logs auth_token is actually session_id: {auth_token}")
                auth_token = agent_state.get("auth_token")  # Force re-fetch from state
                print(f"[CRITICAL FIX] Using correct auth_token from state: {auth_token[:30] if auth_token else 'None'}...")
            
            try:
                params = {
                    "start_date": start_date,
                    "end_date": end_date,
                    "limit": limit,
                    "include_details": include_details
                }
                if disposition:
                    params["disposition"] = disposition
                if direction:
                    params["direction"] = direction
                    
                async with httpx.AsyncClient() as client:
                    response = await client.get(
                        f"{base_url}/call-records/logs",
                        params=params,
                        headers={"Authorization": f"Bearer {auth_token}"},
                        timeout=30.0
                    )
                    response.raise_for_status()
                    return json.dumps(response.json())
            except Exception as e:
                return f"Error fetching call logs: {str(e)}"

        @tool
        async def get_recording_info(full_path: str, auth_token: str = None) -> str:
            """
            Get information about a specific call recording file.
            
            Args:
                full_path: Full path to the recording file on SFTP server
                auth_token: Authentication token (optional, will use from state if not provided)
            
            Returns:
                JSON string with file details
            """
            # Get auth_token from state if not provided
            agent_state = agent_instance.current_state
            auth_token = auth_token or agent_state.get("auth_token")
            
            # CRITICAL: Validate that auth_token is not session_id
            if auth_token and auth_token.startswith("agent_session_"):
                print(f"[CRITICAL ERROR] get_recording_info auth_token is actually session_id: {auth_token}")
                auth_token = agent_state.get("auth_token")  # Force re-fetch from state
                print(f"[CRITICAL FIX] Using correct auth_token from state: {auth_token[:30] if auth_token else 'None'}...")
            
            try:
                async with httpx.AsyncClient() as client:
                    response = await client.get(
                        f"{base_url}/sftp-file-details",
                        params={"full_path": full_path},
                        headers={"Authorization": f"Bearer {auth_token}"},
                        timeout=30.0
                    )
                    response.raise_for_status()
                    return json.dumps(response.json())
            except Exception as e:
                return f"Error fetching recording info: {str(e)}"

        @tool
        @observe(name="extract_recording_paths_tool")
        async def extract_recording_paths(json_data: str, auth_token: str = None) -> str:
            """
            Extract all recording file paths from call data JSON.
            
            Args:
                json_data: JSON string or data containing call records
                auth_token: Authentication token (optional, will use from state if not provided)
            
            Returns:
                JSON string with recording paths and count
            """
            # Get auth_token from state if not provided
            agent_state = agent_instance.current_state
            auth_token = auth_token or agent_state.get("auth_token")
            
            # CRITICAL: Validate that auth_token is not session_id
            if auth_token and auth_token.startswith("agent_session_"):
                print(f"[CRITICAL ERROR] extract_recording_paths auth_token is actually session_id: {auth_token}")
                auth_token = agent_state.get("auth_token")  # Force re-fetch from state
                print(f"[CRITICAL FIX] Using correct auth_token from state: {auth_token[:30] if auth_token else 'None'}...")
            
            # Log to Langfuse
            if langfuse_client:
                current_session_id = agent_state.get("session_id", "")
                trace = langfuse_client.trace(
                    name="extract_recording_paths_tool",
                    input={
                        "data_size": len(json_data) if isinstance(json_data, str) else len(str(json_data)),
                        "tool": "extract_recording_paths"
                    },
                    tags=["call_center_agent", "recording_paths", "tool_call"],
                    session_id=current_session_id
                )
            
            try:
                async with httpx.AsyncClient() as client:
                    response = await client.post(
                        f"{base_url}/extract-recording-paths",
                        json={"json_data": json_data},
                        headers={"Authorization": f"Bearer {auth_token}"},
                        timeout=30.0
                    )
                    response.raise_for_status()
                    result = response.json()
                    
                    # Log success to Langfuse
                    if langfuse_client and 'trace' in locals():
                        trace.update(
                            output={
                                "status": "success",
                                "recording_count": result.get("count", 0),
                                "paths_found": len(result.get("recording_paths", [])),
                                "http_status": response.status_code
                            },
                            level="DEFAULT"
                        )
                    
                    return json.dumps(result)
            except Exception as e:
                error_msg = f"Error extracting recording paths: {str(e)}"
                
                # Log error to Langfuse
                if langfuse_client and 'trace' in locals():
                    trace.update(output={"error": error_msg}, level="ERROR")
                
                return error_msg

        def create_langfuse_wrapped_tool(tool_func, tool_name, tool_tags):
            """Create a Langfuse-wrapped version of a tool function."""
            
            async def wrapped_tool(*args, **kwargs):
                # Get values from state if not provided
                start_date = kwargs.get('start_date') or agent_instance.current_state.get("start_date")
                end_date = kwargs.get('end_date') or agent_instance.current_state.get("end_date")
                auth_token = kwargs.get('auth_token') or agent_instance.current_state.get("auth_token")
                session_id = agent_instance.current_state.get("session_id", "")
                
                # Update kwargs with state values
                kwargs.update({
                    'start_date': start_date,
                    'end_date': end_date,
                    'auth_token': auth_token
                })
                
                # Log to Langfuse
                trace = None
                if langfuse_client:
                    trace = langfuse_client.trace(
                        name=f"{tool_name}_tool",
                        input={k: v for k, v in kwargs.items() if k != 'auth_token'},
                        tags=["call_center_agent"] + tool_tags,
                        session_id=session_id  # Add session_id to group with main session
                    )
                
                try:
                    result = await tool_func(*args, **kwargs)
                    
                    # Log success to Langfuse
                    if trace and not result.startswith("Error"):
                        trace.update(
                            output={
                                "status": "success",
                                "data_size": len(result),
                                "tool": tool_name,
                                "session_id": session_id
                            },
                            level="DEFAULT"
                        )
                    elif trace:
                        trace.update(
                            output={"error": result, "session_id": session_id},
                            level="ERROR"
                        )
                    
                    return result
                    
                except Exception as e:
                    error_msg = f"Error in {tool_name}: {str(e)}"
                    if trace:
                        trace.update(
                            output={"error": error_msg, "session_id": session_id}, 
                            level="ERROR"
                        )
                    return error_msg
            
            return wrapped_tool

        return [
            get_dashboard_data,
            get_call_metrics,
            get_direction_analysis,
            get_caller_analysis,
            get_performance_metrics,
            get_agent_performance,
            get_extension_performance,
            get_hourly_distribution,
            get_call_logs,
            get_recording_info,
            extract_recording_paths
        ]
    
    def _create_graph(self):
        """Create the agent workflow graph with Gemini final formatting."""
        workflow = StateGraph(CallCenterState)
        
        # Add nodes - 3 nodes for the enhanced flow
        workflow.add_node("tool_calling_llm", self._agent_node)
        workflow.add_node("tools", ToolNode(self.tools))
        workflow.add_node("gemini_formatter", self._gemini_formatter_node)
        
        # Add edges
        workflow.add_edge(START, "tool_calling_llm")
        
        # Add conditional edges using custom tools condition
        workflow.add_conditional_edges(
            "tool_calling_llm",
            self._tools_condition,  # Use custom tools condition
            {
                "tools": "tools",
                "gemini_formatter": "gemini_formatter",  # Route to Gemini instead of END
            }
        )
        workflow.add_edge("tools", "tool_calling_llm")  # Loop back to LLM
        
        # Add conditional edges from Gemini formatter
        workflow.add_conditional_edges(
            "gemini_formatter",
            self._gemini_condition,
            {
                "restart": "tool_calling_llm",  # Restart if answer not found
                "__end__": END,  # End if answer is complete
            }
        )
        
        return workflow.compile(checkpointer=redis_memory_saver)
    
    def _tools_condition(self, state: CallCenterState):
        """Custom tools condition that properly handles message sequences."""
        messages = state.get("messages", [])
        if not messages:
            return "gemini_formatter"
        
        last_message = messages[-1]
        
        # Check if the last message has tool calls
        if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
            print(f"[DEBUG] Tools condition: Found {len(last_message.tool_calls)} tool calls, routing to tools")
            return "tools"
        else:
            print(f"[DEBUG] Tools condition: No tool calls found, routing to Gemini formatter")
            return "gemini_formatter"
    
    def _agent_node(self, state: CallCenterState):
        """The main agent reasoning node in the cyclic flow."""
        
        # Extract context from state and store in agent instance
        self.current_state = {
            "auth_token": state.get("auth_token", ""),
            "start_date": state.get("start_date", ""),
            "end_date": state.get("end_date", ""),
            "session_id": state.get("session_id", ""),
            "original_question": state.get("original_question", ""),
            "recursion_count": state.get("recursion_count", 0)
        }
        
        auth_token = self.current_state["auth_token"]
        start_date = self.current_state["start_date"]
        end_date = self.current_state["end_date"]
        session_id = self.current_state["session_id"]
        original_question = self.current_state["original_question"]
        recursion_count = self.current_state["recursion_count"]
        
        messages = state["messages"]
        
        print(f"[DEBUG] Agent node called with {len(messages)} messages (recursion: {recursion_count}/10)")
        print(f"  start_date: {start_date}")
        print(f"  end_date: {end_date}")
        print(f"  auth_token: {auth_token[:30] if auth_token else 'None/Empty'}...")
        print(f"  session_id: {session_id}")
        print(f"  original_question: {original_question[:30] if original_question else 'None'}...")
        
        # Check if we're restarting (indicated by a RESTART_NEEDED message)
        restart_reason = None
        is_restart = False
        
        for msg in messages:
            if hasattr(msg, 'content') and msg.content and msg.content.startswith("RESTART_NEEDED:"):
                is_restart = True
                # Extract the specific restart reason
                restart_reason = msg.content.replace("RESTART_NEEDED:", "").strip()
                break
        
        if is_restart:
            recursion_count += 1
            print(f"[DEBUG] Restart detected, incrementing recursion count to {recursion_count}")
            print(f"[DEBUG] Restart reason: {restart_reason}")
            
            # Check recursion limit
            if recursion_count >= 10:
                print(f"[WARNING] Recursion limit reached ({recursion_count}), forcing final response")
                final_msg = "I apologize, but I'm having difficulty providing a complete answer to your question after multiple attempts. Please try rephrasing your question or contact support for assistance."
                return {
                    "messages": [HumanMessage(content=final_msg)],
                    "recursion_count": recursion_count
                }
        
        # Check if we have tool results to process
        has_tool_results = any(
            hasattr(msg, 'tool_call_id') for msg in messages
            if not (hasattr(msg, 'content') and msg.content and msg.content.startswith("RESTART_NEEDED:"))
        )
        
        if has_tool_results:
            print(f"[DEBUG] Found tool results in messages - processing for final response")
        else:
            print(f"[DEBUG] No tool results found - initial request processing")
        
        # Validate critical parameters before proceeding
        if not auth_token or len(str(auth_token).strip()) == 0:
            error_msg = "CRITICAL ERROR: auth_token is missing from agent state!"
            print(f"[ERROR] {error_msg}")
            return {
                "messages": [HumanMessage(content=f"Authentication Error: {error_msg}. Please check your authentication token and try again.")],
                "recursion_count": recursion_count
            }
        
        if not start_date or not end_date:
            error_msg = "CRITICAL ERROR: start_date or end_date is missing from agent state!"
            print(f"[ERROR] {error_msg}")
            return {
                "messages": [HumanMessage(content=f"Date Error: {error_msg}. Please provide valid start_date and end_date.")],
                "recursion_count": recursion_count
            }
        
        # Enhanced system prompt for cyclic flow
        if has_tool_results:
            # When we have tool results, focus on formatting the response
            system_prompt = f"""
            You are an AI-powered Call Center Analytics Agent. You have just received tool results and need to format them into a comprehensive, human-readable response.
            
            **CURRENT CONTEXT:**
            - Date Range: {start_date} to {end_date}
            - Session ID: {session_id}
            - Original Question: {original_question}
            - Recursion Count: {recursion_count}/10
            - Tool results are available in the conversation
            
            **YOUR TASK:**
            Analyze the tool results in the conversation and provide a well-formatted, comprehensive response that:
            
            1. **Summarizes key findings** from the tool data
            2. **Extracts important metrics** and presents them clearly
            3. **Provides actionable insights** based on the data
            4. **Structures information** with headers, bullet points, and clear formatting
            5. **Mentions which tools were used** for transparency
            
            **FORMATTING GUIDELINES:**
            - Use markdown formatting with headers (##, ###)
            - Present numbers and metrics prominently
            - Create bullet points for key insights
            - Include summary statistics where relevant
            - For recording file requests: provide a clean, numbered list
            - For performance analysis: rank and compare metrics
            - Always end with actionable recommendations
            
            **IMPORTANT:**
            - Do NOT make additional tool calls - just format the existing results
            - Focus on creating a final, comprehensive response
            - Extract maximum value from the available tool data
            - Be thorough since this may be evaluated for completeness
            """
        else:
            # Initial request processing or restart
            restart_guidance = ""
            if is_restart:
                restart_guidance = f"""
            
            **RESTART CONTEXT:**
            This is restart attempt {recursion_count}/10. The previous response was deemed incomplete.
            
            **SPECIFIC ISSUE IDENTIFIED:** {restart_reason}
            
            **RESTART STRATEGY:**
            Based on the issue above, please:
            - Address the specific deficiency mentioned
            - Try a different approach or additional tools to gather more comprehensive data
            - Focus on providing complete, actionable insights
            - Ensure your response directly answers the original question
            """
            
            system_prompt = f"""
            You are an AI-powered Call Center Analytics Agent with access to comprehensive call center data.
            
            **AUTHENTICATION INFO:**
            - Auth Token: Available and verified
            - Date Range: {start_date} to {end_date}
            - Session ID: {session_id}
            - Original Question: {original_question}
            - Recursion Count: {recursion_count}/10{restart_guidance}
            
            **Available Tools:**
            1. **get_dashboard_data** - Comprehensive dashboard with key metrics and trends
            2. **get_call_metrics** - Detailed call analytics and aggregated metrics
            3. **get_direction_analysis** - Analysis of inbound, outbound, and internal calls
            4. **get_caller_analysis** - Caller behavior patterns and top caller insights
            5. **get_performance_metrics** - Call duration trends and performance indicators
            6. **get_agent_performance** - Individual agent metrics and comparisons
            7. **get_extension_performance** - Extension-level performance analytics
            8. **get_hourly_distribution** - Peak time analysis and call volume patterns
            9. **get_call_logs** - Detailed call records with filtering
            10. **get_recording_info** - Call recording file information for specific files
            11. **extract_recording_paths** - Extract recording file paths from call data JSON
            
            **TOOL SELECTION STRATEGY:**
            
            🎯 **FOR RECORDING FILE REQUESTS:**
            When user asks for "recording file names", "recording paths", "list recordings", "audio files", etc.:
            - Call: get_call_logs(include_details=True)
            - Then: extract_recording_paths(json_data=<call_logs_result>)
            
            🎯 **FOR AGENT PERFORMANCE REQUESTS:**
            When user asks for "best performing agent", "top agent", "agent performance", etc.:
            - Call: get_agent_performance()
            
            🎯 **FOR GENERAL ANALYSIS:**
            - Dashboard overview → get_dashboard_data
            - Call metrics → get_call_metrics
            - Performance analysis → get_agent_performance, get_extension_performance
            - Time patterns → get_hourly_distribution
            
            **YOUR TASK:**
            Analyze the user's request and call the appropriate tools to gather the necessary data.
            Be thorough in your tool selection to provide comprehensive answers.
            """
        
        # Create the full message sequence
        full_messages = [SystemMessage(content=system_prompt)] + messages
        
        try:
            response = self.llm_with_tools.invoke(full_messages)
            print(f"[DEBUG] LLM response generated, has tool_calls: {bool(hasattr(response, 'tool_calls') and response.tool_calls)}")
            return {
                "messages": [response],
                "recursion_count": recursion_count,
                "final_model_used": "gpt-4o"  # Track that GPT-4 was used
            }
        except Exception as e:
            print(f"[ERROR] LLM invocation failed: {e}")
            
            # Check if this is an OpenAI API authentication error
            error_str = str(e)
            if "401" in error_str and ("invalid_api_key" in error_str or "Incorrect API key" in error_str):
                print(f"[DEBUG] OpenAI API authentication error detected - creating terminal error")
                error_response = HumanMessage(content="AUTHENTICATION_TERMINAL_ERROR: OpenAI API key is invalid or expired. Please check your OpenAI API credentials and try again with a valid key.")
            else:
                error_response = HumanMessage(content=f"Error processing request: {str(e)}")
            
            return {
                "messages": [error_response],
                "recursion_count": recursion_count,
                "final_model_used": "gpt-4o"  # Track model even for errors
            }
    
    def _gemini_formatter_node(self, state: CallCenterState):
        """Gemini node for final answer formatting and validation."""
        
        if not self.gemini_llm:
            print("[WARNING] Gemini LLM not available, using fallback formatting")
            return self._fallback_formatting(state)
        
        messages = state["messages"]
        original_question = state.get("original_question", "")
        recursion_count = state.get("recursion_count", 0)
        session_id = state.get("session_id", "")
        
        print(f"[DEBUG] Gemini formatter processing (recursion: {recursion_count}/10)")
        
        # Log to Langfuse
        gemini_trace = None
        if langfuse_client:
            gemini_trace = langfuse_client.trace(
                name="gemini_formatter_node",
                input={
                    "original_question": original_question,
                    "recursion_count": recursion_count,
                    "node": "gemini_formatter"
                },
                tags=["call_center_agent", "gemini_formatter", "final_validation"],
                session_id=session_id
            )
        
        # If we're near the recursion limit, be more lenient with completion
        near_limit = recursion_count >= 8
        if near_limit:
            print(f"[DEBUG] Near recursion limit ({recursion_count}/10), being more lenient with completion")
        
        # Get the last assistant message (the one that would have gone to END)
        last_assistant_message = None
        for msg in reversed(messages):
            if hasattr(msg, 'content') and msg.content and not (hasattr(msg, 'tool_calls') and msg.tool_calls):
                # Skip RESTART_NEEDED messages
                if not msg.content.startswith("RESTART_NEEDED:"):
                    last_assistant_message = msg
                    break
        
        if not last_assistant_message or not last_assistant_message.content:
            print("[DEBUG] No suitable assistant message found, requesting restart")
            result = {
                "messages": [HumanMessage(content="RESTART_NEEDED: No complete response found")],
                "recursion_count": recursion_count,
                "final_model_used": "gpt-4o"  # Keep original model attribution for restart
            }
            
            # Log restart to Langfuse
            if gemini_trace:
                gemini_trace.update(
                    output={"decision": "restart", "reason": "No complete response found"},
                    level="DEFAULT"
                )
            
            return result
        
        assistant_response = last_assistant_message.content
        
        # Check for terminal authentication errors - end gracefully instead of restarting
        if "AUTHENTICATION_TERMINAL_ERROR:" in assistant_response:
            print("[DEBUG] Authentication terminal error detected - ending gracefully")
            
            # Create a user-friendly final message
            final_message = """## Authentication Error

I apologize, but I cannot access the call center data due to an authentication issue. The provided authentication token is invalid or has expired.

**To resolve this issue:**
1. Please verify your authentication credentials
2. Ensure your token has the necessary permissions
3. Try again with a valid authentication token

If you continue to experience authentication issues, please contact your system administrator for assistance."""
            
            result = {
                "messages": [HumanMessage(content=final_message)],
                "recursion_count": recursion_count,
                "final_model_used": "gemini-2.5-flash"  # Track that Gemini handled the terminal error
            }
            
            # Log terminal error to Langfuse
            if gemini_trace:
                gemini_trace.update(
                    output={
                        "decision": "terminal_auth_error",
                        "message": "Authentication error detected - ended gracefully",
                        "no_restart": True
                    },
                    level="DEFAULT"
                )
            
            return result
        
        print(f"[DEBUG] Gemini formatter processing:")
        print(f"  Original question: {original_question[:50]}...")
        print(f"  Assistant response length: {len(assistant_response)} chars")
        print(f"  Recursion count: {recursion_count}/10")
        
        # Create Gemini prompt for final formatting and validation
        recursion_context = ""
        if recursion_count > 0:
            recursion_context = f"""
        
        **RECURSION CONTEXT:**
        This is attempt {recursion_count + 1}/10 to provide a complete answer. Previous attempts were deemed incomplete.
        {"Be more lenient in your evaluation as we're approaching the limit." if near_limit else ""}
        """
        
        gemini_prompt = f"""
        You are a professional call center analytics consultant tasked with creating a natural, conversational final response.
        
        **ORIGINAL USER QUESTION:**
        {original_question}
        
        **ASSISTANT'S RESPONSE:**
        {assistant_response}
        
        **RECURSION COUNT:** {recursion_count}/10{recursion_context}
        
        **YOUR TASK:**
        Analyze the assistant's response and determine if it adequately answers the original question.
        
        **DECISION CRITERIA:**
        
        ✅ **COMPLETE ANSWER** - If the response contains:
        - Direct answer to the user's question
        - Relevant data and metrics
        - Clear insights or analysis
        - Actionable information
        {"- ANY reasonable attempt at answering (be lenient due to recursion limit)" if near_limit else ""}
        
        ❌ **INCOMPLETE ANSWER** - If the response contains:
        - Only error messages
        - Generic fallback text
        - No specific data or insights
        - Doesn't address the original question
        {"(Be more strict only if response is completely useless)" if near_limit else ""}
        
        **OUTPUT FORMAT:**
        
        If the answer is COMPLETE:
        ```
        STATUS: COMPLETE
        
        [Create a natural, conversational response that follows these guidelines:
        
        1. START WITH THE DIRECT ANSWER - Lead with the key finding/result immediately
        2. NO QUESTION REPETITION - Never repeat the user's question 
        3. NO FORMAL HEADERS - Avoid "**Question:**", "**Answer:**", "**Analysis:**" etc.
        4. CONVERSATIONAL TONE - Write as if speaking to a colleague
        5. STRUCTURE: Answer → Supporting Data → Context/Insights
        
        Example format:
        "Ronny Tawiah (Agent 1010) received the most calls with 45 total calls, and those calls averaged 116.96 seconds (about 1 minute 57 seconds) each.
        
        Looking at the data from April 21st to May 1st, all of Ronny's calls were outbound calls. This represents a significant call volume compared to other agents during this period, with his calls maintaining a consistent average duration."
        
        Focus on:
        - Clear, direct communication
        - Natural flow between facts
        - Relevant context that adds value
        - Professional but conversational tone]
        ```
        
        If the answer is INCOMPLETE:
        ```
        STATUS: INCOMPLETE
        REASON: [Specific, actionable explanation of what's missing - e.g., "Response lacks specific metrics for the requested time period", "No recording file paths provided despite user asking for file list", "Contains only error messages without any call center data", "Doesn't address the original question about agent performance"]
        ```
        
        **IMPORTANT:**
        - {"Be more lenient in your evaluation due to recursion limit" if near_limit else "Be strict in your evaluation"}
        - Only mark as COMPLETE if the response truly answers the user's question
        - Focus on whether the user would be satisfied with this answer
        - Create natural, engaging responses that feel conversational for complete answers
        {"- PREFER COMPLETE unless response is completely useless" if near_limit else ""}
        """
        
        try:
            # Create Langfuse generation span for Gemini call
            gemini_generation = None
            if gemini_trace:
                gemini_generation = gemini_trace.generation(
                    name="gemini_validation_generation",
                    input=gemini_prompt,
                    model="gemini-2.5-flash",
                    usage={
                        "promptTokens": len(gemini_prompt.split()),
                        "completionTokens": 0,  # Will update after response
                        "totalTokens": len(gemini_prompt.split())
                    }
                )
            
            gemini_response = self.gemini_llm.invoke([HumanMessage(content=gemini_prompt)])
            gemini_content = gemini_response.content.strip()
            
            # Update Langfuse generation with response
            if gemini_generation:
                completion_tokens = len(gemini_content.split())
                gemini_generation.update(
                    output=gemini_content,
                    usage={
                        "promptTokens": len(gemini_prompt.split()),
                        "completionTokens": completion_tokens,
                        "totalTokens": len(gemini_prompt.split()) + completion_tokens
                    }
                )
            
            print(f"[DEBUG] Gemini response length: {len(gemini_content)} chars")
            
            # Parse Gemini's decision
            if "STATUS: COMPLETE" in gemini_content:
                print("[DEBUG] Gemini marked response as COMPLETE")
                # Extract the final answer (everything after STATUS: COMPLETE)
                final_answer = gemini_content.split("STATUS: COMPLETE", 1)[1].strip()
                
                result = {
                    "messages": [HumanMessage(content=final_answer)],
                    "recursion_count": recursion_count,
                    "final_model_used": "gemini-2.5-flash"  # Track that Gemini was used for final response
                }
                
                # Log completion to Langfuse
                if gemini_trace:
                    gemini_trace.update(
                        output={
                            "decision": "complete",
                            "final_answer_length": len(final_answer),
                            "model_used": "gemini-2.5-flash"
                        },
                        level="DEFAULT"
                    )
                
                return result
            else:
                print("[DEBUG] Gemini marked response as INCOMPLETE - requesting restart")
                reason = "Answer incomplete or insufficient"
                if "REASON:" in gemini_content:
                    reason = gemini_content.split("REASON:", 1)[1].strip()
                
                result = {
                    "messages": [HumanMessage(content=f"RESTART_NEEDED: {reason}")],
                    "recursion_count": recursion_count,
                    "final_model_used": "gpt-4o"  # Keep original model attribution for restart
                }
                
                # Log restart to Langfuse
                if gemini_trace:
                    gemini_trace.update(
                        output={"decision": "restart", "reason": reason},
                        level="DEFAULT"
                    )
                
                return result
                
        except Exception as e:
            print(f"[ERROR] Gemini formatting failed: {e}")
            
            # Log error to Langfuse
            if gemini_trace:
                gemini_trace.update(
                    output={"error": str(e), "fallback_used": True},
                    level="ERROR"
                )
            
            return self._fallback_formatting(state)
    
    def _gemini_condition(self, state: CallCenterState):
        """Condition to determine if Gemini wants to restart or end."""
        messages = state.get("messages", [])
        if not messages:
            return "__end__"
        
        last_message = messages[-1]
        if hasattr(last_message, 'content') and last_message.content:
            content = last_message.content
            if content.startswith("RESTART_NEEDED:"):
                print(f"[DEBUG] Gemini condition: Restart requested - {content}")
                return "restart"
            else:
                print(f"[DEBUG] Gemini condition: Answer complete, ending")
                return "__end__"
        
        return "__end__"
    
    def _fallback_formatting(self, state: CallCenterState):
        """Fallback formatting when Gemini is not available."""
        messages = state["messages"]
        recursion_count = state.get("recursion_count", 0)
        
        print(f"[DEBUG] Fallback formatting (recursion: {recursion_count}/10)")
        
        # Get the last assistant message
        last_assistant_message = None
        for msg in reversed(messages):
            if hasattr(msg, 'content') and msg.content and not (hasattr(msg, 'tool_calls') and msg.tool_calls):
                # Skip RESTART_NEEDED messages
                if not msg.content.startswith("RESTART_NEEDED:"):
                    last_assistant_message = msg
                    break
        
        if last_assistant_message and last_assistant_message.content:
            # Check for terminal authentication errors - end gracefully instead of restarting
            if "AUTHENTICATION_TERMINAL_ERROR:" in last_assistant_message.content:
                print("[DEBUG] Authentication terminal error detected in fallback - ending gracefully")
                
                # Create a user-friendly final message
                final_message = """## Authentication Error

I apologize, but I cannot access the call center data due to an authentication issue. The provided authentication token is invalid or has expired.

**To resolve this issue:**
1. Please verify your authentication credentials
2. Ensure your token has the necessary permissions
3. Try again with a valid authentication token

If you continue to experience authentication issues, please contact your system administrator for assistance."""
                
                return {
                    "messages": [HumanMessage(content=final_message)],
                    "recursion_count": recursion_count,
                    "final_model_used": "gpt-4o"  # Fallback handled the terminal error
                }
            
            # Simple check for error messages or incomplete responses
            content = last_assistant_message.content.lower()
            # Be more lenient near recursion limit
            if recursion_count >= 8:
                print("[DEBUG] Near recursion limit, accepting response in fallback")
                return {
                    "messages": [last_assistant_message],
                    "recursion_count": recursion_count,
                    "final_model_used": "gpt-4o"  # Fallback uses original GPT-4 response
                }
            elif any(keyword in content for keyword in ["error", "failed", "cannot", "unable", "missing"]):
                # Provide specific restart reason based on error type
                if "authentication" in content or "token" in content:
                    restart_reason = "Authentication errors detected - need to verify API access and retry data collection"
                elif "timeout" in content or "connection" in content:
                    restart_reason = "Connection/timeout errors detected - need to retry API calls with different approach"
                elif "missing" in content or "not found" in content:
                    restart_reason = "Missing data errors detected - need to try alternative data sources or adjust query parameters"
                else:
                    restart_reason = "General errors detected in response - need to retry with different tool selection or approach"
                
                return {
                    "messages": [HumanMessage(content=f"RESTART_NEEDED: {restart_reason}")],
                    "recursion_count": recursion_count,
                    "final_model_used": "gpt-4o"  # Keep original model attribution for restart
                }
            else:
                return {
                    "messages": [last_assistant_message],
                    "recursion_count": recursion_count,
                    "final_model_used": "gpt-4o"  # Fallback uses original GPT-4 response
                }
        else:
            return {
                "messages": [HumanMessage(content="RESTART_NEEDED: No valid response content found - need to retry with proper tool calls and data collection")],
                "recursion_count": recursion_count,
                "final_model_used": "gpt-4o"  # Keep original model attribution for restart
            }

    @observe(name="call_center_agent_analysis")
    async def analyze(self, request: str, start_date: str, end_date: str, 
                     auth_token: str, thread_id: str = "default") -> Dict[str, Any]:
        """
        Main method to perform call center analysis using the agent.
        
        Args:
            request: Analysis request from user
            start_date: Start date for analysis
            end_date: End date for analysis
            auth_token: Authentication token for API calls
            thread_id: Thread ID for conversation tracking (used as session_id)
        
        Returns:
            Analysis results with recommendations
        """
        # Log memory stats before analysis
        logger.info(f"Starting analysis for thread_id: {thread_id}")
        pre_analysis_stats = get_agent_memory_stats(thread_id)
        logger.info(f"Pre-analysis memory stats: {pre_analysis_stats}")
        
        # Create Langfuse session for this analysis
        session = None
        # Generate shorter session ID to avoid tool call ID length issues (max 40 chars)
        session_hash = hashlib.md5(f"{thread_id}_{datetime.now().isoformat()}".encode()).hexdigest()[:8]
        session_id = f"agent_{session_hash}"
        
        if langfuse_client:
            session = langfuse_client.trace(
                name="call_center_analysis_session",
                input={
                    "user_request": request,
                    "start_date": start_date,
                    "end_date": end_date,
                    "thread_id": thread_id,
                    "session_id": session_id,
                    "redis_memory_enabled": redis_memory_saver.redis_available,
                    "fallback_enabled": redis_memory_saver.fallback_to_memory
                },
                tags=["call_center_agent", "analysis_session", "redis_memory"],
                user_id=thread_id,
                session_id=session_id  # Add session_id to the trace
            )
            
            # Update langfuse context with session information
            try:
                from langfuse.decorators import langfuse_context
                langfuse_context.update_current_trace(session_id=session_id)
                logger.info(f"Langfuse session created: {session_id}")
            except ImportError:
                logger.warning("langfuse_context not available for session tracking")
        
        try:
            # Prepare the analysis request with context
            analysis_prompt = f"""
            **Analysis Request:** {request}
            
            **Date Range:** {start_date} to {end_date}
            **Session ID:** {session_id}
            **Thread ID:** {thread_id}
            
            **Instructions:**
            Please analyze our call center data for the specified period. Use the available tools to gather relevant data and provide comprehensive insights.
            
            Focus on:
            - Key performance indicators and trends
            - Areas of strength and improvement opportunities  
            - Specific, actionable recommendations
            - Data-driven insights that can improve operations
            
            Use your tools strategically to gather the most relevant data for this analysis.
            """
            
            # Create initial state with CallCenterState schema
            initial_state = {
                "messages": [HumanMessage(content=analysis_prompt)],
                "auth_token": auth_token,
                "start_date": start_date,
                "end_date": end_date,
                "session_id": session_id,  # Add session_id to state
                "original_question": request,
                "recursion_count": 0,
                "final_model_used": "gpt-4o"  # Initialize with default model
            }
            
            print(f"[DEBUG] Initializing agent with state:")
            print(f"  start_date: {start_date}")
            print(f"  end_date: {end_date}")
            print(f"  session_id: {session_id}")
            print(f"  thread_id: {thread_id}")
            print(f"  auth_token: {auth_token[:30] if auth_token else 'None/Empty'}...")
            print(f"  original_question: {request[:30] if request else 'None'}...")
            print(f"  redis_memory_available: {redis_memory_saver.redis_available}")
            print(f"  fallback_enabled: {redis_memory_saver.fallback_to_memory}")
            
            # Create config for thread management
            config = {"configurable": {"thread_id": thread_id}}
            
            print(f"[DEBUG] About to invoke graph with config: {config}")
            print(f"[DEBUG] Graph type: {type(self.graph)}")
            print(f"[DEBUG] Graph compiled: {hasattr(self.graph, 'ainvoke')}")
            
            # Run the agent
            try:
                result = await self.graph.ainvoke(initial_state, config=config)
                print(f"[DEBUG] Graph invocation completed successfully")
                print(f"[DEBUG] Result keys: {list(result.keys()) if isinstance(result, dict) else 'Not a dict'}")
            except Exception as graph_error:
                print(f"[ERROR] Graph invocation failed: {graph_error}")
                print(f"[ERROR] Graph error type: {type(graph_error)}")
                print(f"[ERROR] Graph error args: {graph_error.args}")
                print(f"[ERROR] Graph error traceback:")
                traceback.print_exc()
                raise graph_error  # Re-raise to be caught by outer try/except
            
            # Log memory stats after analysis
            post_analysis_stats = get_agent_memory_stats(thread_id)
            logger.info(f"Post-analysis memory stats: {post_analysis_stats}")
            
            # Extract the final analysis - handle multiple message types
            final_message = None
            for message in reversed(result["messages"]):
                # Look for the last assistant message without tool calls
                if hasattr(message, 'content') and message.content and not (hasattr(message, 'tool_calls') and message.tool_calls):
                    final_message = message
                    break
            
            # If no suitable message found, get the last message
            if not final_message:
                final_message = result["messages"][-1] if result["messages"] else None
            
            # Ensure we have content
            analysis_content = ""
            if final_message and hasattr(final_message, 'content'):
                analysis_content = final_message.content or ""
            
            # If analysis is still empty, create a summary from tool results
            if not analysis_content.strip():
                analysis_content = self._create_fallback_analysis(result["messages"])
            
            tools_used = self._extract_tool_calls(result["messages"])
            final_model_used = result.get("final_model_used", "gpt-4o")
            
            # Log successful completion to Langfuse
            if session:
                session.update(
                    output={
                        "status": "success",
                        "analysis_length": len(analysis_content),
                        "tools_used": tools_used,
                        "session_id": session_id,
                        "recursion_count": result.get("recursion_count", 0),
                        "final_model_used": final_model_used,
                        "analysis_preview": analysis_content[:200] + "..." if len(analysis_content) > 200 else analysis_content,
                        "memory_stats": {
                            "pre_analysis": pre_analysis_stats,
                            "post_analysis": post_analysis_stats,
                            "redis_available": redis_memory_saver.redis_available,
                            "fallback_used": not redis_memory_saver.redis_available
                        }
                    },
                    level="DEFAULT"
                )
                
                # Create a generation span for the final LLM response with correct model attribution
                session.generation(
                    name="final_analysis_generation",
                    input=analysis_prompt,
                    output=analysis_content,
                    model=final_model_used,  # Use the actual model that generated the response
                    usage={
                        "promptTokens": len(analysis_prompt.split()),
                        "completionTokens": len(analysis_content.split()),
                        "totalTokens": len(analysis_prompt.split()) + len(analysis_content.split())
                    }
                )
            
            return {
                "status": "success",
                "analysis": analysis_content,
                "tools_used": tools_used,
                "session_id": session_id,  # Include session_id in response
                "recursion_count": result.get("recursion_count", 0),
                "final_model_used": final_model_used,  # Include which model generated the final response
                "timestamp": datetime.now().isoformat(),
                "date_range": {
                    "start_date": start_date,
                    "end_date": end_date
                },
                "original_question": request,
                "memory_info": {
                    "redis_available": redis_memory_saver.redis_available,
                    "fallback_enabled": redis_memory_saver.fallback_to_memory,
                    "thread_id": thread_id,
                    "pre_analysis_stats": pre_analysis_stats,
                    "post_analysis_stats": post_analysis_stats
                }
            }
            
        except Exception as e:
            print(f"[ERROR] Exception in analyze method:")
            print(f"[ERROR] Exception type: {type(e)}")
            print(f"[ERROR] Exception message: '{str(e)}'")
            print(f"[ERROR] Exception args: {e.args}")
            print(f"[ERROR] Exception repr: {repr(e)}")
            
            import traceback
            print(f"[ERROR] Full traceback:")
            traceback.print_exc()
            
            logger.error(f"Agent analysis error: {e}")
            logger.error(f"Agent analysis error type: {type(e)}")
            logger.error(f"Agent analysis error args: {e.args}")
            
            # Create a more informative error message
            error_message = str(e) if str(e) else f"Unknown error of type {type(e).__name__}"
            if not error_message or error_message.strip() == "":
                error_message = f"Empty error message from {type(e).__name__} exception"
            
            # Log error to Langfuse
            if session:
                session.update(
                    output={
                        "error": error_message, 
                        "error_type": type(e).__name__,
                        "error_args": str(e.args),
                        "session_id": session_id,
                        "redis_available": redis_memory_saver.redis_available,
                        "fallback_enabled": redis_memory_saver.fallback_to_memory
                    },
                    level="ERROR"
                )
            
            return {
                "status": "error",
                "error": error_message,
                "error_type": type(e).__name__,
                "error_details": {
                    "message": str(e),
                    "args": str(e.args),
                    "type": type(e).__name__
                },
                "session_id": session_id,
                "timestamp": datetime.now().isoformat(),
                "memory_info": {
                    "redis_available": redis_memory_saver.redis_available,
                    "fallback_enabled": redis_memory_saver.fallback_to_memory,
                    "thread_id": thread_id
                }
            }
    
    def _extract_tool_calls(self, messages) -> list:
        """Extract which tools were called during analysis."""
        tool_calls = []
        for message in messages:
            # Handle different message types that might contain tool calls
            if hasattr(message, 'tool_calls') and message.tool_calls:
                for tool_call in message.tool_calls:
                    # Handle different tool_call formats
                    if isinstance(tool_call, dict):
                        tool_name = tool_call.get('name') or tool_call.get('function', {}).get('name')
                    else:
                        tool_name = getattr(tool_call, 'name', None)
                    
                    if tool_name:
                        tool_calls.append(tool_name)
            
            # Also check for tool responses to identify tools that were used
            if hasattr(message, 'tool_call_id') and hasattr(message, 'name'):
                tool_calls.append(message.name)
        
        return list(set(tool_calls))

    def _create_fallback_analysis(self, messages) -> str:
        """Create a fallback analysis if the final message is empty or missing."""
        tool_results = []
        tool_calls_made = []
        
        for message in messages:
            # Extract tool calls
            if hasattr(message, 'tool_calls') and message.tool_calls:
                for tool_call in message.tool_calls:
                    tool_calls_made.append(tool_call.get('name', 'unknown_tool'))
            
            # Extract tool results (ToolMessage content)
            if hasattr(message, 'content') and message.content and hasattr(message, 'tool_call_id'):
                # This is a tool response message
                tool_results.append(message.content)
        
        if not tool_results and not tool_calls_made:
            return "No analysis could be generated. Please check your request and try again."
        
        fallback_content = "## Call Center Analysis Results\n\n"
        
        if tool_calls_made:
            fallback_content += f"**Tools Used:** {', '.join(set(tool_calls_made))}\n\n"
        
        if tool_results:
            fallback_content += "**Data Retrieved:**\n"
            for i, result in enumerate(tool_results[:3], 1):  # Limit to first 3 results
                # Try to parse JSON and extract key info
                try:
                    import json
                    data = json.loads(result)
                    if isinstance(data, dict):
                        # Extract key metrics if available
                        if 'recording_paths' in data:
                            fallback_content += f"- Found {len(data['recording_paths'])} recording files\n"
                        elif 'summary' in data:
                            summary = data['summary']
                            fallback_content += f"- Total calls: {summary.get('total_calls', 'N/A')}\n"
                            fallback_content += f"- Answer rate: {summary.get('answer_rate', 'N/A')}%\n"
                        elif 'total_count' in data:
                            fallback_content += f"- Total records: {data.get('total_count', 'N/A')}\n"
                except (json.JSONDecodeError, TypeError):
                    # If not JSON, include first 100 chars
                    fallback_content += f"- Tool result {i}: {result[:100]}...\n"
        
        fallback_content += "\n*Note: This is a summary due to analysis processing issues. Please try your request again for detailed insights.*"
        
        return fallback_content

# Initialize the global agent instance
_agent_instance = None

def get_agent(openai_api_key: str, base_url: str = "http://localhost:8000") -> CallCenterAgent:
    """Get or create the global agent instance."""
    global _agent_instance
    if _agent_instance is None:
        _agent_instance = CallCenterAgent(openai_api_key, base_url)
    return _agent_instance 