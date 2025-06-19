from fastapi import APIRouter, HTTPException, status, Depends, Body, Request
from pydantic import BaseModel, validator
from typing import Dict, Any, Optional, Union
import logging
import json
import os
from utils.llm import LLM
from .auth import get_current_user
from api.database import with_mongodb_retry
from api.core.dashboard_agent import get_agent

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Initialize router
router = APIRouter(prefix="/ai", tags=["ai-generation"])

# Pydantic models for request and response
class InterpretDataRequest(BaseModel):
    data: Union[str, Dict[str, Any]]
    model: Optional[str] = "gpt-4-turbo"  # Default to gpt-4-turbo
    
    @validator('data')
    def validate_data(cls, v):
        """
        Validates and normalizes the data field to ensure it's a string.
        If it's already a dict, it will convert it to a JSON string.
        """
        if isinstance(v, dict):
            return json.dumps(v)
        return v
    
    @validator('model')
    def validate_model(cls, v):
        """
        Validates that the model is supported.
        """
        supported_models = [
            # OpenAI models
            "gpt-4-turbo",
            "gpt-4",
            "gpt-3.5-turbo",
            # Gemini models
            "gemini-2.0-flash",
            "gemini-1.5-pro"
        ]
        
        if v not in supported_models:
            raise ValueError(f"Unsupported model: {v}. Supported models are: {', '.join(supported_models)}")
        
        return v

class InterpretDataResponse(BaseModel):
    status: str
    interpretation: Optional[Union[str, Dict[str, Any]]] = None
    model_used: Optional[str] = None
    model_provider: Optional[str] = None  # "openai" or "gemini"
    tokens_used: Optional[Dict[str, int]] = None
    error: Optional[str] = None
    message: Optional[str] = None
    
    class Config:
        # Allow arbitrary types to support JSON objects
        arbitrary_types_allowed = True

class AgentAnalysisRequest(BaseModel):
    request: str
    start_date: str
    end_date: str
    thread_id: Optional[str] = "default"

class AgentAnalysisResponse(BaseModel):
    status: str
    analysis: Optional[str] = None
    tools_used: Optional[list] = None
    timestamp: Optional[str] = None
    date_range: Optional[Dict[str, str]] = None
    error: Optional[str] = None

# Create the LLM instance
def get_llm(model: str = "gpt-4-turbo") -> LLM:
    """
    Get an LLM instance for the specified model.
    Each model gets its own instance since they may use different APIs.
    """
    try:
        # Create a new instance for each model to ensure proper initialization
        llm_instance = LLM(model=model)
        return llm_instance
    except Exception as e:
        logger.error(f"Error initializing LLM with model {model}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Could not initialize LLM with model {model}: {str(e)}"
        )

@router.post("/interpret", response_model=InterpretDataResponse)
@with_mongodb_retry(max_retries=3)
async def interpret_data(
    request: InterpretDataRequest = Body(...),
    current_user: dict = Depends(get_current_user)
):
    """
    Interpret call center data using an AI model.
    Requires authentication.
    """
    try:
        # Log the request
        logger.info(f"Interpret data request received from user: {current_user['username']}")
        
        # Prepare user information for Langfuse tracing
        user_info = {
            "user_id": str(current_user.get("_id", "")),
            "username": current_user.get("username", ""),
            "company_id": current_user.get("company_id", ""),
            "company_name": current_user.get("company_name", "")
        }
        
        # Get the LLM instance with the requested model
        llm = get_llm(model=request.model)
        
        # Process the data with user info for tracing
        result = llm.interpret_data(request.data, user_info=user_info)
        
        # Log success
        logger.info(f"Successfully interpreted data for user {current_user['username']}")
        
        return result
        
    except Exception as e:
        logger.error(f"Error interpreting data: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error interpreting data: {str(e)}"
        )

@router.post("/agent-analysis", response_model=AgentAnalysisResponse)
async def agent_analysis(
    agent_request: AgentAnalysisRequest = Body(...),
    current_user: dict = Depends(get_current_user),
    request: Request = None
):
    """
    AI Agent-powered comprehensive call center analysis.
    
    This endpoint uses an AI agent that can call multiple analytics endpoints as tools
    to gather comprehensive data and provide intelligent analysis.
    
    The agent will:
    - Understand your analysis request
    - Determine which data sources to query
    - Call relevant endpoints to gather data
    - Analyze patterns and correlations
    - Provide actionable insights and recommendations
    """
    try:
        # Validate date format
        from datetime import datetime
        try:
            datetime.strptime(agent_request.start_date, "%Y-%m-%d")
            datetime.strptime(agent_request.end_date, "%Y-%m-%d")
        except ValueError:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid date format. Use YYYY-MM-DD"
            )
        
        # Get OpenAI API key
        openai_api_key = os.getenv("OPENAI_API_KEY")
        if not openai_api_key:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="OpenAI API key not configured"
            )
        
        # Initialize the agent
        agent = get_agent(openai_api_key)
        
        # Extract the original JWT token from the request
        auth_header = request.headers.get("Authorization") if request else None
        original_token = None
        
        if auth_header and auth_header.startswith("Bearer "):
            original_token = auth_header.split(" ")[1]
        
        # If we can't get the original token, generate a new one
        if not original_token:
            # Use the same SECRET_KEY as the rest of the application
            import jwt
            
            # Get the SECRET_KEY from environment (same as auth module)
            SECRET_KEY = os.getenv("JWT_SECRET_KEY")
            if not SECRET_KEY:
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail="JWT secret key not configured"
                )
            
            # Create a token that exactly matches the structure expected by get_current_user
            original_token = jwt.encode({
                "sub": str(current_user.get("_id", "")),  # This should match the user ID
                "email": current_user.get("email", ""),
                "username": current_user.get("username", ""),
                "company_id": str(current_user.get("company_id", "")),
                "company_code": current_user.get("company_code", ""),
                "is_verified": current_user.get("is_verified", True),
                "exp": int(datetime.utcnow().timestamp()) + 3600  # 1 hour expiry as integer
            }, SECRET_KEY, algorithm="HS256")
        
        # Log the request
        logger.info(f"Agent analysis request received from user: {current_user['username']}")
        logger.info(f"Analysis request: {agent_request.request}")
        logger.info(f"Date range: {agent_request.start_date} to {agent_request.end_date}")
        
        # Run the agent analysis
        result = await agent.analyze(
            request=agent_request.request,
            start_date=agent_request.start_date,
            end_date=agent_request.end_date,
            auth_token=original_token,
            thread_id=f"{current_user.get('_id', 'default')}_{agent_request.thread_id}"
        )
        
        # Log success
        if result["status"] == "success":
            logger.info(f"Agent analysis completed successfully for user {current_user['username']}")
            logger.info(f"Tools used: {result.get('tools_used', [])}")
        else:
            logger.error(f"Agent analysis failed: {result.get('error', 'Unknown error')}")
        
        return AgentAnalysisResponse(**result)
        
    except Exception as e:
        logger.error(f"Agent analysis error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Agent analysis failed: {str(e)}"
        )
