import os
import logging
from typing import Dict, Any, Optional, List
from openai import OpenAI
from dotenv import load_dotenv
from pathlib import Path
from uuid import uuid4
import json
import traceback
import re
import random
import time
from datetime import datetime, timedelta
from collections import defaultdict, Counter
import calendar

# Add Google Generative AI import
from google import genai

# Model Context Limits (tokens) - Updated with gemini-2.5-flash-preview-05-20
MODEL_CONTEXT_LIMITS = {
    "gpt-4-turbo": 128000,
    "gpt-4o": 128000, 
    "gpt-4o-mini": 128000,
    "gpt-4": 8192,
    "gpt-3.5-turbo": 16385,
    "gemini-2.0-flash": 1000000,
    "gemini-2.5-flash-preview-05-20": 1000000,  # Added new model
    "gemini-1.5-pro": 2000000,
    "gemini-1.5-flash": 1000000
}

# Model fallback chains - ordered by preference
MODEL_FALLBACK_CHAINS = {
    "gpt-4-turbo": ["gpt-4o", "gpt-4o-mini", "gpt-3.5-turbo"],
    "gpt-4o": ["gpt-4-turbo", "gpt-4o-mini", "gpt-3.5-turbo"],
    "gpt-4o-mini": ["gpt-3.5-turbo", "gpt-4o", "gpt-4-turbo"],
    "gpt-4": ["gpt-4-turbo", "gpt-4o", "gpt-3.5-turbo"],
    "gpt-3.5-turbo": ["gpt-4o-mini", "gpt-4o", "gpt-4-turbo"],
    "gemini-2.0-flash": ["gemini-1.5-flash", "gemini-1.5-pro", "gpt-4o"],
    "gemini-2.5-flash-preview-05-20": ["gemini-2.0-flash", "gemini-1.5-pro", "gpt-4o"],
    "gemini-1.5-pro": ["gemini-2.0-flash", "gemini-1.5-flash", "gpt-4-turbo"],
    "gemini-1.5-flash": ["gemini-2.0-flash", "gemini-1.5-pro", "gpt-4o-mini"]
}

# Reserve tokens for system prompt, user prompt, and response
RESERVED_TOKENS = 15000  # ~10K for prompts + 5K for response

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Langfuse initialization - import only when needed to avoid issues
langfuse_client = None
try:
    from langfuse import Langfuse
    # Only create the client if we have the necessary credentials
    if all([os.getenv("LANGFUSE_PUBLIC_KEY"), os.getenv("LANGFUSE_SECRET_KEY")]):
        langfuse_client = Langfuse(
            public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
            secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
            host=os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")
        )
        logger.info("Langfuse initialized successfully.")
    else:
        logger.warning("Langfuse credentials missing, tracing disabled")
except ImportError:
    logger.warning("Langfuse package not installed, tracing disabled")
except Exception as e:
    logger.error(f"Failed to initialize Langfuse: {e}")

class ModelRetryError(Exception):
    """Custom exception for model retry scenarios"""
    def __init__(self, message: str, original_error: Exception, failed_models: List[str]):
        self.message = message
        self.original_error = original_error
        self.failed_models = failed_models
        super().__init__(self.message)

class LLM:
    """
    LLM class for interacting with OpenAI's GPT models and Google's Gemini models for call center data interpretation.
    Enhanced with user alerting and automatic model retry/fallback functionality.
    """
    
    def __init__(self, model: str = "gpt-4-turbo"):
        """
        Initialize the LLM class.
        
        Args:
            model (str): The model to use. Default is gpt-4-turbo.
                        Supports OpenAI models (gpt-4-turbo, gpt-4, gpt-3.5-turbo) 
                        and Gemini models (gemini-2.0-flash, gemini-1.5-pro).
        """
        self.model = model
        self.original_model = model  # Keep track of original model for reporting
        
        # Determine if this is a Gemini or OpenAI model
        self.is_gemini = model.startswith("gemini")
        
        if self.is_gemini:
            # Initialize Gemini client
            self.google_api_key = os.getenv("GOOGLE_API_KEY")
            if not self.google_api_key:
                logger.error("Google API key not found. Please set the GOOGLE_API_KEY environment variable.")
                raise ValueError("Google API key not found")
            
            self.client = genai.Client(api_key=self.google_api_key)
            logger.info(f"Gemini client initialized with model: {self.model}")
        else:
            # Initialize OpenAI client
            self.openai_api_key = os.getenv("OPENAI_API_KEY")
            if not self.openai_api_key:
                logger.error("OpenAI API key not found. Please set the OPENAI_API_KEY environment variable.")
                raise ValueError("OpenAI API key not found")
                
            self.client = OpenAI(api_key=self.openai_api_key)
            logger.info(f"OpenAI client initialized with model: {self.model}")
        
        # Get base directory and build absolute paths
        base_dir = Path(__file__).resolve().parent.parent
        system_prompt_path = os.path.join(base_dir, "prompts", "interprete_system_prompt.txt")
        user_prompt_path = os.path.join(base_dir, "prompts", "interprete_user_prompt.txt")
        
        logger.info(f"Loading prompts from: {system_prompt_path} and {user_prompt_path}")
        
        # Load prompts
        try:
            with open(system_prompt_path, "r") as f:
                self.system_prompt = f.read()
            with open(user_prompt_path, "r") as f:
                self.user_prompt_template = f.read()
        except FileNotFoundError as e:
            logger.error(f"Error loading prompts: {e}")
            raise ValueError(f"Prompt file not found: {e}")
            
        logger.info(f"LLM initialized with model: {self.model}")
    
    def _create_user_alert(self, alert_type: str, message: str, model_used: str = None, fallback_model: str = None) -> Dict[str, Any]:
        """
        Create a standardized user alert message.
        
        Args:
            alert_type (str): Type of alert (error, warning, info, success)
            message (str): Alert message
            model_used (str): Original model that was attempted
            fallback_model (str): Fallback model that was used
            
        Returns:
            Dict[str, Any]: Formatted alert
        """
        alert = {
            "alert_type": alert_type,
            "message": message,
            "timestamp": time.time(),
            "original_model": model_used or self.original_model
        }
        
        if fallback_model:
            alert["fallback_model"] = fallback_model
            alert["model_switched"] = True
        else:
            alert["model_switched"] = False
            
        return alert
    
    def _attempt_model_with_retry(self, messages: list, max_retries: int = 3, retry_delay: float = 1.0) -> Dict[str, Any]:
        """
        Attempt to call a model with retry logic and automatic fallback.
        
        Args:
            messages (list): Messages to send to the model
            max_retries (int): Maximum retry attempts per model
            retry_delay (float): Delay between retries in seconds
            
        Returns:
            Dict[str, Any]: Model response with additional metadata
        """
        failed_models = []
        alerts = []
        
        # Start with the original model
        models_to_try = [self.model] + MODEL_FALLBACK_CHAINS.get(self.model, [])
        
        for model_attempt in models_to_try:
            logger.info(f"Attempting model: {model_attempt}")
            
            # Update the current model
            original_model = self.model
            self.model = model_attempt
            self._update_client_for_model(model_attempt)
            
            for attempt in range(max_retries):
                try:
                    # Call the appropriate model
                    if self.is_gemini:
                        response = self._call_gemini_model(messages)
                    else:
                        response = self._call_openai_model(messages)
                    
                    # Success! Add metadata and return
                    response["model_metadata"] = {
                        "final_model_used": model_attempt,
                        "original_model": self.original_model,
                        "model_switched": model_attempt != self.original_model,
                        "attempts_made": attempt + 1,
                        "failed_models": failed_models
                    }
                    
                    if model_attempt != self.original_model:
                        alert = self._create_user_alert(
                            "warning", 
                            f"Successfully switched from {self.original_model} to {model_attempt} due to availability issues.",
                            self.original_model,
                            model_attempt
                        )
                        alerts.append(alert)
                    elif attempt > 0:
                        alert = self._create_user_alert(
                            "info",
                            f"Successfully completed request with {model_attempt} after {attempt + 1} attempts."
                        )
                        alerts.append(alert)
                    
                    response["user_alerts"] = alerts
                    return response
                    
                except Exception as e:
                    error_msg = str(e).lower()
                    
                    # Check for specific error types
                    if "rate limit" in error_msg or "quota" in error_msg:
                        if attempt < max_retries - 1:
                            alert = self._create_user_alert(
                                "warning",
                                f"Rate limit reached for {model_attempt}. Retrying in {retry_delay} seconds... (Attempt {attempt + 1}/{max_retries})"
                            )
                            alerts.append(alert)
                            logger.warning(f"Rate limit hit for {model_attempt}, retrying in {retry_delay}s")
                            time.sleep(retry_delay)
                            retry_delay *= 1.5  # Exponential backoff
                            continue
                        else:
                            failed_models.append(f"{model_attempt} (rate limited)")
                            break
                    elif "authentication" in error_msg or "api key" in error_msg or "401" in error_msg:
                        failed_models.append(f"{model_attempt} (auth error)")
                        alert = self._create_user_alert(
                            "error",
                            f"Authentication failed for {model_attempt}. Trying alternative model..."
                        )
                        alerts.append(alert)
                        break
                    elif "not found" in error_msg or "404" in error_msg:
                        failed_models.append(f"{model_attempt} (not found)")
                        alert = self._create_user_alert(
                            "error", 
                            f"Model {model_attempt} not available. Trying alternative model..."
                        )
                        alerts.append(alert)
                        break
                    else:
                        if attempt < max_retries - 1:
                            alert = self._create_user_alert(
                                "warning",
                                f"Temporary error with {model_attempt}: {str(e)[:100]}... Retrying in {retry_delay} seconds..."
                            )
                            alerts.append(alert)
                            logger.warning(f"Error with {model_attempt}, retrying: {e}")
                            time.sleep(retry_delay)
                            retry_delay *= 1.5
                            continue
                        else:
                            failed_models.append(f"{model_attempt} ({str(e)[:50]}...)")
                            break
            
            # Reset the model to original for next attempt
            self.model = original_model
            self._update_client_for_model(original_model)
        
        # If we get here, all models failed
        error_alert = self._create_user_alert(
            "error",
            f"All available models failed. Attempted: {', '.join([m.split(' (')[0] for m in failed_models])}. Please try again later or contact support."
        )
        alerts.append(error_alert)
        
        raise ModelRetryError(
            f"All model attempts failed: {failed_models}",
            Exception("All fallback models exhausted"),
            failed_models
        )
    
    def _update_client_for_model(self, model: str):
        """
        Update the client configuration for the specified model.
        
        Args:
            model (str): Model to configure client for
        """
        # Update is_gemini flag
        self.is_gemini = model.startswith("gemini")
        
        # Update client if switching between providers
        if self.is_gemini and not hasattr(self, '_gemini_client'):
            if not self.google_api_key:
                raise ValueError("Google API key required for Gemini models")
            self._gemini_client = genai.Client(api_key=self.google_api_key)
            self.client = self._gemini_client
        elif not self.is_gemini and not hasattr(self, '_openai_client'):
            if not self.openai_api_key:
                raise ValueError("OpenAI API key required for OpenAI models")
            self._openai_client = OpenAI(api_key=self.openai_api_key)
            self.client = self._openai_client
        elif self.is_gemini:
            self.client = getattr(self, '_gemini_client', self.client)
        else:
            self.client = getattr(self, '_openai_client', self.client)
    
    def _call_gemini_model(self, messages: list) -> Dict[str, Any]:
        """
        Call Gemini model with the given messages.
        
        Args:
            messages (list): List of messages in OpenAI format
            
        Returns:
            Dict[str, Any]: Response in OpenAI-compatible format
        """
        try:
            # Convert OpenAI format to Gemini format
            # For Gemini, we combine system and user messages into a single prompt
            combined_prompt = ""
            for msg in messages:
                if msg["role"] == "system":
                    combined_prompt += f"System: {msg['content']}\n\n"
                elif msg["role"] == "user":
                    combined_prompt += f"User: {msg['content']}\n\n"
            
            # Call Gemini API
            response = self.client.models.generate_content(
                model=self.model,
                contents=combined_prompt.strip()
            )
            
            # Convert response to OpenAI-compatible format
            return {
                "choices": [{
                    "message": {
                        "content": response.text
                    }
                }],
                "usage": {
                    # Gemini doesn't provide detailed token counts in the same way
                    # We'll estimate based on text length
                    "prompt_tokens": len(combined_prompt) // 4,  # Rough estimate: 4 chars per token
                    "completion_tokens": len(response.text) // 4,
                    "total_tokens": (len(combined_prompt) + len(response.text)) // 4
                }
            }
        except Exception as e:
            logger.error(f"Error calling Gemini model: {e}")
            raise e
    
    def _call_openai_model(self, messages: list) -> Dict[str, Any]:
        """
        Call OpenAI model with the given messages.
        
        Args:
            messages (list): List of messages in OpenAI format
            
        Returns:
            Dict[str, Any]: OpenAI response
        """
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.3,
                max_tokens=2000
            )
            
            # Convert to dict format for consistency
            return {
                "choices": [{
                    "message": {
                        "content": response.choices[0].message.content
                    }
                }],
                "usage": {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens
                }
            }
        except Exception as e:
            logger.error(f"Error calling OpenAI model: {e}")
            raise e
        
    def interpret_data(self, data: str, user_info: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Interpret call center data using the selected model (OpenAI or Gemini).
        
        Args:
            data (str): The call center data to interpret.
            user_info (Dict[str, Any]): Optional user information for tracing.
            
        Returns:
            Dict[str, Any]: The interpreted results.
        """
        # Generate a unique trace ID
        trace_id = str(uuid4())
        trace = None
        generation = None
        
        try:
            # Safely handle the data to avoid string formatting issues
            user_prompt = self.user_prompt_template.replace("{data}", str(data))
            
            # Prepare messages in OpenAI format
            messages = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": user_prompt}
            ]
            
            # Create Langfuse trace if available
            if langfuse_client is not None:
                try:
                    # Create a trace following Langfuse API
                    trace = langfuse_client.trace(
                        name="AI-generate",
                        id=trace_id,
                        metadata={
                            "model": self.model,
                            "model_provider": "gemini" if self.is_gemini else "openai",
                            "user_id": user_info.get("user_id", "unknown") if user_info else "unknown",
                            "username": user_info.get("username", "unknown") if user_info else "unknown",
                            "company_id": user_info.get("company_id", "unknown") if user_info else "unknown",
                        }
                    )
                    
                    # Create a generation directly in the trace
                    generation = trace.generation(
                        name="call-center-interpretation",
                        model=self.model,
                        model_parameters={
                            "temperature": 0.3,
                            "max_tokens": 2000
                        },
                        input=messages,
                        metadata={"data_type": "call_center_data"}
                    )
                except Exception as trace_error:
                    logger.error(f"Error creating Langfuse trace: {trace_error}")
                    logger.error(traceback.format_exc())
                    trace = None
                    generation = None
            
            # Use the new retry mechanism with automatic fallback
            start_time = time.time()
            try:
                response = self._attempt_model_with_retry(messages)
            except ModelRetryError as retry_error:
                # All models failed, create comprehensive error response
                end_time = time.time()
                
                error_result = {
                    "status": "error",
                    "error": str(retry_error.message),
                    "message": "Failed to interpret data - all available models unavailable",
                    "user_alerts": [{
                        "alert_type": "error",
                        "message": f"Unable to process your request. All AI models are currently unavailable. Failed models: {', '.join(retry_error.failed_models)}",
                        "timestamp": time.time(),
                        "original_model": self.original_model,
                        "model_switched": False,
                        "support_message": "Please try again in a few minutes or contact support if the issue persists."
                    }],
                    "model_metadata": {
                        "original_model": self.original_model,
                        "failed_models": retry_error.failed_models,
                        "total_processing_time": end_time - start_time
                    }
                }
                
                # Log comprehensive error to Langfuse
                if langfuse_client is not None and trace is not None:
                    try:
                        trace.event(
                            name="all-models-failed",
                            input={"failed_models": retry_error.failed_models, "original_error": str(retry_error.original_error)},
                            level="ERROR"
                        )
                        
                        if generation is not None:
                            generation.end(
                                output={"error": str(retry_error.message), "failed_models": retry_error.failed_models},
                                status="error"
                            )
                    except Exception as log_error:
                        logger.error(f"Error logging retry failure to Langfuse: {log_error}")
                
                return error_result
            
            end_time = time.time()
            
            # Extract the response
            interpretation_text = response["choices"][0]["message"]["content"]
            
            # Parse the interpretation as JSON
            try:
                # Extract JSON from the response if it's wrapped in markdown code blocks
                if "```json" in interpretation_text:
                    # Extract content between ```json and ``` markers
                    json_content = re.search(r'```json\s*([\s\S]*?)\s*```', interpretation_text)
                    if json_content:
                        interpretation_text = json_content.group(1)
                elif "```" in interpretation_text:
                    # Extract content between ``` markers (generic code block)
                    json_content = re.search(r'```\s*([\s\S]*?)\s*```', interpretation_text)
                    if json_content:
                        interpretation_text = json_content.group(1)
                
                # Parse the text as JSON
                interpretation_json = json.loads(interpretation_text)
                logger.info("Successfully parsed interpretation as JSON")
            except json.JSONDecodeError as e:
                logger.warning(f"Failed to parse interpretation as JSON: {e}")
                # If parsing fails, keep the original text but wrap it in a JSON object
                interpretation_json = {"raw_text": interpretation_text}
            
            result = {
                "status": "success",
                "interpretation": interpretation_json,
                "model_used": response.get("model_metadata", {}).get("final_model_used", self.model),
                "model_provider": "gemini" if response.get("model_metadata", {}).get("final_model_used", self.model).startswith("gemini") else "openai",
                "tokens_used": {
                    "prompt_tokens": response["usage"]["prompt_tokens"],
                    "completion_tokens": response["usage"]["completion_tokens"],
                    "total_tokens": response["usage"]["total_tokens"]
                },
                "user_alerts": response.get("user_alerts", []),
                "model_metadata": response.get("model_metadata", {}),
                "processing_time": end_time - start_time
            }
            
            # Update Langfuse generation if available
            if langfuse_client is not None and trace is not None and generation is not None:
                try:
                    # End the generation with the output and token usage
                    generation.end(
                        output=interpretation_json,  # Use the parsed JSON
                        usage={
                            "prompt_tokens": response["usage"]["prompt_tokens"],
                            "completion_tokens": response["usage"]["completion_tokens"],
                            "total_tokens": response["usage"]["total_tokens"]
                        }
                    )
                    
                    # Add an event to the trace with the full result
                    trace.event(
                        name="interpretation-complete",
                        output=result
                    )
                    
                except Exception as span_error:
                    logger.error(f"Error logging to Langfuse: {span_error}")
                    logger.error(traceback.format_exc())
            
            return result
            
        except Exception as e:
            logger.error(f"Error interpreting data: {e}")
            logger.error(traceback.format_exc())
            
            error_result = {
                "status": "error",
                "error": str(e),
                "message": "Failed to interpret data"
            }
            
            # Log error to Langfuse
            if langfuse_client is not None and trace is not None:
                try:
                    # Create error event
                    trace.event(
                        name="error",
                        input={"error": str(e), "traceback": traceback.format_exc()},
                        level="ERROR"
                    )
                    
                    # If generation was created, end it with error
                    if generation is not None:
                        generation.end(
                            output={"error": str(e)},
                            status="error"
                        )
                except Exception as log_error:
                    logger.error(f"Error logging to Langfuse: {log_error}")
                    logger.error(traceback.format_exc())
            
            return error_result
    
    def analyze_business_intelligence(
        self, 
        call_data: Dict[str, Any], 
        company_name: str = "Unknown Company",
        industry: str = "Call Center",
        user_info: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Generate comprehensive business intelligence insights from call center data.
        
        Args:
            call_data (Dict[str, Any]): The call center data to analyze
            company_name (str): Name of the company being analyzed
            industry (str): Industry type of the company
            user_info (Dict[str, Any]): Optional user information for tracing
            
        Returns:
            Dict[str, Any]: Comprehensive business intelligence analysis
        """
        # Generate a unique trace ID
        trace_id = str(uuid4())
        trace = None
        generation = None
        
        try:
            # Load BI analysis prompts and schema
            base_dir = Path(__file__).resolve().parent.parent
            bi_system_prompt_path = os.path.join(base_dir, "prompts", "bi_analysis_system_prompt.txt")
            bi_user_prompt_path = os.path.join(base_dir, "prompts", "bi_analysis_user_prompt.txt")
            bi_json_schema_path = os.path.join(base_dir, "prompts", "bi_analysis_json_schema.json")
            
            logger.info(f"Loading BI prompts from: {bi_system_prompt_path}, {bi_user_prompt_path}, and {bi_json_schema_path}")
            
            # Load BI prompts and schema
            try:
                with open(bi_system_prompt_path, "r") as f:
                    bi_system_prompt = f.read()
                with open(bi_user_prompt_path, "r") as f:
                    bi_user_prompt_template = f.read()
                with open(bi_json_schema_path, "r") as f:
                    bi_json_schema = f.read()
            except FileNotFoundError as e:
                logger.error(f"Error loading BI prompts/schema: {e}")
                raise ValueError(f"BI prompt/schema file not found: {e}")
            
            # Enhance the system prompt with the JSON schema
            bi_system_prompt_enhanced = bi_system_prompt + f"""

**EXACT JSON OUTPUT FORMAT REQUIRED:**
You must return your response in this exact JSON structure:

{bi_json_schema}

**CRITICAL FORMATTING REQUIREMENTS:**
- Return ONLY valid JSON, no markdown code blocks
- Follow the exact structure shown above
- Include all required sections
- Use the specified data types and formats
- Ensure all arrays and objects are properly formatted
"""
            
            # Extract data summary for prompt variables
            time_period = call_data.get("time_period", {})
            start_date = time_period.get("start_date", "Unknown")
            end_date = time_period.get("end_date", "Unknown")
            
            summary = call_data.get("summary", {})
            total_calls = summary.get("total_calls", 0)
            
            # Apply intelligent sampling if dataset is too large for context window
            sampled_call_data, sampling_info = self._apply_intelligent_sampling(
                call_data, 
                target_ratio_transcript=0.7
            )
            
            # Use sampled data for analysis
            actual_total_calls = sampling_info["sampled_calls"]
            
            # Perform temporal analysis to detect multiple months/years
            temporal_analysis = self._analyze_temporal_patterns(sampled_call_data)
            
            # Format the user prompt with sampled data
            bi_user_prompt = bi_user_prompt_template.format(
                company_name=company_name,
                industry=industry,
                start_date=start_date,
                end_date=end_date,
                total_calls=actual_total_calls,
                call_data_json=json.dumps(sampled_call_data, indent=2)
            )
            
            # Add sampling info to user prompt if sampling was applied
            if sampling_info["sampling_applied"]:
                sampling_note = f"""

**DATA SAMPLING INFORMATION:**
- Original dataset: {sampling_info['original_calls']} calls
- Sampled for analysis: {sampling_info['sampled_calls']} calls ({sampling_info['sampling_percentage']:.1f}%)
- With transcripts: {sampling_info['transcripts_sampled']}/{sampling_info['transcripts_available']} ({sampling_info['transcript_percentage']:.1f}%)
- Without transcripts: {sampling_info['no_transcripts_sampled']}/{sampling_info['no_transcripts_available']} ({sampling_info['no_transcript_percentage']:.1f}%)
- Sampling method: Random stratified sampling maintaining statistical representation
- **IMPORTANT**: Scale your insights appropriately - this represents a statistical sample of the full dataset.
"""
                bi_user_prompt += sampling_note
            
            # Generate quality assurance evidence data
            qa_evidence = self._prepare_qa_evidence(sampled_call_data)
            
            # Add QA evidence information to prompt if available
            if qa_evidence.get("qa_evidence_available"):
                qa_section = f"""

**QUALITY ASSURANCE EVIDENCE DATA AVAILABLE:**
{json.dumps(qa_evidence, indent=2)}

**MANDATORY EVIDENCE REQUIREMENTS:**
For every insight, claim, and recommendation in your response, you MUST include:
- Call ID from actual data (use the call_examples above)
- Customer phone number for tracking 
- Recording URL for quality assurance listening
- Exact date/time for context
- Human-readable explanation (not technical field names)
- Clear reason why this call supports the insight

**EVIDENCE EXAMPLES TO USE:**
Use the actual call examples provided above. For instance:
- High Volume Analysis: Use examples from qa_evidence.call_examples.high_volume_calls
- Duration Analysis: Use examples from qa_evidence.call_examples.long_duration_calls  
- Recording Review: Use examples from qa_evidence.call_examples.recording_examples

This enables quality assurance teams to track conversations, listen to recordings, and validate AI insights.
"""
                bi_user_prompt += qa_section
            
            # Add temporal analysis information if available
            if temporal_analysis.get("temporal_analysis_available"):
                analysis_period = temporal_analysis["analysis_period"]
                period_insights = temporal_analysis["period_specific_insights"]
                
                temporal_note = f"""

**TEMPORAL ANALYSIS - CRITICAL REQUIREMENT:**
The data spans {analysis_period['total_days']} days from {analysis_period['start_date']} to {analysis_period['end_date']}.

**MULTIPLE TIME PERIODS DETECTED:**
- Multiple Months: {'YES' if analysis_period['spans_multiple_months'] else 'NO'}
- Multiple Years: {'YES' if analysis_period['spans_multiple_years'] else 'NO'}
- Months Covered: {', '.join(analysis_period['months_covered'])}
- Years Covered: {', '.join(map(str, analysis_period['years_covered']))}
- Quarters Covered: {', '.join(analysis_period['quarters_covered'])}

**MANDATORY TEMPORAL INSIGHTS REQUIREMENTS:**
1. **ACKNOWLEDGE ALL TIME PERIODS**: Explicitly mention each month/year/quarter in your analysis
2. **PERIOD-SPECIFIC INSIGHTS**: Provide detailed insights for each individual month and year
3. **COMPARATIVE ANALYSIS**: Compare performance between different months/years
4. **TEMPORAL TRENDS**: Identify patterns and trends across time periods
5. **SEASONAL PATTERNS**: Highlight seasonal variations and recurring patterns

**PERIOD-SPECIFIC DATA SUMMARY:**
"""
                
                # Add monthly insights
                if period_insights.get("monthly_insights"):
                    temporal_note += "\n**MONTHLY BREAKDOWN:**\n"
                    for insight in period_insights["monthly_insights"]:
                        temporal_note += f"- {insight['period']}: {insight['call_volume']} calls, {insight['answer_rate']}% answer rate, {insight['avg_duration']}s avg duration ({insight['performance_indicator']} performance)\n"
                
                # Add yearly insights
                if period_insights.get("yearly_insights"):
                    temporal_note += "\n**YEARLY BREAKDOWN:**\n"
                    for insight in period_insights["yearly_insights"]:
                        temporal_note += f"- {insight['period']}: {insight['call_volume']} calls, {insight['answer_rate']}% answer rate, {insight['avg_duration']}s avg duration ({insight['performance_indicator']} performance)\n"
                
                # Add comparative analysis
                if period_insights.get("comparative_analysis"):
                    comp_analysis = period_insights["comparative_analysis"]
                    if comp_analysis.get("best_month"):
                        temporal_note += f"\n**PERFORMANCE COMPARISON:**\n"
                        temporal_note += f"- Best performing month: {comp_analysis['best_month'][0]} ({comp_analysis['best_month'][1]:.1f}% answer rate)\n"
                        temporal_note += f"- Worst performing month: {comp_analysis['worst_month'][0]} ({comp_analysis['worst_month'][1]:.1f}% answer rate)\n"
                    
                    if comp_analysis.get("volume_trend"):
                        trend = comp_analysis["volume_trend"]
                        temporal_note += f"- Volume trend: {trend['direction']} ({trend['change_percentage']:+.1f}% change from {trend['first_period'][0]} to {trend['last_period'][0]})\n"
                
                temporal_note += f"""

**YOUR ANALYSIS MUST INCLUDE:**
1. Executive summary acknowledging all {len(analysis_period['months_covered'])} months and {len(analysis_period['years_covered'])} years
2. Individual insights for each month: {', '.join(analysis_period['months_covered'])}
3. Individual insights for each year: {', '.join(map(str, analysis_period['years_covered']))}
4. Month-to-month comparisons and trends
5. Year-over-year comparisons (if multiple years)
6. Seasonal patterns and recurring behaviors
7. Period-specific recommendations for each month/year
8. Forecasting based on temporal trends observed

**TEMPORAL DATA AVAILABLE:**
{json.dumps(temporal_analysis, indent=2)}
"""
                
                bi_user_prompt += temporal_note
            
            # Prepare messages for BI analysis with enhanced system prompt
            messages = [
                {"role": "system", "content": bi_system_prompt_enhanced},
                {"role": "user", "content": bi_user_prompt}
            ]
            
            # Create Langfuse trace if available
            if langfuse_client is not None:
                try:
                    # Create a trace for BI analysis
                    trace = langfuse_client.trace(
                        name="BI-Analysis",
                        id=trace_id,
                        metadata={
                            "model": self.model,
                            "model_provider": "gemini" if self.is_gemini else "openai",
                            "analysis_type": "business_intelligence",
                            "company_name": company_name,
                            "industry": industry,
                            "user_id": user_info.get("user_id", "unknown") if user_info else "unknown",
                            "username": user_info.get("username", "unknown") if user_info else "unknown",
                            "company_id": user_info.get("company_id", "unknown") if user_info else "unknown",
                            "data_period": f"{start_date} to {end_date}",
                            "total_calls_analyzed": total_calls
                        }
                    )
                    
                    # Create a generation for BI analysis
                    generation = trace.generation(
                        name="business-intelligence-analysis",
                        model=self.model,
                        model_parameters={
                            "temperature": 0.1,  # Lower temperature for more consistent BI analysis
                            "max_tokens": 4000   # Higher token limit for comprehensive analysis
                        },
                        input=messages,
                        metadata={
                            "analysis_type": "business_intelligence",
                            "data_type": "call_center_data",
                            "company_context": {
                                "name": company_name,
                                "industry": industry,
                                "calls_analyzed": total_calls
                            }
                        }
                    )
                except Exception as trace_error:
                    logger.error(f"Error creating Langfuse trace: {trace_error}")
                    logger.error(traceback.format_exc())
                    trace = None
                    generation = None
            
            # Use the new retry mechanism with automatic fallback
            start_time = time.time()
            try:
                # For BI analysis, use specialized methods with retry wrapper
                response = self._attempt_model_with_retry_bi(messages)
            except ModelRetryError as retry_error:
                # All models failed, create comprehensive error response
                end_time = time.time()
                
                error_result = {
                    "status": "error",
                    "error": str(retry_error.message),
                    "message": "Failed to generate business intelligence analysis - all available models unavailable",
                    "user_alerts": [{
                        "alert_type": "error",
                        "message": f"Unable to generate business intelligence analysis. All AI models are currently unavailable. Failed models: {', '.join(retry_error.failed_models)}",
                        "timestamp": time.time(),
                        "original_model": self.original_model,
                        "model_switched": False,
                        "support_message": "Please try again in a few minutes or contact support if the issue persists."
                    }],
                    "model_metadata": {
                        "original_model": self.original_model,
                        "failed_models": retry_error.failed_models,
                        "total_processing_time": end_time - start_time
                    },
                    "analysis_metadata": {
                        "company_name": company_name,
                        "industry": industry,
                        "trace_id": trace_id
                    }
                }
                
                # Log comprehensive error to Langfuse
                if langfuse_client is not None and trace is not None:
                    try:
                        trace.event(
                            name="bi-analysis-all-models-failed",
                            input={"failed_models": retry_error.failed_models, "original_error": str(retry_error.original_error)},
                            level="ERROR"
                        )
                        
                        if generation is not None:
                            generation.end(
                                output={"error": str(retry_error.message), "failed_models": retry_error.failed_models},
                                status="error"
                            )
                    except Exception as log_error:
                        logger.error(f"Error logging retry failure to Langfuse: {log_error}")
                
                return error_result
            
            end_time = time.time()
            
            # Extract the BI analysis response
            analysis_text = response["choices"][0]["message"]["content"]
            
            # Parse the BI analysis as JSON
            try:
                # Extract JSON from the response if it's wrapped in markdown code blocks
                original_text = analysis_text
                if "```json" in analysis_text:
                    json_content = re.search(r'```json\s*\n?([\s\S]*?)\n?\s*```', analysis_text)
                    if json_content:
                        analysis_text = json_content.group(1).strip()
                        logger.info("Extracted JSON from ```json code block")
                elif "```" in analysis_text:
                    json_content = re.search(r'```\s*\n?([\s\S]*?)\n?\s*```', analysis_text)
                    if json_content:
                        analysis_text = json_content.group(1).strip()
                        logger.info("Extracted JSON from ``` code block")
                
                # Validate we have content after extraction
                if not analysis_text or analysis_text.isspace():
                    logger.error("JSON extraction resulted in empty content")
                    analysis_text = original_text  # Fall back to original
                
                # Apply basic JSON fixing for common issues
                analysis_text = self._clean_json_response(analysis_text)
                
                # Parse the text as JSON
                analysis_json = json.loads(analysis_text)
                logger.info("Successfully parsed BI analysis as JSON")
                
                # Validate that required sections are present (based on our schema)
                required_sections = [
                    "analysis_metadata", "executive_summary", "customer_experience",
                    "agent_performance", "operational_efficiency", "business_intelligence",
                    "risk_assessment", "recommendations", "visualization_dashboard", "benchmarks_and_targets"
                ]
                
                missing_sections = [section for section in required_sections if section not in analysis_json]
                if missing_sections:
                    logger.warning(f"BI analysis missing sections: {missing_sections}")
                
                # Additional schema validation
                schema_validation_results = self._validate_bi_schema(analysis_json)
                if not schema_validation_results["valid"]:
                    logger.warning(f"BI analysis schema validation issues: {schema_validation_results['issues']}")
                
                # Add validation results to the analysis
                analysis_json["_schema_validation"] = schema_validation_results
                
                # Convert visualizations to structured text format
                analysis_json = self._convert_visualizations_to_structured_text(analysis_json)
                    
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse BI analysis as JSON: {e}")
                
                # Try multiple recovery strategies
                recovery_success = False
                recovery_strategy = ""
                
                # Strategy 1: More aggressive JSON cleaning
                try:
                    logger.info("Attempting recovery strategy 1: More aggressive JSON cleaning")
                    recovered_text = self._extract_json_from_mixed_content(analysis_text)
                    analysis_json = json.loads(recovered_text)
                    recovery_success = True
                    recovery_strategy = "extracted_from_mixed_content"
                    logger.info("Recovery strategy 1 successful: Extracted JSON from mixed content")
                except Exception as recovery_e1:
                    logger.debug(f"Recovery strategy 1 failed: {recovery_e1}")
                
                # Strategy 2: Try to fix specific issues around the error position
                error_position = getattr(e, 'pos', None)
                
                # If we don't have a direct position, try to extract from error message
                if error_position is None:
                    error_msg = str(e)
                    # Look for patterns like "char 21509" in the error message
                    char_match = re.search(r'char (\d+)', error_msg)
                    if char_match:
                        error_position = int(char_match.group(1))
                        logger.debug(f"Extracted error position {error_position} from error message")
                
                if not recovery_success and error_position is not None:
                    try:
                        logger.info(f"Attempting recovery strategy 2: Fix around error position {error_position}")
                        # Use the enhanced control character removal with error position
                        fixed_text = self._remove_control_characters(analysis_text, error_position=error_position)
                        analysis_json = json.loads(fixed_text)
                        recovery_success = True
                        recovery_strategy = "fixed_around_error_position"
                        logger.info("Recovery strategy 2 successful: Fixed around error position")
                    except Exception as recovery_e2:
                        logger.debug(f"Recovery strategy 2 failed: {recovery_e2}")
                
                # Strategy 3: Try ultra-aggressive character-by-character cleaning
                if not recovery_success:
                    try:
                        logger.info("Attempting recovery strategy 3: Ultra-aggressive character cleaning")
                        # Ultra-aggressive cleaning: remove ALL control characters and fix encoding
                        cleaned_chars = []
                        for i, char in enumerate(analysis_text):
                            char_code = ord(char)
                            # Allow only printable ASCII (32-126), newlines (10), and Unicode (>127)
                            if char_code >= 32:
                                cleaned_chars.append(char)
                            elif char_code == 10:  # Newline
                                cleaned_chars.append(char)
                            elif char_code in [9, 13]:  # Tab, carriage return -> space
                                cleaned_chars.append(' ')
                            # Skip all other control characters (0-31)
                            else:
                                logger.debug(f"Removed control character ASCII {char_code} at position {i}")
                        
                        cleaned_text = ''.join(cleaned_chars)
                        
                        # Additional regex cleanup for any remaining problematic characters
                        cleaned_text = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F]', '', cleaned_text)
                        
                        # Fix common encoding issues that might introduce control characters
                        cleaned_text = cleaned_text.replace('\u0000', '')  # Null bytes
                        cleaned_text = cleaned_text.replace('\u0008', '')  # Backspace
                        cleaned_text = cleaned_text.replace('\u000c', '')  # Form feed
                        cleaned_text = cleaned_text.replace('\u001f', '')  # Unit separator
                        
                        analysis_json = json.loads(cleaned_text)
                        recovery_success = True
                        recovery_strategy = "ultra_aggressive_cleaning"
                        logger.info("Recovery strategy 3 successful: Ultra-aggressive character cleaning")
                    except Exception as recovery_e3:
                        logger.debug(f"Recovery strategy 3 failed: {recovery_e3}")
                
                # Strategy 4: Try JSON reconstruction with line-by-line cleaning
                if not recovery_success:
                    try:
                        logger.info("Attempting recovery strategy 4: Line-by-line JSON reconstruction")
                        lines = analysis_text.split('\n')
                        cleaned_lines = []
                        
                        for line_num, line in enumerate(lines):
                            # Clean each line individually
                            cleaned_line = ''.join(char for char in line if ord(char) >= 32 or char in '\t\r')
                            # Remove any Unicode control characters
                            cleaned_line = re.sub(r'[\u0000-\u001F\u007F-\u009F]', '', cleaned_line)
                            cleaned_lines.append(cleaned_line)
                        
                        reconstructed_text = '\n'.join(cleaned_lines)
                        analysis_json = json.loads(reconstructed_text)
                        recovery_success = True
                        recovery_strategy = "line_by_line_reconstruction"
                        logger.info("Recovery strategy 4 successful: Line-by-line reconstruction")
                    except Exception as recovery_e4:
                        logger.debug(f"Recovery strategy 4 failed: {recovery_e4}")
                
                # If all recovery strategies failed
                if not recovery_success:
                    logger.error("All JSON recovery strategies failed")
                    
                    # NEW RECOVERY STRATEGY 5: Property Name Error Fix
                    try:
                        logger.info("Attempting recovery strategy 5: Property name error fix")
                        fixed_text = self._fix_property_name_errors(analysis_text, error_position)
                        analysis_json = json.loads(fixed_text)
                        recovery_success = True
                        recovery_strategy = "property_name_error_fix"
                        logger.info("Recovery strategy 5 successful: Property name error fix")
                    except Exception as recovery_e5:
                        logger.debug(f"Recovery strategy 5 failed: {recovery_e5}")
                
                    # If still failed, try NEW RECOVERY STRATEGY 6: Incremental parsing
                    if not recovery_success:
                        try:
                            logger.info("Attempting recovery strategy 6: Incremental JSON parsing")
                            analysis_json = self._incremental_json_parse(analysis_text, error_position)
                            recovery_success = True
                            recovery_strategy = "incremental_json_parsing"
                            logger.info("Recovery strategy 6 successful: Incremental JSON parsing")
                        except Exception as recovery_e6:
                            logger.debug(f"Recovery strategy 6 failed: {recovery_e6}")
                
                    # If all recovery strategies failed
                    if not recovery_success:
                        logger.error("All enhanced JSON recovery strategies failed")
                        # Store more of the raw text for debugging large JSON responses
                        raw_text_sample = analysis_text[:2000] + "..." if len(analysis_text) > 2000 else analysis_text
                        if error_position and error_position < len(analysis_text):
                            # Include text around error position for debugging
                            start_context = max(0, error_position - 500)
                            end_context = min(len(analysis_text), error_position + 500)
                            error_context = analysis_text[start_context:end_context]
                            raw_text_sample += f"\n\nERROR CONTEXT (chars {start_context}-{end_context}):\n{error_context}"
                        
                        analysis_json = {
                            "status": "parsing_error",
                            "error": f"JSON parsing failed: {str(e)}",
                            "raw_text": raw_text_sample,
                            "recovery_attempts": [
                                "mixed_content_extraction", 
                                "error_position_fix", 
                                "ultra_aggressive_cleaning", 
                                "line_by_line_reconstruction",
                                "property_name_error_fix",
                                "incremental_json_parsing"
                            ],
                            "error_position": error_position
                        }
                else:
                    logger.info(f"JSON parsing recovered using strategy: {recovery_strategy}")
                    # Add recovery metadata to the analysis
                    if isinstance(analysis_json, dict):
                        analysis_json["_parsing_recovery"] = {
                            "recovery_successful": True,
                            "recovery_strategy": recovery_strategy,
                            "original_error": str(e)
                        }
            
            result = {
                "status": "success",
                "analysis": analysis_json,
                "model_used": response.get("model_metadata", {}).get("final_model_used", self.model),
                "model_provider": "gemini" if response.get("model_metadata", {}).get("final_model_used", self.model).startswith("gemini") else "openai",
                "analysis_metadata": {
                    "company_name": company_name,
                    "industry": industry,
                    "analysis_period": f"{start_date} to {end_date}",
                    "total_calls_analyzed": actual_total_calls,
                    "original_calls_available": total_calls,
                    "processing_time_seconds": round(end_time - start_time, 2),
                    "trace_id": trace_id,
                    "sampling_info": sampling_info
                },
                "tokens_used": {
                    "prompt_tokens": response["usage"]["prompt_tokens"],
                    "completion_tokens": response["usage"]["completion_tokens"],
                    "total_tokens": response["usage"]["total_tokens"]
                },
                "user_alerts": response.get("user_alerts", []),
                "model_metadata": response.get("model_metadata", {}),
                "temporal_analysis": temporal_analysis
            }
            
            # Update Langfuse generation if available
            if langfuse_client is not None and trace is not None and generation is not None:
                try:
                    # End the generation with the output and token usage
                    generation.end(
                        output=analysis_json,
                        usage={
                            "prompt_tokens": response["usage"]["prompt_tokens"],
                            "completion_tokens": response["usage"]["completion_tokens"],
                            "total_tokens": response["usage"]["total_tokens"]
                        }
                    )
                    
                    # Add events to the trace for key insights
                    if isinstance(analysis_json, dict) and "executive_summary" in analysis_json:
                        exec_summary = analysis_json["executive_summary"]
                        trace.event(
                            name="executive-summary-generated",
                            output={
                                "overall_health_score": exec_summary.get("overall_health_score"),
                                "key_findings": exec_summary.get("key_findings", []),
                                "critical_alerts": exec_summary.get("critical_alerts", [])
                            }
                        )
                    
                    # Add completion event
                    trace.event(
                        name="bi-analysis-complete",
                        output={
                            "status": "success",
                            "processing_time": round(end_time - start_time, 2),
                            "total_tokens": response["usage"]["total_tokens"]
                        }
                    )
                    
                except Exception as span_error:
                    logger.error(f"Error logging to Langfuse: {span_error}")
                    logger.error(traceback.format_exc())
            
            return result
            
        except Exception as e:
            logger.error(f"Error in business intelligence analysis: {e}")
            logger.error(traceback.format_exc())
            
            error_result = {
                "status": "error",
                "error": str(e),
                "message": "Failed to generate business intelligence analysis",
                "model_used": self.model,
                "model_provider": "gemini" if self.is_gemini else "openai",
                "analysis_metadata": {
                    "company_name": company_name,
                    "industry": industry,
                    "trace_id": trace_id
                }
            }
            
            # Log error to Langfuse
            if langfuse_client is not None and trace is not None:
                try:
                    # Create error event
                    trace.event(
                        name="bi-analysis-error",
                        input={"error": str(e), "traceback": traceback.format_exc()},
                        level="ERROR"
                    )
                    
                    # If generation was created, end it with error
                    if generation is not None:
                        generation.end(
                            output={"error": str(e)},
                            status="error"
                        )
                except Exception as log_error:
                    logger.error(f"Error logging error to Langfuse: {log_error}")
            
            return error_result
    
    def _prepare_qa_evidence(self, call_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Prepare quality assurance evidence data with call IDs, recordings, and human-readable explanations.
        Uses actual MongoDB transcript data with public_url for recordings.
        
        Args:
            call_data (Dict[str, Any]): Call center data with transcripts from MongoDB
            
        Returns:
            Dict[str, Any]: QA evidence data for prompt injection
        """
        try:
            calls = call_data.get("calls", [])
            transcripts = call_data.get("transcripts", [])
            
            if not calls:
                return {"qa_evidence_available": False}
            
            # Create transcript lookup by call_id for MongoDB public_url access
            transcript_lookup = {}
            for transcript in transcripts:
                # Handle different call_id formats
                call_id = transcript.get("transcript_id") or transcript.get("call_id") or transcript.get("_id")
                if call_id:
                    transcript_lookup[str(call_id)] = transcript
            
            # Select representative calls for different categories
            qa_examples = {
                "high_volume_calls": [],
                "long_duration_calls": [],
                "repeat_callers": [],
                "peak_hour_calls": [],
                "recording_examples": []
            }
            
            # Process calls and extract examples
            for call in calls[:50]:  # Limit to first 50 for performance
                try:
                    # Extract call details
                    call_id = call.get("uniqueid") or call.get("call_id") or f"CDR_{datetime.now().strftime('%Y_%m%d_%H%M%S')}"
                    customer_phone = call.get("src", "Unknown")
                    duration = call.get("duration", 0)
                    calldate = call.get("calldate")
                    
                    # Format date/time
                    formatted_datetime = "Unknown"
                    formatted_date_only = "Unknown"
                    if calldate:
                        try:
                            if isinstance(calldate, str):
                                dt = datetime.fromisoformat(calldate.replace('Z', '+00:00'))
                            else:
                                dt = calldate
                            formatted_datetime = dt.strftime("%B %d, %Y at %I:%M %p EST")
                            formatted_date_only = dt.strftime("%B %d, %Y")
                        except:
                            formatted_datetime = str(calldate)
                            formatted_date_only = str(calldate)
                    
                    # Get recording URL from MongoDB transcript data
                    recording_url = self._get_mongodb_recording_url(call_id, transcript_lookup)
                    
                    # Create call example
                    call_example = {
                        "call_id": call_id,
                        "customer_phone": customer_phone,
                        "date_time": formatted_datetime,
                        "date_only": formatted_date_only,
                        "recording_url": recording_url,
                        "duration_seconds": duration,
                        "duration_minutes": f"{duration // 60}:{duration % 60:02d}" if duration else "0:00"
                    }
                    
                    # Categorize calls
                    if duration > 180:  # Long calls (>3 minutes)
                        call_example["why_relevant"] = "Long duration call - analyze for efficiency opportunities and training needs"
                        qa_examples["long_duration_calls"].append(call_example)
                    
                    if recording_url and not recording_url.startswith("https://recordings.company.com"):  # Real recordings
                        call_example["why_relevant"] = "Call with available recording for quality assurance review and coaching"
                        qa_examples["recording_examples"].append(call_example)
                    
                    # Add to general examples if not already categorized
                    if len(qa_examples["high_volume_calls"]) < 3:
                        call_example["why_relevant"] = "Representative call for general performance analysis"
                        qa_examples["high_volume_calls"].append(call_example)
                
                except Exception as e:
                    logger.debug(f"Error processing call for QA evidence: {e}")
                    continue
            
            # Identify repeat callers with dates for enhanced evidence
            repeat_caller_examples = self._identify_repeat_callers_with_dates(calls)
            qa_examples["repeat_callers"] = repeat_caller_examples
            
            return {
                "qa_evidence_available": True,
                "call_examples": qa_examples,
                "total_calls_with_recordings": len(qa_examples["recording_examples"]),
                "total_repeat_callers": len(qa_examples["repeat_callers"]),
                "evidence_instructions": {
                    "recording_source": "Use actual public_url from MongoDB transcript collection for recordings",
                    "call_id_format": "Use actual call IDs from CDR data for complete traceability",
                    "customer_references": "Always include dates when referencing customer interactions",
                    "recording_access": "Provide direct MongoDB public_url for quality assurance listening",
                    "human_explanations": "Replace technical field names with clear business language",
                    "relevance_context": "Explain why each call example supports the specific insight"
                }
            }
            
        except Exception as e:
            logger.error(f"Error preparing QA evidence: {e}")
            return {"qa_evidence_available": False, "error": str(e)}
    
    def _get_mongodb_recording_url(self, call_id: str, transcript_lookup: Dict[str, Any]) -> str:
        """
        Get recording URL from MongoDB transcript data using public_url field.
        Enhanced to handle various data structures and field names.
        
        Args:
            call_id (str): Call ID to look up
            transcript_lookup (Dict[str, Any]): Transcript lookup dictionary
            
        Returns:
            str: Recording URL from MongoDB public_url or fallback
        """
        try:
            # Try to find transcript data for this call_id
            transcript = transcript_lookup.get(str(call_id))
            if transcript:
                # Check for public_url in various possible locations
                if isinstance(transcript, dict):
                    # Direct public_url field
                    if "public_url" in transcript and transcript["public_url"]:
                        return transcript["public_url"]
                    
                    # Check nested structures
                    if "recording" in transcript and isinstance(transcript["recording"], dict):
                        if "public_url" in transcript["recording"]:
                            return transcript["recording"]["public_url"]
                    
                    # Check for alternative field names
                    for field_name in ["recording_url", "audio_url", "file_url", "gcs_url"]:
                        if field_name in transcript and transcript[field_name]:
                            return transcript[field_name]
                    
                    # Check if transcript itself contains URL-like strings
                    for key, value in transcript.items():
                        if isinstance(value, str) and ("http" in value or "gs://" in value):
                            return value
            
            # Try alternative call_id formats
            for alt_id in [call_id, f"CDR_{call_id}", call_id.replace("CDR_", ""), call_id.split("_")[-1] if "_" in call_id else call_id]:
                transcript_data = transcript_lookup.get(str(alt_id))
                if transcript_data and isinstance(transcript_data, dict):
                    # Same checks as above for alternative IDs
                    if "public_url" in transcript_data and transcript_data["public_url"]:
                        return transcript_data["public_url"]
                    
                    for field_name in ["recording_url", "audio_url", "file_url", "gcs_url"]:
                        if field_name in transcript_data and transcript_data[field_name]:
                            return transcript_data[field_name]
            
            # Check if call_id is referenced in any transcript (broader search)
            for transcript_id, transcript_data in transcript_lookup.items():
                if isinstance(transcript_data, dict):
                    # Check if call_id matches any identifier in the transcript
                    if (str(call_id) in str(transcript_data.get("filename", "")) or
                        str(call_id) in str(transcript_data.get("file_path", "")) or
                        str(call_id) == str(transcript_data.get("uniqueid", ""))):
                        if "public_url" in transcript_data and transcript_data["public_url"]:
                            return transcript_data["public_url"]
            
            # Log the available fields for debugging
            if transcript:
                available_fields = list(transcript.keys()) if isinstance(transcript, dict) else "non-dict data"
                logger.debug(f"No public_url found for call_id {call_id}. Available fields: {available_fields}")
            
            # Fallback to generated URL
            return self._generate_recording_url(call_id)
            
        except Exception as e:
            logger.debug(f"Error getting MongoDB recording URL for {call_id}: {e}")
            return self._generate_recording_url(call_id)
    
    def _identify_repeat_callers_with_dates(self, calls: List[Dict]) -> List[Dict]:
        """
        Identify repeat callers and provide examples with dates for better evidence.
        
        Args:
            calls (List[Dict]): List of call records
            
        Returns:
            List[Dict]: Repeat caller examples with date information
        """
        try:
            # Group calls by customer phone number
            caller_groups = {}
            for call in calls:
                phone = call.get("src", "")
                if phone and phone != "Unknown":
                    if phone not in caller_groups:
                        caller_groups[phone] = []
                    caller_groups[phone].append(call)
            
            # Find repeat callers (customers with multiple calls)
            repeat_caller_examples = []
            for phone, phone_calls in caller_groups.items():
                if len(phone_calls) > 1:
                    # Sort calls by date
                    sorted_calls = sorted(phone_calls, key=lambda x: x.get("calldate", ""))
                    
                    # Format dates for the calls
                    call_dates = []
                    for call in sorted_calls:
                        try:
                            calldate = call.get("calldate")
                            if calldate:
                                if isinstance(calldate, str):
                                    dt = datetime.fromisoformat(calldate.replace('Z', '+00:00'))
                                else:
                                    dt = calldate
                                formatted_date = dt.strftime("%B %d, %Y at %I:%M %p")
                                call_dates.append(formatted_date)
                        except:
                            call_dates.append(str(calldate))
                    
                    # Create repeat caller example
                    first_call = sorted_calls[0]
                    last_call = sorted_calls[-1]
                    
                    repeat_example = {
                        "customer_phone": phone,
                        "total_calls": len(phone_calls),
                        "call_dates": call_dates,
                        "first_call_id": first_call.get("uniqueid", ""),
                        "last_call_id": last_call.get("uniqueid", ""),
                        "date_range": f"{call_dates[0]} to {call_dates[-1]}" if len(call_dates) > 1 else call_dates[0],
                        "why_relevant": f"Customer called {len(phone_calls)} times, indicating potential unresolved issues or complex service needs"
                    }
                    
                    repeat_caller_examples.append(repeat_example)
            
            # Limit to top 5 repeat callers
            return repeat_caller_examples[:5]
            
        except Exception as e:
            logger.debug(f"Error identifying repeat callers: {e}")
            return []
    
    def _generate_recording_url(self, call_id: str, recording_file: str = "") -> str:
        """
        Generate a recording URL for call playback (fallback method).
        
        Args:
            call_id (str): Unique call identifier
            recording_file (str): Recording file name
            
        Returns:
            str: Recording URL
        """
        if recording_file and recording_file.strip():
            # If recording file is provided, use it
            base_url = "https://recordings.company.com"
            if recording_file.startswith(('http://', 'https://')):
                return recording_file
            else:
                return f"{base_url}/{recording_file}"
        else:
            # Generate placeholder URL based on call ID
            return f"https://recordings.company.com/call_{call_id}.wav"
    
    def _attempt_model_with_retry_bi(self, messages: list, max_retries: int = 3, retry_delay: float = 1.0) -> Dict[str, Any]:
        """
        Attempt to call a model with retry logic and automatic fallback for BI analysis.
        Uses BI-specific parameters (lower temperature, higher token limits).
        
        Args:
            messages (list): Messages to send to the model
            max_retries (int): Maximum retry attempts per model
            retry_delay (float): Delay between retries in seconds
            
        Returns:
            Dict[str, Any]: Model response with additional metadata
        """
        failed_models = []
        alerts = []
        
        # Start with the original model
        models_to_try = [self.model] + MODEL_FALLBACK_CHAINS.get(self.model, [])
        
        for model_attempt in models_to_try:
            logger.info(f"Attempting BI analysis with model: {model_attempt}")
            
            # Update the current model
            original_model = self.model
            self.model = model_attempt
            self._update_client_for_model(model_attempt)
            
            for attempt in range(max_retries):
                try:
                    # Call the appropriate BI-specific model
                    if self.is_gemini:
                        response = self._call_gemini_model_bi(messages)
                    else:
                        response = self._call_openai_model_bi(messages)
                    
                    # Success! Add metadata and return
                    response["model_metadata"] = {
                        "final_model_used": model_attempt,
                        "original_model": self.original_model,
                        "model_switched": model_attempt != self.original_model,
                        "attempts_made": attempt + 1,
                        "failed_models": failed_models,
                        "analysis_type": "business_intelligence"
                    }
                    
                    if model_attempt != self.original_model:
                        alert = self._create_user_alert(
                            "warning", 
                            f"Successfully switched from {self.original_model} to {model_attempt} for business intelligence analysis due to availability issues.",
                            self.original_model,
                            model_attempt
                        )
                        alerts.append(alert)
                    elif attempt > 0:
                        alert = self._create_user_alert(
                            "info",
                            f"Successfully completed BI analysis with {model_attempt} after {attempt + 1} attempts."
                        )
                        alerts.append(alert)
                    
                    response["user_alerts"] = alerts
                    return response
                    
                except Exception as e:
                    error_msg = str(e).lower()
                    
                    # Check for specific error types
                    if "rate limit" in error_msg or "quota" in error_msg:
                        if attempt < max_retries - 1:
                            alert = self._create_user_alert(
                                "warning",
                                f"Rate limit reached for {model_attempt} during BI analysis. Retrying in {retry_delay} seconds... (Attempt {attempt + 1}/{max_retries})"
                            )
                            alerts.append(alert)
                            logger.warning(f"BI analysis rate limit hit for {model_attempt}, retrying in {retry_delay}s")
                            time.sleep(retry_delay)
                            retry_delay *= 1.5  # Exponential backoff
                            continue
                        else:
                            failed_models.append(f"{model_attempt} (rate limited)")
                            break
                    elif "authentication" in error_msg or "api key" in error_msg or "401" in error_msg:
                        failed_models.append(f"{model_attempt} (auth error)")
                        alert = self._create_user_alert(
                            "error",
                            f"Authentication failed for {model_attempt} during BI analysis. Trying alternative model..."
                        )
                        alerts.append(alert)
                        break
                    elif "not found" in error_msg or "404" in error_msg:
                        failed_models.append(f"{model_attempt} (not found)")
                        alert = self._create_user_alert(
                            "error", 
                            f"Model {model_attempt} not available for BI analysis. Trying alternative model..."
                        )
                        alerts.append(alert)
                        break
                    else:
                        if attempt < max_retries - 1:
                            alert = self._create_user_alert(
                                "warning",
                                f"Temporary error with {model_attempt} during BI analysis: {str(e)[:100]}... Retrying in {retry_delay} seconds..."
                            )
                            alerts.append(alert)
                            logger.warning(f"BI analysis error with {model_attempt}, retrying: {e}")
                            time.sleep(retry_delay)
                            retry_delay *= 1.5
                            continue
                        else:
                            failed_models.append(f"{model_attempt} ({str(e)[:50]}...)")
                            break
            
            # Reset the model to original for next attempt
            self.model = original_model
            self._update_client_for_model(original_model)
        
        # If we get here, all models failed
        error_alert = self._create_user_alert(
            "error",
            f"All available models failed for business intelligence analysis. Attempted: {', '.join([m.split(' (')[0] for m in failed_models])}. Please try again later or contact support."
        )
        alerts.append(error_alert)
        
        raise ModelRetryError(
            f"All BI analysis model attempts failed: {failed_models}",
            Exception("All fallback models exhausted for BI analysis"),
            failed_models
        )

    def _call_openai_model_bi(self, messages: list) -> Dict[str, Any]:
        """
        Call OpenAI model with BI-specific parameters for comprehensive analysis.
        
        Args:
            messages (list): List of messages in OpenAI format
            
        Returns:
            Dict[str, Any]: OpenAI response
        """
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.1,  # Lower temperature for consistent BI analysis
                max_tokens=4000   # Higher limit for comprehensive analysis
            )
            
            # Convert to dict format for consistency
            return {
                "choices": [{
                    "message": {
                        "content": response.choices[0].message.content
                    }
                }],
                "usage": {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens
                }
            }
        except Exception as e:
            logger.error(f"Error calling OpenAI model for BI analysis: {e}")
            raise e
    
    def _call_gemini_model_bi(self, messages: list) -> Dict[str, Any]:
        """
        Call Gemini model with BI-specific parameters for comprehensive analysis.
        
        Args:
            messages (list): List of messages in OpenAI format
            
        Returns:
            Dict[str, Any]: Response in OpenAI-compatible format
        """
        try:
            # Convert OpenAI format to Gemini format
            combined_prompt = ""
            for msg in messages:
                if msg["role"] == "system":
                    combined_prompt += f"System: {msg['content']}\n\n"
                elif msg["role"] == "user":
                    combined_prompt += f"User: {msg['content']}\n\n"
            
            # Call Gemini API with BI-specific configuration
            response = self.client.models.generate_content(
                model=self.model,
                contents=combined_prompt.strip()
            )
            
            # Convert response to OpenAI-compatible format
            return {
                "choices": [{
                    "message": {
                        "content": response.text
                    }
                }],
                "usage": {
                    # Gemini doesn't provide detailed token counts
                    # Estimate based on text length for BI analysis
                    "prompt_tokens": len(combined_prompt) // 4,
                    "completion_tokens": len(response.text) // 4,
                    "total_tokens": (len(combined_prompt) + len(response.text)) // 4
                }
            }
        except Exception as e:
            logger.error(f"Error calling Gemini model for BI analysis: {e}")
            raise e

    def compare_bi_analysis(
        self,
        call_data: Dict[str, Any],
        company_name: str = "Unknown Company",
        industry: str = "Call Center",
        openai_model: str = "gpt-4-turbo",
        gemini_model: str = "gemini-2.0-flash",
        user_info: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Compare business intelligence analysis results from both OpenAI and Gemini models.
        
        Args:
            call_data (Dict[str, Any]): The call center data to analyze
            company_name (str): Name of the company being analyzed
            industry (str): Industry type of the company
            openai_model (str): OpenAI model to use for comparison
            gemini_model (str): Gemini model to use for comparison
            user_info (Dict[str, Any]): Optional user information for tracing
            
        Returns:
            Dict[str, Any]: Comparison results from both models
        """
        comparison_trace_id = str(uuid4())
        comparison_trace = None
        
        try:
            # Create a comparison trace in Langfuse
            if langfuse_client is not None:
                try:
                    comparison_trace = langfuse_client.trace(
                        name="BI-Analysis-Comparison",
                        id=comparison_trace_id,
                        metadata={
                            "analysis_type": "bi_comparison",
                            "models_compared": [openai_model, gemini_model],
                            "company_name": company_name,
                            "industry": industry,
                            "user_id": user_info.get("user_id", "unknown") if user_info else "unknown",
                            "comparison_id": comparison_trace_id
                        }
                    )
                    
                    comparison_trace.event(
                        name="comparison-started",
                        output={
                            "models": [openai_model, gemini_model],
                            "company": company_name,
                            "industry": industry
                        }
                    )
                except Exception as trace_error:
                    logger.error(f"Error creating comparison trace: {trace_error}")
                    comparison_trace = None
            
            logger.info(f"🔄 Starting BI analysis comparison: {openai_model} vs {gemini_model}")
            
            # Create OpenAI LLM instance and run analysis
            openai_llm = LLM(model=openai_model)
            logger.info(f"🤖 Running OpenAI analysis with {openai_model}")
            openai_result = openai_llm.analyze_business_intelligence(
                call_data=call_data,
                company_name=company_name,
                industry=industry,
                user_info=user_info
            )
            
            # Create Gemini LLM instance and run analysis
            gemini_llm = LLM(model=gemini_model)
            logger.info(f"🧠 Running Gemini analysis with {gemini_model}")
            gemini_result = gemini_llm.analyze_business_intelligence(
                call_data=call_data,
                company_name=company_name,
                industry=industry,
                user_info=user_info
            )
            
            # Calculate comparison metrics
            comparison_metrics = self._calculate_comparison_metrics(openai_result, gemini_result)
            
            comparison_result = {
                "comparison_id": comparison_trace_id,
                "comparison_metadata": {
                    "company_name": company_name,
                    "industry": industry,
                    "models_compared": {
                        "openai": openai_model,
                        "gemini": gemini_model
                    },
                    "generated_at": __import__('datetime').datetime.utcnow().isoformat()
                },
                "results": {
                    "openai": openai_result,
                    "gemini": gemini_result
                },
                "comparison_metrics": comparison_metrics,
                "summary": {
                    "both_successful": (
                        openai_result.get("status") == "success" and 
                        gemini_result.get("status") == "success"
                    ),
                    "openai_success": openai_result.get("status") == "success",
                    "gemini_success": gemini_result.get("status") == "success",
                    "total_tokens_openai": openai_result.get("tokens_used", {}).get("total_tokens", 0),
                    "total_tokens_gemini": gemini_result.get("tokens_used", {}).get("total_tokens", 0),
                    "processing_time_openai": openai_result.get("analysis_metadata", {}).get("processing_time_seconds", 0),
                    "processing_time_gemini": gemini_result.get("analysis_metadata", {}).get("processing_time_seconds", 0)
                }
            }
            
            # Log comparison results to Langfuse
            if langfuse_client is not None and comparison_trace is not None:
                try:
                    comparison_trace.event(
                        name="comparison-completed",
                        output={
                            "summary": comparison_result["summary"],
                            "metrics": comparison_metrics
                        }
                    )
                    
                    # Create separate events for model performance
                    comparison_trace.event(
                        name="openai-performance",
                        output={
                            "model": openai_model,
                            "success": openai_result.get("status") == "success",
                            "tokens": openai_result.get("tokens_used", {}),
                            "processing_time": openai_result.get("analysis_metadata", {}).get("processing_time_seconds", 0)
                        }
                    )
                    
                    comparison_trace.event(
                        name="gemini-performance",
                        output={
                            "model": gemini_model,
                            "success": gemini_result.get("status") == "success",
                            "tokens": gemini_result.get("tokens_used", {}),
                            "processing_time": gemini_result.get("analysis_metadata", {}).get("processing_time_seconds", 0)
                        }
                    )
                    
                except Exception as log_error:
                    logger.error(f"Error logging comparison to Langfuse: {log_error}")
            
            logger.info(f"✅ BI analysis comparison completed successfully")
            logger.info(f"📊 OpenAI ({openai_model}): {openai_result.get('status', 'unknown')} - {openai_result.get('tokens_used', {}).get('total_tokens', 0)} tokens")
            logger.info(f"📊 Gemini ({gemini_model}): {gemini_result.get('status', 'unknown')} - {gemini_result.get('tokens_used', {}).get('total_tokens', 0)} tokens")
            
            return comparison_result
            
        except Exception as e:
            logger.error(f"Error in BI analysis comparison: {e}")
            logger.error(traceback.format_exc())
            
            error_result = {
                "comparison_id": comparison_trace_id,
                "status": "error",
                "error": str(e),
                "message": "Failed to complete BI analysis comparison",
                "comparison_metadata": {
                    "company_name": company_name,
                    "industry": industry,
                    "models_compared": {
                        "openai": openai_model,
                        "gemini": gemini_model
                    }
                }
            }
            
            # Log error to Langfuse
            if langfuse_client is not None and comparison_trace is not None:
                try:
                    comparison_trace.event(
                        name="comparison-error",
                        input={"error": str(e), "traceback": traceback.format_exc()},
                        level="ERROR"
                    )
                except Exception as log_error:
                    logger.error(f"Error logging comparison error to Langfuse: {log_error}")
            
            return error_result
    
    def _calculate_comparison_metrics(self, openai_result: Dict[str, Any], gemini_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculate comparison metrics between OpenAI and Gemini results.
        
        Args:
            openai_result (Dict[str, Any]): OpenAI analysis result
            gemini_result (Dict[str, Any]): Gemini analysis result
            
        Returns:
            Dict[str, Any]: Comparison metrics
        """
        try:
            metrics = {
                "performance_comparison": {
                    "token_efficiency": {
                        "openai_tokens": openai_result.get("tokens_used", {}).get("total_tokens", 0),
                        "gemini_tokens": gemini_result.get("tokens_used", {}).get("total_tokens", 0),
                        "more_efficient": "unknown"
                    },
                    "speed_comparison": {
                        "openai_time": openai_result.get("analysis_metadata", {}).get("processing_time_seconds", 0),
                        "gemini_time": gemini_result.get("analysis_metadata", {}).get("processing_time_seconds", 0),
                        "faster_model": "unknown"
                    }
                },
                "analysis_quality": {
                    "both_parsed_successfully": False,
                    "openai_json_valid": False,
                    "gemini_json_valid": False,
                    "structure_completeness": {
                        "openai_sections": 0,
                        "gemini_sections": 0
                    }
                },
                "content_comparison": {
                    "executive_summary_comparison": "not_available",
                    "recommendations_comparison": "not_available",
                    "insights_depth_comparison": "not_available"
                }
            }
            
            # Performance metrics
            openai_tokens = openai_result.get("tokens_used", {}).get("total_tokens", 0)
            gemini_tokens = gemini_result.get("tokens_used", {}).get("total_tokens", 0)
            
            if openai_tokens > 0 and gemini_tokens > 0:
                if openai_tokens < gemini_tokens:
                    metrics["performance_comparison"]["token_efficiency"]["more_efficient"] = "openai"
                elif gemini_tokens < openai_tokens:
                    metrics["performance_comparison"]["token_efficiency"]["more_efficient"] = "gemini"
                else:
                    metrics["performance_comparison"]["token_efficiency"]["more_efficient"] = "equal"
            
            openai_time = openai_result.get("analysis_metadata", {}).get("processing_time_seconds", 0)
            gemini_time = gemini_result.get("analysis_metadata", {}).get("processing_time_seconds", 0)
            
            if openai_time > 0 and gemini_time > 0:
                if openai_time < gemini_time:
                    metrics["performance_comparison"]["speed_comparison"]["faster_model"] = "openai"
                elif gemini_time < openai_time:
                    metrics["performance_comparison"]["speed_comparison"]["faster_model"] = "gemini"
                else:
                    metrics["performance_comparison"]["speed_comparison"]["faster_model"] = "equal"
            
            # Analysis quality metrics
            openai_analysis = openai_result.get("analysis", {})
            gemini_analysis = gemini_result.get("analysis", {})
            
            openai_valid = isinstance(openai_analysis, dict) and "analysis_metadata" in openai_analysis
            gemini_valid = isinstance(gemini_analysis, dict) and "analysis_metadata" in gemini_analysis
            
            metrics["analysis_quality"]["openai_json_valid"] = openai_valid
            metrics["analysis_quality"]["gemini_json_valid"] = gemini_valid
            metrics["analysis_quality"]["both_parsed_successfully"] = openai_valid and gemini_valid
            
            # Count sections in each analysis
            expected_sections = [
                "executive_summary", "customer_experience", "agent_performance", 
                "operational_efficiency", "business_intelligence", "common_products",
                "customer_keywords", "flagged_conversations", "interesting_facts", "forecasts", 
                "risk_assessment", "recommendations", "visualization_dashboard", 
                "benchmarks_and_targets", "analysis_metadata"
            ]
            
            if openai_valid:
                metrics["analysis_quality"]["structure_completeness"]["openai_sections"] = len([
                    k for k in openai_analysis.keys() if k in expected_sections
                ])
            
            if gemini_valid:
                metrics["analysis_quality"]["structure_completeness"]["gemini_sections"] = len([
                    k for k in gemini_analysis.keys() if k in expected_sections
                ])
            
            # Content comparison (basic)
            if openai_valid and gemini_valid:
                openai_exec = openai_analysis.get("executive_summary", {})
                gemini_exec = gemini_analysis.get("executive_summary", {})
                
                if "key_findings" in openai_exec and "key_findings" in gemini_exec:
                    openai_findings_count = len(openai_exec.get("key_findings", []))
                    gemini_findings_count = len(gemini_exec.get("key_findings", []))
                    metrics["content_comparison"]["executive_summary_comparison"] = {
                        "openai_findings": openai_findings_count,
                        "gemini_findings": gemini_findings_count,
                        "more_detailed": "openai" if openai_findings_count > gemini_findings_count 
                                       else "gemini" if gemini_findings_count > openai_findings_count 
                                       else "equal"
                    }
                
                openai_recs = openai_analysis.get("recommendations", {})
                gemini_recs = gemini_analysis.get("recommendations", {})
                
                if "immediate_actions" in openai_recs and "immediate_actions" in gemini_recs:
                    openai_actions_count = len(openai_recs.get("immediate_actions", []))
                    gemini_actions_count = len(gemini_recs.get("immediate_actions", []))
                    metrics["content_comparison"]["recommendations_comparison"] = {
                        "openai_actions": openai_actions_count,
                        "gemini_actions": gemini_actions_count,
                        "more_actionable": "openai" if openai_actions_count > gemini_actions_count 
                                          else "gemini" if gemini_actions_count > openai_actions_count 
                                          else "equal"
                    }
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error calculating comparison metrics: {e}")
            return {
                "error": str(e),
                "message": "Failed to calculate comparison metrics"
            }
    
    def _validate_bi_schema(self, analysis_json: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate the BI analysis JSON against our expected schema structure.
        
        Args:
            analysis_json (Dict[str, Any]): The parsed BI analysis JSON
            
        Returns:
            Dict[str, Any]: Validation results with issues found
        """
        validation_result = {
            "valid": True,
            "issues": [],
            "completeness_score": 0,
            "total_sections": 15,
            "present_sections": 0
        }
        
        try:
            # Define expected structure based on our schema
            expected_structure = {
                "analysis_metadata": {
                    "required_fields": ["company_name", "analysis_period", "data_summary", "generated_at"],
                    "type": "object"
                },
                "executive_summary": {
                    "required_fields": ["overall_health_score", "key_findings", "critical_alerts", "business_impact"],
                    "type": "object"
                },
                "customer_experience": {
                    "required_fields": ["satisfaction_score", "sentiment_distribution", "top_pain_points", "visualizations"],
                    "type": "object"
                },
                "agent_performance": {
                    "required_fields": ["overall_effectiveness_score", "heat_model_breakdown", "performance_distribution", "visualizations"],
                    "type": "object"
                },
                "operational_efficiency": {
                    "required_fields": ["efficiency_score", "key_metrics", "bottlenecks_identified", "visualizations"],
                    "type": "object"
                },
                "business_intelligence": {
                    "required_fields": ["strategic_insights", "customer_behavior_patterns", "revenue_impact_analysis"],
                    "type": "object"
                },
                "common_products": {
                    "required_fields": ["top_products_mentioned", "product_demand_trends", "product_performance_metrics"],
                    "type": "object"
                },
                "customer_keywords": {
                    "required_fields": ["top_keywords", "trending_topics", "sentiment_by_keyword"],
                    "type": "object"
                },
                "flagged_conversations": {
                    "required_fields": ["total_flagged", "flag_categories", "flagged_conversations_details", "flagging_summary"],
                    "type": "object"
                },
                "interesting_facts": {
                    "required_fields": ["surprising_patterns", "notable_correlations", "unusual_behaviors"],
                    "type": "object"
                },
                "forecasts": {
                    "required_fields": ["call_volume_forecast", "product_demand_forecast", "risk_predictions"],
                    "type": "object"
                },
                "risk_assessment": {
                    "required_fields": ["overall_risk_score", "critical_risks", "early_warning_indicators"],
                    "type": "object"
                },
                "recommendations": {
                    "required_fields": ["immediate_actions", "strategic_initiatives"],
                    "type": "object"
                },
                "visualization_dashboard": {
                    "required_fields": ["executive_dashboard", "operational_dashboard"],
                    "type": "object"
                },
                "benchmarks_and_targets": {
                    "required_fields": ["industry_comparisons", "improvement_targets"],
                    "type": "object"
                }
            }
            
            # Check each main section
            for section_name, section_spec in expected_structure.items():
                if section_name in analysis_json:
                    validation_result["present_sections"] += 1
                    section_data = analysis_json[section_name]
                    
                    # Check if section is the right type
                    if section_spec["type"] == "object" and not isinstance(section_data, dict):
                        validation_result["issues"].append(f"Section '{section_name}' should be an object but got {type(section_data).__name__}")
                        validation_result["valid"] = False
                        continue
                    
                    # Check required fields within the section
                    if isinstance(section_data, dict):
                        missing_fields = []
                        for required_field in section_spec["required_fields"]:
                            if required_field not in section_data:
                                missing_fields.append(required_field)
                        
                        if missing_fields:
                            validation_result["issues"].append(f"Section '{section_name}' missing required fields: {missing_fields}")
                            validation_result["valid"] = False
                else:
                    validation_result["issues"].append(f"Missing required section: {section_name}")
                    validation_result["valid"] = False
            
            # Calculate completeness score
            validation_result["completeness_score"] = (validation_result["present_sections"] / validation_result["total_sections"]) * 100
            
            # Check specific data types and values
            if "executive_summary" in analysis_json:
                exec_summary = analysis_json["executive_summary"]
                if "overall_health_score" in exec_summary:
                    health_score = exec_summary["overall_health_score"]
                    if not isinstance(health_score, (int, float)) or not (0 <= health_score <= 100):
                        validation_result["issues"].append("overall_health_score should be a number between 0-100")
                        validation_result["valid"] = False
            
            if "customer_experience" in analysis_json:
                cust_exp = analysis_json["customer_experience"]
                if "satisfaction_score" in cust_exp:
                    sat_score = cust_exp["satisfaction_score"]
                    if not isinstance(sat_score, (int, float)) or not (0 <= sat_score <= 100):
                        validation_result["issues"].append("satisfaction_score should be a number between 0-100")
                        validation_result["valid"] = False
            
            # Check that visualizations are properly structured
            viz_sections = ["customer_experience", "agent_performance", "operational_efficiency"]
            for section in viz_sections:
                if section in analysis_json and "visualizations" in analysis_json[section]:
                    visualizations = analysis_json[section]["visualizations"]
                    if not isinstance(visualizations, list):
                        validation_result["issues"].append(f"'{section}.visualizations' should be an array")
                        validation_result["valid"] = False
            
            logger.info(f"Schema validation completed: {validation_result['completeness_score']:.1f}% complete, {len(validation_result['issues'])} issues found")
            
        except Exception as e:
            validation_result["valid"] = False
            validation_result["issues"].append(f"Schema validation error: {str(e)}")
            logger.error(f"Error during schema validation: {e}")
        
        return validation_result
    
    def _clean_json_response(self, response_text: str) -> str:
        """
        Clean common JSON formatting issues from LLM responses.
        
        Args:
            response_text (str): Raw response text that should contain JSON
            
        Returns:
            str: Cleaned JSON text
        """
        if not response_text:
            return response_text
        
        try:
            # Remove any leading/trailing whitespace
            cleaned = response_text.strip()
            
            # Remove comments (// style)
            cleaned = re.sub(r'//.*?\n', '\n', cleaned)
            
            # Fix invalid control characters that cause JSON parsing errors
            # First, use ultra-aggressive approach to clean all control characters
            cleaned = self._remove_control_characters(cleaned)
            
            # Additional aggressive cleaning for large JSON responses
            # Remove Unicode control characters that might be causing issues
            cleaned = re.sub(r'[\u0000-\u001F\u007F-\u009F]', lambda m: '' if ord(m.group()) != 10 else '\n', cleaned)
            
            # Remove common problematic characters that AI models sometimes generate
            cleaned = cleaned.replace('\u0000', '')  # Null bytes
            cleaned = cleaned.replace('\u0008', '')  # Backspace
            cleaned = cleaned.replace('\u000c', '')  # Form feed
            cleaned = cleaned.replace('\u001f', '')  # Unit separator
            cleaned = cleaned.replace('\u007f', '')  # DEL character
            
            # Then replace common control characters with their JSON-safe equivalents
            # (only if they're meant to be literal strings, not control characters)
            cleaned = cleaned.replace('\t', ' ')     # Tab -> space (safer than \\t in large JSON)
            cleaned = cleaned.replace('\r', '')     # Remove carriage returns
            cleaned = cleaned.replace('\b', '')     # Remove backspace
            cleaned = cleaned.replace('\f', '')     # Remove form feed
            
            # Final regex cleanup for any remaining ASCII control characters
            # Remove ASCII control characters 0-8, 11-12, 14-31 (keep \n=10)
            cleaned = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F]', '', cleaned)
            
            # Fix unescaped quotes within string values (common LLM error)
            # This is tricky - we need to escape quotes that are inside string values but not property names
            cleaned = self._fix_unescaped_quotes(cleaned)
            
            # Fix trailing commas before closing brackets/braces
            cleaned = re.sub(r',(\s*[}\]])', r'\1', cleaned)
            
            # Fix common missing comma issues between properties
            # Pattern: "value"\n  "property": (missing comma)
            cleaned = re.sub(r'(["\}\]])\s*\n(\s*)("[\w_]+"\s*:)', r'\1,\n\2\3', cleaned)
            
            # Ensure the response starts with { or [
            if not cleaned.startswith(('{', '[')):
                # Try to find the first { or [
                start_brace = cleaned.find('{')
                start_bracket = cleaned.find('[')
                
                if start_brace != -1 and (start_bracket == -1 or start_brace < start_bracket):
                    cleaned = cleaned[start_brace:]
                elif start_bracket != -1:
                    cleaned = cleaned[start_bracket:]
            
            # Remove any text after the final } or ]
            if cleaned.startswith('{'):
                # Find the matching closing brace
                brace_count = 0
                for i, char in enumerate(cleaned):
                    if char == '{':
                        brace_count += 1
                    elif char == '}':
                        brace_count -= 1
                        if brace_count == 0:
                            cleaned = cleaned[:i+1]
                            break
            elif cleaned.startswith('['):
                # Find the matching closing bracket
                bracket_count = 0
                for i, char in enumerate(cleaned):
                    if char == '[':
                        bracket_count += 1
                    elif char == ']':
                        bracket_count -= 1
                        if bracket_count == 0:
                            cleaned = cleaned[:i+1]
                            break
            
            return cleaned
            
        except Exception as e:
            logger.warning(f"Error cleaning JSON response: {e}")
            # Last resort: try to extract just the JSON content if there's mixed content
            return self._extract_json_from_mixed_content(response_text)
    
    def _fix_unescaped_quotes(self, json_text: str) -> str:
        """
        Fix unescaped quotes within JSON string values.
        
        Args:
            json_text (str): JSON text that may have unescaped quotes
            
        Returns:
            str: JSON text with quotes properly escaped
        """
        try:
            # This is a simplified approach - we'll look for patterns where quotes
            # appear inside string values and escape them
            
            # Pattern: "some text with "quotes" inside"
            # We need to be careful not to break valid JSON structure
            
            lines = json_text.split('\n')
            fixed_lines = []
            
            for line in lines:
                # Skip lines that don't contain string values
                if ':' not in line or '"' not in line:
                    fixed_lines.append(line)
                    continue
                
                # Look for pattern: "key": "value with "quotes" inside"
                # This is a basic fix - more sophisticated parsing would be needed for complex cases
                if line.count('"') > 4:  # More than the expected key:value pair
                    # Try to fix by escaping quotes that are clearly inside string values
                    # Look for the pattern: ": "..." where ... might contain unescaped quotes
                    
                    # Find the position after ": "
                    colon_quote_pattern = r':\s*"'
                    match = re.search(colon_quote_pattern, line)
                    
                    if match:
                        # Split at the start of the value string
                        prefix = line[:match.end()]
                        remainder = line[match.end():]
                        
                        # Find the closing quote of the value (last quote on the line usually)
                        # But be careful about commas and other JSON structure
                        
                        # Simple heuristic: if the line ends with ", or " or just "
                        # then the last quote is the closing quote
                        if remainder.endswith('",') or remainder.endswith('"'):
                            # Find the last quote
                            last_quote_pos = remainder.rfind('"')
                            if last_quote_pos > 0:
                                # Escape any quotes between start and the last quote
                                value_part = remainder[:last_quote_pos]
                                end_part = remainder[last_quote_pos:]
                                
                                # Escape internal quotes
                                value_part = value_part.replace('"', '\\"')
                                
                                fixed_line = prefix + value_part + end_part
                                fixed_lines.append(fixed_line)
                                continue
                
                # If we couldn't fix it with the above logic, keep the original line
                fixed_lines.append(line)
            
            return '\n'.join(fixed_lines)
            
        except Exception as e:
            logger.warning(f"Error fixing unescaped quotes: {e}")
            return json_text
    
    def _remove_control_characters(self, text: str, error_position: int = None) -> str:
        """
        Remove invalid control characters from JSON text while preserving structure.
        Enhanced to handle specific error positions with aggressive cleaning.
        
        Args:
            text (str): Text that may contain invalid control characters
            error_position (int): Specific character position where error occurred
            
        Returns:
            str: Text with control characters removed or escaped
        """
        try:
            # If we have an error position, apply extra aggressive cleaning around that area
            if error_position is not None:
                logger.debug(f"Applying targeted cleaning around error position {error_position}")
                
                # Expand cleaning window around error position
                start_window = max(0, error_position - 500)
                end_window = min(len(text), error_position + 500)
                
                # Extract the problematic section
                before_section = text[:start_window]
                problem_section = text[start_window:end_window]
                after_section = text[end_window:]
                
                # Ultra-aggressive cleaning of the problem section
                cleaned_problem = []
                for char in problem_section:
                    char_code = ord(char)
                    if char_code >= 32:  # Normal printable characters
                        cleaned_problem.append(char)
                    elif char_code == 10:  # Newline (essential)
                        cleaned_problem.append(char)
                    # Replace ALL other control characters with spaces or remove
                    elif char_code in [9, 13]:  # Tab, carriage return -> space
                        cleaned_problem.append(' ')
                    # Remove all other control characters completely
                
                text = before_section + ''.join(cleaned_problem) + after_section
                logger.debug(f"Applied targeted cleaning to {len(problem_section)} characters around error position")
            
            # Now apply general control character removal to entire text
            result = []
            control_chars_found = []
            
            for i, char in enumerate(text):
                char_code = ord(char)
                
                # Allow normal printable characters (32-126) plus Unicode (>126)
                if char_code >= 32:
                    result.append(char)
                # Allow newlines (essential for JSON formatting)
                elif char_code == 10:  # \n
                    result.append(char)
                # Replace tabs and carriage returns with spaces
                elif char_code in [9, 13]:  # \t, \r
                    result.append(' ')
                # Remove all other control characters (0-31 except \n)
                else:
                    control_chars_found.append(f"ASCII {char_code} at position {i}")
                    logger.debug(f"Removed control character: ASCII {char_code} at position {i}")
                    # Don't add anything - completely remove the character
            
            # If we found problematic control characters, log them for debugging
            if control_chars_found:
                logger.warning(f"Removed {len(control_chars_found)} control characters: {control_chars_found[:10]}")
            
            cleaned_text = ''.join(result)
            
            # Final check: if we still have suspicious characters, do one more pass
            if error_position is not None:
                # Use regex to remove any remaining problematic characters
                cleaned_text = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F]', '', cleaned_text)
                
                # For very large files (like the 48,032 character position case), 
                # also check for Unicode control characters
                cleaned_text = re.sub(r'[\u0000-\u001F\u007F-\u009F]', lambda m: '' if m.group() not in '\n\t' else m.group(), cleaned_text)
            
            return cleaned_text
            
        except Exception as e:
            logger.warning(f"Error removing control characters: {e}")
            return text
    
    def _extract_json_from_mixed_content(self, text: str) -> str:
        """
        Extract JSON content from mixed content that may have non-JSON text.
        
        Args:
            text (str): Mixed content that contains JSON
            
        Returns:
            str: Extracted JSON content
        """
        try:
            # Remove all control characters aggressively first - use our enhanced method
            cleaned = self._remove_control_characters(text)
            
            # Try to find the first complete JSON object or array
            start_pos = -1
            end_pos = -1
            
            # Look for opening brace or bracket
            for i, char in enumerate(cleaned):
                if char in ['{', '[']:
                    start_pos = i
                    break
            
            if start_pos == -1:
                logger.error("No JSON start found in mixed content")
                return text
            
            # Count braces/brackets to find the matching closing one
            if cleaned[start_pos] == '{':
                brace_count = 0
                for i in range(start_pos, len(cleaned)):
                    if cleaned[i] == '{':
                        brace_count += 1
                    elif cleaned[i] == '}':
                        brace_count -= 1
                        if brace_count == 0:
                            end_pos = i
                            break
            else:  # starting with '['
                bracket_count = 0
                for i in range(start_pos, len(cleaned)):
                    if cleaned[i] == '[':
                        bracket_count += 1
                    elif cleaned[i] == ']':
                        bracket_count -= 1
                        if bracket_count == 0:
                            end_pos = i
                            break
            
            if end_pos == -1:
                logger.error("No matching JSON end found in mixed content")
                return cleaned[start_pos:]  # Return from start to end
            
            extracted = cleaned[start_pos:end_pos+1]
            logger.info(f"Extracted JSON from mixed content: {len(extracted)} characters")
            return extracted
            
        except Exception as e:
            logger.error(f"Error extracting JSON from mixed content: {e}")
            return text
    
    def _fix_property_name_errors(self, text: str, error_position: int = None) -> str:
        """
        Fix JSON property name errors specifically targeting 'Expecting property name enclosed in double quotes'.
        
        Args:
            text (str): JSON text with property name errors
            error_position (int): Character position where error occurred
            
        Returns:
            str: Fixed JSON text
        """
        try:
            logger.debug(f"Fixing property name errors around position {error_position}")
            
            # If we have error position, work around that area
            if error_position:
                # Look for common property name errors around the error position
                start_window = max(0, error_position - 200)
                end_window = min(len(text), error_position + 200)
                
                before_section = text[:start_window]
                problem_section = text[start_window:end_window]
                after_section = text[end_window:]
                
                # Fix common property name issues in the problem section
                fixed_section = problem_section
                
                # Pattern 1: Missing quotes around property names
                # Fix: propertyName: "value" -> "propertyName": "value"
                fixed_section = re.sub(r'(\s*)([a-zA-Z_][a-zA-Z0-9_]*)\s*:', r'\1"\2":', fixed_section)
                
                # Pattern 2: Single quotes instead of double quotes for property names
                # Fix: 'propertyName': "value" -> "propertyName": "value"
                fixed_section = re.sub(r"'([^']+)'(\s*):", r'"\1"\2:', fixed_section)
                
                # Pattern 3: Extra characters before property names
                # Fix: ,extra "propertyName": -> , "propertyName":
                fixed_section = re.sub(r'([,{]\s*)[^"{\[\s]*\s*("[\w_]+"\s*:)', r'\1\2', fixed_section)
                
                # Pattern 4: Missing commas between properties
                # Fix: "prop1": "val1" "prop2": -> "prop1": "val1", "prop2":
                fixed_section = re.sub(r'(["\}\]])\s*\n\s*("[\w_]+"\s*:)', r'\1,\n\2', fixed_section)
                
                return before_section + fixed_section + after_section
            
            # If no specific error position, apply fixes to entire text
            fixed_text = text
            
            # Apply all property name fixes globally
            fixed_text = re.sub(r'(\s*)([a-zA-Z_][a-zA-Z0-9_]*)\s*:', r'\1"\2":', fixed_text)
            fixed_text = re.sub(r"'([^']+)'(\s*):", r'"\1"\2:', fixed_text)
            fixed_text = re.sub(r'([,{]\s*)[^"{\[\s]*\s*("[\w_]+"\s*:)', r'\1\2', fixed_text)
            fixed_text = re.sub(r'(["\}\]])\s*\n\s*("[\w_]+"\s*:)', r'\1,\n\2', fixed_text)
            
            logger.debug("Applied property name error fixes")
            return fixed_text
            
        except Exception as e:
            logger.error(f"Error fixing property name errors: {e}")
            return text

    def _incremental_json_parse(self, text: str, error_position: int = None) -> Dict[str, Any]:
        """
        Attempt to parse JSON incrementally by building a valid structure from partial content.
        
        Args:
            text (str): JSON text that failed to parse
            error_position (int): Character position where error occurred
            
        Returns:
            Dict[str, Any]: Partial JSON structure that could be parsed
        """
        try:
            logger.debug("Attempting incremental JSON parsing")
            
            # Clean the text first
            cleaned_text = self._remove_control_characters(text, error_position)
            
            # Try to find the largest valid JSON substring
            if error_position and error_position > 100:
                # Start from before the error and work backwards to find valid JSON
                for cutoff in range(error_position, max(0, error_position - 10000), -100):
                    try:
                        partial_text = cleaned_text[:cutoff]
                        
                        # Try to close any open braces/brackets
                        if partial_text.count('{') > partial_text.count('}'):
                            partial_text += '}' * (partial_text.count('{') - partial_text.count('}'))
                        
                        if partial_text.count('[') > partial_text.count(']'):
                            partial_text += ']' * (partial_text.count('[') - partial_text.count(']'))
                        
                        # Remove trailing commas
                        partial_text = re.sub(r',(\s*[}\]])', r'\1', partial_text)
                        
                        result = json.loads(partial_text)
                        
                        # Add metadata about the partial parsing
                        if isinstance(result, dict):
                            result['_parsing_metadata'] = {
                                'status': 'partial_success',
                                'original_length': len(text),
                                'parsed_length': cutoff,
                                'parsing_method': 'incremental_parsing'
                            }
                        
                        logger.info(f"Incremental parsing successful: parsed {cutoff} of {len(text)} characters")
                        return result
                        
                    except json.JSONDecodeError:
                        continue
            
            # If position-based parsing failed, try chunk-based parsing
            # Parse the JSON in smaller chunks and combine what works
            lines = cleaned_text.split('\n')
            partial_json = "{"
            
            for i, line in enumerate(lines):
                test_json = partial_json + "\n" + line
                
                # Try to make it valid JSON by closing brackets
                test_complete = test_json
                if test_complete.count('{') > test_complete.count('}'):
                    test_complete += '\n' + '}' * (test_complete.count('{') - test_complete.count('}'))
                
                # Remove trailing commas
                test_complete = re.sub(r',(\s*[}\]])', r'\1', test_complete)
                
                try:
                    json.loads(test_complete)
                    partial_json = test_json  # This line is safe to add
                except json.JSONDecodeError:
                    # This line broke it, so finalize what we have
                    break
            
            # Finalize the partial JSON
            if partial_json.count('{') > partial_json.count('}'):
                partial_json += '}' * (partial_json.count('{') - partial_json.count('}'))
            
            partial_json = re.sub(r',(\s*[}\]])', r'\1', partial_json)
            
            result = json.loads(partial_json)
            
            # Add metadata
            if isinstance(result, dict):
                result['_parsing_metadata'] = {
                    'status': 'partial_success_line_based',
                    'original_length': len(text),
                    'parsed_lines': i,
                    'total_lines': len(lines),
                    'parsing_method': 'incremental_line_parsing'
                }
            
            logger.info(f"Line-based incremental parsing successful: parsed {i} of {len(lines)} lines")
            return result
            
        except Exception as e:
            logger.error(f"Incremental JSON parsing failed: {e}")
            
            # Last resort: return a minimal valid structure with error info
            return {
                "status": "partial_parsing_failed",
                "error": str(e),
                "parsing_method": "incremental_fallback",
                "original_text_length": len(text)
            }
    
    def _apply_intelligent_sampling(
        self, 
        call_data: Dict[str, Any], 
        target_ratio_transcript: float = 0.7
    ) -> tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Apply intelligent sampling to call data only if it would exceed context limits.
        Maintains 70:30 ratio of calls with/without transcripts.
        
        Args:
            call_data (Dict[str, Any]): Original call center data
            target_ratio_transcript (float): Target ratio of calls with transcripts (0.7 = 70%)
            
        Returns:
            tuple: (sampled_call_data, sampling_info)
        """
        try:
            # First, estimate if the current data would exceed context limits
            estimated_tokens = self._estimate_data_tokens(call_data)
            available_tokens = self._get_available_context_tokens()
            
            sampling_info = {
                "sampling_applied": False,
                "original_calls": 0,
                "sampled_calls": 0,
                "sampling_percentage": 100.0,
                "transcripts_available": 0,
                "transcripts_sampled": 0,
                "no_transcripts_available": 0,
                "no_transcripts_sampled": 0,
                "transcript_percentage": 0.0,
                "no_transcript_percentage": 0.0,
                "estimated_tokens": estimated_tokens,
                "available_tokens": available_tokens,
                "reason": "no_sampling_needed"
            }
            
            # If data fits within context limits, return as-is
            if estimated_tokens <= available_tokens:
                logger.info(f"📊 Data size: {estimated_tokens:,} tokens ≤ {available_tokens:,} available → No sampling needed")
                
                # Still populate sampling info for transparency
                calls = call_data.get("calls", [])
                transcripts = call_data.get("transcripts", [])
                
                sampling_info.update({
                    "original_calls": len(calls),
                    "sampled_calls": len(calls),
                    "transcripts_available": len(transcripts),
                    "transcripts_sampled": len(transcripts),
                    "no_transcripts_available": len(calls) - len(transcripts),
                    "no_transcripts_sampled": len(calls) - len(transcripts),
                    "transcript_percentage": 100.0,
                    "no_transcript_percentage": 100.0
                })
                
                return call_data, sampling_info
            
            # Data is too large - apply intelligent sampling
            logger.warning(f"⚠️ Data size: {estimated_tokens:,} tokens > {available_tokens:,} available → Applying intelligent sampling")
            
            sampled_data, sampling_details = self._perform_stratified_sampling(
                call_data, 
                available_tokens, 
                target_ratio_transcript
            )
            
            sampling_info.update(sampling_details)
            sampling_info["sampling_applied"] = True
            sampling_info["reason"] = "context_limit_exceeded"
            
            logger.info(f"✅ Sampling complete: {sampling_info['original_calls']} → {sampling_info['sampled_calls']} calls ({sampling_info['sampling_percentage']:.1f}%)")
            
            return sampled_data, sampling_info
            
        except Exception as e:
            logger.error(f"Error in intelligent sampling: {e}")
            logger.error(traceback.format_exc())
            
            # Return original data with error info
            calls = call_data.get("calls", [])
            sampling_info.update({
                "sampling_applied": False,
                "original_calls": len(calls),
                "sampled_calls": len(calls),
                "reason": f"sampling_error: {str(e)}"
            })
            
            return call_data, sampling_info
    
    def _estimate_data_tokens(self, call_data: Dict[str, Any]) -> int:
        """
        Estimate the number of tokens in the call data JSON.
        
        Args:
            call_data (Dict[str, Any]): Call center data
            
        Returns:
            int: Estimated token count
        """
        try:
            # Convert to JSON string and estimate tokens
            json_str = json.dumps(call_data, indent=2)
            
            # Rough estimation: 4 characters per token (industry standard)
            estimated_tokens = len(json_str) // 4
            
            logger.debug(f"Data size estimation: {len(json_str):,} chars ≈ {estimated_tokens:,} tokens")
            
            return estimated_tokens
            
        except Exception as e:
            logger.error(f"Error estimating data tokens: {e}")
            # Return conservative high estimate
            return 200000
    
    def _get_available_context_tokens(self) -> int:
        """
        Get available context tokens for the current model.
        
        Returns:
            int: Available tokens for data
        """
        try:
            # Get model's total context limit
            total_limit = MODEL_CONTEXT_LIMITS.get(self.model, 128000)  # Default to 128K
            
            # Subtract reserved tokens for prompts and response
            available_tokens = total_limit - RESERVED_TOKENS
            
            logger.debug(f"Model {self.model}: {total_limit:,} total - {RESERVED_TOKENS:,} reserved = {available_tokens:,} available")
            
            return max(available_tokens, 10000)  # Minimum 10K tokens
            
        except Exception as e:
            logger.error(f"Error calculating available tokens: {e}")
            return 50000  # Conservative fallback
    
    def _perform_stratified_sampling(
        self, 
        call_data: Dict[str, Any], 
        target_tokens: int, 
        transcript_ratio: float
    ) -> tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Perform stratified sampling maintaining transcript ratio.
        
        Args:
            call_data (Dict[str, Any]): Original call data
            target_tokens (int): Target token count
            transcript_ratio (float): Desired ratio of calls with transcripts
            
        Returns:
            tuple: (sampled_data, sampling_details)
        """
        try:
            calls = call_data.get("calls", [])
            transcripts = call_data.get("transcripts", [])
            
            if not calls:
                return call_data, {"error": "No calls found in data"}
            
            # Create sets for efficient lookup
            transcript_call_ids = {t.get("call_id") or t.get("uniqueid") for t in transcripts if t.get("call_id") or t.get("uniqueid")}
            
            # Separate calls by transcript availability
            calls_with_transcripts = []
            calls_without_transcripts = []
            
            for call in calls:
                call_id = call.get("uniqueid") or call.get("call_id")
                if call_id in transcript_call_ids:
                    calls_with_transcripts.append(call)
                else:
                    calls_without_transcripts.append(call)
            
            logger.info(f"📊 Original data: {len(calls)} calls ({len(calls_with_transcripts)} with transcripts, {len(calls_without_transcripts)} without)")
            
            # Calculate target sample size iteratively
            sample_size = self._calculate_optimal_sample_size(calls, transcripts, target_tokens)
            
            # Calculate target counts maintaining ratio
            target_with_transcripts = int(sample_size * transcript_ratio)
            target_without_transcripts = sample_size - target_with_transcripts
            
            # Adjust if we don't have enough of either type
            available_with = len(calls_with_transcripts)
            available_without = len(calls_without_transcripts)
            
            if target_with_transcripts > available_with:
                target_with_transcripts = available_with
                target_without_transcripts = min(sample_size - target_with_transcripts, available_without)
            
            if target_without_transcripts > available_without:
                target_without_transcripts = available_without
                target_with_transcripts = min(sample_size - target_without_transcripts, available_with)
            
            # Perform random sampling
            sampled_with_transcripts = random.sample(calls_with_transcripts, target_with_transcripts) if target_with_transcripts > 0 else []
            sampled_without_transcripts = random.sample(calls_without_transcripts, target_without_transcripts) if target_without_transcripts > 0 else []
            
            # Combine sampled calls
            sampled_calls = sampled_with_transcripts + sampled_without_transcripts
            
            # Get corresponding transcripts for sampled calls
            sampled_call_ids = {call.get("uniqueid") or call.get("call_id") for call in sampled_calls}
            sampled_transcripts = [t for t in transcripts if (t.get("call_id") or t.get("uniqueid")) in sampled_call_ids]
            
            # Create sampled dataset
            sampled_data = call_data.copy()
            sampled_data["calls"] = sampled_calls
            sampled_data["transcripts"] = sampled_transcripts
            
            # Update summary
            if "summary" in sampled_data:
                original_summary = sampled_data["summary"].copy()
                sampled_data["summary"]["total_calls"] = len(sampled_calls)
                sampled_data["summary"]["calls_with_recordings"] = len(sampled_calls)  # Assuming all sampled calls have recordings
                
                # Scale other metrics proportionally
                scale_factor = len(sampled_calls) / len(calls) if len(calls) > 0 else 1
                for key in ["total_talk_time", "average_call_duration"]:
                    if key in original_summary:
                        if key == "average_call_duration":
                            # Keep average as-is (don't scale averages)
                            continue
                        else:
                            sampled_data["summary"][key] = int(original_summary[key] * scale_factor)
            
            # Calculate sampling details
            actual_sample_size = len(sampled_calls)
            sampling_percentage = (actual_sample_size / len(calls)) * 100 if len(calls) > 0 else 0
            
            sampling_details = {
                "original_calls": len(calls),
                "sampled_calls": actual_sample_size,
                "sampling_percentage": sampling_percentage,
                "transcripts_available": len(calls_with_transcripts),
                "transcripts_sampled": len(sampled_with_transcripts),
                "no_transcripts_available": len(calls_without_transcripts),
                "no_transcripts_sampled": len(sampled_without_transcripts),
                "transcript_percentage": (len(sampled_with_transcripts) / len(calls_with_transcripts)) * 100 if len(calls_with_transcripts) > 0 else 0,
                "no_transcript_percentage": (len(sampled_without_transcripts) / len(calls_without_transcripts)) * 100 if len(calls_without_transcripts) > 0 else 0,
                "target_tokens": target_tokens,
                "sample_method": "stratified_random"
            }
            
            logger.info(f"🎯 Stratified sampling: {transcript_ratio*100:.0f}% transcript ratio maintained")
            logger.info(f"📈 With transcripts: {len(sampled_with_transcripts)}/{len(calls_with_transcripts)} ({sampling_details['transcript_percentage']:.1f}%)")
            logger.info(f"📉 Without transcripts: {len(sampled_without_transcripts)}/{len(calls_without_transcripts)} ({sampling_details['no_transcript_percentage']:.1f}%)")
            
            return sampled_data, sampling_details
            
        except Exception as e:
            logger.error(f"Error in stratified sampling: {e}")
            logger.error(traceback.format_exc())
            
            return call_data, {"error": f"Sampling failed: {str(e)}"}
    
    def _calculate_optimal_sample_size(
        self, 
        calls: list, 
        transcripts: list, 
        target_tokens: int
    ) -> int:
        """
        Calculate optimal sample size that fits within token limits.
        
        Args:
            calls (list): List of call records
            transcripts (list): List of transcript records
            target_tokens (int): Target token count
            
        Returns:
            int: Optimal sample size
        """
        try:
            total_calls = len(calls)
            
            if total_calls == 0:
                return 0
            
            # Start with a reasonable minimum sample size
            min_sample = min(50, total_calls)  # At least 50 calls or all available
            max_sample = total_calls
            
            # Binary search for optimal sample size
            optimal_size = min_sample
            
            for sample_size in range(min_sample, max_sample + 1, max(1, (max_sample - min_sample) // 20)):
                # Estimate tokens for this sample size
                estimated_tokens = self._estimate_sample_tokens(calls, transcripts, sample_size)
                
                if estimated_tokens <= target_tokens:
                    optimal_size = sample_size
                else:
                    break  # Found the limit
            
            logger.debug(f"Optimal sample size: {optimal_size}/{total_calls} calls for {target_tokens:,} token limit")
            
            return optimal_size
            
        except Exception as e:
            logger.error(f"Error calculating optimal sample size: {e}")
            return min(50, len(calls))  # Conservative fallback
    
    def _estimate_sample_tokens(self, calls: list, transcripts: list, sample_size: int) -> int:
        """
        Estimate tokens for a given sample size.
        
        Args:
            calls (list): All call records
            transcripts (list): All transcript records
            sample_size (int): Proposed sample size
            
        Returns:
            int: Estimated token count
        """
        try:
            if not calls or sample_size <= 0:
                return 0
            
            # Calculate average sizes
            avg_call_size = sum(len(json.dumps(call)) for call in calls[:min(10, len(calls))]) // min(10, len(calls))
            avg_transcript_size = sum(len(json.dumps(t)) for t in transcripts[:min(10, len(transcripts))]) // max(1, min(10, len(transcripts)))
            
            # Estimate total size for sample
            calls_size = avg_call_size * sample_size
            transcripts_size = avg_transcript_size * min(sample_size, len(transcripts))  # Assume some calls have transcripts
            
            # Add overhead for JSON structure
            overhead = 1000  # Basic JSON structure overhead
            
            total_chars = calls_size + transcripts_size + overhead
            estimated_tokens = total_chars // 4  # 4 chars per token
            
            return estimated_tokens
            
        except Exception as e:
            logger.error(f"Error estimating sample tokens: {e}")
            return sample_size * 100  # Conservative estimate: 100 tokens per call
    
    def _analyze_temporal_patterns(self, call_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze temporal patterns in call data to detect multiple months/years and provide period-specific insights.
        
        Args:
            call_data (Dict[str, Any]): Call center data with time information
            
        Returns:
            Dict[str, Any]: Temporal analysis with period-specific insights
        """
        try:
            calls = call_data.get("calls", [])
            time_period = call_data.get("time_period", {})
            
            # Parse start and end dates
            start_date_str = time_period.get("start_date", "")
            end_date_str = time_period.get("end_date", "")
            
            if not start_date_str or not end_date_str:
                return {"temporal_analysis_available": False, "reason": "Missing time period information"}
            
            # Parse dates
            try:
                start_date = datetime.strptime(start_date_str, "%Y-%m-%d")
                end_date = datetime.strptime(end_date_str, "%Y-%m-%d")
            except ValueError:
                return {"temporal_analysis_available": False, "reason": "Invalid date format"}
            
            # Calculate time span details
            total_days = (end_date - start_date).days + 1
            
            # Detect multiple periods
            temporal_periods = self._detect_temporal_periods(start_date, end_date)
            
            # Analyze call distribution by periods
            period_analysis = self._analyze_calls_by_periods(calls, temporal_periods)
            
            # Generate period-specific insights
            period_insights = self._generate_period_insights(period_analysis, temporal_periods)
            
            # Detect seasonal patterns
            seasonal_patterns = self._detect_seasonal_patterns(calls, temporal_periods)
            
            # Generate temporal summary
            temporal_summary = {
                "analysis_period": {
                    "start_date": start_date_str,
                    "end_date": end_date_str,
                    "total_days": total_days,
                    "spans_multiple_months": len(temporal_periods["months"]) > 1,
                    "spans_multiple_years": len(temporal_periods["years"]) > 1,
                    "months_covered": temporal_periods["months"],
                    "years_covered": temporal_periods["years"],
                    "quarters_covered": temporal_periods["quarters"]
                },
                "period_breakdown": period_analysis,
                "period_specific_insights": period_insights,
                "seasonal_patterns": seasonal_patterns,
                "temporal_trends": self._analyze_temporal_trends(period_analysis),
                "temporal_analysis_available": True
            }
            
            logger.info(f"🕐 Temporal analysis complete: {len(temporal_periods['months'])} months, {len(temporal_periods['years'])} years")
            
            return temporal_summary
            
        except Exception as e:
            logger.error(f"Error in temporal analysis: {e}")
            logger.error(traceback.format_exc())
            return {"temporal_analysis_available": False, "error": str(e)}
    
    def _detect_temporal_periods(self, start_date: datetime, end_date: datetime) -> Dict[str, List]:
        """
        Detect all temporal periods (months, years, quarters) covered by the date range.
        
        Args:
            start_date (datetime): Start date
            end_date (datetime): End date
            
        Returns:
            Dict[str, List]: Lists of months, years, and quarters covered
        """
        periods = {
            "months": [],
            "years": [],
            "quarters": [],
            "month_year_combinations": []
        }
        
        current_date = start_date.replace(day=1)  # Start from first day of start month
        
        while current_date <= end_date:
            # Month and year
            month_name = current_date.strftime("%B")
            year = current_date.year
            month_year = f"{month_name} {year}"
            
            if month_year not in periods["month_year_combinations"]:
                periods["month_year_combinations"].append(month_year)
            
            if month_name not in periods["months"]:
                periods["months"].append(month_name)
            
            if year not in periods["years"]:
                periods["years"].append(year)
            
            # Quarter
            quarter = f"Q{((current_date.month - 1) // 3) + 1} {year}"
            if quarter not in periods["quarters"]:
                periods["quarters"].append(quarter)
            
            # Move to next month
            if current_date.month == 12:
                current_date = current_date.replace(year=current_date.year + 1, month=1)
            else:
                current_date = current_date.replace(month=current_date.month + 1)
        
        return periods
    
    def _analyze_calls_by_periods(self, calls: List[Dict], temporal_periods: Dict[str, List]) -> Dict[str, Any]:
        """
        Analyze call distribution across different temporal periods.
        
        Args:
            calls (List[Dict]): List of call records
            temporal_periods (Dict[str, List]): Detected temporal periods
            
        Returns:
            Dict[str, Any]: Analysis by periods
        """
        period_data = {
            "monthly_breakdown": defaultdict(lambda: {"calls": 0, "duration": 0, "answered": 0}),
            "yearly_breakdown": defaultdict(lambda: {"calls": 0, "duration": 0, "answered": 0}),
            "quarterly_breakdown": defaultdict(lambda: {"calls": 0, "duration": 0, "answered": 0}),
            "daily_patterns_by_month": defaultdict(lambda: defaultdict(int)),
            "hourly_patterns_by_month": defaultdict(lambda: defaultdict(int))
        }
        
        for call in calls:
            try:
                # Extract call date
                call_date_str = call.get("calldate")
                if not call_date_str:
                    continue
                
                # Parse call date
                if isinstance(call_date_str, str):
                    # Handle various date formats
                    try:
                        call_date = datetime.fromisoformat(call_date_str.replace('Z', '+00:00'))
                    except:
                        try:
                            call_date = datetime.strptime(call_date_str[:19], "%Y-%m-%d %H:%M:%S")
                        except:
                            continue
                else:
                    call_date = call_date_str
                
                # Extract metrics
                duration = call.get("duration", 0)
                if isinstance(duration, str):
                    try:
                        duration = float(duration)
                    except:
                        duration = 0
                
                disposition = call.get("disposition", "")
                is_answered = disposition == "ANSWERED"
                
                # Monthly breakdown
                month_key = call_date.strftime("%B %Y")
                period_data["monthly_breakdown"][month_key]["calls"] += 1
                period_data["monthly_breakdown"][month_key]["duration"] += duration
                if is_answered:
                    period_data["monthly_breakdown"][month_key]["answered"] += 1
                
                # Yearly breakdown
                year_key = str(call_date.year)
                period_data["yearly_breakdown"][year_key]["calls"] += 1
                period_data["yearly_breakdown"][year_key]["duration"] += duration
                if is_answered:
                    period_data["yearly_breakdown"][year_key]["answered"] += 1
                
                # Quarterly breakdown
                quarter = f"Q{((call_date.month - 1) // 3) + 1} {call_date.year}"
                period_data["quarterly_breakdown"][quarter]["calls"] += 1
                period_data["quarterly_breakdown"][quarter]["duration"] += duration
                if is_answered:
                    period_data["quarterly_breakdown"][quarter]["answered"] += 1
                
                # Daily patterns by month
                month_key = call_date.strftime("%B %Y")
                day_of_week = call_date.strftime("%A")
                period_data["daily_patterns_by_month"][month_key][day_of_week] += 1
                
                # Hourly patterns by month
                hour = call_date.hour
                period_data["hourly_patterns_by_month"][month_key][hour] += 1
                
            except Exception as e:
                logger.debug(f"Error processing call date: {e}")
                continue
        
        # Convert defaultdicts to regular dicts for JSON serialization
        return {
            "monthly_breakdown": dict(period_data["monthly_breakdown"]),
            "yearly_breakdown": dict(period_data["yearly_breakdown"]),
            "quarterly_breakdown": dict(period_data["quarterly_breakdown"]),
            "daily_patterns_by_month": {k: dict(v) for k, v in period_data["daily_patterns_by_month"].items()},
            "hourly_patterns_by_month": {k: dict(v) for k, v in period_data["hourly_patterns_by_month"].items()}
        }
    
    def _generate_period_insights(self, period_analysis: Dict[str, Any], temporal_periods: Dict[str, List]) -> Dict[str, Any]:
        """
        Generate specific insights for each time period.
        
        Args:
            period_analysis (Dict[str, Any]): Analysis data by periods
            temporal_periods (Dict[str, List]): Detected temporal periods
            
        Returns:
            Dict[str, Any]: Period-specific insights
        """
        insights = {
            "monthly_insights": [],
            "yearly_insights": [],
            "quarterly_insights": [],
            "comparative_analysis": {},
            "period_performance_ranking": {}
        }
        
        # Monthly insights
        monthly_data = period_analysis.get("monthly_breakdown", {})
        for month, data in monthly_data.items():
            answer_rate = (data["answered"] / data["calls"] * 100) if data["calls"] > 0 else 0
            avg_duration = data["duration"] / data["calls"] if data["calls"] > 0 else 0
            
            insights["monthly_insights"].append({
                "period": month,
                "call_volume": data["calls"],
                "answer_rate": round(answer_rate, 1),
                "avg_duration": round(avg_duration, 1),
                "total_duration": data["duration"],
                "performance_indicator": "High" if answer_rate > 80 else "Medium" if answer_rate > 60 else "Low"
            })
        
        # Yearly insights
        yearly_data = period_analysis.get("yearly_breakdown", {})
        for year, data in yearly_data.items():
            answer_rate = (data["answered"] / data["calls"] * 100) if data["calls"] > 0 else 0
            avg_duration = data["duration"] / data["calls"] if data["calls"] > 0 else 0
            
            insights["yearly_insights"].append({
                "period": year,
                "call_volume": data["calls"],
                "answer_rate": round(answer_rate, 1),
                "avg_duration": round(avg_duration, 1),
                "total_duration": data["duration"],
                "performance_indicator": "High" if answer_rate > 80 else "Medium" if answer_rate > 60 else "Low"
            })
        
        # Quarterly insights
        quarterly_data = period_analysis.get("quarterly_breakdown", {})
        for quarter, data in quarterly_data.items():
            answer_rate = (data["answered"] / data["calls"] * 100) if data["calls"] > 0 else 0
            avg_duration = data["duration"] / data["calls"] if data["calls"] > 0 else 0
            
            insights["quarterly_insights"].append({
                "period": quarter,
                "call_volume": data["calls"],
                "answer_rate": round(answer_rate, 1),
                "avg_duration": round(avg_duration, 1),
                "total_duration": data["duration"],
                "performance_indicator": "High" if answer_rate > 80 else "Medium" if answer_rate > 60 else "Low"
            })
        
        # Comparative analysis
        if len(monthly_data) > 1:
            # Find best and worst performing months
            monthly_performance = [(month, data["answered"] / data["calls"] * 100 if data["calls"] > 0 else 0) 
                                 for month, data in monthly_data.items()]
            monthly_performance.sort(key=lambda x: x[1], reverse=True)
            
            insights["comparative_analysis"]["best_month"] = monthly_performance[0]
            insights["comparative_analysis"]["worst_month"] = monthly_performance[-1]
            
            # Calculate growth/decline
            monthly_volumes = [(month, data["calls"]) for month, data in monthly_data.items()]
            if len(monthly_volumes) >= 2:
                first_month_volume = monthly_volumes[0][1]
                last_month_volume = monthly_volumes[-1][1]
                volume_change = ((last_month_volume - first_month_volume) / first_month_volume * 100) if first_month_volume > 0 else 0
                insights["comparative_analysis"]["volume_trend"] = {
                    "change_percentage": round(volume_change, 1),
                    "direction": "Increasing" if volume_change > 5 else "Decreasing" if volume_change < -5 else "Stable",
                    "first_period": monthly_volumes[0],
                    "last_period": monthly_volumes[-1]
                }
        
        return insights
    
    def _detect_seasonal_patterns(self, calls: List[Dict], temporal_periods: Dict[str, List]) -> Dict[str, Any]:
        """
        Detect seasonal patterns in call data.
        
        Args:
            calls (List[Dict]): List of call records
            temporal_periods (Dict[str, List]): Detected temporal periods
            
        Returns:
            Dict[str, Any]: Seasonal pattern analysis
        """
        seasonal_data = {
            "monthly_patterns": defaultdict(list),
            "day_of_week_patterns": defaultdict(int),
            "hourly_patterns": defaultdict(int),
            "seasonal_trends": {}
        }
        
        for call in calls:
            try:
                call_date_str = call.get("calldate")
                if not call_date_str:
                    continue
                
                # Parse call date
                if isinstance(call_date_str, str):
                    try:
                        call_date = datetime.fromisoformat(call_date_str.replace('Z', '+00:00'))
                    except:
                        try:
                            call_date = datetime.strptime(call_date_str[:19], "%Y-%m-%d %H:%M:%S")
                        except:
                            continue
                else:
                    call_date = call_date_str
                
                # Collect patterns
                month_name = call_date.strftime("%B")
                seasonal_data["monthly_patterns"][month_name].append(call)
                
                day_of_week = call_date.strftime("%A")
                seasonal_data["day_of_week_patterns"][day_of_week] += 1
                
                hour = call_date.hour
                seasonal_data["hourly_patterns"][hour] += 1
                
            except Exception as e:
                logger.debug(f"Error in seasonal pattern analysis: {e}")
                continue
        
        # Analyze patterns
        seasonal_data["seasonal_trends"] = {
            "busiest_month": max(seasonal_data["monthly_patterns"].items(), 
                               key=lambda x: len(x[1])) if seasonal_data["monthly_patterns"] else None,
            "busiest_day": max(seasonal_data["day_of_week_patterns"].items(), 
                             key=lambda x: x[1]) if seasonal_data["day_of_week_patterns"] else None,
            "peak_hour": max(seasonal_data["hourly_patterns"].items(), 
                           key=lambda x: x[1]) if seasonal_data["hourly_patterns"] else None
        }
        
        return {
            "monthly_call_counts": {k: len(v) for k, v in seasonal_data["monthly_patterns"].items()},
            "day_of_week_distribution": dict(seasonal_data["day_of_week_patterns"]),
            "hourly_distribution": dict(seasonal_data["hourly_patterns"]),
            "seasonal_trends": seasonal_data["seasonal_trends"]
        }
    
    def _analyze_temporal_trends(self, period_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze trends across temporal periods.
        
        Args:
            period_analysis (Dict[str, Any]): Period analysis data
            
        Returns:
            Dict[str, Any]: Trend analysis
        """
        trends = {
            "monthly_trends": [],
            "yearly_trends": [],
            "overall_trend": "stable"
        }
        
        # Analyze monthly trends
        monthly_data = period_analysis.get("monthly_breakdown", {})
        if len(monthly_data) > 1:
            monthly_items = list(monthly_data.items())
            for i in range(1, len(monthly_items)):
                prev_month, prev_data = monthly_items[i-1]
                curr_month, curr_data = monthly_items[i]
                
                volume_change = ((curr_data["calls"] - prev_data["calls"]) / prev_data["calls"] * 100) if prev_data["calls"] > 0 else 0
                
                trends["monthly_trends"].append({
                    "from_period": prev_month,
                    "to_period": curr_month,
                    "volume_change_percent": round(volume_change, 1),
                    "direction": "increasing" if volume_change > 5 else "decreasing" if volume_change < -5 else "stable"
                })
        
        # Determine overall trend
        if trends["monthly_trends"]:
            increasing_trends = sum(1 for t in trends["monthly_trends"] if t["direction"] == "increasing")
            decreasing_trends = sum(1 for t in trends["monthly_trends"] if t["direction"] == "decreasing")
            
            if increasing_trends > decreasing_trends:
                trends["overall_trend"] = "increasing"
            elif decreasing_trends > increasing_trends:
                trends["overall_trend"] = "decreasing"
            else:
                trends["overall_trend"] = "stable"
        
        return trends
    
    def _convert_visualizations_to_structured_text(self, analysis_json: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert visualization data to structured text format instead of chart specifications.
        This addresses issues with problematic visualizations by providing clear text summaries.
        
        Args:
            analysis_json (Dict[str, Any]): Analysis with visualization data
            
        Returns:
            Dict[str, Any]: Analysis with visualizations converted to structured text
        """
        try:
            # Create a copy to avoid modifying the original
            text_analysis = analysis_json.copy()
            
            # Convert visualization dashboard to structured text
            if "visualization_dashboard" in text_analysis:
                viz_dashboard = text_analysis["visualization_dashboard"]
                
                # Convert executive dashboard
                if "executive_dashboard" in viz_dashboard:
                    exec_dash = viz_dashboard["executive_dashboard"]
                    text_analysis["executive_summary_text"] = self._format_executive_dashboard_text(exec_dash)
                
                # Convert operational dashboard
                if "operational_dashboard" in viz_dashboard:
                    ops_dash = viz_dashboard["operational_dashboard"]
                    text_analysis["operational_summary_text"] = self._format_operational_dashboard_text(ops_dash)
                
                # Remove the original visualization dashboard
                del text_analysis["visualization_dashboard"]
            
            # Convert section-level visualizations to text
            sections_with_viz = ["customer_experience", "agent_performance", "operational_efficiency", "business_intelligence"]
            
            for section_name in sections_with_viz:
                if section_name in text_analysis and isinstance(text_analysis[section_name], dict):
                    section_data = text_analysis[section_name]
                    
                    if "visualizations" in section_data:
                        # Convert visualizations to structured text
                        viz_text = self._format_visualizations_as_text(section_data["visualizations"], section_name)
                        section_data["insights_summary"] = viz_text
                        
                        # Remove the original visualizations
                        del section_data["visualizations"]
            
            logger.info("Successfully converted visualizations to structured text format")
            return text_analysis
            
        except Exception as e:
            logger.error(f"Error converting visualizations to text: {e}")
            return analysis_json
    
    def _format_executive_dashboard_text(self, exec_dashboard: Dict[str, Any]) -> str:
        """Format executive dashboard data as structured text."""
        try:
            text_parts = []
            text_parts.append("=== EXECUTIVE DASHBOARD SUMMARY ===")
            
            # Key metrics
            if "key_metrics" in exec_dashboard:
                text_parts.append("\n📊 KEY PERFORMANCE INDICATORS:")
                for metric in exec_dashboard["key_metrics"]:
                    if isinstance(metric, dict):
                        name = metric.get("name", "Unknown Metric")
                        value = metric.get("value", "N/A")
                        trend = metric.get("trend", "")
                        text_parts.append(f"   • {name}: {value} {trend}")
            
            # Health scores
            if "health_scores" in exec_dashboard:
                text_parts.append("\n🎯 HEALTH SCORES:")
                for score in exec_dashboard["health_scores"]:
                    if isinstance(score, dict):
                        category = score.get("category", "Unknown")
                        value = score.get("score", "N/A")
                        status = score.get("status", "")
                        text_parts.append(f"   • {category}: {value}/100 ({status})")
            
            # Trends
            if "trends" in exec_dashboard:
                text_parts.append("\n📈 PERFORMANCE TRENDS:")
                for trend in exec_dashboard["trends"]:
                    if isinstance(trend, dict):
                        metric = trend.get("metric", "Unknown")
                        direction = trend.get("direction", "")
                        change = trend.get("change", "")
                        text_parts.append(f"   • {metric}: {direction} {change}")
            
            return "\n".join(text_parts)
            
        except Exception as e:
            logger.error(f"Error formatting executive dashboard text: {e}")
            return "Executive dashboard data formatting error"
    
    def _format_operational_dashboard_text(self, ops_dashboard: Dict[str, Any]) -> str:
        """Format operational dashboard data as structured text."""
        try:
            text_parts = []
            text_parts.append("=== OPERATIONAL DASHBOARD SUMMARY ===")
            
            # Call volume metrics
            if "call_volume" in ops_dashboard:
                text_parts.append("\n📞 CALL VOLUME ANALYSIS:")
                vol_data = ops_dashboard["call_volume"]
                if isinstance(vol_data, dict):
                    total = vol_data.get("total_calls", "N/A")
                    peak_hour = vol_data.get("peak_hour", "N/A")
                    avg_per_hour = vol_data.get("average_per_hour", "N/A")
                    text_parts.append(f"   • Total Calls: {total}")
                    text_parts.append(f"   • Peak Hour: {peak_hour}")
                    text_parts.append(f"   • Average per Hour: {avg_per_hour}")
            
            # Agent performance
            if "agent_metrics" in ops_dashboard:
                text_parts.append("\n👥 AGENT PERFORMANCE:")
                agent_data = ops_dashboard["agent_metrics"]
                if isinstance(agent_data, dict):
                    top_performer = agent_data.get("top_performer", "N/A")
                    avg_handle_time = agent_data.get("average_handle_time", "N/A")
                    resolution_rate = agent_data.get("resolution_rate", "N/A")
                    text_parts.append(f"   • Top Performer: {top_performer}")
                    text_parts.append(f"   • Avg Handle Time: {avg_handle_time}")
                    text_parts.append(f"   • Resolution Rate: {resolution_rate}")
            
            # Customer satisfaction
            if "customer_satisfaction" in ops_dashboard:
                text_parts.append("\n😊 CUSTOMER SATISFACTION:")
                csat_data = ops_dashboard["customer_satisfaction"]
                if isinstance(csat_data, dict):
                    score = csat_data.get("overall_score", "N/A")
                    trend = csat_data.get("trend", "")
                    feedback_count = csat_data.get("feedback_count", "N/A")
                    text_parts.append(f"   • Overall Score: {score}/100")
                    text_parts.append(f"   • Trend: {trend}")
                    text_parts.append(f"   • Feedback Count: {feedback_count}")
            
            return "\n".join(text_parts)
            
        except Exception as e:
            logger.error(f"Error formatting operational dashboard text: {e}")
            return "Operational dashboard data formatting error"
    
    def _format_visualizations_as_text(self, visualizations: List[Dict[str, Any]], section_name: str) -> str:
        """Format visualization data as structured text."""
        try:
            text_parts = []
            text_parts.append(f"=== {section_name.upper().replace('_', ' ')} INSIGHTS ===")
            
            for i, viz in enumerate(visualizations, 1):
                if not isinstance(viz, dict):
                    continue
                    
                title = viz.get("title", f"Insight {i}")
                chart_type = viz.get("type", "analysis")
                
                text_parts.append(f"\n📊 {title} ({chart_type.upper()})")
                
                # Format data points
                if "data" in viz:
                    data = viz["data"]
                    if isinstance(data, list):
                        text_parts.append("   Data Points:")
                        for point in data[:10]:  # Limit to first 10 points
                            if isinstance(point, dict):
                                x_val = point.get("x", point.get("label", ""))
                                y_val = point.get("y", point.get("value", ""))
                                text_parts.append(f"     • {x_val}: {y_val}")
                
                # Format insights
                if "insights" in viz:
                    insights = viz["insights"]
                    if isinstance(insights, list):
                        text_parts.append("   Key Insights:")
                        for insight in insights:
                            text_parts.append(f"     • {insight}")
                    elif isinstance(insights, str):
                        text_parts.append(f"   Insight: {insights}")
                
                # Format summary
                if "summary" in viz:
                    text_parts.append(f"   Summary: {viz['summary']}")
            
            return "\n".join(text_parts)
            
        except Exception as e:
            logger.error(f"Error formatting visualizations as text: {e}")
            return f"{section_name} visualization data formatting error"