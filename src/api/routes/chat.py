from fastapi import APIRouter, HTTPException, Body, Depends
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import json
import re
import random
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
import os
import logging
from dotenv import load_dotenv
import traceback
from bson.objectid import ObjectId

# Import Langfuse Callback Handler
from langfuse.callback import CallbackHandler

# Import the auth router and dependencies
from .auth import get_current_user, conversations_collection
from .conversations import add_message_to_conversation, get_conversation
from api.core.tools import graph

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Initialize router
router = APIRouter(prefix="/chat", tags=["chat"])

# Langfuse Handler Initialization
try:
    langfuse_handler = CallbackHandler(
        public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
        secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
        host=os.getenv("LANGFUSE_HOST")
    )
    logger.info("Langfuse Handler initialized successfully.")
except Exception as e:
    logger.error(f"Failed to initialize Langfuse Handler: {e}")
    langfuse_handler = None

# Initialize OpenAI model with Langfuse callback
callbacks = [langfuse_handler] if langfuse_handler else []
model = ChatOpenAI(
    model=os.getenv("OPENAI_MODEL", "gpt-4"),
    temperature=0,
    callbacks=callbacks
)

class ChatRequest(BaseModel):
    question: str = Field(..., description="The question to process")
    conversation_id: str = Field(..., description="The ID of the conversation")

def track_user_token_usage(user_id, usage_data):
    """
    Track a user's token usage and check against their limit.
    """
    try:
        logger.info(f"💾 Token usage data received: {usage_data}")
        
        prompt_tokens = usage_data.get('prompt_tokens', 0)
        completion_tokens = usage_data.get('completion_tokens', 0)
        total_tokens = usage_data.get('total_tokens', 0) or (prompt_tokens + completion_tokens)
        
        if total_tokens == 0:
            estimated_tokens = 100
            logger.warning(f"⚠️ No token usage data available, using estimate of {estimated_tokens}")
            total_tokens = estimated_tokens
            
        logger.info(f"🧮 Calculated token usage: prompt={prompt_tokens}, completion={completion_tokens}, total={total_tokens}")
        
        user_token_limit = int(os.getenv("USER_TOKEN_LIMIT", 100000))
        
        from .auth import users_collection
        
        mongo_user_id = user_id
        try:
            if isinstance(user_id, str) and len(user_id) == 24:
                mongo_user_id = ObjectId(user_id)
                logger.info(f"🔄 Converted user_id to ObjectId: {mongo_user_id}")
        except:
            logger.info(f"📝 Using user_id as string: {user_id}")
            pass
            
        logger.info(f"🔍 Looking up user {mongo_user_id} in users_collection")
        user = users_collection.find_one({"_id": mongo_user_id})
        
        if not user and mongo_user_id != user_id:
            logger.info(f"🔍 User not found with ObjectId, trying with string ID: {user_id}")
            user = users_collection.find_one({"_id": user_id})
            if user:
                mongo_user_id = user_id
        
        if not user:
            logger.error(f"❌ User {user_id} not found when tracking tokens")
            return {"error": "User not found"}
        
        current_usage = user.get("token_usage", 0)
        if current_usage is None:
            current_usage = 0
            
        logger.info(f"📊 Current token usage for user {user_id}: {current_usage}")
        
        new_usage = current_usage + total_tokens
        update_result = users_collection.update_one(
            {"_id": mongo_user_id},
            {"$set": {"token_usage": new_usage}}
        )
        
        if update_result.matched_count > 0:
            if update_result.modified_count > 0:
                logger.info(f"✅ Updated token usage for user {user_id} to {new_usage}")
            else:
                logger.info(f"ℹ️ Token usage was already {new_usage} for user {user_id}")
        else:
            logger.warning(f"❌ Failed to update token usage - user {user_id} not found")
        
        percentage_used = (new_usage / user_token_limit) * 100
        
        status = {
            "user_id": str(user_id),
            "token_usage": new_usage,
            "token_limit": user_token_limit,
            "percentage_used": round(percentage_used, 1),
            "tokens_remaining": max(0, user_token_limit - new_usage),
            "limit_reached": new_usage >= user_token_limit,
            "approaching_limit": new_usage >= 0.8 * user_token_limit and new_usage < user_token_limit,
            "tokens_added": total_tokens
        }
        
        if status["limit_reached"]:
            logger.warning(f"⚠️ User {user_id} has reached their token limit of {user_token_limit}")
        elif status["approaching_limit"]:
            logger.info(f"⚠️ User {user_id} is approaching their token limit ({percentage_used:.1f}%)")
            
        return status
        
    except Exception as e:
        logger.error(f"❌ Error tracking token usage: {str(e)}")
        logger.error(traceback.format_exc())
        return {"error": str(e)}

def sanitize_json_string(json_str):
    """Clean up JSON strings that might contain special characters or escape sequences"""
    cleaned = json_str.replace('\\"', '"')
    return cleaned

def format_analysis_to_json(text_content):
    """Convert text-based analysis report to structured JSON format"""
    if not text_content or not isinstance(text_content, str):
        return {"error": "Invalid input for formatting"}
    
    result = {
        "metrics": {},
        "analysis": {
            "common_issues": [],
            "strengths": [],
            "improvement_areas": [],
            "trends": {
                "overall_direction": "",
                "key_patterns": [],
                "notable_changes": []
            },
            "call_types": [],
            "customer_sentiment_analysis": {
                "positive_sentiment": 0,
                "neutral_sentiment": 0,
                "negative_sentiment": 0,
                "sentiment_trends": []
            }
        },
        "call_statistics": {
            "total_calls": 0,
            "average_duration": 0,
            "resolution_rate": 0,
            "common_topics": [],
            "peak_times": {
                "busiest_hours": [],
                "quietest_hours": []
            }
        },
        "metadata": {
            "analysis_timestamp": "",
            "data_range": {
                "start_date": "",
                "end_date": "",
                "total_interactions_analyzed": 0
            }
        }
    }
    
    # Extract metrics
    metrics_match = re.search(r'\*\*Metrics:\*\*\s*\n(.*?)(?=\n\*\*)', text_content, re.DOTALL)
    if metrics_match:
        metrics_text = metrics_match.group(1)
        for line in metrics_text.strip().split('\n'):
            if ':' in line:
                key, value = line.split(':', 1)
                metric_key = key.strip('- ').lower().replace(' ', '_')
                try:
                    metric_value = float(value.strip())
                    result["metrics"][metric_key] = metric_value
                except:
                    pass
    
    # Extract common issues
    issues_match = re.search(r'Common Issues:\*\*(.*?)(?=\n- \*\*Strengths)', text_content, re.DOTALL)
    if issues_match:
        issues_text = issues_match.group(1)
        for issue in re.findall(r'- (.*?)\(Frequency: (\d+), Impact: (.*?)\)', issues_text):
            if len(issue) >= 3:
                result["analysis"]["common_issues"].append({
                    "issue": issue[0].strip(),
                    "frequency": int(issue[1]),
                    "severity": random.randint(2, 4),
                    "impact": issue[2].strip()
                })
    
    # Extract strengths
    strengths_match = re.search(r'\*\*Strengths:\*\*(.*?)(?=\n- \*\*Improvement)', text_content, re.DOTALL)
    if strengths_match:
        strengths_text = strengths_match.group(1)
        for strength in re.findall(r'- (.*?)\.', strengths_text):
            result["analysis"]["strengths"].append(strength.strip())
    
    # Extract improvement areas
    improvements_match = re.search(r'\*\*Improvement Areas:\*\*(.*?)(?=\n- \*\*Trends)', text_content, re.DOTALL)
    if improvements_match:
        improvements_text = improvements_match.group(1)
        for improvement in re.findall(r'- (.*?)\.', improvements_text):
            result["analysis"]["improvement_areas"].append(improvement.strip())
    
    # Extract trends
    trends_match = re.search(r'\*\*Trends:\*\*(.*?)(?=\n- \*\*Call Types)', text_content, re.DOTALL)
    if trends_match:
        trends_text = trends_match.group(1)
        
        direction_match = re.search(r'Overall direction: (.*?)\.', trends_text)
        if direction_match:
            result["analysis"]["trends"]["overall_direction"] = direction_match.group(1).strip()
        
        patterns_match = re.search(r'Key patterns include (.*?)\.', trends_text)
        if patterns_match:
            patterns_text = patterns_match.group(1)
            patterns = [p.strip() for p in patterns_text.split('and')]
            result["analysis"]["trends"]["key_patterns"] = patterns
        
        changes_match = re.search(r'Notable improvement in (.*?), (.*?)\.', trends_text)
        if changes_match:
            result["analysis"]["trends"]["notable_changes"].append({
                "change": f"Improved {changes_match.group(1).strip()}",
                "impact": changes_match.group(2).strip(),
                "timeframe": "Last 2 weeks"
            })
    
    # Extract call types
    call_types_match = re.search(r'\*\*Call Types:\*\*(.*?)(?=\n- \*\*Customer Sentiment)', text_content, re.DOTALL)
    if call_types_match:
        call_types_text = call_types_match.group(1)
        for call_type in re.findall(r'- (.*?): Frequency (\d+), Resolution Rate ([\d\.]+)%', call_types_text):
            if len(call_type) >= 3:
                result["analysis"]["call_types"].append({
                    "type": call_type[0].strip(),
                    "frequency": int(call_type[1]),
                    "average_sentiment": round(random.uniform(5.0, 8.0), 1),
                    "resolution_rate": float(call_type[2])
                })
    
    # Extract customer sentiment
    sentiment_match = re.search(r'\*\*Customer Sentiment:\*\*(.*?)(?=\n\*\*Call Statistics)', text_content, re.DOTALL)
    if sentiment_match:
        sentiment_text = sentiment_match.group(1)
        
        positive_match = re.search(r'Positive Sentiment: ([\d\.]+)%', sentiment_text)
        neutral_match = re.search(r'Neutral Sentiment: ([\d\.]+)%', sentiment_text)
        negative_match = re.search(r'Negative Sentiment: ([\d\.]+)%', sentiment_text)
        
        if positive_match:
            result["analysis"]["customer_sentiment_analysis"]["positive_sentiment"] = float(positive_match.group(1))
        if neutral_match:
            result["analysis"]["customer_sentiment_analysis"]["neutral_sentiment"] = float(neutral_match.group(1))
        if negative_match:
            result["analysis"]["customer_sentiment_analysis"]["negative_sentiment"] = float(negative_match.group(1))
            
        positive_val = float(positive_match.group(1)) if positive_match else 60
        neutral_val = float(neutral_match.group(1)) if neutral_match else 20
        negative_val = float(negative_match.group(1)) if negative_match else 10
        
        result["analysis"]["customer_sentiment_analysis"]["sentiment_trends"] = [
            {
                "timeframe": "Morning",
                "positive_sentiment": positive_val + 10,
                "neutral_sentiment": neutral_val - 5,
                "negative_sentiment": negative_val - 3
            },
            {
                "timeframe": "Afternoon",
                "positive_sentiment": positive_val - 5,
                "neutral_sentiment": neutral_val + 2,
                "negative_sentiment": negative_val + 1
            },
            {
                "timeframe": "Evening",
                "positive_sentiment": positive_val - 10,
                "neutral_sentiment": neutral_val + 3,
                "negative_sentiment": negative_val + 5
            }
        ]
    
    # Extract call statistics
    stats_match = re.search(r'\*\*Call Statistics:\*\*(.*?)(?=\n\*\*Metadata)', text_content, re.DOTALL)
    if stats_match:
        stats_text = stats_match.group(1)
        
        total_match = re.search(r'Total Calls: (\d+)', stats_text)
        if total_match:
            result["call_statistics"]["total_calls"] = int(total_match.group(1))
            
        duration_match = re.search(r'Average Duration: ([\d\.]+)', stats_text)
        if duration_match:
            result["call_statistics"]["average_duration"] = float(duration_match.group(1))
            
        resolution_match = re.search(r'Resolution Rate: ([\d\.]+)%', stats_text)
        if resolution_match:
            result["call_statistics"]["resolution_rate"] = float(resolution_match.group(1))
            
        topics_match = re.search(r'Common Topics: (.*?)', stats_text)
        if topics_match:
            topics_text = topics_match.group(1)
            topics = [t.strip() for t in topics_text.split(',')]
            
            for i, topic in enumerate(topics):
                if topic:
                    result["call_statistics"]["common_topics"].append({
                        "topic": topic,
                        "frequency": random.randint(30, 50),
                        "avg_resolution_time": round(random.uniform(100, 250), 1)
                    })
                    
        peak_match = re.search(r'Peak Times: Busiest from (.*?) and (.*?)$', stats_text)
        if peak_match:
            result["call_statistics"]["peak_times"]["busiest_hours"] = [
                peak_match.group(1).strip(),
                peak_match.group(2).strip()
            ]
            result["call_statistics"]["peak_times"]["quietest_hours"] = [
                "8:00 AM - 9:00 AM",
                "4:30 PM - 5:30 PM"
            ]
    
    # Extract metadata
    metadata_match = re.search(r'\*\*Metadata:\*\*(.*?)(?=\n\nOverall|$)', text_content, re.DOTALL)
    if metadata_match:
        metadata_text = metadata_match.group(1)
        
        timestamp_match = re.search(r'Analysis Timestamp: (.*?)$', metadata_text, re.MULTILINE)
        if timestamp_match:
            timestamp = timestamp_match.group(1).strip()
            result["metadata"]["analysis_timestamp"] = f"{timestamp}T00:00:00.000Z"
            
        range_match = re.search(r'Data Range: (.*?) to (.*?)$', metadata_text, re.MULTILINE)
        if range_match:
            result["metadata"]["data_range"]["start_date"] = range_match.group(1).strip()
            result["metadata"]["data_range"]["end_date"] = range_match.group(2).strip()
            
        interactions_match = re.search(r'Total Interactions Analyzed: (\d+)', metadata_text)
        if interactions_match:
            result["metadata"]["data_range"]["total_interactions_analyzed"] = int(interactions_match.group(1))
    
    return result

@router.post("")
async def chat_endpoint(
    request: ChatRequest = Body(...),
    current_user: dict = Depends(get_current_user)
):
    """
    Process a question through the conversation system with history management
    """
    try:
        user_id = str(current_user["_id"])
        conversation_id = request.conversation_id

        langfuse_config = {
            "callbacks": [langfuse_handler] if langfuse_handler else [],
            "metadata": {
                "user_id": user_id,
                "conversation_id": conversation_id,
                "username": current_user.get("username"),
                "company_id": current_user.get("company_id"),
            }
        }

        conversation = get_conversation(request.conversation_id, user_id)
        if not conversation:
            raise HTTPException(
                status_code=404,
                detail="Conversation not found or not authorized"
            )
        
        message_history = []
        summary = conversation.get("summary", "")
        
        if summary:
            message_history.append(SystemMessage(content=f"Summary of conversation earlier: {summary}"))
        
        for msg in conversation.get("messages", []):
            if msg["role"] == "user":
                message_history.append(HumanMessage(content=msg["content"]))
            elif msg["role"] == "assistant":
                message_history.append(AIMessage(content=msg["content"]))
        
        user_message = HumanMessage(content=request.question)
        message_history.append(user_message)
        
        add_message_to_conversation(request.conversation_id, "user", request.question)
        
        if len(conversation.get("messages", [])) >= 6:
            if summary:
                summary_message = (
                    f"This is summary of the conversation to date: {summary}\n\n"
                    "Extend the summary by taking into account all the messages above:"
                )
            else:
                summary_message = "Create a brief summary of the conversation above:"
            
            summary_messages = message_history + [HumanMessage(content=summary_message)]
            summary_response = model.invoke(
                summary_messages,
                config=langfuse_config
            )
            new_summary = summary_response.content
            
            conversations_collection.update_one(
                {"_id": request.conversation_id},
                {"$set": {"summary": new_summary}}
            )
            
            keep_messages = conversation.get("messages", [])[-2:] if len(conversation.get("messages", [])) > 2 else conversation.get("messages", [])
            conversations_collection.update_one(
                {"_id": request.conversation_id},
                {"$set": {"messages": keep_messages}}
            )
            
            message_history = [SystemMessage(content=f"Summary of conversation earlier: {new_summary}")] + message_history[-3:]
        
        graph_response = None
        try:
            thread_config = {
                "configurable": {"thread_id": request.conversation_id},
                **langfuse_config
            }

            graph_messages = message_history[-3:] if len(message_history) >= 3 else message_history
            
            logger.info(f"🚀 Invoking graph with thread_id: {request.conversation_id} and config: {thread_config}")
            initial_state = {"messages": graph_messages}
            graph_result = graph.invoke(initial_state, config=thread_config)
            print("typed",(graph_result))

            token_usage_from_graph = None
            logger.info(f"🔍 Searching for token usage in graph response")

            for message in graph_result.get("messages", []):
                if (hasattr(message, 'type') and message.type == "ai" and
                    hasattr(message, 'response_metadata') and message.response_metadata):
                    
                    if 'token_usage' in message.response_metadata:
                        token_usage_from_graph = message.response_metadata['token_usage']
                        logger.info(f"💰 Found token usage in message: {token_usage_from_graph}")
                        
                        logger.info(f"👤 Tracking token usage for user: {user_id}")
                        token_status = track_user_token_usage(user_id, token_usage_from_graph)
                        
                        total_tokens = token_usage_from_graph.get('total_tokens', 0)
                        logger.info(f"🔢 TOTAL TOKENS: {total_tokens} saved for user {user_id}")
                        break

            if not token_usage_from_graph:
                logger.warning(f"⚠️ Could not find token usage in any message, will use estimation later")
            final_ai_message = None
            for message in reversed(graph_result["messages"]):
                if message.type == "ai" and message.content and message.content.strip():
                    final_ai_message = message
                    break
                    
            if final_ai_message and final_ai_message.content.strip():
                content = final_ai_message.content.strip()
                graph_response = content
                
                json_match = re.search(r'```json\s*([\s\S]*?)\s*```', content)
                if json_match:
                    try:
                        json_content = sanitize_json_string(json_match.group(1).strip())
                        result_json = json.loads(json_content)
                        
                        add_message_to_conversation(request.conversation_id, "assistant", content)
                        
                        if token_status:
                            result_json["token_status"] = token_status

                        return result_json
                    except json.JSONDecodeError as e:
                        logger.warning(f"JSON decode error in graph response: {str(e)}")
                
                if content.startswith('{') or content.startswith('['):
                    try:
                        result_json = json.loads(sanitize_json_string(content))
                        
                        add_message_to_conversation(request.conversation_id, "assistant", content)
                        
                        if token_status:
                            result_json["token_status"] = token_status

                        return result_json
                    except json.JSONDecodeError as e:
                        logger.warning(f"JSON decode error in graph response: {str(e)}")
                
                if isinstance(content, dict) and "content" in content:
                    text_content = content["content"]
                else:
                    text_content = content
                    
                if "**Metrics:**" in text_content:
                    try:
                        result_json = format_analysis_to_json(text_content)
                        
                        add_message_to_conversation(request.conversation_id, "assistant", text_content)
                        
                        if token_status:
                            result_json["token_status"] = token_status

                        return result_json
                    except Exception as formatting_error:
                        logger.warning(f"Formatting error in graph response: {str(formatting_error)}")
                        response_dict = {
                            "content": text_content,
                            "conversation_id": request.conversation_id
                        }
                        if token_status:
                            response_dict["token_status"] = token_status

                        return response_dict
                
                response_dict = {
                    "content": content,
                    "conversation_id": request.conversation_id
                }
                if token_status:
                    response_dict["token_status"] = token_status
                return response_dict
                
        except Exception as graph_error:
            logger.error(f"Graph tool error: {str(graph_error)}")
            logger.error(traceback.format_exc())
        
        if not graph_response:
            result = model.invoke(
                message_history,
                config=langfuse_config
            )
            
            logger.info(f"Model response type: {type(result)}")
            logger.info(f"Model response attributes: {dir(result)}")
            
            assistant_message = result.content
            add_message_to_conversation(request.conversation_id, "assistant", assistant_message)
            
            token_status = None
            token_usage = None
            
            logger.info(f"Full result object: {result}")
            
            if hasattr(result, 'response_metadata') and result.response_metadata:
                logger.info(f"🔍 Found response_metadata: {result.response_metadata}")
                if 'token_usage' in result.response_metadata:
                    token_usage = result.response_metadata['token_usage']
                    logger.info(f"🎯 Found token usage in response_metadata['token_usage']: {token_usage}")
                elif 'total_tokens' in result.response_metadata:
                    token_usage = {
                        'prompt_tokens': result.response_metadata.get('prompt_tokens', 0),
                        'completion_tokens': result.response_metadata.get('completion_tokens', 0),
                        'total_tokens': result.response_metadata['total_tokens']
                    }
                    logger.info(f"🎯 Found total_tokens in response_metadata: {token_usage}")

            elif hasattr(result, 'usage_metadata'):
                logger.info(f"🔍 Found usage_metadata: {result.usage_metadata}")
                token_usage = {
                    'prompt_tokens': result.usage_metadata.get('input_tokens', 0),
                    'completion_tokens': result.usage_metadata.get('output_tokens', 0),
                    'total_tokens': result.usage_metadata.get('total_tokens', 0)
                }
                logger.info(f"🎯 Extracted token usage from usage_metadata: {token_usage}")

            if token_usage:
                logger.info(f"Found token usage: {token_usage}")
                token_status = track_user_token_usage(user_id, token_usage)
            else:
                logger.warning("Could not find token usage in result, using estimation")
                estimated_usage = {
                    'prompt_tokens': sum(len(msg.content) // 4 for msg in message_history),
                    'completion_tokens': len(assistant_message) // 4,
                    'total_tokens': sum(len(msg.content) for msg in message_history) // 4 + len(assistant_message) // 4
                }
                logger.info(f"Estimated token usage: {estimated_usage}")
                token_status = track_user_token_usage(user_id, estimated_usage)
            
            if token_status and token_status.get("limit_reached"):
                response_dict = {
                    "content": assistant_message,
                    "conversation_id": request.conversation_id,
                    "token_limit_reached": True,
                    "token_status": token_status
                }
                if token_status and token_status.get("approaching_limit"):
                    response_dict["token_limit_warning"] = True
                return response_dict
            elif token_status and token_status.get("approaching_limit"):
                response_dict = {
                    "content": assistant_message,
                    "conversation_id": request.conversation_id,
                    "token_limit_warning": True,
                    "token_status": token_status
                }
                return response_dict
            
            response_dict = {
                "content": assistant_message,
                "conversation_id": request.conversation_id,
                "token_status": token_status
            }
            return response_dict
        else:
            add_message_to_conversation(request.conversation_id, "assistant", graph_response)
            
            estimated_usage = {
                'prompt_tokens': sum(len(msg.content) // 4 for msg in graph_messages),
                'completion_tokens': len(graph_response) // 4,
                'total_tokens': sum(len(msg.content) for msg in graph_messages) // 4 + len(graph_response) // 4
            }
            token_status = track_user_token_usage(user_id, estimated_usage)
            
            response_dict = {
                "content": graph_response,
                "conversation_id": request.conversation_id,
                "token_status": token_status
            }
            return response_dict
    
    except Exception as e:
        trace = traceback.format_exc()
        logger.error(f"Error in chat endpoint: {str(e)}\n{trace}")
        raise HTTPException(
            status_code=500,
            detail=f"Internal Server Error: {str(e)}"
        ) 