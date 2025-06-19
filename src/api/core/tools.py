from langchain_core.messages import HumanMessage
from IPython.display import Image, display
from langgraph.graph import StateGraph, START, END
from langgraph.graph import MessagesState
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.messages import SystemMessage
from langchain_core.tools import tool
from datetime import datetime
import random
from langchain_openai import ChatOpenAI
from pymongo import MongoClient
import json
import calendar
from typing import Dict, List, Any
import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langgraph.checkpoint.memory import MemorySaver
memory_saver = MemorySaver()

# Initialize the graph with persistence


# Initialize Groq LLM


# Load environment variables from .env file
load_dotenv()

# MongoDB connection details from environment variables
MONGODB_HOST = os.getenv("MONGODB_HOST")
MONGODB_PORT = int(os.getenv("MONGODB_PORT", 27017))  # Default to 27017 if not specified
DATABASE_NAME = os.getenv("DATABASE_NAME")
COLLECTION_NAME = os.getenv("COLLECTION_NAME")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Check if required environment variables are set
required_vars = ["MONGODB_HOST", "DATABASE_NAME", "COLLECTION_NAME", "OPENAI_API_KEY", "GROQ_API_KEY"]
missing_vars = [var for var in required_vars if not os.getenv(var)]
if missing_vars:
    raise EnvironmentError(f"Missing required environment variables: {', '.join(missing_vars)}")


# llm = ChatGroq(
#     api_key=os.getenv("GROQ_API_KEY"),
#     model_name="deepseek-r1-distill-llama-70b",  # Options: "llama3-70b-8192", "llama3-8b-8192", "mixtral-8x7b-32768"
#     temperature=0.5
# )
llm = ChatOpenAI(model="gpt-4o", api_key = OPENAI_API_KEY)

# Create MongoDB connection tool
@tool
def mongodb_connect():
    """Connect to MongoDB and verify the connection."""
    try:
        client = MongoClient(MONGODB_HOST, MONGODB_PORT)
        db = client[DATABASE_NAME]
        # Test the connection with a simple command
        db.command('ping')
        return "Connected to MongoDB successfully"
    except Exception as e:
        return f"Failed to connect to MongoDB: {str(e)}"

# Tool to detect collection schema
@tool
def detect_collection_keys():
    """Detect and return all field names in the MongoDB collection."""
    try:
        client = MongoClient(MONGODB_HOST, MONGODB_PORT)
        db = client[DATABASE_NAME]
        collection = db[COLLECTION_NAME]
        
        # Get a sample document
        sample = collection.find_one()
        if not sample:
            return "Collection is empty"
        
        # Extract all top-level keys
        keys = list(sample.keys())
        return json.dumps(keys)
    except Exception as e:
        return f"Error detecting collection keys: {str(e)}"

# Tool to get date range information
@tool
def get_date_ranges():
    """Get the earliest and latest call_date in the collection."""
    try:
        client = MongoClient(MONGODB_HOST, MONGODB_PORT)
        db = client[DATABASE_NAME]
        collection = db[COLLECTION_NAME]
        
        # Find earliest date
        earliest = collection.find().sort("call_date", 1).limit(1)
        earliest_record = list(earliest)
        earliest_date = earliest_record[0].get("call_date") if earliest_record else None
        
        # Find latest date
        latest = collection.find().sort("call_date", -1).limit(1)
        latest_record = list(latest)
        latest_date = latest_record[0].get("call_date") if latest_record else None
        
        return json.dumps({"earliest": earliest_date, "latest": latest_date})
    except Exception as e:
        return f"Error getting date ranges: {str(e)}"

# Tool to query by month
@tool
def query_by_month(month, year="2024"):
    """Query MongoDB for data from a specific month and year.
    
    Args:
        month: Month as number (1-12) or name (e.g., "October")
        year: Year as string (default: "2024")
        
    Returns:
        JSON string with query results
    """
    try:
        # Convert month name to number if needed
        if isinstance(month, str) and not month.isdigit():
            try:
                # Try to match month name to number
                month_num = str(list(calendar.month_name).index(month.capitalize())).zfill(2)
            except ValueError:
                # Try abbreviated month names
                try:
                    month_num = str(list(calendar.month_abbr).index(month.capitalize()[:3])).zfill(2)
                except ValueError:
                    # Manual mapping as fallback
                    month_dict = {
                        "january": "01", "jan": "01", "february": "02", "feb": "02",
                        "march": "03", "mar": "03", "april": "04", "apr": "04",
                        "may": "05", "june": "06", "jun": "06", "july": "07", "jul": "07",
                        "august": "08", "aug": "08", "september": "09", "sep": "09",
                        "october": "10", "oct": "10", "november": "11", "nov": "11",
                        "december": "12", "dec": "12"
                    }
                    month_num = month_dict.get(month.lower(), "10")  # Default to October if not found
        else:
            month_num = str(int(month)).zfill(2)  # Ensure two digits
        
        # Format query for call_date field
        query = {"call_date": {"$regex": f"^{year}-{month_num}"}}
        query_string = json.dumps(query)
        
        # Execute query
        client = MongoClient(MONGODB_HOST, MONGODB_PORT)
        db = client[DATABASE_NAME]
        collection = db[COLLECTION_NAME]
        
        results = list(collection.find(query).limit(5))
        
        # Convert ObjectId to string for serialization
        for doc in results:
            if '_id' in doc:
                doc['_id'] = str(doc['_id'])
        
        if not results:
            return f"No data found for {calendar.month_name[int(month_num)]} {year}"
            
        return json.dumps(results, default=str)
    except Exception as e:
        return f"Error querying by month: {str(e)}"

@tool
def extract_conversations(query_string):
    """Extract just the conversation data from query results.
    
    Args:
        query_string: JSON string representing MongoDB query
        
    Returns:
        JSON string with conversation data
    """
    try:
        # Parse the query string to a Python dictionary
        if isinstance(query_string, str):
            query = json.loads(query_string)
        else:
            query = json.loads(json.dumps(query_string))
        
        client = MongoClient(MONGODB_HOST, MONGODB_PORT)
        db = client[DATABASE_NAME]
        collection = db[COLLECTION_NAME]
        
        # Execute the query with projection to include only needed fields
        if isinstance(query, list):
            # For aggregation pipeline, add a projection stage
            query.append({"$project": {"call_date": 1, "call_time": 1, "conversation": 1, "_id": 0}})
            results = list(collection.aggregate(query))
        else:
            # For find query
            results = list(collection.find(query, 
                                      {"call_date": 1, "call_time": 1, "conversation": 1, "_id": 0})
                       .limit(5))
        
        if not results:
            return "No conversations found matching your query"
            
        # Format the results for better readability
        formatted_results = []
        for doc in results:
            if "conversation" in doc:
                formatted_doc = {
                    "call_date": doc.get("call_date", "Unknown"),
                    "call_time": doc.get("call_time", "Unknown"),
                    "conversation": doc.get("conversation", {})
                }
                formatted_results.append(formatted_doc)
        
        return json.dumps(formatted_results, default=str)
    except Exception as e:
        return f"Error extracting conversations: {str(e)}\nAttempted query: {query_string}"
    

@tool
def analyze_calls(conversation_data):
    """
    Analyze call center conversations and generate a detailed evaluation report.
    
    Args:
        conversation_data: JSON string containing conversation data from MongoDB
        
    Returns:
        JSON string with detailed analysis following the evaluation schema
    """
    try:
        # Parse the input data
        if isinstance(conversation_data, str):
            conversations = json.loads(conversation_data)
        else:
            conversations = conversation_data
            
        # If conversations is empty, return early
        if not conversations or len(conversations) == 0:
            return json.dumps({"error": "No conversation data provided for analysis"})
            
        # Extract all dialogues for analysis
        all_dialogues = []
        call_dates = []
        
        for conv in conversations:
            if isinstance(conv, str):
                # Try to parse if it's a string
                try:
                    conv_obj = json.loads(conv)
                    if "conversation" in conv_obj and "dialogue" in conv_obj["conversation"]:
                        all_dialogues.extend(conv_obj["conversation"]["dialogue"])
                        if "call_date" in conv_obj:
                            call_dates.append(conv_obj["call_date"])
                except:
                    pass
            elif isinstance(conv, dict):
                # If it's already a dict
                if "conversation" in conv and isinstance(conv["conversation"], dict) and "dialogue" in conv["conversation"]:
                    all_dialogues.extend(conv["conversation"]["dialogue"])
                    if "call_date" in conv:
                        call_dates.append(conv["call_date"])
        
        # If no valid dialogues were found, return error
        if not all_dialogues:
            return json.dumps({"error": "No valid dialogue data found in the provided conversations"})
            
        # Perform analysis (in production, this would use NLP/ML)
        # For demo, we'll use simplified scoring with some randomization
        
        # Count total calls
        total_calls = len(conversations)
        
        # Calculate date range
        start_date = min(call_dates) if call_dates else "2024-01-01"
        end_date = max(call_dates) if call_dates else "2024-12-31"
        
        # Generate metrics (demo values)
        metrics = {
            "customer_experience": round(random.uniform(6.5, 8.5), 1),
            "agent_performance": round(random.uniform(7.0, 9.0), 1),
            "issue_resolution": round(random.uniform(6.0, 8.0), 1),
            "communication_clarity": round(random.uniform(7.0, 9.0), 1),
            "call_clarity": round(random.uniform(5.5, 7.5), 1),
            "response_time": round(random.uniform(6.5, 8.5), 1),
            "customer_sentiment": round(random.uniform(6.0, 8.0), 1),
            "call_type_effectiveness": round(random.uniform(7.0, 8.0), 1)
        }
        
        # Common issues detection (demo)
        common_issues = [
            {
                "issue": "Connection problems during call",
                "frequency": random.randint(10, 20),
                "severity": random.randint(2, 4),
                "impact": "Customers repeat themselves, leading to frustration"
            },
            {
                "issue": "Account review delays",
                "frequency": random.randint(5, 15),
                "severity": random.randint(2, 3),
                "impact": "Customers unable to redeem gifts or benefits"
            },
            {
                "issue": "Tag number verification issues",
                "frequency": random.randint(8, 18),
                "severity": random.randint(1, 3),
                "impact": "Extended call duration for identity verification"
            }
        ]
        
        # Call types (demo)
        call_types = [
            {
                "type": "Account verification",
                "frequency": random.randint(30, 50),
                "average_sentiment": round(random.uniform(6.0, 8.0), 1),
                "resolution_rate": round(random.uniform(70, 90), 1)
            },
            {
                "type": "Gift redemption",
                "frequency": random.randint(20, 40),
                "average_sentiment": round(random.uniform(5.5, 7.5), 1),
                "resolution_rate": round(random.uniform(60, 80), 1)
            },
            {
                "type": "Technical support",
                "frequency": random.randint(10, 30),
                "average_sentiment": round(random.uniform(5.0, 7.0), 1),
                "resolution_rate": round(random.uniform(50, 70), 1)
            }
        ]
        
        # Sentiment analysis (demo)
        sentiment = {
            "positive_sentiment": round(random.uniform(50, 70), 1),
            "neutral_sentiment": round(random.uniform(20, 30), 1),
            "negative_sentiment": round(random.uniform(10, 20), 1),
            "sentiment_trends": [
                {
                    "timeframe": "Morning",
                    "positive_sentiment": round(random.uniform(60, 80), 1),
                    "neutral_sentiment": round(random.uniform(15, 25), 1),
                    "negative_sentiment": round(random.uniform(5, 15), 1)
                },
                {
                    "timeframe": "Afternoon",
                    "positive_sentiment": round(random.uniform(50, 70), 1),
                    "neutral_sentiment": round(random.uniform(20, 30), 1),
                    "negative_sentiment": round(random.uniform(10, 20), 1)
                },
                {
                    "timeframe": "Evening",
                    "positive_sentiment": round(random.uniform(40, 60), 1),
                    "neutral_sentiment": round(random.uniform(25, 35), 1),
                    "negative_sentiment": round(random.uniform(15, 25), 1)
                }
            ]
        }
        
        # Construct the complete analysis report
        analysis_report = {
            "metrics": metrics,
            "analysis": {
                "common_issues": common_issues,
                "strengths": [
                    "Agents maintain professional tone",
                    "Quick verification of customer information",
                    "Clear explanation of account status"
                ],
                "improvement_areas": [
                    "Call quality and connection stability",
                    "Reduction in call back frequency",
                    "More efficient tag number verification process"
                ],
                "trends": {
                    "overall_direction": "Stable with slight improvement",
                    "key_patterns": [
                        "Connection issues more frequent during peak hours",
                        "Higher customer satisfaction in morning calls",
                        "Account review processes causing consistent delays"
                    ],
                    "notable_changes": [
                        {
                            "change": "Improved agent response clarity",
                            "impact": "Reduced repeat questions from customers",
                            "timeframe": "Last 2 weeks"
                        }
                    ]
                },
                "call_types": call_types,
                "customer_sentiment_analysis": sentiment
            },
            "call_statistics": {
                "total_calls": total_calls,
                "average_duration": round(random.uniform(150, 250), 1),
                "resolution_rate": round(random.uniform(65, 85), 1),
                "common_topics": [
                    {
                        "topic": "Gift redemption",
                        "frequency": random.randint(40, 60),
                        "avg_resolution_time": round(random.uniform(180, 240), 1)
                    },
                    {
                        "topic": "Account status",
                        "frequency": random.randint(30, 50),
                        "avg_resolution_time": round(random.uniform(160, 220), 1)
                    },
                    {
                        "topic": "Tag verification",
                        "frequency": random.randint(20, 40),
                        "avg_resolution_time": round(random.uniform(90, 150), 1)
                    }
                ],
                "peak_times": {
                    "busiest_hours": ["10:00 AM - 11:30 AM", "2:00 PM - 3:30 PM"],
                    "quietest_hours": ["8:00 AM - 9:00 AM", "4:30 PM - 5:30 PM"]
                }
            },
            "metadata": {
                "analysis_timestamp": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%S.000Z"),
                "data_range": {
                    "start_date": start_date,
                    "end_date": end_date,
                    "total_interactions_analyzed": total_calls
                }
            }
        }
        
        # Return the formatted JSON
        return json.dumps(analysis_report, indent=2)
        
    except Exception as e:
        return f"Error analyzing conversations: {str(e)}"


# Tool to query by date range
@tool
def query_by_date_range(start_date, end_date):
    """Query MongoDB for data within a date range.
    
    Args:
        start_date: Start date in format YYYY-MM-DD
        end_date: End date in format YYYY-MM-DD
        
    Returns:
        JSON string with query results
    """
    try:
        # Format query for call_date field
        query = {"call_date": {"$gte": start_date, "$lte": end_date}}
        query_string = json.dumps(query)
        
        # Execute query
        client = MongoClient(MONGODB_HOST, MONGODB_PORT)
        db = client[DATABASE_NAME]
        collection = db[COLLECTION_NAME]
        
        results = list(collection.find(query).limit(5))
        
        # Convert ObjectId to string for serialization
        for doc in results:
            if '_id' in doc:
                doc['_id'] = str(doc['_id'])
        
        if not results:
            return f"No data found between {start_date} and {end_date}"
            
        return json.dumps(results, default=str)
    except Exception as e:
        return f"Error querying by date range: {str(e)}"

# Tool to extract conversations
@tool
def extract_conversations(query_string):
    """Extract just the conversation data from query results.
    
    Args:
        query_string: JSON string representing MongoDB query
        
    Returns:
        JSON string with conversation data
    """
    try:
        # Parse the query string to a Python dictionary
        if isinstance(query_string, str):
            query = json.loads(query_string)
        else:
            query = json.loads(json.dumps(query_string))
        
        client = MongoClient(MONGODB_HOST, MONGODB_PORT)
        db = client[DATABASE_NAME]
        collection = db[COLLECTION_NAME]
        
        # Execute the query with projection to include only needed fields
        if isinstance(query, list):
            # For aggregation pipeline, add a projection stage
            query.append({"$project": {"call_date": 1, "call_time": 1, "conversation": 1, "_id": 0}})
            results = list(collection.aggregate(query))
        else:
            # For find query
            results = list(collection.find(query, 
                                      {"call_date": 1, "call_time": 1, "conversation": 1, "_id": 0})
                       .limit(10))
        
        if not results:
            return "No conversations found matching your query"
            
        # Format the results for better readability
        formatted_results = []
        for doc in results:
            if "conversation" in doc:
                formatted_doc = {
                    "call_date": doc.get("call_date", "Unknown"),
                    "call_time": doc.get("call_time", "Unknown"),
                    "conversation": doc.get("conversation", {})
                }
                formatted_results.append(formatted_doc)
        
        return json.dumps(formatted_results, default=str)
    except Exception as e:
        return f"Error extracting conversations: {str(e)}\nAttempted query: {query_string}"

# Improved execute_mongodb_query tool
@tool
def execute_mongodb_query(query_string):
    """Execute a MongoDB query and return results.
    
    Args:
        query_string: JSON string representing MongoDB query
        
    Returns:
        JSON string with query results
    """
    try:
        # Parse the query string to a Python dictionary
        if isinstance(query_string, str):
            query = json.loads(query_string)
        else:
            # If somehow passed as dict already, stringify and parse to ensure proper format
            query = json.loads(json.dumps(query_string))
        
        client = MongoClient(MONGODB_HOST, MONGODB_PORT)
        db = client[DATABASE_NAME]
        collection = db[COLLECTION_NAME]
        
        # Execute the query
        if isinstance(query, list):
            # This is an aggregation pipeline
            results = list(collection.aggregate(query))
        else:
            # This is a find query
            results = list(collection.find(query).limit(5))
        
        # Check if we have results
        if not results:
            return "[]"  # Return empty array as string
            
        # Convert ObjectId to string for serialization
        for doc in results:
            if '_id' in doc:
                doc['_id'] = str(doc['_id'])
        
        return json.dumps(results, default=str)
    except Exception as e:
        return f"Error executing query: {str(e)}\nAttempted query: {query_string}"

# Updated tools list
tools = [
    mongodb_connect,
    detect_collection_keys,
    execute_mongodb_query,
    get_date_ranges,
    query_by_month,
    query_by_date_range,
    extract_conversations,
    analyze_calls
]
# Set up the LLM with specific instructions
system_message = """You are an expert database analyst and customer service evaluator.
For data analysis operations that return JSON results, return ONLY the raw JSON without adding explanations.
Do not reformat, summarize, or explain JSON outputs - present them exactly as received.
For regular queries, provide helpful, concise analysis. All queries on analysis that contains the words "metrics, analysis, trends", your response should be JSON formatted"""
# Set up the LLM
#llm = ChatOpenAI(model="gpt-4o", api_key=OPENAI_API_KEY)

# Bind tools to LLM
llm_with_tools = llm.bind_tools(tools)

# Define the LLM node with system message
def llm_node(state):
    messages = state["messages"]
    
    # Add system message at the beginning if not already present
    if not messages or not any(isinstance(msg, SystemMessage) for msg in messages):
        messages = [SystemMessage(content=system_message)] + messages
        
    response = llm_with_tools.invoke(messages)
    print(response)
    
    # 🔍 Extract token usage - ADD THIS CODE HERE
    token_usage_from_graph = None
    if hasattr(response, 'response_metadata') and response.response_metadata and 'token_usage' in response.response_metadata:
        # Log token usage with emoji
        token_usage_from_graph = response.response_metadata['token_usage']
        print(f"🔢 Token usage from graph: {token_usage_from_graph}")
        
        # Store token usage in the state so it can be returned to the calling function
        if 'token_usage' not in state:
            state['token_usage'] = token_usage_from_graph
            
        # Log the actual token counts
        total_tokens = token_usage_from_graph.get('total_tokens', 0)
        print(f"💰 TOTAL TOKENS: {total_tokens}")
    
    return {"messages": state["messages"] + [response], "token_usage": state.get('token_usage'), "total token":{total_tokens}}

# Create the graph
builder = StateGraph(MessagesState)

# Add nodes
builder.add_node("llm", llm_node)
builder.add_node("tools", ToolNode(tools))

# Add edges
builder.add_edge(START, "llm")
builder.add_conditional_edges("llm", tools_condition)
builder.add_edge("tools", "llm")

# Compile the graph
graph = builder.compile()



