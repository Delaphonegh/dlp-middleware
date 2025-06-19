from motor.motor_asyncio import AsyncIOMotorClient
from fastapi import Depends, HTTPException
import os
from dotenv import load_dotenv
import logging
import motor.motor_asyncio
import time
import asyncio
from functools import wraps
from typing import Optional

# OpenTelemetry imports
from opentelemetry import trace
from opentelemetry.trace import Status, StatusCode

load_dotenv()
logger = logging.getLogger(__name__)

# Get OpenTelemetry tracer for database operations
tracer = trace.get_tracer(__name__)

# MongoDB connection settings
MONGODB_HOST = os.getenv("MONGODB_HOST", "localhost")
MONGODB_PORT = int(os.getenv("MONGODB_PORT", 37037))
DATABASE_NAME = os.getenv("DATABASE_NAME", "delaphone")

# MongoDB connection retry settings
MAX_RETRIES = int(os.getenv("MONGODB_MAX_RETRIES", 5))
RETRY_DELAY = float(os.getenv("MONGODB_RETRY_DELAY", 1.0))  # seconds
RETRY_BACKOFF = float(os.getenv("MONGODB_RETRY_BACKOFF", 2.0))  # exponential backoff factor

# Create a motor client with a connection pool
client = None

# Construct the MongoDB URL
MONGODB_URL = f"{MONGODB_HOST}:{MONGODB_PORT}"
print(MONGODB_URL)
logger.info(f"Connecting to MongoDB at: {MONGODB_URL}, Database: {DATABASE_NAME}")

# Create MongoDB client with retry logic
@tracer.start_as_current_span("mongodb_client_connection")
async def get_mongodb_client(max_retries=MAX_RETRIES, retry_delay=RETRY_DELAY, backoff_factor=RETRY_BACKOFF) -> Optional[motor.motor_asyncio.AsyncIOMotorClient]:
    """
    Get a MongoDB client with retry logic.
    
    Args:
        max_retries (int): Maximum number of connection retries
        retry_delay (float): Initial delay between retries in seconds
        backoff_factor (float): Exponential backoff factor
        
    Returns:
        Optional[motor.motor_asyncio.AsyncIOMotorClient]: MongoDB client or None if connection fails
    """
    global client
    current_span = trace.get_current_span()
    
    # Add connection attributes to span
    if current_span:
        current_span.set_attribute("db.system", "mongodb")
        current_span.set_attribute("db.connection.string", f"{MONGODB_HOST}:{MONGODB_PORT}")
        current_span.set_attribute("db.name", DATABASE_NAME)
        current_span.set_attribute("db.operation", "connect")
        current_span.set_attribute("db.mongodb.max_retries", max_retries)
    
    # Return existing client if already connected
    if client:
        try:
            # Test if the connection is still valid
            await client.admin.command('ping')
            if current_span:
                current_span.set_attribute("db.connection.reused", True)
                current_span.add_event("connection_reused", {"status": "valid"})
            return client
        except Exception as e:
            logger.warning(f"Existing MongoDB connection appears to be invalid: {str(e)}")
            if current_span:
                current_span.add_event("connection_validation_failed", {"error": str(e)})
            # Continue to reconnection attempt
    
    # Initialize retry variables
    retries = 0
    current_delay = retry_delay
    
    if current_span:
        current_span.set_attribute("db.connection.retry_attempts", 0)
    
    while retries <= max_retries:
        try:
            if current_span:
                current_span.set_attribute("db.connection.retry_attempts", retries)
                current_span.add_event("connection_attempt", {"attempt": retries + 1, "max_retries": max_retries})
            
            # Create a new client with connection pooling options
            mongo_client = motor.motor_asyncio.AsyncIOMotorClient(
                host=MONGODB_HOST, 
                port=MONGODB_PORT,
                serverSelectionTimeoutMS=5000,
                connectTimeoutMS=5000,
                socketTimeoutMS=10000,
                maxPoolSize=50,
                minPoolSize=10,
                maxIdleTimeMS=30000,  # Close idle connections after 30 seconds
                retryWrites=True,     # Enable automatic retry for write operations
                retryReads=True       # Enable automatic retry for read operations
            )
            
            # Test connection
            await mongo_client.admin.command('ping')
            logger.info(f"Successfully connected to MongoDB at {MONGODB_HOST}:{MONGODB_PORT}")
            
            if current_span:
                current_span.set_attribute("db.connection.successful", True)
                current_span.set_attribute("db.connection.pool_size", 50)
                current_span.add_event("connection_established", {
                    "host": MONGODB_HOST, 
                    "port": MONGODB_PORT,
                    "attempts_required": retries + 1
                })
                current_span.set_status(Status(StatusCode.OK))
            
            client = mongo_client
            return client
            
        except Exception as e:
            retries += 1
            if retries > max_retries:
                logger.error(f"Failed to connect to MongoDB after {max_retries} attempts: {str(e)}")
                if current_span:
                    current_span.set_attribute("db.connection.successful", False)
                    current_span.set_attribute("db.connection.final_error", str(e))
                    current_span.add_event("connection_failed_permanently", {
                        "error": str(e),
                        "total_attempts": retries,
                        "max_retries": max_retries
                    })
                    current_span.set_status(Status(StatusCode.ERROR, f"MongoDB connection failed: {str(e)}"))
                return None
            
            logger.warning(f"MongoDB connection attempt {retries}/{max_retries} failed: {str(e)}")
            logger.info(f"Retrying in {current_delay:.2f} seconds...")
            
            if current_span:
                current_span.add_event("connection_retry", {
                    "error": str(e),
                    "attempt": retries,
                    "retry_delay": current_delay,
                    "next_delay": current_delay * backoff_factor
                })
            
            # Wait before retrying with exponential backoff
            await asyncio.sleep(current_delay)
            current_delay *= backoff_factor

# Database instance getter with retry
@tracer.start_as_current_span("get_database")
async def get_database():
    """
    Get the database instance with connection retry logic.
    
    Returns:
        AsyncIOMotorDatabase: MongoDB database instance
    """
    global client
    current_span = trace.get_current_span()
    
    if current_span:
        current_span.set_attribute("db.operation", "get_database")
        current_span.set_attribute("db.name", DATABASE_NAME)
    
    if not client:
        if current_span:
            current_span.add_event("creating_new_client")
        client = await get_mongodb_client()
        
        # If connection still fails after all retries, raise an exception
        if not client:
            logger.critical("Could not establish MongoDB connection after retries")
            if current_span:
                current_span.set_status(Status(StatusCode.ERROR, "Failed to connect to MongoDB database"))
            raise ConnectionError("Failed to connect to MongoDB database")
    
    if current_span:
        current_span.set_attribute("db.connection.available", True)
        current_span.add_event("database_accessed", {"database": DATABASE_NAME})
        current_span.set_status(Status(StatusCode.OK))
    
    return client[DATABASE_NAME]

# Retry decorator for MongoDB operations
def with_mongodb_retry(max_retries=3, retry_delay=0.5, backoff_factor=1.5):
    """
    Decorator for retrying MongoDB operations on failure.
    
    Args:
        max_retries (int): Maximum number of retries
        retry_delay (float): Initial delay in seconds
        backoff_factor (float): Factor to increase delay after each retry
    """
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Create a span for the MongoDB operation
            with tracer.start_as_current_span(f"mongodb_operation_{func.__name__}") as span:
                retries = 0
                current_delay = retry_delay
                
                span.set_attribute("db.operation", func.__name__)
                span.set_attribute("db.mongodb.max_retries", max_retries)
                span.set_attribute("db.mongodb.retry_delay", retry_delay)
                
                while True:
                    try:
                        span.set_attribute("db.mongodb.attempt", retries + 1)
                        result = await func(*args, **kwargs)
                        
                        if retries > 0:
                            span.add_event("operation_succeeded_after_retry", {
                                "attempts_required": retries + 1,
                                "total_delay": sum(retry_delay * (backoff_factor ** i) for i in range(retries))
                            })
                        
                        span.set_status(Status(StatusCode.OK))
                        return result
                        
                    except HTTPException as http_ex:
                        # Don't retry HTTP exceptions (including rate limits)
                        # These are intentional responses that should be passed to the client
                        logger.info(f"Not retrying HTTP exception ({http_ex.status_code}): {http_ex.detail}")
                        span.set_attribute("db.mongodb.http_exception", True)
                        span.set_attribute("db.mongodb.http_status", http_ex.status_code)
                        span.add_event("http_exception_not_retried", {"status_code": http_ex.status_code, "detail": http_ex.detail})
                        span.set_status(Status(StatusCode.ERROR, f"HTTP {http_ex.status_code}: {http_ex.detail}"))
                        raise
                        
                    except Exception as e:
                        retries += 1
                        if retries > max_retries:
                            logger.error(f"Operation failed after {max_retries} retries: {str(e)}")
                            span.set_attribute("db.mongodb.final_failure", True)
                            span.add_event("operation_failed_permanently", {
                                "error": str(e),
                                "total_attempts": retries,
                                "max_retries": max_retries
                            })
                            span.set_status(Status(StatusCode.ERROR, f"MongoDB operation failed after {max_retries} retries: {str(e)}"))
                            raise
                        
                        logger.warning(f"MongoDB operation failed (attempt {retries}/{max_retries}): {str(e)}")
                        logger.info(f"Retrying in {current_delay:.2f} seconds...")
                        
                        span.add_event("operation_retry", {
                            "error": str(e),
                            "attempt": retries,
                            "retry_delay": current_delay,
                            "connection_error": "connection" in str(e).lower()
                        })
                        
                        await asyncio.sleep(current_delay)
                        current_delay *= backoff_factor
                        
                        # If this is a connection-related error, try to get a fresh client
                        if "not connected" in str(e).lower() or "connection" in str(e).lower():
                            global client
                            client = None  # Force a new connection on next get_database() call
                            span.add_event("forcing_new_connection", {"reason": "connection_error"})
        
        return wrapper
    return decorator

# Example of using the retry decorator for a database operation
@with_mongodb_retry()
async def find_user_by_id(user_id: str):
    """Example function using retry decorator with OpenTelemetry tracing"""
    current_span = trace.get_current_span()
    
    # Add operation-specific attributes
    if current_span:
        current_span.set_attribute("db.collection.name", "users")
        current_span.set_attribute("db.operation", "find_one")
        current_span.set_attribute("db.mongodb.user_id", user_id)
    
    db = await get_database()
    
    # Add event before the operation
    if current_span:
        current_span.add_event("executing_find_operation", {"collection": "users", "query": "by_id"})
    
    result = await db.users.find_one({"_id": user_id})
    
    # Add result information to span
    if current_span:
        current_span.set_attribute("db.result.found", result is not None)
        if result:
            current_span.add_event("user_found", {"user_id": user_id})
        else:
            current_span.add_event("user_not_found", {"user_id": user_id})
    
    return result

# Initialize MongoDB connection on module load
async def init_mongodb():
    """Initialize MongoDB connection when the module is loaded"""
    try:
        await get_mongodb_client()
        logger.info("MongoDB connection initialized")
    except Exception as e:
        logger.error(f"Failed to initialize MongoDB connection: {str(e)}")

# Create an event loop to initialize MongoDB on module load
loop = asyncio.get_event_loop()
try:
    loop.run_until_complete(init_mongodb())
except Exception as e:
    logger.error(f"MongoDB initialization error: {str(e)}") 