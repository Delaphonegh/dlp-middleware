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

load_dotenv()
logger = logging.getLogger(__name__)

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
    
    # Return existing client if already connected
    if client:
        try:
            # Test if the connection is still valid
            await client.admin.command('ping')
            return client
        except Exception as e:
            logger.warning(f"Existing MongoDB connection appears to be invalid: {str(e)}")
            # Continue to reconnection attempt
    
    # Initialize retry variables
    retries = 0
    current_delay = retry_delay
    
    while retries <= max_retries:
        try:
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
            
            client = mongo_client
            return client
            
        except Exception as e:
            retries += 1
            if retries > max_retries:
                logger.error(f"Failed to connect to MongoDB after {max_retries} attempts: {str(e)}")
                return None
            
            logger.warning(f"MongoDB connection attempt {retries}/{max_retries} failed: {str(e)}")
            logger.info(f"Retrying in {current_delay:.2f} seconds...")
            
            # Wait before retrying with exponential backoff
            await asyncio.sleep(current_delay)
            current_delay *= backoff_factor

# Database instance getter with retry
async def get_database():
    """
    Get the database instance with connection retry logic.
    
    Returns:
        AsyncIOMotorDatabase: MongoDB database instance
    """
    global client
    
    if not client:
        client = await get_mongodb_client()
        
        # If connection still fails after all retries, raise an exception
        if not client:
            logger.critical("Could not establish MongoDB connection after retries")
            raise ConnectionError("Failed to connect to MongoDB database")
    
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
            retries = 0
            current_delay = retry_delay
            
            while True:
                try:
                    return await func(*args, **kwargs)
                except HTTPException as http_ex:
                    # Don't retry HTTP exceptions (including rate limits)
                    # These are intentional responses that should be passed to the client
                    logger.info(f"Not retrying HTTP exception ({http_ex.status_code}): {http_ex.detail}")
                    raise
                except Exception as e:
                    retries += 1
                    if retries > max_retries:
                        logger.error(f"Operation failed after {max_retries} retries: {str(e)}")
                        raise
                    
                    logger.warning(f"MongoDB operation failed (attempt {retries}/{max_retries}): {str(e)}")
                    logger.info(f"Retrying in {current_delay:.2f} seconds...")
                    
                    await asyncio.sleep(current_delay)
                    current_delay *= backoff_factor
                    
                    # If this is a connection-related error, try to get a fresh client
                    if "not connected" in str(e).lower() or "connection" in str(e).lower():
                        global client
                        client = None  # Force a new connection on next get_database() call
        
        return wrapper
    return decorator

# Example of using the retry decorator for a database operation
@with_mongodb_retry()
async def find_user_by_id(user_id: str):
    """Example function using retry decorator"""
    db = await get_database()
    return await db.users.find_one({"_id": user_id})

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