"""
Redis connection handling with retry functionality for improved resilience.
"""
import redis
import logging
import asyncio
import os
import time
from functools import wraps
from typing import Optional, Any, Callable
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

# Redis connection settings
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))
REDIS_DB = int(os.getenv("REDIS_DB", 0))
REDIS_PASSWORD = os.getenv("REDIS_PASSWORD", None)

# Redis retry settings
REDIS_MAX_RETRIES = int(os.getenv("REDIS_MAX_RETRIES", 3))
REDIS_RETRY_DELAY = float(os.getenv("REDIS_RETRY_DELAY", 0.5))  # seconds
REDIS_RETRY_BACKOFF = float(os.getenv("REDIS_RETRY_BACKOFF", 1.5))  # exponential backoff factor

# Global Redis client
redis_client = None

def get_redis_client(max_retries=REDIS_MAX_RETRIES, retry_delay=REDIS_RETRY_DELAY, 
                     backoff_factor=REDIS_RETRY_BACKOFF) -> Optional[redis.Redis]:
    """
    Get a Redis client with retry logic for connection failures.
    
    Args:
        max_retries (int): Maximum number of connection retries
        retry_delay (float): Initial delay between retries in seconds
        backoff_factor (float): Exponential backoff factor
        
    Returns:
        Optional[redis.Redis]: Redis client or None if connection fails
    """
    global redis_client
    
    # Return existing client if it's already connected
    if redis_client:
        try:
            # Test if the connection is still valid
            redis_client.ping()
            return redis_client
        except Exception as e:
            logger.warning(f"Existing Redis connection appears to be invalid: {str(e)}")
            # Continue to reconnection attempt
    
    # Initialize retry variables
    retries = 0
    current_delay = retry_delay
    
    while retries <= max_retries:
        try:
            # Create a new client with connection pooling
            client = redis.Redis(
                host=REDIS_HOST,
                port=REDIS_PORT,
                db=REDIS_DB,
                password=REDIS_PASSWORD,
                socket_timeout=5.0,
                socket_connect_timeout=5.0,
                decode_responses=True,  # Automatically decode responses to Python strings
                health_check_interval=30  # Check connection health every 30 seconds
            )
            
            # Test connection
            if not client.ping():
                raise ConnectionError("Redis ping failed")
                
            logger.info(f"✅ Successfully connected to Redis at {REDIS_HOST}:{REDIS_PORT}")
            
            redis_client = client
            return client
            
        except Exception as e:
            retries += 1
            if retries > max_retries:
                logger.error(f"Failed to connect to Redis after {max_retries} attempts: {str(e)}")
                return None
            
            logger.warning(f"Redis connection attempt {retries}/{max_retries} failed: {str(e)}")
            logger.info(f"Retrying in {current_delay:.2f} seconds...")
            
            # Wait before retrying with exponential backoff
            time.sleep(current_delay)
            current_delay *= backoff_factor
    
    # If we got here, all connection attempts failed
    return None

def with_redis_retry(max_retries=REDIS_MAX_RETRIES, retry_delay=REDIS_RETRY_DELAY, backoff_factor=REDIS_RETRY_BACKOFF):
    """
    Decorator for retrying Redis operations on failure.
    
    Args:
        max_retries (int): Maximum number of retries
        retry_delay (float): Initial delay in seconds
        backoff_factor (float): Factor to increase delay after each retry
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            retries = 0
            current_delay = retry_delay
            last_error = None
            
            while retries <= max_retries:
                try:
                    # Ensure we have a Redis client
                    global redis_client
                    if not redis_client:
                        redis_client = get_redis_client()
                        if not redis_client:
                            raise ConnectionError("Unable to connect to Redis")
                    
                    # Call the original function
                    return func(*args, **kwargs)
                    
                except (redis.ConnectionError, redis.TimeoutError, ConnectionError) as e:
                    # Only retry connection-related errors
                    retries += 1
                    last_error = e
                    
                    if retries > max_retries:
                        logger.error(f"Redis operation failed after {max_retries} retries: {str(e)}")
                        raise
                    
                    logger.warning(f"Redis operation failed (attempt {retries}/{max_retries}): {str(e)}")
                    logger.info(f"Retrying in {current_delay:.2f} seconds...")
                    
                    # Reset client on connection error to force reconnection
                    redis_client = None
                    
                    # Wait before retrying with exponential backoff
                    time.sleep(current_delay)
                    current_delay *= backoff_factor
                
                except Exception as e:
                    # Don't retry other types of errors
                    logger.error(f"Redis operation failed with non-connection error: {str(e)}")
                    raise
            
            # If we got here, all retry attempts failed
            raise last_error or Exception("Redis operation failed after all retries")
            
        return wrapper
    return decorator

# Async version of the retry decorator for use with FastAPI
def with_async_redis_retry(max_retries=REDIS_MAX_RETRIES, retry_delay=REDIS_RETRY_DELAY, backoff_factor=REDIS_RETRY_BACKOFF):
    """
    Async decorator for retrying Redis operations on failure in async contexts.
    
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
            last_error = None
            
            while retries <= max_retries:
                try:
                    # Ensure we have a Redis client
                    global redis_client
                    if not redis_client:
                        redis_client = get_redis_client()
                        if not redis_client:
                            raise ConnectionError("Unable to connect to Redis")
                    
                    # Call the original function
                    return await func(*args, **kwargs)
                    
                except (redis.ConnectionError, redis.TimeoutError, ConnectionError) as e:
                    # Only retry connection-related errors
                    retries += 1
                    last_error = e
                    
                    if retries > max_retries:
                        logger.error(f"Redis operation failed after {max_retries} retries: {str(e)}")
                        raise
                    
                    logger.warning(f"Redis operation failed (attempt {retries}/{max_retries}): {str(e)}")
                    logger.info(f"Retrying in {current_delay:.2f} seconds...")
                    
                    # Reset client on connection error to force reconnection
                    redis_client = None
                    
                    # Wait before retrying with exponential backoff
                    await asyncio.sleep(current_delay)
                    current_delay *= backoff_factor
                
                except Exception as e:
                    # Don't retry other types of errors
                    logger.error(f"Redis operation failed with non-connection error: {str(e)}")
                    raise
            
            # If we got here, all retry attempts failed
            raise last_error or Exception("Redis operation failed after all retries")
            
        return wrapper
    return decorator

# Initialize Redis on module load
try:
    redis_client = get_redis_client()
    if not redis_client:
        logger.warning("Failed to initialize Redis connection at module load")
except Exception as e:
    logger.error(f"Redis initialization error: {str(e)}") 