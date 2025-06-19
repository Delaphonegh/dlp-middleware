import redis
import json
import logging
import os
from datetime import timedelta
import hashlib
from .redis_retry import redis_client, with_redis_retry, with_async_redis_retry
from .json_utils import json_serialize, json_deserialize

# Configure logging
logger = logging.getLogger(__name__)

# Cache expiration times (in seconds)
CACHE_EXPIRY = {
    "short": 300,  # 5 minutes
    "medium": 1800,  # 30 minutes
    "long": 86400,  # 24 hours
}

# Note: We now use the redis_client from redis_retry.py
# so we no longer need to initialize it here

def generate_cache_key(prefix, params):
    """
    Generate a cache key from a prefix and parameters.
    
    Args:
        prefix (str): The prefix for the key (e.g., 'call_records')
        params (dict): Parameters that determine the uniqueness of the data
        
    Returns:
        str: A unique cache key
    """
    # Convert params to a sorted string to ensure consistent key generation
    param_str = json_serialize(params)
    
    # Create a hash of the parameters for a shorter key
    param_hash = hashlib.md5(param_str.encode()).hexdigest()
    
    return f"{prefix}:{param_hash}"

@with_redis_retry()
def get_cached_data(key):
    """
    Get data from Redis cache with retry functionality.
    
    Args:
        key (str): The cache key
        
    Returns:
        dict or None: The cached data or None if not found
    """
    if not redis_client:
        return None
    
    data = redis_client.get(key)
    if data:
        logger.info(f"Cache hit for key: {key}")
        return json_deserialize(data)
    logger.info(f"Cache miss for key: {key}")
    return None

@with_redis_retry()
def set_cached_data(key, data, expiry_type="medium"):
    """
    Store data in Redis cache with retry functionality.
    
    Args:
        key (str): The cache key
        data (dict): The data to cache
        expiry_type (str): The expiration time type ('short', 'medium', 'long')
        
    Returns:
        bool: True if successful, False otherwise
    """
    if not redis_client:
        return False
    
    try:
        # Use custom JSON serializer to handle Decimal objects
        serialized = json_serialize(data)
        expiry = CACHE_EXPIRY.get(expiry_type, CACHE_EXPIRY["medium"])
        redis_client.setex(key, expiry, serialized)
        logger.info(f"Cached data with key: {key}, expiry: {expiry}s")
        return True
    except Exception as e:
        logger.error(f"Error serializing data for caching: {str(e)}")
        return False

@with_redis_retry()
def invalidate_cache(prefix):
    """
    Invalidate all cache entries with the given prefix with retry functionality.
    
    Args:
        prefix (str): The prefix to match
        
    Returns:
        int: Number of keys removed
    """
    if not redis_client:
        return 0
    
    # Find all keys matching the pattern
    pattern = f"{prefix}:*"
    keys = redis_client.keys(pattern)
    
    if not keys:
        return 0
    
    # Delete all matching keys
    count = redis_client.delete(*keys)
    logger.info(f"Invalidated {count} cache entries with prefix: {prefix}")
    return count

@with_redis_retry()
def get_cache_stats():
    """
    Get cache statistics with retry functionality.
    
    Returns:
        dict: Cache statistics
    """
    if not redis_client:
        return {"status": "disconnected"}
    
    info = redis_client.info()
    stats = {
        "status": "connected",
        "used_memory_human": info.get("used_memory_human", "unknown"),
        "total_keys": sum(db.get("keys", 0) for db_name, db in info.items() if db_name.startswith("db")),
        "uptime_days": round(info.get("uptime_in_seconds", 0) / 86400, 2),
        "hit_rate": round(info.get("keyspace_hits", 0) / (info.get("keyspace_hits", 0) + info.get("keyspace_misses", 1)) * 100, 2),
        "connected_clients": info.get("connected_clients", 0)
    }
    return stats 