from fastapi import APIRouter, Depends, status, HTTPException
from typing import Dict, Any
import logging
from datetime import datetime
import asyncio
import os

from api.database import get_database, get_mongodb_client
from api.core.redis_cache import get_cache_stats
from api.core.redis_retry import redis_client, with_redis_retry, with_async_redis_retry

# Set up logging
logger = logging.getLogger(__name__)

# Create router
router = APIRouter(prefix="/health", tags=["health"])

@router.get("", response_model=Dict[str, Any])
async def health_check():
    """
    System health check endpoint.
    
    Returns a comprehensive status report of all system components:
    - Overall system status
    - MongoDB connection status
    - Redis cache status
    - Environment information
    """
    health_report = {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "components": {},
        "environment": {
            "hostname": os.environ.get("HOSTNAME", "unknown"),
            "app_env": os.environ.get("APP_ENV", "development")
        }
    }
    
    # Check MongoDB health
    mongo_status = await check_mongodb_health()
    health_report["components"]["mongodb"] = mongo_status
    
    # Check Redis health
    redis_status = await check_redis_health()
    health_report["components"]["redis"] = redis_status
    
    # Update overall status based on component health
    if mongo_status["status"] == "unhealthy" or redis_status["status"] == "unhealthy":
        health_report["status"] = "degraded"
        
    if mongo_status["status"] == "unhealthy" and redis_status["status"] == "unhealthy":
        health_report["status"] = "unhealthy"
    
    return health_report

@router.get("/mongodb", response_model=Dict[str, Any])
async def mongodb_health():
    """Check MongoDB connection health"""
    return await check_mongodb_health()

@router.get("/redis", response_model=Dict[str, Any])
async def redis_health():
    """Check Redis connection health"""
    return await check_redis_health()

async def check_mongodb_health() -> Dict[str, Any]:
    """Check MongoDB connection status with detailed diagnostics"""
    try:
        # Attempt to get MongoDB client
        client = await get_mongodb_client(max_retries=1)
        
        if not client:
            return {
                "status": "unhealthy",
                "message": "Could not establish MongoDB connection",
                "timestamp": datetime.utcnow().isoformat()
            }
        
        # Test basic operations
        start_time = datetime.utcnow()
        await client.admin.command('ping')
        response_time = (datetime.utcnow() - start_time).total_seconds() * 1000  # in milliseconds
        
        # Get server info
        server_info = await client.admin.command('serverStatus')
        
        # Extract useful metrics
        metrics = {
            "response_time_ms": round(response_time, 2),
            "connections": {
                "current": server_info.get("connections", {}).get("current", 0),
                "available": server_info.get("connections", {}).get("available", 0),
                "totalCreated": server_info.get("connections", {}).get("totalCreated", 0)
            },
            "uptime_seconds": server_info.get("uptime", 0),
            "uptime_days": round(server_info.get("uptime", 0) / 86400, 2),
            "version": server_info.get("version", "unknown")
        }
        
        return {
            "status": "healthy",
            "message": "MongoDB connection is healthy",
            "timestamp": datetime.utcnow().isoformat(),
            "metrics": metrics
        }
    except Exception as e:
        logger.error(f"MongoDB health check failed: {str(e)}")
        return {
            "status": "unhealthy",
            "message": f"MongoDB health check failed: {str(e)}",
            "timestamp": datetime.utcnow().isoformat(),
            "error": str(e)
        }

@with_async_redis_retry(max_retries=1)  # Use async retry with only 1 retry for health checks
async def check_redis_health() -> Dict[str, Any]:
    """Check Redis connection status with detailed diagnostics using retry mechanism"""
    if not redis_client:
        return {
            "status": "unhealthy",
            "message": "Redis client is not initialized",
            "timestamp": datetime.utcnow().isoformat()
        }
    
    # Test basic operations
    start_time = datetime.utcnow()
    ping_result = redis_client.ping()
    response_time = (datetime.utcnow() - start_time).total_seconds() * 1000  # in milliseconds
    
    if not ping_result:
        return {
            "status": "unhealthy",
            "message": "Redis ping failed",
            "timestamp": datetime.utcnow().isoformat()
        }
    
    # Get Redis info
    info = redis_client.info()
    
    # Get cache stats
    cache_stats = get_cache_stats()
    
    # Extract useful metrics
    metrics = {
        "response_time_ms": round(response_time, 2),
        "uptime_seconds": info.get("uptime_in_seconds", 0),
        "uptime_days": round(info.get("uptime_in_seconds", 0) / 86400, 2),
        "memory_used": info.get("used_memory_human", "unknown"),
        "connected_clients": info.get("connected_clients", 0),
        "version": info.get("redis_version", "unknown"),
        "hit_rate": cache_stats.get("hit_rate", 0)
    }
    
    return {
        "status": "healthy",
        "message": "Redis connection is healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "metrics": metrics
    } 