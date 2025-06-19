"""
Utility functions for managing and monitoring Redis-based agent memory.

This module provides tools for:
- Monitoring memory usage across threads
- Cleaning up old conversation data
- Debugging serialization issues
- Managing memory configuration
"""

import json
import logging
import time
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta

from .redis_memory_config import (
    get_redis_checkpoint_saver, 
    get_agent_memory_stats, 
    cleanup_agent_memory,
    MEMORY_CONFIG
)
from .redis_retry import redis_client

logger = logging.getLogger(__name__)

class AgentMemoryManager:
    """Manager for Redis-based agent memory operations."""
    
    def __init__(self):
        self.checkpoint_saver = get_redis_checkpoint_saver()
    
    def get_all_thread_stats(self) -> Dict[str, Any]:
        """Get memory statistics for all threads."""
        try:
            if not self.checkpoint_saver.redis_available or not redis_client:
                return {"error": "Redis not available", "fallback_active": True}
            
            # Get all agent memory keys
            all_keys = redis_client.keys("langgraph:agent:*")
            
            # Group by thread_id
            thread_data = {}
            for key in all_keys:
                parts = key.split(":")
                if len(parts) >= 4:
                    thread_id = parts[-2]  # Assuming format: langgraph:agent:type:thread_id:checkpoint_id
                    
                    if thread_id not in thread_data:
                        thread_data[thread_id] = {
                            "thread_id": thread_id,
                            "checkpoints": [],
                            "metadata_keys": [],
                            "total_keys": 0,
                            "last_activity": None
                        }
                    
                    thread_data[thread_id]["total_keys"] += 1
                    
                    if "checkpoint" in key:
                        checkpoint_id = parts[-1]
                        thread_data[thread_id]["checkpoints"].append(checkpoint_id)
                        
                        # Try to get timestamp from checkpoint data
                        try:
                            data = redis_client.get(key)
                            if data:
                                parsed_data = json.loads(data)
                                timestamp = parsed_data.get("timestamp")
                                if timestamp:
                                    if not thread_data[thread_id]["last_activity"] or timestamp > thread_data[thread_id]["last_activity"]:
                                        thread_data[thread_id]["last_activity"] = timestamp
                        except Exception:
                            pass  # Skip if can't parse
                    
                    elif "metadata" in key:
                        thread_data[thread_id]["metadata_keys"].append(key)
            
            # Convert timestamps to readable format
            for thread_id, data in thread_data.items():
                if data["last_activity"]:
                    data["last_activity_human"] = datetime.fromtimestamp(data["last_activity"]).isoformat()
                    data["hours_since_activity"] = (time.time() - data["last_activity"]) / 3600
                data["checkpoint_count"] = len(data["checkpoints"])
            
            return {
                "total_threads": len(thread_data),
                "total_keys": len(all_keys),
                "threads": thread_data,
                "redis_available": True,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error getting thread stats: {e}")
            return {"error": str(e), "redis_available": False}
    
    def cleanup_inactive_threads(self, hours_inactive: int = 48) -> Dict[str, Any]:
        """
        Clean up threads that have been inactive for a specified time.
        
        Args:
            hours_inactive: Hours of inactivity before cleanup
        
        Returns:
            Cleanup results
        """
        try:
            if not self.checkpoint_saver.redis_available or not redis_client:
                return {"error": "Redis not available"}
            
            cutoff_time = time.time() - (hours_inactive * 3600)
            threads_cleaned = 0
            keys_deleted = 0
            
            thread_stats = self.get_all_thread_stats()
            if "error" in thread_stats:
                return thread_stats
            
            for thread_id, data in thread_stats["threads"].items():
                last_activity = data.get("last_activity")
                
                # Clean up if no activity or activity is older than cutoff
                if not last_activity or last_activity < cutoff_time:
                    pattern = f"langgraph:agent:*:{thread_id}:*"
                    keys_to_delete = redis_client.keys(pattern)
                    
                    if keys_to_delete:
                        redis_client.delete(*keys_to_delete)
                        threads_cleaned += 1
                        keys_deleted += len(keys_to_delete)
                        logger.info(f"Cleaned up inactive thread {thread_id}: {len(keys_to_delete)} keys deleted")
            
            return {
                "status": "success",
                "threads_cleaned": threads_cleaned,
                "keys_deleted": keys_deleted,
                "cutoff_hours": hours_inactive,
                "cutoff_time": datetime.fromtimestamp(cutoff_time).isoformat(),
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error cleaning up inactive threads: {e}")
            return {"error": str(e)}
    
    def debug_thread_memory(self, thread_id: str) -> Dict[str, Any]:
        """
        Debug memory issues for a specific thread.
        
        Args:
            thread_id: Thread ID to debug
        
        Returns:
            Debug information
        """
        try:
            if not self.checkpoint_saver.redis_available or not redis_client:
                return {"error": "Redis not available"}
            
            debug_info = {
                "thread_id": thread_id,
                "timestamp": datetime.now().isoformat(),
                "keys": [],
                "serialization_tests": [],
                "redis_info": {}
            }
            
            # Get all keys for this thread
            pattern = f"langgraph:agent:*:{thread_id}:*"
            keys = redis_client.keys(pattern)
            
            for key in keys:
                key_info = {
                    "key": key,
                    "type": "unknown",
                    "size_bytes": 0,
                    "ttl": redis_client.ttl(key),
                    "can_deserialize": False,
                    "data_preview": None
                }
                
                # Determine key type
                if "checkpoint" in key:
                    key_info["type"] = "checkpoint"
                elif "metadata" in key:
                    key_info["type"] = "metadata"
                elif "writes" in key:
                    key_info["type"] = "writes"
                
                # Get data and test deserialization
                try:
                    data = redis_client.get(key)
                    if data:
                        key_info["size_bytes"] = len(data)
                        key_info["data_preview"] = data[:100] + "..." if len(data) > 100 else data
                        
                        # Test JSON deserialization
                        try:
                            parsed = json.loads(data)
                            key_info["can_deserialize"] = True
                            key_info["parsed_type"] = type(parsed).__name__
                            if isinstance(parsed, dict):
                                key_info["dict_keys"] = list(parsed.keys())
                        except json.JSONDecodeError as e:
                            key_info["deserialize_error"] = str(e)
                
                except Exception as e:
                    key_info["redis_error"] = str(e)
                
                debug_info["keys"].append(key_info)
            
            # Test serialization with sample data
            sample_data = {
                "test_timestamp": time.time(),
                "test_string": "Hello World",
                "test_number": 42,
                "test_list": [1, 2, 3],
                "test_dict": {"nested": "value"}
            }
            
            try:
                serialized = self.checkpoint_saver._serialize_data(sample_data, "debug_test")
                if serialized:
                    deserialized = self.checkpoint_saver._deserialize_data(serialized, "debug_test")
                    debug_info["serialization_tests"].append({
                        "test": "basic_serialization",
                        "success": True,
                        "original_size": len(str(sample_data)),
                        "serialized_size": len(serialized),
                        "round_trip_success": deserialized == sample_data
                    })
                else:
                    debug_info["serialization_tests"].append({
                        "test": "basic_serialization",
                        "success": False,
                        "error": "Serialization returned None"
                    })
            except Exception as e:
                debug_info["serialization_tests"].append({
                    "test": "basic_serialization",
                    "success": False,
                    "error": str(e)
                })
            
            # Redis connection info
            try:
                redis_info = redis_client.info()
                debug_info["redis_info"] = {
                    "connected_clients": redis_info.get("connected_clients"),
                    "used_memory_human": redis_info.get("used_memory_human"),
                    "redis_version": redis_info.get("redis_version"),
                    "uptime_in_seconds": redis_info.get("uptime_in_seconds")
                }
            except Exception as e:
                debug_info["redis_info"]["error"] = str(e)
            
            return debug_info
            
        except Exception as e:
            logger.error(f"Error debugging thread memory: {e}")
            return {"error": str(e), "thread_id": thread_id}
    
    def force_memory_cleanup(self) -> Dict[str, Any]:
        """Force a comprehensive memory cleanup."""
        try:
            # Run global cleanup
            cleanup_agent_memory()
            
            # Get stats after cleanup
            stats = get_agent_memory_stats()
            
            return {
                "status": "success",
                "message": "Forced memory cleanup completed",
                "post_cleanup_stats": stats,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error in force cleanup: {e}")
            return {"error": str(e)}
    
    def get_memory_health_report(self) -> Dict[str, Any]:
        """Generate a comprehensive memory health report."""
        try:
            report = {
                "timestamp": datetime.now().isoformat(),
                "redis_status": "unknown",
                "overall_health": "unknown",
                "warnings": [],
                "recommendations": []
            }
            
            # Check Redis availability
            if self.checkpoint_saver.redis_available:
                report["redis_status"] = "available"
            else:
                report["redis_status"] = "unavailable"
                report["warnings"].append("Redis is not available - using fallback memory storage")
                report["recommendations"].append("Check Redis connection and server status")
            
            # Get global stats
            global_stats = get_agent_memory_stats()
            report["global_stats"] = global_stats
            
            # Get thread stats
            thread_stats = self.get_all_thread_stats()
            if "error" not in thread_stats:
                report["thread_stats"] = thread_stats
                
                # Analyze thread health
                total_threads = thread_stats.get("total_threads", 0)
                total_keys = thread_stats.get("total_keys", 0)
                
                if total_threads > 100:
                    report["warnings"].append(f"High number of active threads: {total_threads}")
                    report["recommendations"].append("Consider running cleanup for inactive threads")
                
                if total_keys > 1000:
                    report["warnings"].append(f"High number of Redis keys: {total_keys}")
                    report["recommendations"].append("Consider reducing TTL or running more frequent cleanups")
                
                # Check for old threads
                old_threads = 0
                for thread_data in thread_stats.get("threads", {}).values():
                    hours_since = thread_data.get("hours_since_activity", 0)
                    if hours_since > 48:  # Older than 2 days
                        old_threads += 1
                
                if old_threads > 10:
                    report["warnings"].append(f"Many inactive threads: {old_threads}")
                    report["recommendations"].append("Run cleanup for inactive threads")
            else:
                report["thread_stats_error"] = thread_stats["error"]
            
            # Overall health assessment
            if len(report["warnings"]) == 0:
                report["overall_health"] = "good"
            elif len(report["warnings"]) <= 2:
                report["overall_health"] = "fair"
            else:
                report["overall_health"] = "poor"
            
            return report
            
        except Exception as e:
            logger.error(f"Error generating health report: {e}")
            return {"error": str(e), "timestamp": datetime.now().isoformat()}

# Global manager instance
_memory_manager = None

def get_memory_manager() -> AgentMemoryManager:
    """Get or create the global memory manager instance."""
    global _memory_manager
    if _memory_manager is None:
        _memory_manager = AgentMemoryManager()
    return _memory_manager

# Utility functions for easy access
def get_memory_health_report() -> Dict[str, Any]:
    """Get a comprehensive memory health report."""
    manager = get_memory_manager()
    return manager.get_memory_health_report()

def cleanup_old_conversations(hours_inactive: int = 48) -> Dict[str, Any]:
    """Clean up conversations inactive for specified hours."""
    manager = get_memory_manager()
    return manager.cleanup_inactive_threads(hours_inactive)

def debug_conversation_memory(thread_id: str) -> Dict[str, Any]:
    """Debug memory issues for a specific conversation."""
    manager = get_memory_manager()
    return manager.debug_thread_memory(thread_id) 