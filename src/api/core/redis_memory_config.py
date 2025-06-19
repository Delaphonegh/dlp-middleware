"""
Redis-based memory configuration for Call Center Agent.

This module provides Redis-based checkpoint saving and loading for LangGraph agents,
allowing conversation state to persist across API calls and server restarts.
Each thread_id gets its own isolated memory namespace.
"""

import json
import logging
import time
import hashlib
from typing import Dict, Any, Optional, List, Iterator, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict

from langgraph.checkpoint.base import BaseCheckpointSaver, Checkpoint, CheckpointMetadata, CheckpointTuple
from langgraph.checkpoint.memory import MemorySaver
from .redis_retry import redis_client, with_redis_retry

logger = logging.getLogger(__name__)

# Redis key prefixes for agent memory
AGENT_MEMORY_PREFIX = "langgraph:agent"
CHECKPOINT_PREFIX = f"{AGENT_MEMORY_PREFIX}:checkpoint"
WRITES_PREFIX = f"{AGENT_MEMORY_PREFIX}:writes"
METADATA_PREFIX = f"{AGENT_MEMORY_PREFIX}:metadata"

# Memory configuration
MEMORY_CONFIG = {
    "default_ttl_hours": 24,  # Default TTL for conversation memory
    "max_checkpoints_per_thread": 100,  # Maximum checkpoints to keep per thread
    "cleanup_interval_hours": 6,  # How often to run cleanup
    "serialization_debug": True,  # Enable debug logging for serialization errors
}

@dataclass
class AgentCheckpoint:
    """Agent checkpoint data structure."""
    checkpoint_id: str
    thread_id: str
    timestamp: float
    state: Dict[str, Any]
    metadata: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for Redis storage."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AgentCheckpoint':
        """Create from dictionary loaded from Redis."""
        return cls(**data)

class RedisCheckpointSaver(BaseCheckpointSaver):
    """
    Redis-based checkpoint saver for LangGraph agents.
    
    Provides persistent conversation memory tied to thread_id with:
    - Automatic TTL management
    - Serialization error handling with debug logging
    - Fallback to in-memory storage on Redis failures
    - Thread isolation
    - Cleanup of old checkpoints
    """
    
    def __init__(self, fallback_to_memory: bool = True, ttl_hours: int = None):
        """
        Initialize Redis checkpoint saver.
        
        Args:
            fallback_to_memory: Use MemorySaver as fallback when Redis fails
            ttl_hours: TTL for checkpoints in hours (default from config)
        """
        super().__init__()
        self.fallback_to_memory = fallback_to_memory
        self.memory_fallback = MemorySaver() if fallback_to_memory else None
        self.ttl_seconds = (ttl_hours or MEMORY_CONFIG["default_ttl_hours"]) * 3600
        self.redis_available = self._check_redis_availability()
        
        logger.info(f"RedisCheckpointSaver initialized:")
        logger.info(f"  Redis available: {self.redis_available}")
        logger.info(f"  Fallback enabled: {fallback_to_memory}")
        logger.info(f"  TTL: {self.ttl_seconds}s ({self.ttl_seconds/3600}h)")
    
    def _check_redis_availability(self) -> bool:
        """Check if Redis is available for use."""
        try:
            if not redis_client:
                logger.warning("Redis client not initialized")
                return False
            
            # Test Redis connection
            redis_client.ping()
            logger.info("Redis connection verified for agent memory")
            return True
            
        except Exception as e:
            logger.error(f"Redis not available for agent memory: {e}")
            return False
    
    def _get_checkpoint_key(self, thread_id: str, checkpoint_id: str) -> str:
        """Generate Redis key for checkpoint."""
        return f"{CHECKPOINT_PREFIX}:{thread_id}:{checkpoint_id}"
    
    def _get_writes_key(self, thread_id: str, checkpoint_id: str) -> str:
        """Generate Redis key for checkpoint writes."""
        return f"{WRITES_PREFIX}:{thread_id}:{checkpoint_id}"
    
    def _get_metadata_key(self, thread_id: str) -> str:
        """Generate Redis key for thread metadata."""
        return f"{METADATA_PREFIX}:{thread_id}"
    
    def _get_thread_pattern(self, thread_id: str) -> str:
        """Generate Redis pattern for all thread data."""
        return f"{AGENT_MEMORY_PREFIX}:*:{thread_id}:*"
    
    def _serialize_data(self, data: Any, context: str = "unknown") -> Optional[str]:
        """
        Serialize data with comprehensive error handling and debug logging.
        
        Args:
            data: Data to serialize
            context: Context for debugging (e.g., "checkpoint", "writes")
        
        Returns:
            Serialized JSON string or None on error
        """
        try:
            # Handle special LangGraph types
            if hasattr(data, 'dict'):
                # Pydantic models
                serialized = json.dumps(data.dict())
            elif hasattr(data, '__dict__'):
                # Regular objects with __dict__
                serialized = json.dumps(data.__dict__)
            else:
                # Regular JSON serialization
                serialized = json.dumps(data, default=str)
            
            if MEMORY_CONFIG["serialization_debug"]:
                logger.debug(f"Serialization success [{context}]: {len(serialized)} chars")
            
            return serialized
            
        except (TypeError, ValueError) as e:
            if MEMORY_CONFIG["serialization_debug"]:
                logger.error(f"Serialization error [{context}]: {type(e).__name__}: {e}")
                logger.error(f"Data type: {type(data)}")
                logger.error(f"Data preview: {str(data)[:200]}...")
                
                # Try to identify problematic fields
                if hasattr(data, '__dict__'):
                    for key, value in data.__dict__.items():
                        try:
                            json.dumps(value, default=str)
                        except Exception as field_error:
                            logger.error(f"Problematic field '{key}': {type(value)} - {field_error}")
            
            logger.warning(f"Using in-memory fallback due to serialization error in {context}")
            return None
            
        except Exception as e:
            logger.error(f"Unexpected serialization error [{context}]: {e}")
            return None
    
    def _deserialize_data(self, serialized: str, context: str = "unknown") -> Optional[Any]:
        """
        Deserialize data with error handling and debug logging.
        
        Args:
            serialized: JSON string to deserialize
            context: Context for debugging
        
        Returns:
            Deserialized data or None on error
        """
        try:
            data = json.loads(serialized)
            
            if MEMORY_CONFIG["serialization_debug"]:
                logger.debug(f"Deserialization success [{context}]: {type(data)}")
            
            return data
            
        except (json.JSONDecodeError, ValueError) as e:
            if MEMORY_CONFIG["serialization_debug"]:
                logger.error(f"Deserialization error [{context}]: {e}")
                logger.error(f"Data preview: {serialized[:200]}...")
            
            logger.warning(f"Using in-memory fallback due to deserialization error in {context}")
            return None
            
        except Exception as e:
            logger.error(f"Unexpected deserialization error [{context}]: {e}")
            return None
    
    @with_redis_retry()
    def put(self, config: Dict[str, Any], checkpoint: Checkpoint, metadata: CheckpointMetadata) -> None:
        """
        Save checkpoint to Redis with fallback handling.
        
        Args:
            config: Configuration containing thread_id
            checkpoint: Checkpoint data to save
            metadata: Checkpoint metadata
        """
        thread_id = config.get("configurable", {}).get("thread_id")
        if not thread_id:
            logger.error("No thread_id found in config for checkpoint save")
            if self.memory_fallback:
                return self.memory_fallback.put(config, checkpoint, metadata)
            return
        
        checkpoint_id = checkpoint.get("id") or f"checkpoint_{int(time.time() * 1000)}"
        
        if MEMORY_CONFIG["serialization_debug"]:
            logger.debug(f"Saving checkpoint: thread_id={thread_id}, checkpoint_id={checkpoint_id}")
        
        # Try Redis first
        if self.redis_available and redis_client:
            try:
                # Serialize checkpoint data
                checkpoint_data = {
                    "checkpoint_id": checkpoint_id,
                    "thread_id": thread_id,
                    "timestamp": time.time(),
                    "checkpoint": checkpoint,
                    "metadata": metadata or {}
                }
                
                serialized_checkpoint = self._serialize_data(checkpoint_data, "checkpoint")
                if serialized_checkpoint:
                    checkpoint_key = self._get_checkpoint_key(thread_id, checkpoint_id)
                    redis_client.setex(checkpoint_key, self.ttl_seconds, serialized_checkpoint)
                    
                    # Update thread metadata
                    self._update_thread_metadata(thread_id, checkpoint_id)
                    
                    if MEMORY_CONFIG["serialization_debug"]:
                        logger.info(f"Checkpoint saved to Redis: {checkpoint_key}")
                    return
                else:
                    logger.warning("Checkpoint serialization failed, using fallback")
                    
            except Exception as e:
                logger.error(f"Error saving checkpoint to Redis: {e}")
                self.redis_available = False
        
        # Fallback to memory
        if self.memory_fallback:
            logger.info(f"Using memory fallback for checkpoint save: thread_id={thread_id}")
            self.memory_fallback.put(config, checkpoint, metadata)
        else:
            logger.error(f"No fallback available for checkpoint save: thread_id={thread_id}")
    
    @with_redis_retry()
    def get(self, config: Dict[str, Any]) -> Optional[Checkpoint]:
        """
        Load checkpoint from Redis with fallback handling.
        
        Args:
            config: Configuration containing thread_id
        
        Returns:
            Latest checkpoint or None if not found
        """
        thread_id = config.get("configurable", {}).get("thread_id")
        if not thread_id:
            logger.error("No thread_id found in config for checkpoint load")
            if self.memory_fallback:
                return self.memory_fallback.get(config)
            return None
        
        if MEMORY_CONFIG["serialization_debug"]:
            logger.debug(f"Loading checkpoint: thread_id={thread_id}")
        
        # Try Redis first
        if self.redis_available and redis_client:
            try:
                # Get latest checkpoint for thread
                latest_checkpoint_id = self._get_latest_checkpoint_id(thread_id)
                if latest_checkpoint_id:
                    checkpoint_key = self._get_checkpoint_key(thread_id, latest_checkpoint_id)
                    serialized_data = redis_client.get(checkpoint_key)
                    
                    if serialized_data:
                        checkpoint_data = self._deserialize_data(serialized_data, "checkpoint")
                        if checkpoint_data:
                            if MEMORY_CONFIG["serialization_debug"]:
                                logger.info(f"Checkpoint loaded from Redis: {checkpoint_key}")
                            return checkpoint_data.get("checkpoint")
                    
            except Exception as e:
                logger.error(f"Error loading checkpoint from Redis: {e}")
                self.redis_available = False
        
        # Fallback to memory
        if self.memory_fallback:
            logger.info(f"Using memory fallback for checkpoint load: thread_id={thread_id}")
            return self.memory_fallback.get(config)
        
        logger.warning(f"No checkpoint found: thread_id={thread_id}")
        return None
    
    def list(self, config: Dict[str, Any], before: Optional[str] = None, limit: Optional[int] = None) -> Iterator[Tuple[str, Checkpoint]]:
        """
        List checkpoints for a thread.
        
        Args:
            config: Configuration containing thread_id
            before: List checkpoints before this ID
            limit: Maximum number of checkpoints to return
        
        Yields:
            Tuples of (checkpoint_id, checkpoint)
        """
        thread_id = config.get("configurable", {}).get("thread_id")
        if not thread_id:
            if self.memory_fallback:
                yield from self.memory_fallback.list(config, before, limit)
            return
        
        # Try Redis first
        if self.redis_available and redis_client:
            try:
                checkpoint_ids = self._get_checkpoint_ids(thread_id, before, limit)
                
                for checkpoint_id in checkpoint_ids:
                    checkpoint_key = self._get_checkpoint_key(thread_id, checkpoint_id)
                    serialized_data = redis_client.get(checkpoint_key)
                    
                    if serialized_data:
                        checkpoint_data = self._deserialize_data(serialized_data, "checkpoint_list")
                        if checkpoint_data:
                            yield (checkpoint_id, checkpoint_data.get("checkpoint"))
                
                return
                
            except Exception as e:
                logger.error(f"Error listing checkpoints from Redis: {e}")
                self.redis_available = False
        
        # Fallback to memory
        if self.memory_fallback:
            yield from self.memory_fallback.list(config, before, limit)
    
    @with_redis_retry()
    def _update_thread_metadata(self, thread_id: str, checkpoint_id: str):
        """Update thread metadata with latest checkpoint info."""
        try:
            metadata_key = self._get_metadata_key(thread_id)
            metadata = {
                "thread_id": thread_id,
                "latest_checkpoint_id": checkpoint_id,
                "last_updated": time.time(),
                "total_checkpoints": self._count_checkpoints(thread_id)
            }
            
            serialized_metadata = self._serialize_data(metadata, "metadata")
            if serialized_metadata:
                redis_client.setex(metadata_key, self.ttl_seconds, serialized_metadata)
                
        except Exception as e:
            logger.error(f"Error updating thread metadata: {e}")
    
    @with_redis_retry()
    def _get_latest_checkpoint_id(self, thread_id: str) -> Optional[str]:
        """Get the latest checkpoint ID for a thread."""
        try:
            metadata_key = self._get_metadata_key(thread_id)
            serialized_metadata = redis_client.get(metadata_key)
            
            if serialized_metadata:
                metadata = self._deserialize_data(serialized_metadata, "metadata")
                if metadata:
                    return metadata.get("latest_checkpoint_id")
            
            # Fallback: scan for checkpoints
            pattern = f"{CHECKPOINT_PREFIX}:{thread_id}:*"
            keys = redis_client.keys(pattern)
            
            if keys:
                # Sort by timestamp (extract from key or get from data)
                latest_key = max(keys)
                return latest_key.split(":")[-1]
            
            return None
            
        except Exception as e:
            logger.error(f"Error getting latest checkpoint ID: {e}")
            return None
    
    @with_redis_retry()
    def _get_checkpoint_ids(self, thread_id: str, before: Optional[str] = None, limit: Optional[int] = None) -> List[str]:
        """Get checkpoint IDs for a thread with optional filtering."""
        try:
            pattern = f"{CHECKPOINT_PREFIX}:{thread_id}:*"
            keys = redis_client.keys(pattern)
            
            # Extract checkpoint IDs
            checkpoint_ids = [key.split(":")[-1] for key in keys]
            
            # Sort by timestamp (assuming checkpoint IDs are sortable)
            checkpoint_ids.sort(reverse=True)  # Most recent first
            
            # Apply before filter
            if before:
                try:
                    before_index = checkpoint_ids.index(before)
                    checkpoint_ids = checkpoint_ids[before_index + 1:]
                except ValueError:
                    pass  # before not found, continue with all
            
            # Apply limit
            if limit:
                checkpoint_ids = checkpoint_ids[:limit]
            
            return checkpoint_ids
            
        except Exception as e:
            logger.error(f"Error getting checkpoint IDs: {e}")
            return []
    
    @with_redis_retry()
    def _count_checkpoints(self, thread_id: str) -> int:
        """Count checkpoints for a thread."""
        if not self.redis_available:
            return self.fallback_memory._count_checkpoints(thread_id) if hasattr(self.fallback_memory, '_count_checkpoints') else 0
        
        try:
            pattern = self._get_thread_pattern(thread_id)
            keys = redis_client.keys(pattern)
            return len([k for k in keys if "checkpoint" in k])
        except Exception as e:
            logger.error(f"Error counting checkpoints for thread {thread_id}: {e}")
            return 0

    # Async methods required by LangGraph
    def get_tuple(self, config: Dict[str, Any]) -> Optional[CheckpointTuple]:
        """
        Get checkpoint tuple (config, checkpoint, metadata) for the given config.
        
        Args:
            config: Configuration containing thread_id
        
        Returns:
            CheckpointTuple or None if not found
        """
        checkpoint = self.get(config)
        if checkpoint:
            # Create proper CheckpointMetadata with all required fields
            metadata = CheckpointMetadata(
                source="input",  # Required: 'input', 'loop', 'update', or 'fork'
                step=0,         # Required: step number
                writes={},      # Required: any pending writes
                parents={}      # Required: parent checkpoint references
            )
            
            # Create CheckpointTuple with all required fields
            return CheckpointTuple(
                config=config,
                checkpoint=checkpoint,
                metadata=metadata,
                parent_config=None,
                pending_writes=[]
            )
        return None

    async def aget_tuple(self, config: Dict[str, Any]) -> Optional[CheckpointTuple]:
        """Async version of get_tuple."""
        try:
            return self.get_tuple(config)
        except Exception as e:
            logger.error(f"Error in aget_tuple: {e}")
            if self.fallback_to_memory and hasattr(self.fallback_memory, 'aget_tuple'):
                return await self.fallback_memory.aget_tuple(config)
            return None

    async def aput(self, config: Dict[str, Any], checkpoint: Checkpoint, metadata: CheckpointMetadata, new_versions: Dict[str, Any]) -> Dict[str, Any]:
        """Async version of put."""
        try:
            self.put(config, checkpoint, metadata)
            return config  # Return the config as expected by LangGraph
        except Exception as e:
            logger.error(f"Error in aput: {e}")
            if self.fallback_to_memory and hasattr(self.fallback_memory, 'aput'):
                return await self.fallback_memory.aput(config, checkpoint, metadata, new_versions)
            return config

    async def alist(self, config: Dict[str, Any], before: Optional[str] = None, limit: Optional[int] = None) -> Iterator[Tuple[str, Checkpoint]]:
        """Async version of list."""
        try:
            for checkpoint_tuple in self.list(config, before, limit):
                yield checkpoint_tuple
        except Exception as e:
            logger.error(f"Error in alist: {e}")
            if self.fallback_to_memory and hasattr(self.fallback_memory, 'alist'):
                async for checkpoint_tuple in self.fallback_memory.alist(config, before, limit):
                    yield checkpoint_tuple

    async def aput_writes(self, config: Dict[str, Any], writes: List[Tuple[str, Any]], task_id: str) -> None:
        """Async version of put_writes for handling pending writes."""
        try:
            # For now, we'll just log the writes as Redis doesn't need special handling for pending writes
            # The actual checkpoint saving is handled by aput
            thread_id = config.get("configurable", {}).get("thread_id", "default")
            logger.debug(f"Pending writes for thread {thread_id}: {len(writes)} writes, task_id: {task_id}")
            
            # We could store pending writes in Redis if needed, but for now we'll pass
            # since the main checkpoint saving happens in aput
            pass
            
        except Exception as e:
            logger.error(f"Error in aput_writes: {e}")
            if self.fallback_to_memory and hasattr(self.fallback_memory, 'aput_writes'):
                await self.fallback_memory.aput_writes(config, writes, task_id)

    @with_redis_retry()
    def cleanup_old_checkpoints(self, thread_id: str = None):
        """
        Clean up old checkpoints to prevent memory bloat.
        
        Args:
            thread_id: Specific thread to clean up, or None for all threads
        """
        try:
            if thread_id:
                patterns = [f"{AGENT_MEMORY_PREFIX}:*:{thread_id}:*"]
            else:
                patterns = [f"{AGENT_MEMORY_PREFIX}:*"]
            
            total_cleaned = 0
            
            for pattern in patterns:
                keys = redis_client.keys(pattern)
                
                # Group by thread_id
                thread_keys = {}
                for key in keys:
                    parts = key.split(":")
                    if len(parts) >= 4:
                        key_thread_id = parts[-2]  # Assuming format: prefix:type:thread_id:checkpoint_id
                        if key_thread_id not in thread_keys:
                            thread_keys[key_thread_id] = []
                        thread_keys[key_thread_id].append(key)
                
                # Clean up each thread
                for thread_id, thread_key_list in thread_keys.items():
                    checkpoint_keys = [k for k in thread_key_list if CHECKPOINT_PREFIX in k]
                    
                    if len(checkpoint_keys) > MEMORY_CONFIG["max_checkpoints_per_thread"]:
                        # Sort by timestamp and keep only the most recent
                        checkpoint_keys.sort(reverse=True)
                        keys_to_delete = checkpoint_keys[MEMORY_CONFIG["max_checkpoints_per_thread"]:]
                        
                        if keys_to_delete:
                            redis_client.delete(*keys_to_delete)
                            total_cleaned += len(keys_to_delete)
                            logger.info(f"Cleaned up {len(keys_to_delete)} old checkpoints for thread {thread_id}")
            
            if total_cleaned > 0:
                logger.info(f"Total checkpoints cleaned up: {total_cleaned}")
                
        except Exception as e:
            logger.error(f"Error during checkpoint cleanup: {e}")
    
    def get_memory_stats(self, thread_id: str = None) -> Dict[str, Any]:
        """
        Get memory usage statistics.
        
        Args:
            thread_id: Specific thread stats, or None for global stats
        
        Returns:
            Dictionary with memory statistics
        """
        stats = {
            "redis_available": self.redis_available,
            "fallback_enabled": self.fallback_to_memory,
            "ttl_hours": self.ttl_seconds / 3600,
            "timestamp": datetime.now().isoformat()
        }
        
        if self.redis_available and redis_client:
            try:
                if thread_id:
                    # Thread-specific stats
                    pattern = f"{AGENT_MEMORY_PREFIX}:*:{thread_id}:*"
                    keys = redis_client.keys(pattern)
                    stats.update({
                        "thread_id": thread_id,
                        "total_keys": len(keys),
                        "checkpoint_count": len([k for k in keys if CHECKPOINT_PREFIX in k]),
                        "has_metadata": any(METADATA_PREFIX in k for k in keys)
                    })
                else:
                    # Global stats
                    all_keys = redis_client.keys(f"{AGENT_MEMORY_PREFIX}:*")
                    stats.update({
                        "total_keys": len(all_keys),
                        "total_checkpoints": len([k for k in all_keys if CHECKPOINT_PREFIX in k]),
                        "total_metadata": len([k for k in all_keys if METADATA_PREFIX in k]),
                        "unique_threads": len(set(k.split(":")[-2] for k in all_keys if len(k.split(":")) >= 4))
                    })
                    
            except Exception as e:
                stats["redis_error"] = str(e)
        
        return stats

# Global instance
_redis_checkpoint_saver = None

def get_redis_checkpoint_saver(ttl_hours: int = None, fallback_to_memory: bool = True) -> RedisCheckpointSaver:
    """
    Get or create the global Redis checkpoint saver instance.
    
    Args:
        ttl_hours: TTL for checkpoints in hours
        fallback_to_memory: Enable fallback to MemorySaver
    
    Returns:
        RedisCheckpointSaver instance
    """
    global _redis_checkpoint_saver
    
    if _redis_checkpoint_saver is None:
        _redis_checkpoint_saver = RedisCheckpointSaver(
            fallback_to_memory=fallback_to_memory,
            ttl_hours=ttl_hours
        )
        logger.info("Global RedisCheckpointSaver instance created")
    
    return _redis_checkpoint_saver

def cleanup_agent_memory(thread_id: str = None):
    """
    Utility function to clean up agent memory.
    
    Args:
        thread_id: Specific thread to clean, or None for all threads
    """
    saver = get_redis_checkpoint_saver()
    saver.cleanup_old_checkpoints(thread_id)

def get_agent_memory_stats(thread_id: str = None) -> Dict[str, Any]:
    """
    Utility function to get agent memory statistics.
    
    Args:
        thread_id: Specific thread stats, or None for global stats
    
    Returns:
        Memory statistics dictionary
    """
    saver = get_redis_checkpoint_saver()
    return saver.get_memory_stats(thread_id) 