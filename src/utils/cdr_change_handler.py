#!/usr/bin/env python
"""
CDR Change Handler

A specialized handler for CDR table changes that monitors call direction data
and updates related caches or triggers analytics recalculations.
"""

import os
import sys
import json
import logging
from datetime import datetime
import redis

from dotenv import load_dotenv
from db_change_monitor import DBChangeMonitor

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("cdr_change_handler.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("cdr_change_handler")

# Load environment variables
load_dotenv()

# Constants for call direction
DIRECTION_INBOUND = "inbound"
DIRECTION_OUTBOUND = "outbound"
DIRECTION_INTERNAL = "internal"
DIRECTION_UNKNOWN = "unknown"

def determine_call_direction(src, dst):
    """
    Determine call direction based on the following rules:
    - Inbound: dst is ≤ 5 digits (internal extension)
    - Outbound: src is ≤ 5 digits (extension), dst is > 5 digits (external number)
    - Internal: src and dst are both ≤ 5 digits
    - Unknown: any other case
    """
    if not src or not dst:
        return DIRECTION_UNKNOWN
    
    src_is_extension = len(str(src).strip()) <= 5 and str(src).strip().isdigit()
    dst_is_extension = len(str(dst).strip()) <= 5 and str(dst).strip().isdigit()
    
    if src_is_extension and dst_is_extension:
        return DIRECTION_INTERNAL
    elif src_is_extension and not dst_is_extension:
        return DIRECTION_OUTBOUND
    elif not src_is_extension and dst_is_extension:
        return DIRECTION_INBOUND
    else:
        return DIRECTION_UNKNOWN


class CDRChangeHandler(DBChangeMonitor):
    """Specialized monitor for CDR table changes with focus on call direction"""
    
    def __init__(self, connection_settings, server_id=100, redis_client=None):
        """
        Initialize the CDR change handler
        
        Args:
            connection_settings: Dict with MySQL connection settings
            server_id: Unique server ID for replication
            redis_client: Redis client for cache operations
        """
        super().__init__(
            connection_settings=connection_settings,
            server_id=server_id,
            tables=["cdr"],
            schemas=[os.environ.get("MYSQL_SCHEMA", "asteriskcdrdb")]
        )
        
        # Initialize Redis connection if provided
        self.redis = redis_client
        if not self.redis and os.environ.get("REDIS_HOST"):
            self.redis = redis.Redis(
                host=os.environ.get("REDIS_HOST", "localhost"),
                port=int(os.environ.get("REDIS_PORT", 6379)),
                db=int(os.environ.get("REDIS_DB", 0)),
                password=os.environ.get("REDIS_PASSWORD", None),
                decode_responses=True
            )
        
        logger.info("Initialized CDR Change Handler")
    
    def _handle_insert(self, schema, table, timestamp, row):
        """Handle new CDR record insertion"""
        if table != "cdr":
            return
            
        logger.info(f"New call record inserted at {timestamp}")
        
        try:
            # Extract CDR record data
            values = row["values"]
            src = values.get("src", "")
            dst = values.get("dst", "")
            calldate = values.get("calldate", "")
            disposition = values.get("disposition", "")
            uniqueid = values.get("uniqueid", "")
            
            # Determine call direction
            direction = determine_call_direction(src, dst)
            
            logger.info(f"New {direction} call: {src} -> {dst} [{disposition}]")
            
            # Update direction breakdown counts in cache
            if self.redis:
                # Increment direction counter
                direction_key = f"cdr:direction:{direction}"
                self.redis.incr(direction_key)
                
                # Add to daily stats
                if calldate:
                    day_str = calldate.strftime("%Y-%m-%d") if hasattr(calldate, "strftime") else str(calldate).split()[0]
                    daily_key = f"cdr:daily:{day_str}:direction:{direction}"
                    self.redis.incr(daily_key)
                
                # Invalidate dashboard cache
                self.redis.delete("dashboard:call_stats")
                
                # Store call record with direction
                if uniqueid:
                    cdr_data = {**values, "direction": direction}
                    self.redis.hset(f"cdr:{uniqueid}", mapping=self._serialize_dict(cdr_data))
        
        except Exception as e:
            logger.error(f"Error processing CDR insert: {str(e)}")
    
    def _handle_update(self, schema, table, timestamp, row):
        """Handle CDR record update"""
        if table != "cdr":
            return
            
        logger.info(f"Call record updated at {timestamp}")
        
        try:
            # Get before/after values
            before = row["before_values"] 
            after = row["after_values"]
            uniqueid = after.get("uniqueid", "")
            
            # Check if src or dst changed (which would affect direction)
            src_changed = before.get("src", "") != after.get("src", "")
            dst_changed = before.get("dst", "") != after.get("dst", "")
            
            if src_changed or dst_changed:
                # Recalculate direction
                old_direction = determine_call_direction(before.get("src", ""), before.get("dst", ""))
                new_direction = determine_call_direction(after.get("src", ""), after.get("dst", ""))
                
                logger.info(f"Call direction changed: {old_direction} -> {new_direction} for call {uniqueid}")
                
                # Update direction stats in cache
                if self.redis and old_direction != new_direction:
                    # Decrement old direction counter
                    self.redis.decr(f"cdr:direction:{old_direction}")
                    # Increment new direction counter
                    self.redis.incr(f"cdr:direction:{new_direction}")
                    
                    # Update call record with new direction
                    if uniqueid:
                        self.redis.hset(f"cdr:{uniqueid}", "direction", new_direction)
                        
                    # Invalidate dashboard cache
                    self.redis.delete("dashboard:call_stats")
        
        except Exception as e:
            logger.error(f"Error processing CDR update: {str(e)}")
    
    def _handle_delete(self, schema, table, timestamp, row):
        """Handle CDR record deletion"""
        if table != "cdr":
            return
            
        logger.info(f"Call record deleted at {timestamp}")
        
        try:
            # Extract data from deleted record
            values = row["values"]
            src = values.get("src", "")
            dst = values.get("dst", "")
            uniqueid = values.get("uniqueid", "")
            
            # Determine direction of deleted call
            direction = determine_call_direction(src, dst)
            
            # Update direction stats in cache
            if self.redis:
                # Decrement direction counter
                self.redis.decr(f"cdr:direction:{direction}")
                
                # Delete call record from cache
                if uniqueid:
                    self.redis.delete(f"cdr:{uniqueid}")
                
                # Invalidate dashboard cache
                self.redis.delete("dashboard:call_stats")
        
        except Exception as e:
            logger.error(f"Error processing CDR deletion: {str(e)}")
    
    def _serialize_dict(self, data):
        """Convert dictionary values to strings for Redis storage"""
        result = {}
        for key, value in data.items():
            if value is None:
                result[key] = ""
            elif isinstance(value, (datetime, list, dict)):
                result[key] = json.dumps(value, default=str)
            else:
                result[key] = str(value)
        return result


# Example usage
if __name__ == "__main__":
    # Get MySQL connection settings from environment variables
    mysql_settings = {
        "host": os.environ.get("MYSQL_HOST", "localhost"),
        "port": int(os.environ.get("MYSQL_PORT", 3306)),
        "user": os.environ.get("MYSQL_USER", "replica_user"),
        "passwd": os.environ.get("MYSQL_PASSWORD", "password"),
    }
    
    try:
        handler = CDRChangeHandler(
            connection_settings=mysql_settings,
            server_id=int(os.environ.get("MYSQL_SERVER_ID", 100))
        )
        
        # Start monitoring
        handler.start(blocking=True)
        
    except KeyboardInterrupt:
        logger.info("CDR monitoring stopped by user")
    except Exception as e:
        logger.error(f"Error in CDR Change Handler: {str(e)}") 