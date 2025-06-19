#!/usr/bin/env python
"""
MySQL Binlog Change Monitor

This script listens to MySQL database changes through binlog replication.
It detects INSERT, UPDATE, and DELETE operations on specified tables.
"""

import os
import sys
import json
import logging
import time
import signal
from datetime import datetime

from pymysqlreplication import BinLogStreamReader
from pymysqlreplication.row_event import (
    DeleteRowsEvent,
    UpdateRowsEvent,
    WriteRowsEvent,
)
from pymysqlreplication.event import QueryEvent
from dotenv import load_dotenv

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("db_change_monitor.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("db_change_monitor")

# Load environment variables
load_dotenv()

class DBChangeMonitor:
    """Monitor changes to MySQL database using binlog replication"""
    
    def __init__(self, connection_settings, server_id=100, tables=None, schemas=None):
        """
        Initialize the database change monitor.
        
        Args:
            connection_settings: Dict with MySQL connection settings
            server_id: Unique server ID for replication
            tables: List of tables to monitor (optional)
            schemas: List of database schemas to monitor (optional)
        """
        self.connection_settings = connection_settings
        self.server_id = server_id
        self.tables = tables
        self.schemas = schemas
        self.running = False
        self.stream = None
        
        # Register signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self.shutdown)
        signal.signal(signal.SIGTERM, self.shutdown)
        
        logger.info(f"Initialized DB Change Monitor for schemas: {schemas}, tables: {tables}")
    
    def start(self, blocking=True, resume_stream=True):
        """
        Start monitoring database changes.
        
        Args:
            blocking: Whether to block execution while monitoring
            resume_stream: Whether to resume from last known position
        """
        try:
            logger.info("Starting BinLog Stream Reader...")
            
            # Configure the replication stream
            self.stream = BinLogStreamReader(
                connection_settings=self.connection_settings,
                server_id=self.server_id,
                blocking=blocking,
                only_events=[WriteRowsEvent, UpdateRowsEvent, DeleteRowsEvent, QueryEvent],
                only_schemas=self.schemas,
                only_tables=self.tables,
                resume_stream=resume_stream,
                log_file=os.environ.get('MYSQL_BINLOG_FILE', None),
                log_pos=int(os.environ.get('MYSQL_BINLOG_POS', 0)) or None
            )
            
            self.running = True
            self._monitor_changes()
            
        except Exception as e:
            logger.error(f"Error starting BinLog stream: {str(e)}")
            self.shutdown()
            raise
    
    def _monitor_changes(self):
        """Process binlog events and trigger handlers"""
        logger.info("Monitoring database changes...")
        
        try:
            for binlogevent in self.stream:
                if not self.running:
                    break
                
                # Save current binlog position for resuming later
                self._save_binlog_position()
                
                # Skip events for tables we're not interested in
                if hasattr(binlogevent, "table") and self.tables and binlogevent.table not in self.tables:
                    continue
                
                # Process the event
                self._process_event(binlogevent)
                
        except Exception as e:
            logger.error(f"Error processing binlog events: {str(e)}")
            self.shutdown()
            raise
            
        finally:
            self.shutdown()
    
    def _process_event(self, binlogevent):
        """Process a single binlog event"""
        schema = getattr(binlogevent, "schema", "")
        table = getattr(binlogevent, "table", "")
        timestamp = datetime.fromtimestamp(binlogevent.timestamp).isoformat()
        
        event_type = type(binlogevent).__name__
        
        if isinstance(binlogevent, WriteRowsEvent):
            for row in binlogevent.rows:
                self._handle_insert(schema, table, timestamp, row)
                
        elif isinstance(binlogevent, UpdateRowsEvent):
            for row in binlogevent.rows:
                self._handle_update(schema, table, timestamp, row)
                
        elif isinstance(binlogevent, DeleteRowsEvent):
            for row in binlogevent.rows:
                self._handle_delete(schema, table, timestamp, row)
                
        elif isinstance(binlogevent, QueryEvent):
            self._handle_query(schema, timestamp, binlogevent.query)
    
    def _handle_insert(self, schema, table, timestamp, row):
        """Handle INSERT operation"""
        logger.info(f"INSERT on {schema}.{table} at {timestamp}")
        logger.debug(f"Values: {json.dumps(row['values'], default=str)}")
        
        # Implement your custom INSERT handler here
        # Example: notify a service, update a cache, etc.
    
    def _handle_update(self, schema, table, timestamp, row):
        """Handle UPDATE operation"""
        logger.info(f"UPDATE on {schema}.{table} at {timestamp}")
        logger.debug(f"Before: {json.dumps(row['before_values'], default=str)}")
        logger.debug(f"After: {json.dumps(row['after_values'], default=str)}")
        
        # Implement your custom UPDATE handler here
        # Example: notify a service, update a cache, etc.
    
    def _handle_delete(self, schema, table, timestamp, row):
        """Handle DELETE operation"""
        logger.info(f"DELETE from {schema}.{table} at {timestamp}")
        logger.debug(f"Values: {json.dumps(row['values'], default=str)}")
        
        # Implement your custom DELETE handler here
        # Example: notify a service, update a cache, etc.
    
    def _handle_query(self, schema, timestamp, query):
        """Handle raw SQL query"""
        if "CREATE TABLE" in query or "ALTER TABLE" in query or "DROP TABLE" in query:
            logger.info(f"DDL QUERY on {schema} at {timestamp}")
            logger.debug(f"Query: {query}")
            
            # Implement your custom query handler here
            # Example: recreate tables in a replica, etc.
    
    def _save_binlog_position(self):
        """Save current binlog file and position for resuming later"""
        if self.stream and self.stream.log_file and self.stream.log_pos:
            # You can store these values in a file or database
            # to resume from this position later
            os.environ['MYSQL_BINLOG_FILE'] = self.stream.log_file
            os.environ['MYSQL_BINLOG_POS'] = str(self.stream.log_pos)
    
    def shutdown(self, signum=None, frame=None):
        """Gracefully shut down the monitor"""
        if self.running:
            logger.info("Shutting down DB Change Monitor...")
            self.running = False
            
            if self.stream:
                self.stream.close()
                self.stream = None
                
            logger.info("DB Change Monitor stopped")


# Example usage
if __name__ == "__main__":
    # Get MySQL connection settings from environment variables
    mysql_settings = {
        "host": os.environ.get("MYSQL_HOST", "localhost"),
        "port": int(os.environ.get("MYSQL_PORT", 3306)),
        "user": os.environ.get("MYSQL_USER", "replica_user"),
        "passwd": os.environ.get("MYSQL_PASSWORD", "password"),
    }
    
    # Optionally specify schemas and tables to monitor
    schemas = [os.environ.get("MYSQL_SCHEMA", "asteriskcdrdb")]
    tables = ["cdr"]  # Add any specific tables to monitor
    
    try:
        monitor = DBChangeMonitor(
            connection_settings=mysql_settings,
            server_id=int(os.environ.get("MYSQL_SERVER_ID", 100)),
            schemas=schemas,
            tables=tables
        )
        
        # Start monitoring
        monitor.start(blocking=True)
        
    except KeyboardInterrupt:
        logger.info("Monitoring stopped by user")
    except Exception as e:
        logger.error(f"Error in DB Change Monitor: {str(e)}") 