# MySQL Database Change Monitor

This utility monitors changes to the MySQL database by reading the binary log (binlog).

## Prerequisites

1. MySQL must have binary logging enabled
2. You need a user with replication privileges

## Setting Up MySQL for Replication

1. Edit your MySQL server configuration file (my.cnf) to enable binary logging:

```
[mysqld]
server-id=1
log_bin=mysql-bin
binlog_format=ROW
binlog_row_image=FULL
```

2. Create a user with replication privileges:

```sql
CREATE USER 'replica_user'@'%' IDENTIFIED BY 'your_secure_password';
GRANT REPLICATION SLAVE, REPLICATION CLIENT ON *.* TO 'replica_user'@'%';
GRANT SELECT ON *.* TO 'replica_user'@'%';
FLUSH PRIVILEGES;
```

3. Restart MySQL server for changes to take effect.

## Configuration

Create a `.env` file with the following settings:

```
# Database replication settings
MYSQL_HOST=197.221.94.195
MYSQL_PORT=3306
MYSQL_USER=replica_user
MYSQL_PASSWORD=your_replica_password
MYSQL_SCHEMA=asteriskcdrdb
MYSQL_SERVER_ID=100

# Binlog position for resuming (will be updated by the monitor)
MYSQL_BINLOG_FILE=
MYSQL_BINLOG_POS=0
```

## Usage

Run the script:

```bash
python db_change_monitor.py
```

## Customizing Event Handlers

The script includes basic handlers for INSERT, UPDATE, DELETE, and DDL queries. 
To implement custom handling logic, modify the following methods in the `DBChangeMonitor` class:

- `_handle_insert(schema, table, timestamp, row)`
- `_handle_update(schema, table, timestamp, row)`
- `_handle_delete(schema, table, timestamp, row)`
- `_handle_query(schema, timestamp, query)`

## Example: Real-time Cache Invalidation

You can use this monitor to invalidate caches when data changes:

```python
def _handle_update(self, schema, table, timestamp, row):
    """Handle UPDATE operation with cache invalidation"""
    logger.info(f"UPDATE on {schema}.{table} at {timestamp}")
    
    if table == "cdr":
        # Get the ID or unique identifier of the updated record
        record_id = row['after_values'].get('uniqueid')
        
        # Invalidate cache for this record
        cache_key = f"cdr:{record_id}"
        redis_client.delete(cache_key)
        
        # Also invalidate any aggregate caches
        redis_client.delete("cdr:dashboard:daily")
```

## Technical Notes

1. The `server_id` must be unique in your MySQL replication topology.
2. The script saves the current binlog position to resume from in case of restart.
3. For production use, consider saving binlog position to a database instead of environment variables.
4. Handle exceptions appropriately for your specific use case. 