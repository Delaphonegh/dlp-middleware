"""
JSON utilities for handling non-standard JSON serialization
"""
import json
from decimal import Decimal
from datetime import datetime, date
from bson import ObjectId

class CustomJSONEncoder(json.JSONEncoder):
    """
    Custom JSON encoder that handles:
    - Decimal (converts to float)
    - datetime/date (converts to ISO format)
    - ObjectId (converts to string)
    - Other types that might cause issues with JSON serialization
    """
    def default(self, obj):
        if isinstance(obj, Decimal):
            return float(obj)
        if isinstance(obj, (datetime, date)):
            return obj.isoformat()
        if isinstance(obj, ObjectId):
            return str(obj)
        if hasattr(obj, '__dict__'):
            return obj.__dict__
        return super(CustomJSONEncoder, self).default(obj)

def json_serialize(data):
    """
    Serialize data to JSON string using the CustomJSONEncoder.
    Handles Decimal, datetime, date, ObjectId and other complex types.
    
    Args:
        data: The data to serialize to JSON
        
    Returns:
        str: JSON string representation of the data
    """
    return json.dumps(data, cls=CustomJSONEncoder)
    
def json_deserialize(json_string):
    """
    Deserialize JSON string back to Python objects.
    
    Args:
        json_string: The JSON string to deserialize
        
    Returns:
        The deserialized Python object
    """
    return json.loads(json_string) 