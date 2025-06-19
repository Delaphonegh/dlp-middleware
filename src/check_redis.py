import os
import sys
import redis
from dotenv import load_dotenv

load_dotenv()

# Redis connection settings
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))
REDIS_DB = int(os.getenv("REDIS_DB", 0))
REDIS_PASSWORD = os.getenv("REDIS_PASSWORD", None)

def main():
    print(f"Connecting to Redis at {REDIS_HOST}:{REDIS_PORT}, DB: {REDIS_DB}")
    
    try:
        # Connect to Redis
        r = redis.Redis(
            host=REDIS_HOST,
            port=REDIS_PORT,
            db=REDIS_DB,
            password=REDIS_PASSWORD,
            decode_responses=True  # Return strings instead of bytes
        )
        
        # Test connection
        if r.ping():
            print("✅ Connected to Redis successfully")
        else:
            print("❌ Failed to ping Redis")
            return
            
        # Check all keys
        print("\n--- All Redis Keys ---")
        all_keys = r.keys("*")
        print(f"Total keys: {len(all_keys)}")
        for key in all_keys:
            key_type = r.type(key)
            if key_type == "string":
                value = r.get(key)
                print(f"Key: {key} (Type: {key_type}), Value: {value[:100]}..." if len(value) > 100 else f"Key: {key} (Type: {key_type}), Value: {value}")
            elif key_type == "hash":
                value = r.hgetall(key)
                print(f"Key: {key} (Type: {key_type}), Value: {value}")
            elif key_type == "list":
                value = r.lrange(key, 0, -1)
                print(f"Key: {key} (Type: {key_type}), Value: {value}")
            elif key_type == "set":
                value = r.smembers(key)
                print(f"Key: {key} (Type: {key_type}), Value: {value}")
            else:
                print(f"Key: {key} (Type: {key_type})")
        
        # Check user-related patterns
        print("\n--- User & Admin Related Keys ---")
        user_patterns = [
            "user:*",
            "*admin*",
            "*user*",
            "cache:*",
            "ratelimit:*",
            "ipratelimit:*"
        ]
        
        for pattern in user_patterns:
            pattern_keys = r.keys(pattern)
            if pattern_keys:
                print(f"\nPattern '{pattern}' keys: {len(pattern_keys)}")
                for key in pattern_keys:
                    key_type = r.type(key)
                    if key_type == "string":
                        value = r.get(key)
                        print(f"Key: {key} (Type: {key_type}), Value: {value[:100]}..." if len(value) > 100 else f"Key: {key} (Type: {key_type}), Value: {value}")
                    elif key_type == "hash":
                        value = r.hgetall(key)
                        print(f"Key: {key} (Type: {key_type}), Value: {value}")
                    else:
                        print(f"Key: {key} (Type: {key_type})")
        
        # Offer to flush
        if all_keys:
            answer = input("\nDo you want to flush all Redis data? (yes/no): ")
            if answer.lower() == "yes":
                r.flushall()
                print("🧹 Redis database flushed successfully")
            else:
                print("Flush operation cancelled")
        
    except redis.ConnectionError as e:
        print(f"❌ Redis connection error: {e}")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main() 