"""
Script to flush Redis cache entries
"""
import os
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
            
        # Option to flush a specific pattern or all
        choice = input("Flush options:\n1. All keys\n2. Dashboard keys only\nEnter choice (1/2): ")
        
        if choice == "1":
            # Flush all keys
            count = r.flushdb()
            print(f"🧹 Flushed all Redis keys")
        elif choice == "2":
            # Only flush dashboard keys
            dashboard_keys = r.keys("dashboard:*")
            if dashboard_keys:
                count = r.delete(*dashboard_keys)
                print(f"🧹 Deleted {count} dashboard cache entries")
            else:
                print("No dashboard keys found")
        else:
            print("Invalid choice")
            
    except redis.ConnectionError as e:
        print(f"❌ Redis connection error: {e}")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main() 