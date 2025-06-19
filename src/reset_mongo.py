import os
from pymongo import MongoClient
from dotenv import load_dotenv

load_dotenv()

# MongoDB connection settings
MONGODB_HOST = os.getenv("MONGODB_HOST", "localhost") 
MONGODB_PORT = int(os.getenv("MONGODB_PORT", 27017))
DATABASE_NAME = os.getenv("DATABASE_NAME", "users")  # Get database name from environment variables

def main():
    print(f"Connecting to MongoDB at {MONGODB_HOST}:{MONGODB_PORT}")
    print(f"Using database: {DATABASE_NAME}")
    
    try:
        # Connect to MongoDB
        client = MongoClient(host=MONGODB_HOST, port=MONGODB_PORT, serverSelectionTimeoutMS=5000)
        
        # Test connection
        client.admin.command('ping')
        print("✅ Connected to MongoDB successfully")
        
        # Access the database
        db = client[DATABASE_NAME]
        
        # Print collections and their document counts
        print("\n--- Collections ---")
        for collection_name in db.list_collection_names():
            count = db[collection_name].count_documents({})
            print(f"Collection: {collection_name}, Documents: {count}")
        
        # Ask for confirmation before deleting
        answer = input("\nDo you want to delete all users? (yes/no): ")
        if answer.lower() == "yes":
            # Delete all users
            result = db["users"].delete_many({})
            print(f"🧹 Deleted {result.deleted_count} users")
        else:
            print("Delete operation cancelled")
        
        # Print updated collections
        print("\n--- Updated Collections ---")
        for collection_name in db.list_collection_names():
            count = db[collection_name].count_documents({})
            print(f"Collection: {collection_name}, Documents: {count}")
            
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main() 