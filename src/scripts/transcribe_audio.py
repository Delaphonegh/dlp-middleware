import os
import paramiko
from pathlib import Path
from openai import OpenAI
from pymongo import MongoClient
from dotenv import load_dotenv
import io
from datetime import datetime
import concurrent.futures
from typing import List, Dict
import time
import json

# Load environment variables
load_dotenv()

# OpenAI Configuration
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

# SFTP Configuration
SFTP_HOST = "197.221.94.195"
SFTP_PORT = 24611
SFTP_USERNAME = "techops"
SFTP_PASSWORD = "TECOPS!@)(#$*&"
BASE_PATH = "/var/spool/asterisk/monitor/2025/"

# MongoDB Configuration
MONGO_URI = "mongodb://admin:delaphone%4001@102.22.15.141:37037/"
DB_NAME = "shellClub"
COLLECTION_NAME = "2025"

# Batch processing configuration
BATCH_SIZE = 5  # Number of files to process in parallel
MAX_RETRIES = 3  # Maximum number of retries for failed transcriptions

def connect_sftp():
    """Establish SFTP connection"""
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(SFTP_HOST, port=SFTP_PORT, username=SFTP_USERNAME, password=SFTP_PASSWORD)
    return ssh.open_sftp()

def normalize_path(recording_path: str) -> str:
    """Normalize the recording path to ensure it's absolute"""
    if recording_path.startswith('/'):
        return recording_path
    return os.path.join(BASE_PATH, recording_path)

def get_recording_files(json_data: Dict) -> List[str]:
    """Extract recording file paths from JSON data"""
    recording_files = []
    
    if isinstance(json_data, str):
        try:
            json_data = json.loads(json_data)
        except json.JSONDecodeError:
            print("Error: Invalid JSON string")
            return []
    
    if "records" in json_data:
        records = json_data["records"]
    elif isinstance(json_data, list):
        records = json_data
    else:
        records = [json_data]
    
    for record in records:
        if isinstance(record, dict) and "recordingfile" in record:
            recording_path = record["recordingfile"]
            if recording_path and recording_path.strip():
                normalized_path = normalize_path(recording_path)
                recording_files.append(normalized_path)
    
    return recording_files

def stream_file_from_sftp(sftp, remote_path):
    """Stream a file from SFTP server to memory"""
    try:
        buffer = io.BytesIO()
        sftp.getfo(remote_path, buffer)
        buffer.seek(0)
        return buffer
    except Exception as e:
        print(f"Error streaming {remote_path}: {e}")
        return None

def transcribe_audio_stream(audio_buffer):
    """Transcribe audio from memory buffer using OpenAI API"""
    try:
        transcription = client.audio.transcriptions.create(
            model="gpt-4o-transcribe",
            file=audio_buffer,
            response_format="text",
            prompt="This is a phone call recording. Please transcribe the conversation accurately. It's a call from a customer to a support agent. It has Ghanaian accent."
        )
        return transcription
    except Exception as e:
        print(f"Error transcribing audio: {e}")
        return None

def save_to_mongodb(file_path, transcription):
    """Save transcription to MongoDB"""
    try:
        client = MongoClient(MONGO_URI)
        db = client[DB_NAME]
        collection = db[COLLECTION_NAME]
        
        document = {
            "file_path": file_path,
            "transcription": transcription,
            "timestamp": datetime.now(),
            "file_name": os.path.basename(file_path)
        }
        
        collection.insert_one(document)
        print(f"Saved transcription for {file_path} to MongoDB")
    except Exception as e:
        print(f"Error saving to MongoDB: {e}")
    finally:
        client.close()

def process_file(sftp, file_path):
    """Process a single file: stream, transcribe, and save"""
    for attempt in range(MAX_RETRIES):
        try:
            # Stream file from SFTP
            audio_buffer = stream_file_from_sftp(sftp, file_path)
            if not audio_buffer:
                continue

            # Transcribe
            transcription = transcribe_audio_stream(audio_buffer)
            if transcription:
                # Save to MongoDB
                save_to_mongodb(file_path, transcription)
                return True

        except Exception as e:
            print(f"Error processing {file_path} (attempt {attempt + 1}/{MAX_RETRIES}): {e}")
            if attempt < MAX_RETRIES - 1:
                time.sleep(2 ** attempt)  # Exponential backoff
            continue
        finally:
            if 'audio_buffer' in locals():
                audio_buffer.close()
    
    return False

def process_batch(sftp, file_batch):
    """Process a batch of files in parallel"""
    with concurrent.futures.ThreadPoolExecutor(max_workers=BATCH_SIZE) as executor:
        futures = [executor.submit(process_file, sftp, file_path) for file_path in file_batch]
        return [future.result() for future in concurrent.futures.as_completed(futures)]

def main():
    # Example JSON data - replace this with your actual JSON data
    json_data = {
        "records": [
            {
                "recordingfile": "/var/spool/asterisk/monitor/2025/05/30/q-11112-+233546335880-20250530-163843-1748623114.2350.wav"
            },
            {
                "recordingfile": "/var/spool/asterisk/monitor/2025/05/30/q-11112-+233546335880-20250530-163432-1748622862.2348.wav"
            }
            # Add more records as needed
        ]
    }
    
    try:
        # Connect to SFTP
        sftp = connect_sftp()
        
        # Get recording files from JSON
        recording_files = get_recording_files(json_data)
        total_files = len(recording_files)
        print(f"Found {total_files} recording files to process")
        
        # Process files in batches
        for i in range(0, total_files, BATCH_SIZE):
            batch = recording_files[i:i + BATCH_SIZE]
            print(f"\nProcessing batch {i//BATCH_SIZE + 1}/{(total_files + BATCH_SIZE - 1)//BATCH_SIZE}")
            print(f"Files {i+1} to {min(i+BATCH_SIZE, total_files)} of {total_files}")
            
            results = process_batch(sftp, batch)
            successful = sum(1 for r in results if r)
            print(f"Successfully processed {successful}/{len(batch)} files in this batch")
            
            # Small delay between batches to avoid overwhelming the API
            if i + BATCH_SIZE < total_files:
                time.sleep(2)
    
    except Exception as e:
        print(f"Error in main process: {e}")
    finally:
        if 'sftp' in locals():
            sftp.close()

if __name__ == "__main__":
    main() 