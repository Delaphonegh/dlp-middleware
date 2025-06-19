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
from langfuse import Langfuse
import tempfile

# Load environment variables
load_dotenv()

# OpenAI Configuration
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

# Langfuse Configuration
langfuse = Langfuse(
    public_key=os.getenv('LANGFUSE_PUBLIC_KEY'),
    secret_key=os.getenv('LANGFUSE_SECRET_KEY'),
    host=os.getenv('LANGFUSE_HOST', 'https://cloud.langfuse.com')
)

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

def create_sftp_connection():
    """Create a new SFTP connection with proper error handling"""
    try:
        ssh = paramiko.SSHClient()
        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        ssh.connect(
            SFTP_HOST,
            port=SFTP_PORT,
            username=SFTP_USERNAME,
            password=SFTP_PASSWORD,
            timeout=30,
            banner_timeout=30
        )
        return ssh.open_sftp()
    except Exception as e:
        print(f"Error creating SFTP connection: {e}")
        return None

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
        
        # Verify file format
        if not remote_path.lower().endswith(('.wav', '.mp3', '.flac', '.m4a', '.mp4', '.mpeg', '.mpga', '.oga', '.ogg', '.webm')):
            print(f"Unsupported file format for {remote_path}")
            return None
            
        return buffer
    except Exception as e:
        print(f"Error streaming {remote_path}: {e}")
        return None

def transcribe_audio_stream(audio_buffer, trace):
    """Transcribe audio from memory buffer using OpenAI API"""
    try:
        # Start a generation span
        generation = trace.generation(
            name="audio_transcription",
            model="whisper-1",  # Changed to use whisper-1 model
            input={"prompt": "This is a phone call recording. Please transcribe the conversation accurately. It's a call from a customer to a support agent. It has Ghanaian accent."}
        )
        
        # Create a temporary file to ensure proper format handling
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
            temp_file.write(audio_buffer.getvalue())
            temp_file.flush()
            
            # Open the file in binary read mode
            with open(temp_file.name, 'rb') as audio_file:
                transcription = client.audio.transcriptions.create(
                    model="whisper-1",  # Changed to use whisper-1 model
                    file=audio_file,
                    response_format="text",
                    prompt="This is a phone call recording. Please transcribe the conversation accurately. It's a call from a customer to a support agent. It has Ghanaian accent."
                )
        
        # Clean up the temporary file
        os.unlink(temp_file.name)
        
        generation.update(output=transcription)
        return transcription
    except Exception as e:
        print(f"Error transcribing audio: {e}")
        if 'generation' in locals():
            generation.update(error=str(e))
        return None
    finally:
        if 'temp_file' in locals():
            try:
                os.unlink(temp_file.name)
            except:
                pass

def save_to_mongodb(file_path, transcription, trace):
    """Save transcription to MongoDB"""
    try:
        span = trace.span(name="mongodb_save")
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
        span.update(output={"status": "success", "file": file_path})
        print(f"Saved transcription for {file_path} to MongoDB")
    except Exception as e:
        print(f"Error saving to MongoDB: {e}")
        if 'span' in locals():
            span.update(error=str(e))
    finally:
        client.close()

def process_file(sftp, file_path):
    """Process a single file: stream, transcribe, and save"""
    # Create a trace for this file
    trace = langfuse.trace(
        name="transcriptions",
        metadata={
            "file_path": file_path,
            "file_name": os.path.basename(file_path),
            "timestamp": datetime.now().isoformat()
        }
    )
    
    try:
        for attempt in range(MAX_RETRIES):
            try:
                # Stream file from SFTP
                span = trace.span(name="sftp_stream")
                audio_buffer = stream_file_from_sftp(sftp, file_path)
                if not audio_buffer:
                    span.update(error="Failed to stream file")
                    continue
                span.update(output={"status": "success"})

                # Transcribe
                transcription = transcribe_audio_stream(audio_buffer, trace)
                if transcription:
                    # Save to MongoDB
                    save_to_mongodb(file_path, transcription, trace)
                    trace.update(status="success")
                    return True

            except Exception as e:
                print(f"Error processing {file_path} (attempt {attempt + 1}/{MAX_RETRIES}): {e}")
                if attempt < MAX_RETRIES - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff
                continue
            finally:
                if 'audio_buffer' in locals():
                    audio_buffer.close()
        
        trace.update(status="error", error="All retry attempts failed")
        return False
    
    except Exception as e:
        trace.update(status="error", error=str(e))
        return False

def process_batch(sftp, file_batch):
    """Process a batch of files in parallel"""
    with concurrent.futures.ThreadPoolExecutor(max_workers=BATCH_SIZE) as executor:
        futures = [executor.submit(process_file, sftp, file_path) for file_path in file_batch]
        return [future.result() for future in concurrent.futures.as_completed(futures)]

def main():
    # Load JSON data from the notebook
    json_data = {
        "records": [
            {
                "recordingfile": "/var/spool/asterisk/monitor/2025/05/30/q-11112-+233546335880-20250530-163843-1748623114.2350.wav"
            },
            {
                "recordingfile": "/var/spool/asterisk/monitor/2025/05/30/q-11112-+233546335880-20250530-163432-1748622862.2348.wav"
            },
            {
                "recordingfile": "/var/spool/asterisk/monitor/2025/05/30/q-11111-+233202012345-20250530-154607-1748619955.2343.WAV"
            },
            {
                "recordingfile": "/var/spool/asterisk/monitor/2025/05/30/out-0243159411-1010-20250530-153441-1748619281.2341.wav"
            },
            {
                "recordingfile": "/var/spool/asterisk/monitor/2025/05/30/q-11111-+233243159411-20250530-152954-1748618988.2339.WAV"
            },
            {
                "recordingfile": "out-0244689327-1009-20250530-145904-1748617144.2329.wav"
            },
            {
                "recordingfile": "out-0264689327-1009-20250530-145800-1748617080.2327.wav"
            },
            {
                "recordingfile": "out-0266309021-1009-20250530-145730-1748617050.2325.wav"
            }
        ]
    }
    
    try:
        # Get recording files from JSON
        recording_files = get_recording_files(json_data)
        total_files = len(recording_files)
        print(f"Found {total_files} recording files to process")
        
        # Process files in batches
        for i in range(0, total_files, BATCH_SIZE):
            batch = recording_files[i:i + BATCH_SIZE]
            print(f"\nProcessing batch {i//BATCH_SIZE + 1}/{(total_files + BATCH_SIZE - 1)//BATCH_SIZE}")
            print(f"Files {i+1} to {min(i+BATCH_SIZE, total_files)} of {total_files}")
            
            # Create new SFTP connection for each batch
            sftp = create_sftp_connection()
            if not sftp:
                print("Failed to create SFTP connection, retrying in 5 seconds...")
                time.sleep(5)
                continue
            
            try:
                results = process_batch(sftp, batch)
                successful = sum(1 for r in results if r)
                print(f"Successfully processed {successful}/{len(batch)} files in this batch")
            finally:
                sftp.close()
            
            # Small delay between batches to avoid overwhelming the API
            if i + BATCH_SIZE < total_files:
                time.sleep(2)
    
    except Exception as e:
        print(f"Error in main process: {e}")
    finally:
        # Flush any remaining traces
        langfuse.flush()

if __name__ == "__main__":
    main() 