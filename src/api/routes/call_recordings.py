import os
import paramiko
import asyncio
import json
from fastapi import APIRouter, HTTPException, Query, Depends
from pydantic import BaseModel
from dotenv import load_dotenv
from fastapi.responses import StreamingResponse
import io
from typing import List, Union, Dict, Any
from api.routes.auth import get_current_user

load_dotenv()

SFTP_HOST = os.getenv("SFTP_HOST")
SFTP_PORT = int(os.getenv("SFTP_PORT", "22"))
SFTP_USERNAME = os.getenv("SFTP_USERNAME")
SFTP_PASSWORD = os.getenv("SFTP_PASSWORD")

router = APIRouter()

class FileDetails(BaseModel):
    full_path: str
    file_name: str
    file_size: int
    last_modified: float  # or str for ISO

class CallDataRequest(BaseModel):
    json_data: Union[str, Dict[str, Any], List[Dict[str, Any]]]

class RecordingPathsResponse(BaseModel):
    recording_paths: List[str]
    count: int

def get_recording_paths(json_data):
    """
    Extract all recordingfile paths from JSON data.
    
    Args:
        json_data: JSON string or dict containing call data
    
    Returns:
        List of full recording file paths (empty strings filtered out)
    """
    # Always stringify first, then parse to ensure consistent handling
    try:
        if isinstance(json_data, str):
            json_string = json_data
        else:
            json_string = json.dumps(json_data)
        
        data = json.loads(json_string)
    except (json.JSONDecodeError, TypeError):
        return []
    
    recording_paths = []
    
    # Handle different data structures
    if isinstance(data, dict):
        # Check if it's a full API response with records
        if "records" in data:
            records = data["records"]
        else:
            # Single record
            records = [data]
    elif isinstance(data, list):
        records = data
    else:
        return []
    
    # Extract recordingfile values
    for record in records:
        if isinstance(record, dict):
            recording_file = record.get("recordingfile", "")
            if recording_file and recording_file.strip():  # Filter out empty strings
                recording_paths.append(recording_file)
    
    return recording_paths

@router.post("/extract-recording-paths", response_model=RecordingPathsResponse)
async def extract_recording_paths(
    request: CallDataRequest,
    current_user: dict = Depends(get_current_user)
):
    """
    Extract all recording file paths from call data JSON.
    
    Args:
        request: JSON data containing call records
    
    Returns:
        List of recording file paths and count
    """
    try:
        recording_paths = get_recording_paths(request.json_data)
        return RecordingPathsResponse(
            recording_paths=recording_paths,
            count=len(recording_paths)
        )
    except Exception as e:
        raise HTTPException(
            status_code=400, 
            detail=f"Error processing JSON data: {str(e)}"
        )

async def async_sftp_connect():
    def connect():
        ssh = paramiko.SSHClient()
        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        ssh.connect(
            SFTP_HOST,
            port=SFTP_PORT,
            username=SFTP_USERNAME,
            password=SFTP_PASSWORD,
            timeout=20
        )
        return ssh.open_sftp(), ssh
    return await asyncio.to_thread(connect)

@router.get("/sftp-file-details", response_model=FileDetails)
async def sftp_file_details(
    full_path: str = Query(..., description="Full path to the file on the SFTP server")
):
    """
    Given a full file path, return details about the file from the SFTP server.
    """
    directory = os.path.dirname(full_path)
    filename = os.path.basename(full_path)

    sftp, ssh = await async_sftp_connect()
    try:
        # List files in the directory
        try:
            files = await asyncio.to_thread(sftp.listdir_attr, directory)
        except FileNotFoundError:
            raise HTTPException(status_code=404, detail="Directory not found on SFTP server")
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"SFTP error: {str(e)}")

        # Search for the file
        for f in files:
            if f.filename == filename:
                return FileDetails(
                    full_path=full_path,
                    file_name=filename,
                    file_size=f.st_size,
                    last_modified=f.st_mtime
                )
        raise HTTPException(status_code=404, detail="File not found in directory")
    finally:
        sftp.close()
        ssh.close()

@router.get("/sftp-stream-audio")
async def sftp_stream_audio(
    full_path: str = Query(..., description="Full path to the file on the SFTP server"),
    current_user: dict = Depends(get_current_user)
):
    """
    Stream an audio file from the SFTP server to the frontend, efficiently.
    """
    sftp, ssh = await async_sftp_connect()
    try:
        remote_file = await asyncio.to_thread(sftp.open, full_path, 'rb')
    except FileNotFoundError:
        sftp.close()
        ssh.close()
        raise HTTPException(status_code=404, detail="File not found on SFTP server")
    except Exception as e:
        sftp.close()
        ssh.close()
        raise HTTPException(status_code=500, detail=f"SFTP error: {str(e)}")

    def file_iterator(file_obj, chunk_size=1024 * 64):
        try:
            while True:
                data = file_obj.read(chunk_size)
                if not data:
                    break
                yield data
        finally:
            file_obj.close()
            sftp.close()
            ssh.close()

    media_type = "audio/wav"
    filename = os.path.basename(full_path)

    return StreamingResponse(
        file_iterator(remote_file),
        media_type=media_type,
        headers={
            "Content-Disposition": f'inline; filename=\"{filename}\"'
        }
    )
