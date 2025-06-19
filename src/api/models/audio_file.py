from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime

class AudioFileUploadRequest(BaseModel):
    """Request model for uploading audio files"""
    file_paths: List[str] = Field(..., description="List of remote file paths on SFTP server")
    user_id: Optional[str] = Field(None, description="User ID for file organization (uses current user if not provided)")

class AudioFileResult(BaseModel):
    """Result model for individual audio file processing"""
    remote_path: str
    user_id: str
    success: bool
    gcs_url: Optional[str] = None
    error: Optional[str] = None
    processing_time: Optional[float] = None
    file_size: int = 0

class AudioFileBatchResponse(BaseModel):
    """Response model for batch audio file processing"""
    user_id: str
    total_files: int
    successful: int
    failed: int
    files: List[AudioFileResult]
    start_time: str
    end_time: Optional[str] = None
    total_processing_time: Optional[float] = None
    total_size_bytes: int = 0
    task_id: Optional[str] = None

class UserFileInfo(BaseModel):
    """Model for user file information from GCS"""
    name: str
    full_path: str
    size: int
    created: Optional[str] = None
    updated: Optional[str] = None
    gcs_url: str

class AudioFileListResponse(BaseModel):
    """Response model for listing user's audio files"""
    user_id: str
    files: List[UserFileInfo]
    total_files: int
    total_size_bytes: int 