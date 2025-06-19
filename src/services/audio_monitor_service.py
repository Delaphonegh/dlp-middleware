#!/usr/bin/env python3
"""
Background Audio Monitor Service

This service monitors the database for new recording files and automatically processes them:
1. Looks up company data from MongoDB shellClub.companies collection
2. Decrypts and uses company-specific Issabel database credentials
3. Monitors 'recordingfile' column changes in CDR table
4. Downloads files from SFTP server
5. Uploads to GCS bucket (organized by company_id)
6. Transcribes with AssemblyAI
7. Performs OpenAI analysis (topic classification, sentiment, agent performance)
8. Saves results to MongoDB in company-specific collection with Langfuse tracing

Usage:
    python src/services/audio_monitor_service.py --company-id <company_id>
"""

import os
import sys
import asyncio
import time
import hashlib
import logging
import json
from datetime import datetime
from typing import Dict, List, Optional, Set
from dataclasses import dataclass
import argparse
import signal
from pathlib import Path

# Add the src directory to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import pymysql
import paramiko
from google.cloud import storage
from google.oauth2 import service_account
import assemblyai as aai
from motor.motor_asyncio import AsyncIOMotorClient
from dotenv import load_dotenv

# Cryptography for password decryption
from cryptography.fernet import Fernet

# OpenAI and Langfuse imports
import openai
try:
    from langfuse import Langfuse
    from langfuse.openai import openai as langfuse_openai
    LANGFUSE_AVAILABLE = True
except ImportError:
    LANGFUSE_AVAILABLE = False
    print("⚠️ Langfuse not available - tracing will be disabled")

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('audio_monitor.log')
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class CompanyData:
    """Company information from MongoDB"""
    company_id: str
    name: str
    company_code: str
    description: str
    is_active: bool
    issabel_db_host: str
    issabel_db_user: str
    issabel_db_name: str
    issabel_db_port: int
    issabel_db_password: str  # Decrypted password

@dataclass
class MonitorConfig:
    """Configuration for the audio monitor service"""
    company_data: CompanyData
    sftp_config: Dict
    gcs_config: Dict
    assemblyai_api_key: str
    openai_api_key: str
    mongodb_uri: str
    langfuse_public_key: Optional[str] = None
    langfuse_secret_key: Optional[str] = None
    langfuse_host: str = "https://cloud.langfuse.com"
    poll_interval: int = 30
    max_files_per_batch: int = 5
    max_retries: int = 3

class CompanyManager:
    """Manages company data lookup and credential decryption"""
    
    def __init__(self, mongodb_uri: str, fernet_key: bytes):
        self.mongodb_uri = mongodb_uri
        self.fernet = Fernet(fernet_key)
        self.mongo_client = None
        
    async def connect(self):
        """Connect to MongoDB"""
        try:
            self.mongo_client = AsyncIOMotorClient(self.mongodb_uri)
            # Test connection
            await self.mongo_client.admin.command('ismaster')
            logger.info("✅ Connected to MongoDB for company lookup")
            return True
        except Exception as e:
            logger.error(f"❌ Failed to connect to MongoDB: {e}")
            return False
    
    async def get_company_data(self, company_id: str) -> Optional[CompanyData]:
        """Lookup company data and decrypt credentials"""
        try:
            db = self.mongo_client["shellClub"]
            companies_collection = db["companies"]
            
            # Find company by ID
            company_doc = await companies_collection.find_one({"_id": company_id})
            
            if not company_doc:
                logger.error(f"❌ Company with ID '{company_id}' not found")
                return None
            
            # Check if company is active
            if not company_doc.get('is_active', False):
                logger.error(f"❌ Company '{company_doc.get('name', 'Unknown')}' is not active")
                return None
            
            # Decrypt password
            try:
                encrypted_password = company_doc['issabel_db_password']
                logger.info(f"🔐 Attempting to decrypt password for company: {company_doc['name']}")
                
                # Handle different encrypted password formats
                if isinstance(encrypted_password, str):
                    # Try decrypting as string first
                    try:
                        decrypted_password = self.fernet.decrypt(encrypted_password.encode()).decode()
                    except Exception:
                        # If that fails, try decrypting directly (in case it's already bytes)
                        decrypted_password = self.fernet.decrypt(encrypted_password).decode()
                else:
                    # If it's already bytes, decrypt directly
                    decrypted_password = self.fernet.decrypt(encrypted_password).decode()
                
                logger.info("✅ Password decrypted successfully")
                
            except Exception as e:
                logger.error(f"❌ Failed to decrypt Issabel password: {str(e)}")
                logger.error(f"💡 Encrypted password format: {type(encrypted_password)} - {repr(encrypted_password[:50])}...")
                logger.error("💡 Make sure the FERNET_KEY matches the one used to encrypt this password")
                logger.error("💡 You can test decryption with: python scripts/manage_encryption.py decrypt --key YOUR_KEY --encrypted 'encrypted_password'")
                return None
            
            company_data = CompanyData(
                company_id=company_doc['_id'],
                name=company_doc['name'],
                company_code=company_doc['company_code'],
                description=company_doc.get('description', ''),
                is_active=company_doc['is_active'],
                issabel_db_host=company_doc['issabel_db_host'],
                issabel_db_user=company_doc['issabel_db_user'],
                issabel_db_name=company_doc['issabel_db_name'],
                issabel_db_port=company_doc['issabel_db_port'],
                issabel_db_password=decrypted_password
            )
            
            logger.info(f"✅ Found company: {company_data.name} (Code: {company_data.company_code})")
            logger.info(f"🔗 Issabel DB: {company_data.issabel_db_host}:{company_data.issabel_db_port}/{company_data.issabel_db_name}")
            
            return company_data
            
        except Exception as e:
            logger.error(f"❌ Error looking up company data: {e}")
            return None
    
    async def validate_issabel_connection(self, company_data: CompanyData) -> bool:
        """Validate Issabel database connection"""
        try:
            connection = pymysql.connect(
                host=company_data.issabel_db_host,
                port=company_data.issabel_db_port,
                user=company_data.issabel_db_user,
                password=company_data.issabel_db_password,
                database=company_data.issabel_db_name,
                charset='utf8mb4',
                connect_timeout=10
            )
            
            # Test query
            cursor = connection.cursor()
            cursor.execute("SELECT 1")
            cursor.fetchone()
            cursor.close()
            connection.close()
            
            logger.info(f"✅ Issabel database connection validated for {company_data.name}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to connect to Issabel database: {e}")
            return False
    
    async def cleanup(self):
        """Clean up MongoDB connection"""
        if self.mongo_client:
            self.mongo_client.close()
            logger.info("🔌 Company manager MongoDB connection closed")

class DatabaseMonitor:
    """Monitors database changes in the recordingfile column"""
    
    def __init__(self, company_data: CompanyData):
        self.company_data = company_data
        self.connection = None
        self.previous_hashes: Dict[str, str] = {}
        self.previous_values: Dict[str, Set[str]] = {}
        self.shutdown_requested = False
        
    def connect(self):
        """Establish database connection using company-specific credentials"""
        try:
            self.connection = pymysql.connect(
                host=self.company_data.issabel_db_host,
                port=self.company_data.issabel_db_port,
                user=self.company_data.issabel_db_user,
                password=self.company_data.issabel_db_password,
                database=self.company_data.issabel_db_name,
                charset='utf8mb4'
            )
            logger.info(f"✅ Connected to Issabel database: {self.company_data.issabel_db_host}:{self.company_data.issabel_db_port}")
            return True
        except Exception as e:
            logger.error(f"❌ Failed to connect to Issabel database: {e}")
            return False
    
    def disconnect(self):
        """Close database connection"""
        if self.connection:
            self.connection.close()
            logger.info("🔌 Issabel database connection closed")
    
    def get_tables_with_recordingfile(self) -> List[str]:
        """Find all tables with 'recordingfile' column"""
        try:
            cursor = self.connection.cursor()
            
            cursor.execute(f"""
                SELECT TABLE_NAME 
                FROM INFORMATION_SCHEMA.COLUMNS 
                WHERE TABLE_SCHEMA = '{self.company_data.issabel_db_name}' AND COLUMN_NAME = 'recordingfile'
            """)
            
            tables = [row[0] for row in cursor.fetchall()]
            cursor.close()
            return tables
        except Exception as e:
            logger.error(f"❌ Error finding tables with recordingfile: {e}")
            return []
    
    def get_recordingfile_values(self, table: str) -> List[str]:
        """Get all recordingfile values from a table"""
        try:
            cursor = self.connection.cursor()
            cursor.execute(f"SELECT recordingfile FROM {table} WHERE recordingfile IS NOT NULL ORDER BY recordingfile")
            values = [row[0] for row in cursor.fetchall()]
            cursor.close()
            return values
        except Exception as e:
            logger.error(f"❌ Error getting recordingfile values from {table}: {e}")
            return []
    
    def hash_values(self, values: List[str]) -> str:
        """Create hash of recordingfile values"""
        combined = ''.join(sorted(values))
        return hashlib.md5(combined.encode('utf-8')).hexdigest()
    
    def detect_new_files(self) -> List[str]:
        """Detect new recording files across all monitored tables"""
        new_files = []
        
        try:
            tables = self.get_tables_with_recordingfile()
            
            for table in tables:
                current_values = self.get_recordingfile_values(table)
                current_set = set(current_values)
                current_hash = self.hash_values(current_values)
                
                # Check for changes
                previous_hash = self.previous_hashes.get(table, '')
                if current_hash != previous_hash:
                    logger.info(f"🔄 Change detected in table '{table}'")
                    
                    # Find new files
                    previous_set = self.previous_values.get(table, set())
                    new_files_in_table = list(current_set - previous_set)
                    
                    if new_files_in_table:
                        logger.info(f"🆕 Found {len(new_files_in_table)} new files in '{table}':")
                        for file_path in new_files_in_table:
                            logger.info(f"   - {file_path}")
                        new_files.extend(new_files_in_table)
                    
                    # Update stored values
                    self.previous_hashes[table] = current_hash
                    self.previous_values[table] = current_set
                
        except Exception as e:
            logger.error(f"❌ Error detecting changes: {e}")
            # Reconnect on error
            self.connect()
        
        return new_files
    
    def initialize_baseline(self):
        """Initialize baseline hashes for all tables"""
        logger.info("🔍 Initializing baseline for database monitoring...")
        tables = self.get_tables_with_recordingfile()
        
        for table in tables:
            values = self.get_recordingfile_values(table)
            self.previous_hashes[table] = self.hash_values(values)
            self.previous_values[table] = set(values)
            logger.info(f"📊 Table '{table}': {len(values)} existing files")
        
        logger.info(f"✅ Monitoring {len(tables)} tables for recordingfile changes")

class AudioProcessor:
    """Processes audio files: download, upload, transcribe, analyze with OpenAI"""
    
    def __init__(self, config: MonitorConfig):
        self.config = config
        self.gcs_client = None
        self.mongo_client = None
        self.mongo_db = None
        self.mongo_collection = None
        self.openai_client = None
        self.langfuse_client = None
        
    async def initialize_services(self):
        """Initialize external services"""
        try:
            # Initialize GCS client
            if self.config.gcs_config.get('service_account_path'):
                credentials = service_account.Credentials.from_service_account_file(
                    self.config.gcs_config['service_account_path'],
                    scopes=["https://www.googleapis.com/auth/cloud-platform"]
                )
                self.gcs_client = storage.Client(credentials=credentials)
            else:
                self.gcs_client = storage.Client()
            
            # Initialize AssemblyAI
            aai.settings.api_key = self.config.assemblyai_api_key
            
            # Initialize OpenAI client with optional Langfuse tracing
            self.openai_client = self._get_openai_client_with_tracing()
            
            # Initialize Langfuse client if available
            if LANGFUSE_AVAILABLE and self.config.langfuse_public_key and self.config.langfuse_secret_key:
                self.langfuse_client = Langfuse(
                    public_key=self.config.langfuse_public_key,
                    secret_key=self.config.langfuse_secret_key,
                    host=self.config.langfuse_host
                )
                logger.info("🔍 Langfuse tracing initialized")
            
            # Initialize MongoDB with company-specific collection
            self.mongo_client = AsyncIOMotorClient(self.config.mongodb_uri)
            self.mongo_db = self.mongo_client["delaphone_transcriptions"]
            self.mongo_collection = self.mongo_db[self.config.company_data.company_code]
            
            logger.info(f"💾 MongoDB collection: delaphone_transcriptions.{self.config.company_data.company_code}")
            logger.info("☁️ All services initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize services: {e}")
            return False
    
    def _get_openai_client_with_tracing(self):
        """Get OpenAI client with optional Langfuse tracing"""
        if not self.config.openai_api_key:
            raise Exception("OPENAI_API_KEY is required")
        
        if LANGFUSE_AVAILABLE and self.config.langfuse_public_key and self.config.langfuse_secret_key:
            logger.info("🔍 Langfuse tracing enabled for OpenAI API calls")
            return langfuse_openai.OpenAI(api_key=self.config.openai_api_key)
        else:
            logger.info("📝 Using standard OpenAI client (no tracing)")
            return openai.OpenAI(api_key=self.config.openai_api_key)
    
    async def process_file_batch(self, file_paths: List[str]) -> List[Dict]:
        """Process a batch of audio files"""
        logger.info(f"🎵 Processing batch of {len(file_paths)} files")
        results = []
        
        for i, file_path in enumerate(file_paths, 1):
            logger.info(f"📁 Processing file {i}/{len(file_paths)}: {os.path.basename(file_path)}")
            
            try:
                result = await self.process_single_file(file_path)
                results.append(result)
                
                if result['success']:
                    logger.info(f"✅ Successfully processed: {os.path.basename(file_path)}")
                else:
                    logger.error(f"❌ Failed to process: {os.path.basename(file_path)} - {result.get('error')}")
                
                # Small delay between files to avoid overwhelming services
                if i < len(file_paths):
                    await asyncio.sleep(2)
                    
            except Exception as e:
                logger.error(f"❌ Error processing {file_path}: {e}")
                results.append({
                    'file_path': file_path,
                    'success': False,
                    'error': str(e)
                })
        
        successful = len([r for r in results if r['success']])
        logger.info(f"📊 Batch completed: {successful}/{len(file_paths)} files processed successfully")
        return results
    
    async def process_single_file(self, file_path: str) -> Dict:
        """Process a single audio file through the complete pipeline"""
        result = {
            'file_path': file_path,
            'filename': os.path.basename(file_path),
            'success': False,
            'error': None,
            'gcs_url': None,
            'public_url': None,
            'transcript_id': None,
            'mongodb_id': None,
            'processing_time': 0,
            'langfuse_trace_id': None
        }
        
        start_time = time.time()
        trace = None
        
        try:
            # Create Langfuse trace if available
            if self.langfuse_client:
                trace = self.langfuse_client.trace(
                    name="audio-file-processing",
                    metadata={
                        "company_id": self.config.company_data.company_id,
                        "file_path": file_path,
                        "filename": os.path.basename(file_path),
                        "start_time": datetime.now().isoformat()
                    },
                    tags=["audio-monitor", "background-task", self.config.company_data.company_id]
                )
                result['langfuse_trace_id'] = trace.id
                logger.info(f"🔍 Created Langfuse trace: {trace.id}")
            
            # Step 1: Download from SFTP
            logger.info(f"📥 Downloading from SFTP: {file_path}")
            if trace:
                sftp_span = trace.span(name="sftp-download", metadata={"remote_path": file_path})
            
            audio_buffer = await self.download_from_sftp(file_path)
            if not audio_buffer:
                result['error'] = "Failed to download from SFTP"
                if trace:
                    sftp_span.update(level="ERROR", status_message="Download failed")
                return result
            
            if trace:
                sftp_span.update(output={"status": "success", "file_size": len(audio_buffer.getvalue())})
            
            # Step 2: Upload to GCS
            logger.info(f"☁️ Uploading to GCS...")
            if trace:
                gcs_span = trace.span(name="gcs-upload")
            
            gcs_result = await self.upload_to_gcs(audio_buffer, result['filename'])
            if not gcs_result:
                result['error'] = "Failed to upload to GCS"
                if trace:
                    gcs_span.update(level="ERROR", status_message="Upload failed")
                return result
            
            result['gcs_url'] = gcs_result['gcs_url']
            result['public_url'] = gcs_result['public_url']
            
            if trace:
                gcs_span.update(output={
                    "gcs_url": gcs_result['gcs_url'],
                    "public_url": gcs_result['public_url'],
                    "blob_path": gcs_result['blob_path']
                })
            
            # Step 3: Transcribe with AssemblyAI
            logger.info(f"🎙️ Transcribing audio...")
            if trace:
                transcription_span = trace.span(name="assemblyai-transcription")
            
            transcript_result = await self.transcribe_audio(gcs_result['public_url'], trace)
            if not transcript_result:
                result['error'] = "Failed to transcribe audio"
                if trace:
                    transcription_span.update(level="ERROR", status_message="Transcription failed")
                return result
            
            result['transcript_id'] = transcript_result['transcript_id']
            
            if trace:
                transcription_span.update(output={
                    "transcript_id": transcript_result['transcript_id'],
                    "words_count": transcript_result.get('words_count', 0),
                    "speakers_count": transcript_result.get('speakers_count', 0),
                    "audio_duration": transcript_result.get('audio_duration')
                })
            
            # Step 4: OpenAI Analysis
            logger.info(f"🤖 Performing OpenAI analysis...")
            openai_result = await self.analyze_with_openai(transcript_result, trace)
            
            # Step 5: Save to MongoDB
            logger.info(f"💾 Saving to MongoDB...")
            if trace:
                mongo_span = trace.span(name="mongodb-save")
            
            mongo_result = await self.save_to_mongodb(file_path, gcs_result, transcript_result, openai_result)
            if mongo_result:
                result['mongodb_id'] = str(mongo_result.inserted_id)
                
                if trace:
                    mongo_span.update(output={"mongodb_id": str(mongo_result.inserted_id)})
            
            result['success'] = True
            result['processing_time'] = time.time() - start_time
            
            if trace:
                trace.update(
                    output={
                        "success": True,
                        "processing_time": result['processing_time'],
                        "transcript_id": result['transcript_id'],
                        "mongodb_id": result['mongodb_id']
                    }
                )
            
        except Exception as e:
            result['error'] = str(e)
            logger.error(f"❌ Error in processing pipeline: {e}")
            
            if trace:
                trace.update(level="ERROR", status_message=str(e))
        
        return result
    
    async def download_from_sftp(self, remote_path: str):
        """Download file from SFTP server"""
        ssh = None
        sftp = None
        
        try:
            # Create SSH connection
            ssh = paramiko.SSHClient()
            ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            ssh.connect(
                self.config.sftp_config['host'],
                port=self.config.sftp_config['port'],
                username=self.config.sftp_config['username'],
                password=self.config.sftp_config['password'],
                timeout=30
            )
            
            sftp = ssh.open_sftp()
            
            # Download file to memory
            from io import BytesIO
            buffer = BytesIO()
            sftp.getfo(remote_path, buffer)
            buffer.seek(0)
            
            return buffer
            
        except Exception as e:
            logger.error(f"❌ SFTP download error: {e}")
            return None
        finally:
            if sftp:
                sftp.close()
            if ssh:
                ssh.close()
    
    async def upload_to_gcs(self, buffer, filename: str) -> Optional[Dict]:
        """Upload file to Google Cloud Storage"""
        try:
            bucket_name = self.config.gcs_config['bucket_name']
            bucket = self.gcs_client.bucket(bucket_name)
            
            # Create path with company_id organization
            blob_path = f"companies/{self.config.company_data.company_id}/audio_files/{filename}"
            blob = bucket.blob(blob_path)
            
            # Upload file
            buffer.seek(0)
            blob.upload_from_file(buffer, content_type='audio/wav')
            
            # Make blob publicly accessible
            blob.make_public()
            
            return {
                'gcs_url': f"gs://{bucket_name}/{blob_path}",
                'public_url': blob.public_url,
                'blob_path': blob_path
            }
            
        except Exception as e:
            logger.error(f"❌ GCS upload error: {e}")
            return None
    
    async def transcribe_audio(self, public_url: str, trace=None) -> Optional[Dict]:
        """Transcribe audio using AssemblyAI"""
        try:
            # Configure transcription
            config = aai.TranscriptionConfig(
                speaker_labels=True,
                sentiment_analysis=True,
                auto_highlights=True
            )
            
            # Create transcriber and submit job
            transcriber = aai.Transcriber()
            transcript = transcriber.transcribe(public_url, config)
            
            if transcript.status == aai.TranscriptStatus.error:
                logger.error(f"❌ Transcription failed: {transcript.error}")
                return None
            
            # Process utterances
            utterances = []
            speakers_in_transcript = set()
            if hasattr(transcript, 'utterances') and transcript.utterances:
                for utterance in transcript.utterances:
                    speakers_in_transcript.add(utterance.speaker)
                    utterances.append({
                        'speaker': utterance.speaker,
                        'text': utterance.text,
                        'start': utterance.start,
                        'end': utterance.end,
                        'confidence': utterance.confidence
                    })
            
            # Process sentiment analysis
            sentiment_results = []
            if hasattr(transcript, 'sentiment_analysis') and transcript.sentiment_analysis:
                for sentiment in transcript.sentiment_analysis:
                    sentiment_results.append({
                        'text': sentiment.text,
                        'sentiment': sentiment.sentiment,
                        'confidence': sentiment.confidence,
                        'start': sentiment.start,
                        'end': sentiment.end
                    })
            
            return {
                'transcript_id': transcript.id,
                'text': transcript.text,
                'utterances': utterances,
                'sentiment_analysis': sentiment_results,
                'confidence': transcript.confidence,
                'audio_duration': transcript.audio_duration,
                'status': str(transcript.status),
                'speakers_count': len(speakers_in_transcript),
                'words_count': len(transcript.text.split()) if transcript.text else 0
            }
            
        except Exception as e:
            logger.error(f"❌ Transcription error: {e}")
            return None
    
    async def analyze_with_openai(self, transcript_result: Dict, trace=None) -> Dict:
        """Perform OpenAI analysis on the transcript"""
        try:
            if trace:
                openai_span = trace.span(name="openai-analysis")
            
            # Define custom tag categories
            tag_list = {
                'Account Issues': 'Problems related to user accounts, such as login difficulties or account access.',
                'Technical Support': 'Inquiries regarding software or hardware functionality and troubleshooting.',
                'Billing and Payments': 'Questions or problems about invoices, payments, or subscription plans.',
                'Product Inquiry': 'Requests for information about product features, capabilities, or availability.',
                'Service Disruption': 'Reports of outages or interruptions in service performance or availability.'
            }
            
            speakers_count = transcript_result.get('speakers_count', 0)
            transcript_text = transcript_result.get('text', '')
            
            # Build enhanced prompt based on speaker count
            if speakers_count >= 2:
                # Multi-party conversation - include sentiment analysis and agent performance
                enhanced_prompt = f"""
You are a helpful assistant designed to analyze call center conversations with focus on agent performance, customer experience, and call completion quality.

TRANSCRIPT TO ANALYZE:
{transcript_text}

TASK 1 - TOPIC CLASSIFICATION:
I will give you a list of topics and definitions. Select the most relevant topic from the list.

{tag_list}

TASK 2 - AGENT & CUSTOMER IDENTIFICATION + PERFORMANCE ANALYSIS:
Since this is a call center conversation, identify who is the CALL CENTER AGENT and who is the CUSTOMER:
- Analyze the conversation flow and speaking patterns
- The agent typically greets first, asks how they can help, follows procedures
- The customer typically has a problem, question, or request

Then evaluate the AGENT'S PERFORMANCE using the HEAT model:
- H (HALT): Did the agent listen actively and acknowledge the customer's concerns?
- E (EMPATHY): Did the agent show understanding and empathy toward the customer?
- A (APOLOGIZE): Did the agent apologize when appropriate for issues or inconvenience?
- T (TAKE ACTION): Did the agent take concrete steps to resolve the customer's issue or was the agent's response helpful?

Rate each HEAT component with both:
- Categorical: EXCELLENT, GOOD, FAIR, or POOR
- Numeric: Score from 1-10 (1=Very Poor, 3=Poor, 5=Fair, 7=Good, 9=Excellent, 10=Outstanding)

Rate the agent's overall performance as: EXCELLENT, GOOD, FAIR, or POOR
Calculate overall numeric score as average of all HEAT scores (1-10 scale)

TASK 3 - SENTIMENT ANALYSIS:
Analyze the sentiment of both parties:
- CUSTOMER SENTIMENT: How did the customer feel throughout the conversation?
- AGENT SENTIMENT: How professional and positive was the agent's demeanor?
- Rate each as: POSITIVE, NEUTRAL, or NEGATIVE
- Provide explanations for each sentiment rating

TASK 4 - CALL COMPLETION ANALYSIS:
Analyze if this call ended naturally or was truncated/incomplete:
- COMPLETE: Natural conversation flow with proper greeting, issue resolution, and closure
- TRUNCATED: Conversation ended abruptly, technical issues, unresolved customer issue, sudden disconnection
- Look for signs like: repeated "hello?", "can you hear me?", mid-sentence cutoffs, no proper closure, unresolved customer concerns

TASK 5 - CALL SUMMARY:
Generate a concise summary of the conversation that includes:
- What the customer was calling about (main issue/request)
- Key points discussed during the call
- What actions were taken or solutions provided
- Outcome of the call (resolved, escalated, requires follow-up, etc.)
- Keep the summary between 2-4 sentences and focus on factual information

Return your response in this exact JSON format:
{{
    "topic": "selected_topic_name",
    "agent_performance": {{
        "agent_identified": "Speaker A/Speaker B/Unknown",
        "customer_identified": "Speaker A/Speaker B/Unknown", 
        "heat_model_analysis": {{
            "halt_score": "EXCELLENT/GOOD/FAIR/POOR",
            "halt_numeric": "1-10 numeric score",
            "empathy_score": "EXCELLENT/GOOD/FAIR/POOR",
            "empathy_numeric": "1-10 numeric score",
            "apologize_score": "EXCELLENT/GOOD/FAIR/POOR or N/A",
            "apologize_numeric": "1-10 numeric score or N/A",
            "take_action_score": "EXCELLENT/GOOD/FAIR/POOR",
            "take_action_numeric": "1-10 numeric score"
        }},
        "overall_performance": "EXCELLENT/GOOD/FAIR/POOR",
        "overall_numeric_score": "1-10 average of all HEAT scores",
        "performance_explanation": "detailed explanation of agent performance"
    }},
    "sentiment_analysis": {{
        "customer_sentiment": "POSITIVE/NEUTRAL/NEGATIVE",
        "customer_explanation": "brief explanation of customer's emotional state",
        "agent_sentiment": "POSITIVE/NEUTRAL/NEGATIVE",
        "agent_explanation": "brief explanation of agent's professional demeanor"
    }},
    "call_completion": {{
        "status": "COMPLETE/TRUNCATED",
        "explanation": "brief explanation of why the call is considered complete or truncated"
    }},
    "summary": "2-4 sentence summary of the call including main issue, key points, actions taken, and outcome"
}}
"""
            else:
                # Single speaker - only topic classification and call completion
                enhanced_prompt = f"""
You are a helpful assistant designed to analyze audio content with topic tags and call completion quality.

TRANSCRIPT TO ANALYZE:
{transcript_text}

TASK 1 - TOPIC CLASSIFICATION:
I will give you a list of topics and definitions. Select the most relevant topic from the list.

{tag_list}

TASK 2 - CALL COMPLETION ANALYSIS:
Analyze if this call ended naturally or was truncated/incomplete:
- COMPLETE: Natural flow with proper beginning and ending
- TRUNCATED: Abrupt ending, technical issues, incomplete message, sudden disconnection
- Look for signs like: mid-sentence cutoffs, incomplete thoughts, no proper closure

TASK 3 - AUDIO SUMMARY:
Generate a concise summary of the audio content that includes:
- Main topic or purpose of the audio
- Key information conveyed
- Overall context or setting
- Keep the summary between 1-3 sentences and focus on factual information

Return your response in this exact JSON format:
{{
    "topic": "selected_topic_name",
    "sentiment_analysis": null,
    "call_completion": {{
        "status": "COMPLETE/TRUNCATED",
        "explanation": "brief explanation of why the call is considered complete or truncated"
    }},
    "summary": "1-4 sentence summary of the audio content including main topic and key information"
}}
"""
            
            # Prepare metadata for Langfuse tracing
            trace_metadata = {
                "company_id": self.config.company_data.company_id,
                "transcript_id": transcript_result.get('transcript_id'),
                "audio_duration": transcript_result.get('audio_duration'),
                "speakers_count": speakers_count,
                "words_count": transcript_result.get('words_count', 0),
                "analysis_type": "multi_party" if speakers_count >= 2 else "single_party"
            }
            
            # Call OpenAI API with Langfuse tracing
            if trace:
                # Create generation within the trace
                generation = trace.generation(
                    name="openai-gpt-analysis",
                    model="gpt-3.5-turbo",
                    input=[
                        {"role": "system", "content": "You are a helpful assistant that analyzes transcripts and returns JSON responses."},
                        {"role": "user", "content": enhanced_prompt}
                    ],
                    metadata={
                        "temperature": 0.1,
                        "max_tokens": 700,
                        "prompt_type": "multi_party" if speakers_count >= 2 else "single_party",
                        "transcript_length": len(transcript_text)
                    }
                )
            
            response = self.openai_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that analyzes transcripts and returns JSON responses."},
                    {"role": "user", "content": enhanced_prompt}
                ],
                temperature=0.1,
                max_tokens=700
            )
            
            # Update generation with response
            if trace:
                generation.end(
                    output=response.choices[0].message.content,
                    usage={
                        "input": response.usage.prompt_tokens,
                        "output": response.usage.completion_tokens,
                        "total": response.usage.total_tokens
                    }
                )
            
            openai_response_text = response.choices[0].message.content.strip()
            
            # Parse OpenAI response
            try:
                openai_results = json.loads(openai_response_text)
                logger.info(f"🤖 OpenAI Analysis - Topic: {openai_results.get('topic', 'Unknown')}")
                
                if openai_results.get('agent_performance'):
                    agent_perf = openai_results['agent_performance']
                    heat_analysis = agent_perf.get('heat_model_analysis', {})
                    logger.info(f"👨‍💼 Agent Performance: {agent_perf.get('overall_performance', 'Unknown')} (Score: {agent_perf.get('overall_numeric_score', 'N/A')}/10)")
                
                if openai_results.get('sentiment_analysis'):
                    sentiment = openai_results['sentiment_analysis']
                    logger.info(f"😊 Customer Sentiment: {sentiment.get('customer_sentiment', 'N/A')}")
                    logger.info(f"🎯 Agent Sentiment: {sentiment.get('agent_sentiment', 'N/A')}")
                
                if openai_results.get('call_completion'):
                    logger.info(f"📞 Call Completion: {openai_results['call_completion'].get('status', 'Unknown')}")
                
                if openai_results.get('summary'):
                    # Log first 100 characters of summary
                    summary_preview = openai_results['summary'][:100] + "..." if len(openai_results['summary']) > 100 else openai_results['summary']
                    logger.info(f"📋 Summary Generated: {summary_preview}")
                
                if trace:
                    openai_span.update(output=openai_results)
                
                return openai_results
                
            except json.JSONDecodeError:
                logger.warning(f"⚠️ Could not parse OpenAI response as JSON: {openai_response_text}")
                return {
                    'topic': openai_response_text,
                    'agent_performance': None,
                    'sentiment_analysis': None,
                    'call_completion': None,
                    'summary': None,
                    'raw_response': openai_response_text
                }
                
        except Exception as e:
            logger.error(f"❌ OpenAI analysis failed: {e}")
            if trace:
                openai_span.update(level="ERROR", status_message=str(e))
            return {
                'error': str(e),
                'topic': None,
                'agent_performance': None,
                'sentiment_analysis': None,
                'call_completion': None,
                'summary': None
            }
    
    async def save_to_mongodb(self, file_path: str, gcs_result: Dict, transcript_result: Dict, openai_result: Dict):
        """Save transcription results to MongoDB in company-specific collection"""
        try:
            document = {
                'company_id': self.config.company_data.company_id,
                'company_code': self.config.company_data.company_code,
                'company_name': self.config.company_data.name,
                'file_path': file_path,
                'filename': os.path.basename(file_path),
                'gcs_url': gcs_result['gcs_url'],
                'public_url': gcs_result['public_url'],
                'blob_path': gcs_result['blob_path'],
                'transcript_id': transcript_result['transcript_id'],
                'text': transcript_result['text'],
                'utterances': transcript_result['utterances'],
                'sentiment_analysis': transcript_result['sentiment_analysis'],
                'confidence': transcript_result['confidence'],
                'audio_duration': transcript_result['audio_duration'],
                'status': transcript_result['status'],
                'speakers_count': transcript_result.get('speakers_count', 0),
                'words_count': transcript_result.get('words_count', 0),
                'topic_detection': {
                    'results': [],  # Empty when IAB categories disabled
                    'summary': {},  # Empty when IAB categories disabled
                    'iab_categories_enabled': False  # Flag to indicate IAB categories are disabled
                },
                'lemur_analysis': {  # Custom OpenAI analysis
                    'custom_topic': openai_result.get('topic'),
                    'agent_performance': openai_result.get('agent_performance'),
                    'sentiment_analysis': openai_result.get('sentiment_analysis'),
                    'call_completion': openai_result.get('call_completion'),
                    'summary': openai_result.get('summary'),
                    'error': openai_result.get('error'),
                    'raw_response': openai_result.get('raw_response')
                },
                'processed_at': datetime.now(),
                'created_at': datetime.now(),
                'language_model': getattr(transcript_result, 'language_model', None),
                'acoustic_model': getattr(transcript_result, 'acoustic_model', None),
                'sentiments_count': len(transcript_result.get('sentiment_analysis', [])),
                'topics_detected': 0  # Will be 0 when IAB categories disabled
            }
            
            result = await self.mongo_collection.insert_one(document)
            logger.info(f"💾 Saved to MongoDB collection '{self.config.company_data.company_code}' with ID: {result.inserted_id}")
            logger.info(f"🎤 Speakers detected: {document['speakers_count']}")
            logger.info(f"📝 Words transcribed: {document['words_count']}")
            return result
            
        except Exception as e:
            logger.error(f"❌ MongoDB save error: {e}")
            return None
    
    async def cleanup(self):
        """Clean up resources"""
        if self.mongo_client:
            self.mongo_client.close()
            logger.info("🔌 MongoDB connection closed")

class AudioMonitorService:
    """Main service that orchestrates database monitoring and audio processing"""
    
    def __init__(self, config: MonitorConfig):
        self.config = config
        self.db_monitor = DatabaseMonitor(config.company_data)
        self.audio_processor = AudioProcessor(config)
        self.shutdown_requested = False
        
        # Set up signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        logger.info(f"🛑 Received signal {signum}, initiating graceful shutdown...")
        self.shutdown_requested = True
    
    async def start(self):
        """Start the monitoring service"""
        company_data = self.config.company_data
        logger.info(f"🚀 Starting Audio Monitor Service for company: {company_data.name} ({company_data.company_code})")
        
        try:
            # Initialize database connection
            if not self.db_monitor.connect():
                logger.error("❌ Failed to connect to Issabel database")
                return
            
            # Initialize audio processing services
            if not await self.audio_processor.initialize_services():
                logger.error("❌ Failed to initialize audio processing services")
                return
            
            # Initialize baseline
            self.db_monitor.initialize_baseline()
            
            logger.info(f"✅ Service started successfully")
            logger.info(f"🏢 Company: {company_data.name} (ID: {company_data.company_id})")
            logger.info(f"📂 Collection: delaphone_transcriptions.{company_data.company_code}")
            logger.info(f"🔄 Monitoring for changes every {self.config.poll_interval} seconds")
            logger.info(f"📁 Max files per batch: {self.config.max_files_per_batch}")
            logger.info(f"🔍 Langfuse tracing: {'Enabled' if LANGFUSE_AVAILABLE and self.config.langfuse_public_key else 'Disabled'}")
            
            # Main monitoring loop
            iteration = 0
            while not self.shutdown_requested:
                iteration += 1
                logger.info(f"🔍 Monitoring iteration {iteration}")
                
                try:
                    # Detect new files
                    new_files = self.db_monitor.detect_new_files()
                    
                    if new_files:
                        logger.info(f"🎵 Found {len(new_files)} new recording files to process")
                        
                        # Process files in batches
                        for i in range(0, len(new_files), self.config.max_files_per_batch):
                            if self.shutdown_requested:
                                break
                                
                            batch = new_files[i:i + self.config.max_files_per_batch]
                            batch_num = (i // self.config.max_files_per_batch) + 1
                            total_batches = (len(new_files) + self.config.max_files_per_batch - 1) // self.config.max_files_per_batch
                            
                            logger.info(f"🔄 Processing batch {batch_num}/{total_batches}")
                            results = await self.audio_processor.process_file_batch(batch)
                            
                            # Log batch results
                            successful = len([r for r in results if r['success']])
                            logger.info(f"✅ Batch {batch_num} completed: {successful}/{len(batch)} files processed successfully")
                            
                            # Log Langfuse trace IDs for successful processing
                            for result in results:
                                if result['success'] and result.get('langfuse_trace_id'):
                                    logger.info(f"🔍 Langfuse trace for {result['filename']}: {result['langfuse_trace_id']}")
                    else:
                        logger.info("😴 No new files detected")
                    
                    # Wait before next check
                    if not self.shutdown_requested:
                        logger.info(f"⏰ Waiting {self.config.poll_interval} seconds until next check...")
                        await asyncio.sleep(self.config.poll_interval)
                
                except Exception as e:
                    logger.error(f"❌ Error in monitoring loop: {e}")
                    await asyncio.sleep(10)  # Short delay before retrying
            
        except Exception as e:
            logger.error(f"❌ Fatal error in service: {e}")
        finally:
            await self.cleanup()
    
    async def cleanup(self):
        """Clean up resources"""
        logger.info("🧹 Cleaning up resources...")
        self.db_monitor.disconnect()
        await self.audio_processor.cleanup()
        logger.info("✅ Cleanup completed")

async def create_config_from_company_data(company_data: CompanyData, args) -> MonitorConfig:
    """Create configuration from company data and environment variables"""
    
    # SFTP configuration
    sftp_config = {
        'host': os.getenv('SFTP_HOST', '197.221.94.195'),
        'port': int(os.getenv('SFTP_PORT', 24611)),
        'username': os.getenv('SFTP_USERNAME', 'techops'),
        'password': os.getenv('SFTP_PASSWORD', 'TECOPS!@)(#$*&')
    }
    
    # GCS configuration
    gcs_config = {
        'bucket_name': os.getenv('GCS_BUCKET_NAME', 'delaphone-audio-files'),
        'service_account_path': os.getenv('GCS_SERVICE_ACCOUNT_PATH', 
                                        '/Users/redeemer/Desktop/Projectx/delaphone-data/src/dashboard-461709-99d6d4dbc31b.json')
    }
    
    # API Keys
    assemblyai_api_key = os.getenv('ASSEMBLY_API_KEY')
    if not assemblyai_api_key:
        raise ValueError("ASSEMBLY_API_KEY environment variable is required")
    
    openai_api_key = os.getenv('OPENAI_API_KEY')
    if not openai_api_key:
        raise ValueError("OPENAI_API_KEY environment variable is required")
    
    # MongoDB configuration
    mongodb_uri = os.getenv('MONGODB_URI', 'mongodb://admin:delaphone%4001@102.22.15.141:37037/')
    
    # Langfuse configuration (optional)
    langfuse_public_key = os.getenv('LANGFUSE_PUBLIC_KEY')
    langfuse_secret_key = os.getenv('LANGFUSE_SECRET_KEY')
    langfuse_host = os.getenv('LANGFUSE_HOST', 'https://cloud.langfuse.com')
    
    return MonitorConfig(
        company_data=company_data,
        sftp_config=sftp_config,
        gcs_config=gcs_config,
        assemblyai_api_key=assemblyai_api_key,
        openai_api_key=openai_api_key,
        mongodb_uri=mongodb_uri,
        langfuse_public_key=langfuse_public_key,
        langfuse_secret_key=langfuse_secret_key,
        langfuse_host=langfuse_host,
        poll_interval=args.poll_interval,
        max_files_per_batch=args.max_files_per_batch,
        max_retries=args.max_retries
    )

async def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='Audio Monitor Service with Company-Specific Configuration')
    parser.add_argument('--company-id', required=True, help='Company ID to lookup in MongoDB')
    parser.add_argument('--poll-interval', type=int, default=30, help='Polling interval in seconds (default: 30)')
    parser.add_argument('--max-files-per-batch', type=int, default=5, help='Maximum files to process per batch (default: 5)')
    parser.add_argument('--max-retries', type=int, default=3, help='Maximum retry attempts (default: 3)')
    
    args = parser.parse_args()
    
    try:
        # Get required environment variables
        mongodb_uri = os.getenv('MONGODB_URI', 'mongodb://admin:delaphone%4001@102.22.15.141:37037/')
        fernet_key_str = os.getenv('FERNET_KEY')
        
        if not fernet_key_str:
            logger.error("❌ FERNET_KEY environment variable is required")
            sys.exit(1)
        
        # Handle Fernet key format - it should be a base64-encoded string
        try:
            # Remove 'b' prefix and quotes if present (e.g., "b'key'" -> "key")
            if fernet_key_str.startswith("b'") and fernet_key_str.endswith("'"):
                fernet_key_str = fernet_key_str[2:-1]
            elif fernet_key_str.startswith('b"') and fernet_key_str.endswith('"'):
                fernet_key_str = fernet_key_str[2:-1]
            
            # Fernet expects the key as bytes, but from a base64-encoded string
            fernet_key = fernet_key_str.encode('utf-8')
        except Exception as e:
            logger.error(f"❌ Invalid FERNET_KEY format: {e}")
            logger.error("💡 FERNET_KEY should be a 32-byte base64-encoded string like: NP5k5bwjyQYDXNPptSPi5Igah4Q9NikkXYhEmf6ipyg=")
            sys.exit(1)
        
        # Initialize company manager
        company_manager = CompanyManager(mongodb_uri, fernet_key)
        
        # Connect to MongoDB
        if not await company_manager.connect():
            logger.error("❌ Failed to connect to MongoDB for company lookup")
            sys.exit(1)
        
        # Get company data
        company_data = await company_manager.get_company_data(args.company_id)
        if not company_data:
            logger.error(f"❌ Failed to get company data for ID: {args.company_id}")
            await company_manager.cleanup()
            sys.exit(1)
        
        # Validate Issabel connection
        if not await company_manager.validate_issabel_connection(company_data):
            logger.error("❌ Failed to validate Issabel database connection")
            await company_manager.cleanup()
            sys.exit(1)
        
        # Create configuration
        config = await create_config_from_company_data(company_data, args)
        
        # Clean up company manager
        await company_manager.cleanup()
        
        # Create and start service
        service = AudioMonitorService(config)
        await service.start()
        
    except KeyboardInterrupt:
        logger.info("🛑 Service stopped by user")
    except Exception as e:
        logger.error(f"❌ Service failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())