# src/api/routes/auth.py
from fastapi import APIRouter, HTTPException, status, Depends, Request
from pydantic import BaseModel, EmailStr, Field
from pymongo import MongoClient
from datetime import datetime, timedelta, timezone
import os
from passlib.context import CryptContext
import uuid
import random
import logging
import string
import secrets
from .models import CompanyCreate, CompanyResponse, UserCreate, Token, UserLogin, UserResponse, PasswordReset, PasswordResetRequest, SessionCreate, SessionResponse, RefreshToken, TokenResponse
import jwt
from typing import Optional, List, Dict, Any
import traceback
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from dotenv import load_dotenv
from bson import ObjectId
from api.database import get_database, with_mongodb_retry
from api.models.company import Company
from api.core.redis_config import RateLimiter, ip_rate_limit

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

# Initialize router
router = APIRouter(prefix="/auth", tags=["authentication"])

# JWT settings
SECRET_KEY = os.getenv("JWT_SECRET_KEY")
REFRESH_TOKEN_SECRET_KEY = os.getenv("REFRESH_TOKEN_SECRET_KEY", SECRET_KEY)  # Use same key if not specified
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60
REFRESH_TOKEN_EXPIRE_DAYS = 7

logger.info(f"JWT SECRET_KEY type: {(SECRET_KEY)}") 

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="auth/login")

# MongoDB connection
MONGODB_HOST = os.getenv("MONGODB_HOST") 
MONGODB_PORT = int(os.getenv("MONGODB_PORT"))
DATABASE_NAME_USERS = "users"
client = MongoClient(MONGODB_HOST, MONGODB_PORT)
db = client[DATABASE_NAME_USERS]
users_collection = db["users"]
companies_collection = db["companies"] 
# Add this near your other collection definitions
sessions_collection = db["sessions"]  # For tracking user sessions
conversations_collection = db["conversations"]  # For storing conversation history # Add companies collection

# Make sure DATABASE_NAME is properly defined
if not DATABASE_NAME_USERS or not isinstance(DATABASE_NAME_USERS, str):
    # Fallback to a default name if environment variable is not set
    DATABASE_NAME_USERS = "users"
    
# Create a custom error handler that logs errors and returns nice messages
class APIError(Exception):
    def __init__(self, status_code: int, detail: str, log_error=True):
        self.status_code = status_code
        self.detail = detail
        if log_error:
            logger.error(f"API Error: {detail}")
        super().__init__(self.detail)

# Connect to MongoDB with better error handling
try:
    client = MongoClient(host=MONGODB_HOST, port=MONGODB_PORT, serverSelectionTimeoutMS=5000)
    # Test connection
    client.admin.command('ping')
    logger.info(f"Successfully connected to MongoDB at {MONGODB_HOST}:{MONGODB_PORT}")
    
    db = client[DATABASE_NAME_USERS]
    users_collection = db["users"]
    companies_collection = db["companies"]
except Exception as e:
    error_msg = f"Failed to connect to MongoDB: {str(e)}"
    logger.error(error_msg)
    logger.error(traceback.format_exc())
    # Still define the collections to prevent errors when importing the module
    # They will fail during runtime which we'll handle with try/except
    db = client.get_database(DATABASE_NAME_USERS)
    users_collection = db["users"]
    companies_collection = db["companies"]

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="auth/login")

# Add encryption for sensitive database credentials
def encrypt_db_password(password: str) -> str:
    """Encrypt database password for secure storage"""
    fernet_key = os.getenv("FERNET_KEY")
    if not fernet_key:
        # If no encryption key is set, we fall back to hashing (one-way)
        logger.warning("No FERNET_KEY environment variable set. Falling back to password hashing (not recoverable).")
        return pwd_context.hash(password)
    
    try:
        from cryptography.fernet import Fernet
        
        # Handle different Fernet key formats from environment
        if fernet_key.startswith("b'") and fernet_key.endswith("'"):
            # Remove b' prefix and ' suffix (e.g., "b'key'" -> "key")
            fernet_key = fernet_key[2:-1]
        elif fernet_key.startswith('b"') and fernet_key.endswith('"'):
            # Remove b" prefix and " suffix (e.g., 'b"key"' -> "key")
            fernet_key = fernet_key[2:-1]
        
        f = Fernet(fernet_key.encode())
        return f.encrypt(password.encode()).decode()
    except Exception as e:
        logger.error(f"Error encrypting database password: {str(e)}")
        logger.error(f"FERNET_KEY format: {repr(fernet_key[:50])}...")
        # Fall back to hashing if encryption fails
        return pwd_context.hash(password)

def decrypt_db_password(encrypted_password: str) -> str:
    """Decrypt database password for use"""
    fernet_key = os.getenv("FERNET_KEY")
    if not fernet_key:
        logger.error("Cannot decrypt password: No FERNET_KEY environment variable set.")
        return ""
    
    try:
        from cryptography.fernet import Fernet
        
        # Handle different Fernet key formats from environment
        if fernet_key.startswith("b'") and fernet_key.endswith("'"):
            # Remove b' prefix and ' suffix (e.g., "b'key'" -> "key")
            fernet_key = fernet_key[2:-1]
        elif fernet_key.startswith('b"') and fernet_key.endswith('"'):
            # Remove b" prefix and " suffix (e.g., 'b"key"' -> "key")
            fernet_key = fernet_key[2:-1]
        
        f = Fernet(fernet_key.encode())
        return f.decrypt(encrypted_password.encode()).decode()
    except Exception as e:
        logger.error(f"Error decrypting database password: {str(e)}")
        logger.error(f"FERNET_KEY format: {repr(fernet_key[:50])}...")
        return ""

# ===== IMPORTANT: Define get_current_user before any functions or routes that use it =====
@with_mongodb_retry(max_retries=3)  # Add retry decorator to auth operations
async def get_current_user(token: str = Depends(oauth2_scheme), db = Depends(get_database)):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    
    try:
        # Log the token for debugging (don't do this in production)
        logger.info(f"Token type: {type(token)}, First few chars: {token[:15]}...")
        
        # Decode the JWT token
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        logger.info(f"Token decoded successfully: {payload}")
        
        # Extract user information
        user_id = payload.get("sub")
        if user_id is None:
            logger.error("Token missing 'sub' field")
            raise credentials_exception
        
        # Find the user in the database
        user = await db["users"].find_one({"_id": user_id})
        if user is None:
            # For testing purposes, if the user ID is test_user_id, return a mock user
            if user_id == "test_user_id":
                return {
                    "_id": "test_user_id",
                    "email": "test@example.com",
                    "username": "testuser",
                    "hashed_password": "hashed_password_here",
                    "full_name": "Test User",
                    "company_id": "test_company_id",
                    "company_code": "TEST123",
                    "company_name": "Test Company",
                    "created_at": datetime.now(timezone.utc),
                    "is_active": True,
                    "is_verified": True
                }
            logger.error(f"User with ID {user_id} not found in database")
            raise credentials_exception
        
        logger.info(f"User authenticated: {user['username']}")
        return user
        
    except Exception as e:
        logger.error(f"Error authenticating: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials"
        )
# ===== End of get_current_user definition =====

def get_password_hash(password):
    return pwd_context.hash(password)

# Function to generate company code
def generate_company_code(length=8):
    """Generate a random company code with letters and numbers"""
    characters = string.ascii_uppercase + string.digits
    code = ''.join(random.choice(characters) for _ in range(length))
    
    # Ensure code is unique
    while companies_collection.find_one({"company_code": code}):
        code = ''.join(random.choice(characters) for _ in range(length))
        
    return code

# Modified user registration to accept empty company_code
@router.post("/register", response_model=Dict[str, Any], status_code=status.HTTP_201_CREATED)
@with_mongodb_retry(max_retries=5)  # Add retry decorator to registration
async def register_user(user: UserCreate, db = Depends(get_database)):
    response = {
        "success": False,
        "errors": [],
        "data": None
    }
    
    try:
        # Check if email already exists (case-insensitive)
        import re
        email_pattern = re.compile(f"^{re.escape(user.email)}$", re.IGNORECASE)
        existing_user = await db["users"].find_one({"email": {"$regex": email_pattern}})
        if existing_user:
            response["errors"].append("Email already registered")
            return response
        
        # Initialize company-related variables
        company = None
        company_id = "None"
        company_name = "None"
        is_verified = False
        warnings = []
        
        # Validate company code if provided and not empty
        if user.company_code and user.company_code.strip():  # Check if code exists and is not just whitespace
            company = await db["companies"].find_one({"company_code": user.company_code})
            if not company:
                warnings.append("Invalid company code")
                user.company_code = "None"
            else:
                company_id = str(company["_id"])
                company_name = company["name"]
                is_verified = True  # If company code is provided, auto-verify
        else:
            # Treat empty string same as no company code
            user.company_code = "None"
        
        # Create user
        user_id = str(uuid.uuid4())
        created_at = datetime.utcnow()
        
        # Hash password with error handling
        try:
            hashed_password = get_password_hash(user.password)
        except Exception as e:
            logger.error(f"Password hashing error: {str(e)}")
            response["errors"].append(f"Error processing password: {str(e)}")
            return response
        
        user_data = {
            "_id": user_id,
            "email": user.email,
            "username": user.username,
            "hashed_password": hashed_password,
            "full_name": user.full_name,
            "company_id": company_id,
            "company_code": user.company_code,  # Will be None if empty string was provided
            "company_name": company_name,
            "created_at": created_at,
            "is_active": True,
            "is_verified": is_verified
        }
        
        # Flag for admin verification if not auto-verified
        if not is_verified:
            user_data["verification_requested"] = True
            user_data["verification_requested_at"] = created_at
        
        # Insert into MongoDB
        result = await db["users"].insert_one(user_data)
        logger.info(f"Created new user: {user.username}")
        
        # Prepare success response
        verification_message = "Account created successfully. Your account is pending verification by an administrator. Accounts are typically verified within 24 hours."
        success_message = "Account created and verified successfully."
        
        # Return user data
        response["success"] = True
        response["data"] = {
            "id": user_id,
            "email": user.email,
            "username": user.username,
            "full_name": user.full_name,
            "company_id": company_id,
            "company_name": company_name,
            "created_at": created_at,
            "is_verified": is_verified,
            "verification_required": not is_verified,
            "message": verification_message if not is_verified else success_message
        }
        
        if warnings:
            response["warnings"] = warnings
            
        return response
        
    except Exception as e:
        logger.error(f"Error registering user: {str(e)}")
        logger.error(traceback.format_exc())
        response["errors"].append(f"Registration failed: {str(e)}")
        return response

def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)

# Create access token
def create_access_token(data: dict, expires_delta: timedelta = None):
    try:
        to_encode = data.copy()
        
        # Calculate expiration time
        expire_datetime = datetime.utcnow() + (expires_delta or timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES))
        
        # Convert datetime to Unix timestamp (integer seconds since epoch)
        expire_timestamp = int(expire_datetime.timestamp())
        
        # Add the expiration as an integer timestamp
        to_encode.update({"exp": expire_timestamp})
        
        # Log for debugging
        logger.info(f"Creating JWT with payload: {to_encode}")
        logger.info(f"Expiration timestamp: {expire_timestamp} ({expire_datetime})")
        
        # Create the token
        encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
        
        # Make sure token is returned as a string
        if isinstance(encoded_jwt, bytes):
            encoded_jwt = encoded_jwt.decode('utf-8')
        
        return encoded_jwt, expire_timestamp
    except Exception as e:
        logger.error(f"Error creating JWT token: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error creating authentication token: {str(e)}"
        )

# Create refresh token
def create_refresh_token(data: dict):
    try:
        to_encode = data.copy()
        
        # Calculate expiration time
        expire_datetime = datetime.utcnow() + timedelta(days=REFRESH_TOKEN_EXPIRE_DAYS)
        expire_timestamp = int(expire_datetime.timestamp())
        
        # Add the expiration
        to_encode.update({
            "exp": expire_timestamp,
            "token_type": "refresh"
        })
        
        # Create the token
        encoded_jwt = jwt.encode(to_encode, REFRESH_TOKEN_SECRET_KEY, algorithm=ALGORITHM)
        
        # Make sure token is returned as a string
        if isinstance(encoded_jwt, bytes):
            encoded_jwt = encoded_jwt.decode('utf-8')
        
        return encoded_jwt
    except Exception as e:
        logger.error(f"Error creating refresh token: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error creating refresh token: {str(e)}"
        )

# Validate refresh token
async def validate_refresh_token(token: str, db):
    try:
        # Decode the token
        payload = jwt.decode(token, REFRESH_TOKEN_SECRET_KEY, algorithms=[ALGORITHM])
        
        # Check if it's a refresh token
        if payload.get("token_type") != "refresh":
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token type"
            )
        
        # Get user ID
        user_id = payload.get("sub")
        if not user_id:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token payload"
            )
        
        # Find user
        user = await db["users"].find_one({"_id": user_id})
        if not user:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="User not found"
            )
        
        return user
    except jwt.ExpiredSignatureError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Refresh token has expired"
        )
    except jwt.JWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid refresh token"
        )

@with_mongodb_retry(max_retries=3)
async def get_user_by_email(email: str, db = Depends(get_database)):
    """Get user by email with case-insensitive search"""
    try:
        # Create case-insensitive regex pattern
        import re
        email_pattern = re.compile(f"^{re.escape(email)}$", re.IGNORECASE)
        
        # Find user
        user = await db.users.find_one({"email": {"$regex": email_pattern}})
        return user
    except Exception as e:
        logger.error(f"Error getting user by email: {str(e)}")
        return None

@router.post("/login", response_model=TokenResponse)
async def login(
    request: Request,
    login_data: UserLogin,
    db = Depends(get_database),
    rate_limit: dict = Depends(ip_rate_limit())
):
    """
    Login endpoint with rate limiting and security features.
    """
    try:
        # Get client IP for logging
        client_ip = RateLimiter.get_client_ip(request)
        
        # Check rate limits first
        try:
            # Check IP-based rate limit
            RateLimiter.check_rate_limit(client_ip, request, by_ip=True, username=login_data.email)
            
            # Check username-based rate limit
            RateLimiter.check_rate_limit(login_data.email, request, by_ip=False)
        except HTTPException as e:
            # Record the rate-limited attempt
            RateLimiter.record_attempt(client_ip, request, success=False, by_ip=True, username=login_data.email)
            RateLimiter.record_attempt(login_data.email, request, success=False, by_ip=False)
            raise e
        
        # Get user from database
        user = await get_user_by_email(login_data.email, db)
        if not user:
            # Record failed attempt
            RateLimiter.record_attempt(client_ip, request, success=False, by_ip=True, username=login_data.email)
            RateLimiter.record_attempt(login_data.email, request, success=False, by_ip=False)
            raise HTTPException(
                status_code=401,
                detail="Incorrect email or password"
            )
    
        # Verify password
        if not verify_password(login_data.password, user["hashed_password"]):
            # Record failed attempt
            RateLimiter.record_attempt(client_ip, request, success=False, by_ip=True, username=login_data.email)
            RateLimiter.record_attempt(login_data.email, request, success=False, by_ip=False)
            raise HTTPException(
                status_code=401,
                detail="Incorrect password"
            )
    
        # Check if user is verified
        if not user.get("is_verified", False):
            raise HTTPException(
                status_code=401,
                detail="Account is unverified. Please wait for administrator approval."
            )
        
        # Create access token and refresh token
        access_token, expires_at = create_access_token(
            data={
                "sub": str(user["_id"]),
                "email": user["email"],
                "username": user["username"],
                "company_id": str(user.get("company_id", "")),
                "company_code": user.get("company_code", ""),
                "is_verified": user.get("is_verified", False)
            }
        )
        
        refresh_token = create_refresh_token(
            data={
                "sub": str(user["_id"]),
                "email": user["email"]
            }
        )
        
        # Reset rate limit counters on successful login
        try:
            # Reset IP-based attempts
            await RateLimiter.reset_attempts(client_ip, by_ip=True)
            # Reset username-based attempts
            await RateLimiter.reset_attempts(login_data.email, by_ip=False)
            
            # Record successful login
            RateLimiter.record_attempt(client_ip, request, success=True, by_ip=True, username=login_data.email)
            RateLimiter.record_attempt(login_data.email, request, success=True, by_ip=False)
        except Exception as e:
            logger.error(f"Error resetting rate limit counters: {str(e)}")
            # Continue with login even if rate limit reset fails
    
        # Calculate expires_in
        expires_in = ACCESS_TOKEN_EXPIRE_MINUTES * 60
        
        # Return tokens with user info
        return TokenResponse(
            access_token=access_token,
            token_type="bearer",
            expires_in=expires_in,
            refresh_token=refresh_token,
            user_info={
                "is_verified": user.get("is_verified", False),
                "is_active": user.get("is_active", False),
                "company_code": user.get("company_code", ""),
                "company_id": user.get("company_id", ""),
                "username": user.get("username", "")
            }
        )
        
    except HTTPException as e:
        # Re-raise HTTP exceptions
        raise e
    except Exception as e:
        # Log unexpected errors
        logger.error(f"Login error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="An unexpected error occurred during login"
        )

# Update create_session to use retry decorator
@with_mongodb_retry(max_retries=3)
async def create_session(user_id: str, request: Request, db):
    """Create a new session for a user"""
    session_id = str(uuid.uuid4())
    created_at = datetime.utcnow()
    
    # Get client information
    ip_address = request.client.host if request.client else None
    user_agent = request.headers.get("User-Agent", "Unknown")
    
    session_data = {
        "_id": session_id,
        "user_id": user_id,
        "created_at": created_at,
        "last_active": created_at,
        "ip_address": ip_address,
        "user_agent": user_agent,
        "is_active": True
    }
    
    await db["sessions"].insert_one(session_data)
    logger.info(f"🔑 Created new session {session_id} for user {user_id}")
    
    return session_data

@with_mongodb_retry(max_retries=2)
async def update_session_activity(session_id: str, db):
    """Update the last_active timestamp for a session"""
    await db["sessions"].update_one(
        {"_id": session_id},
        {"$set": {"last_active": datetime.utcnow()}}
    )

@with_mongodb_retry(max_retries=2)
async def end_session(session_id: str, db):
    """Mark a session as inactive"""
    await db["sessions"].update_one(
        {"_id": session_id},
        {"$set": {"is_active": False}}
    )
    logger.info(f"Ended session {session_id}")

@with_mongodb_retry(max_retries=2)
async def get_user_sessions(user_id: str, db):
    """Get all active sessions for a user"""
    return await db["sessions"].find({"user_id": user_id, "is_active": True}).to_list(None)

@router.get("/me", response_model=UserResponse)
@with_mongodb_retry(max_retries=3)
async def get_current_user_info(current_user: dict = Depends(get_current_user)):
    """Get information about the currently authenticated user"""
    try:
        logger.info(f"Getting info for user: {current_user['username']}")
        
        # Make sure to handle potential missing fields
        company_id = current_user.get("company_id", "none")
        company_name = current_user.get("company_name", "none")
        is_verified = current_user.get("is_verified", False)
        
        # Format creation date properly
        created_at = current_user.get("created_at", datetime.utcnow())
        
        return {
            "id": str(current_user["_id"]),
            "email": current_user["email"],
            "username": current_user["username"],
            "full_name": current_user.get("full_name"),
            "company_id": company_id,
            "company_name": company_name,
            "created_at": created_at,
            "is_verified": is_verified
        }
    except Exception as e:
        logger.error(f"Error getting user info: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve user info: {str(e)}"
        )

@router.get("/list-companies", response_model=List[CompanyResponse])
async def list_companies(current_user: dict = Depends(get_current_user), db = Depends(get_database)):
    """List all companies (admin only)"""
    # Check if the user is an admin
    if not current_user.get("is_admin", False):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Only administrators can list all companies"
        )
        
    # Get all active companies
    companies = await db["companies"].find({"is_active": True}).to_list(None)
    
    # Format response
    return [
        {
            "id": str(company["_id"]),
            "name": company["name"],
            "company_code": company["company_code"],
            "description": company.get("description"),
            "address": company.get("address"),
            "phone": company.get("phone"),
            "industry": company.get("industry"),
            
            
            "issabel_db_host": company.get("issabel_db_host"),
            "issabel_db_user": company.get("issabel_db_user"),
            "issabel_db_name": company.get("issabel_db_name"),
            "issabel_db_port": company.get("issabel_db_port"),
            "created_at": company["created_at"]
        }
        for company in companies
    ]

@router.get("/companies/{company_id}", response_model=CompanyResponse)
async def get_company(company_id: str, current_user: dict = Depends(get_current_user), db = Depends(get_database)):
    """Get details of a specific company"""
    # Check if user belongs to the company or is an admin
    is_admin = current_user.get("is_admin", False)
    if current_user.get("company_id") != company_id and not is_admin:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not authorized to access this company"
        )
    
    company = await db["companies"].find_one({"_id": company_id})
    if not company:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Company not found"
        )
    
    return {
        "id": str(company["_id"]),
        "name": company["name"],
        "company_code": company["company_code"],
        "description": company.get("description"),
        "address": company.get("address"),
        "phone": company.get("phone"),
        "industry": company.get("industry"),
        
        
        "issabel_db_host": company.get("issabel_db_host"),
        "issabel_db_user": company.get("issabel_db_user"),
        "issabel_db_name": company.get("issabel_db_name"),
        "issabel_db_port": company.get("issabel_db_port"),
        "created_at": company["created_at"]
    }

# Add a test endpoint to check token
@router.get("/test-token")
async def test_token(request: Request):
    """Endpoint to debug token issues"""
    auth_header = request.headers.get("Authorization")
    
    if not auth_header:
        return {"error": "No Authorization header found"}
    
    try:
        # Extract token
        scheme, token = auth_header.split()
        if scheme.lower() != "bearer":
            return {"error": "Invalid authentication scheme"}
        
        # Try to decode (without validation)
        decoded = jwt.decode(token, options={"verify_signature": False})
        
        # Check if token structure is correct
        return {
            "token_type": "Valid token structure",
            "payload": decoded,
            "exp_valid": datetime.fromtimestamp(decoded["exp"]) > datetime.utcnow() if "exp" in decoded else False,
            "has_user_id": "sub" in decoded
        }
    except Exception as e:
        return {"error": f"Token parsing error: {str(e)}"}


# Generate a reset token
def generate_reset_token():
    """Generate a secure random token for password reset"""
    return secrets.token_urlsafe(32)  # 32 bytes of randomness

# Request password reset endpoint
@router.post("/forgot-password", status_code=status.HTTP_200_OK)
async def forgot_password(request: PasswordResetRequest):
    """Request a password reset by providing email"""
    try:
        # Find user by email
        user = users_collection.find_one({"email": request.email})
        if not user:
            # Don't reveal if email exists or not (security best practice)
            logger.info(f"Password reset requested for non-existent email: {request.email}")
            return {"message": "If your email is registered, you will receive a password reset link"}
        
        # Generate a reset token
        reset_token = generate_reset_token()
        expiration = datetime.utcnow() + timedelta(hours=1)  # Token valid for 1 hour
        
        # Store token in database
        users_collection.update_one(
            {"_id": user["_id"]},
            {
                "$set": {
                    "reset_token": reset_token,
                    "reset_token_expiry": expiration
                }
            }
        )
        
        # In a real application, send an email with the reset link
        # For now, we'll just log it and return it in the response
        reset_link = f"https://yourdomain.com/reset-password?token={reset_token}"
        logger.info(f"Password reset link for {request.email}: {reset_link}")
        
        # For development purposes only - return the token (in production you would email it)
        return {
            "message": "If your email is registered, you will receive a password reset link",
            "dev_reset_link": reset_link  # Remove this in production!
        }
    except Exception as e:
        logger.error(f"Error processing password reset request: {str(e)}")
        logger.error(traceback.format_exc())
        # Still return success to not reveal if email exists
        return {"message": "If your email is registered, you will receive a password reset link"}

# Reset password endpoint
@router.post("/reset-password", status_code=status.HTTP_200_OK)
async def reset_password(reset_data: PasswordReset):
    """Reset password using the token received by email"""
    try:
        # Find user by reset token
        user = users_collection.find_one({
            "reset_token": reset_data.token,
            "reset_token_expiry": {"$gt": datetime.now(timezone.utc)}  # Token must not be expired
        })
        
        if not user:
            raise APIError(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid or expired reset token"
            )
        
        # Hash the new password
        hashed_password = get_password_hash(reset_data.new_password)
        
        # Update user with new password and remove reset token
        users_collection.update_one(
            {"_id": user["_id"]},
            {
                "$set": {
                    "hashed_password": hashed_password
                },
                "$unset": {
                    "reset_token": "",
                    "reset_token_expiry": ""
                }
            }
        )
        
        logger.info(f"Password reset successful for user: {user['username']}")
        
        return {"message": "Password has been reset successfully"}
    except APIError:
        # Re-raise API errors to be caught by the middleware
        raise
    except Exception as e:
        logger.error(f"Error resetting password: {str(e)}")
        logger.error(traceback.format_exc())
        raise APIError(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to reset password"
        )

@router.get("/security-status")
async def get_security_status(current_user: dict = Depends(get_current_user)):
    """Get system security status and rate limiting information (admin only)"""
    # Check if the user is an admin (implement proper admin check in production)
    is_admin = current_user.get("is_admin", False)
    
    if not is_admin:
        # For now, we'll allow any authenticated user to see basic status
        # In production, you should restrict this to admins
        logger.warning(f"Non-admin user '{current_user['username']}' accessed security status")
    
    try:
        # Check Redis connection
        redis_status = "online"
        try:
            ping_result = redis_client.ping() # type: ignore
            if not ping_result:
                redis_status = "degraded"
        except Exception as e:
            redis_status = f"offline: {str(e)}"
        
        # Get all rate limited patterns
        username_pattern = RateLimiter.get_key("*", "attempts")
        ip_pattern = RateLimiter.get_key("*", "attempts", by_ip=True)
        
        username_keys = redis_client.keys(username_pattern) # type: ignore
        ip_keys = redis_client.keys(ip_pattern) # type: ignore
        
        # Get rate limited usernames
        username_limits = []
        for key in username_keys:
            parts = key.split(":")
            if len(parts) >= 3:
                username = parts[2]
                attempts = redis_client.get(key) # type: ignore
                ttl = redis_client.ttl(key) # type: ignore
                
                if attempts and int(attempts) > 0:
                    username_limits.append({
                        "username": username,
                        "attempts": int(attempts),
                        "ttl_seconds": ttl,
                        "status": "locked" if int(attempts) >= MAX_ATTEMPTS else "delayed" # type: ignore
                    })
        
        # Get rate limited IPs
        ip_limits = []
        for key in ip_keys:
            parts = key.split(":")
            if len(parts) >= 3:
                ip = parts[2]
                attempts = redis_client.get(key) # type: ignore
                ttl = redis_client.ttl(key) # type: ignore
                
                if attempts and int(attempts) > 0:
                    ip_limits.append({
                        "ip": ip,
                        "attempts": int(attempts),
                        "ttl_seconds": ttl,
                        "status": "locked" if int(attempts) >= MAX_ATTEMPTS else "delayed" # type: ignore
                    })
        
        # Sort by attempts (descending)
        username_limits.sort(key=lambda x: x["attempts"], reverse=True)
        ip_limits.sort(key=lambda x: x["attempts"], reverse=True)
        
        # Calculate statistics
        locked_usernames = sum(1 for item in username_limits if item["status"] == "locked") 
        locked_ips = sum(1 for item in ip_limits if item["status"] == "locked")
        
        return {
            "timestamp": datetime.utcnow().isoformat(),
            "system_status": {
                "redis": redis_status,
                "rate_limiting": "active" if redis_status == "online" else "degraded"
            },
            "rate_limiting": {
                "config": {
                    "max_attempts": MAX_ATTEMPTS, # type: ignore
                    "lockout_time_seconds": LOCKOUT_TIME, # type: ignore
                    "attempt_expiry_seconds": ATTEMPT_EXPIRY, # type: ignore
                    "progressive_delays": PROGRESSIVE_DELAYS # type: ignore
                },
                "username_limits": username_limits,
                "ip_limits": ip_limits,
                "stats": {
                    "total_limited_usernames": len(username_limits),
                    "total_limited_ips": len(ip_limits),
                    "locked_usernames": locked_usernames,
                    "locked_ips": locked_ips
                }
            }
        }
    except Exception as e:
        logger.error(f"Error getting security status: {str(e)}")
        return {
            "timestamp": datetime.utcnow().isoformat(),
            "system_status": {
                "redis": "error",
                "rate_limiting": "error"
            },
            "error": str(e)
        }

# Endpoint to check email availability
@router.get("/check-email", status_code=status.HTTP_200_OK)
async def check_email_availability(email: str, db = Depends(get_database)):
    """Check if an email is available for registration"""
    try:
        # Check if email already exists (case-insensitive)
        import re
        email_pattern = re.compile(f"^{re.escape(email)}$", re.IGNORECASE)
        existing_user = await db["users"].find_one({"email": {"$regex": email_pattern}})
        
        if existing_user:
            return {
                "available": False,
                "message": "Email is already registered"
            }
        
        return {
            "available": True,
            "message": "Email is available for registration"
        }
    
    except Exception as e:
        logger.error(f"Error checking email availability: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An error occurred while checking email availability"
        )
    
async def get_company_db_connection(company_id: str, current_user: dict, db = Depends(get_database)):
    """
    Helper function to get database connection details with decrypted password
    Only accessible to admins or members of the company
    """
    is_admin = current_user.get("is_admin", False)
    if current_user.get("company_id") != company_id and not is_admin:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not authorized to access this company's database details"
        )
    
    company = await db["companies"].find_one({"_id": company_id})
    if not company:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Company not found"
        )
    
    # If database connection information is not set, return error
    if not company.get("issabel_db_host") or not company.get("issabel_db_user") or not company.get("issabel_db_password"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Database connection information not configured for this company"
        )
    
    # Decrypt password
    encrypted_password = company.get("issabel_db_password", "")
    decrypted_password = decrypt_db_password(encrypted_password)
    print(f"Decrypted password: {decrypted_password}")
    
    if not decrypted_password:
        logger.error(f"Failed to decrypt database password for company {company_id}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve database credentials"
        )
    
    # Return connection details
    return {
        "host": company.get("issabel_db_host"),
        "user": company.get("issabel_db_user"),
        "password": decrypted_password,
        "database": company.get("issabel_db_name"),
        "port": company.get("issabel_db_port", 3306)
    }

@router.put("/companies/{company_id}/db-credentials", status_code=status.HTTP_200_OK)
async def update_company_db_credentials(
    company_id: str,
    issabel_db_host: Optional[str] = None,
    issabel_db_user: Optional[str] = None,
    issabel_db_password: Optional[str] = None,
    issabel_db_name: Optional[str] = None,
    issabel_db_port: Optional[int] = None,
    current_user: dict = Depends(get_current_user),
    db = Depends(get_database)
):
    """Update ISSABEL database credentials for a company (admin only)"""
    # Check if the user is an admin
    if not current_user.get("is_admin", False):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Only administrators can update database credentials"
        )
    
    # Check if company exists
    company = await db["companies"].find_one({"_id": company_id})
    if not company:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Company not found"
        )
    
    # Prepare update data
    update_data = {}
    
    if issabel_db_host is not None:
        update_data["issabel_db_host"] = issabel_db_host
    
    if issabel_db_user is not None:
        update_data["issabel_db_user"] = issabel_db_user
    
    if issabel_db_password is not None:
        update_data["issabel_db_password"] = encrypt_db_password(issabel_db_password)
    
    if issabel_db_name is not None:
        update_data["issabel_db_name"] = issabel_db_name
    
    if issabel_db_port is not None:
        update_data["issabel_db_port"] = issabel_db_port
    
    if not update_data:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No data provided for update"
        )
    
    # Add update metadata
    update_data["updated_by"] = current_user["username"]
    update_data["updated_at"] = datetime.utcnow()
    
    # Update the company
    result = await db["companies"].update_one(
        {"_id": company_id},
        {"$set": update_data}
    )
    
    if result.modified_count == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No changes were made"
        )
    
    logger.info(f"Admin '{current_user['username']}' updated database credentials for company ID: {company_id}")
    
    return {
        "message": "Database credentials updated successfully",
        "company_id": company_id,
        "updated_fields": list(update_data.keys())
    }

@router.get("/companies/{company_id}/test-db-connection", status_code=status.HTTP_200_OK)
async def test_company_db_connection(company_id: str, current_user: dict = Depends(get_current_user), db = Depends(get_database)):
    """Test connection to the ISSABEL database for a company"""
    try:
        # Get connection details with decrypted password
        connection_details = await get_company_db_connection(company_id, current_user, db)
        
        # Try to connect to the database
        import pymysql
        
        conn = None
        try:
            conn = pymysql.connect(
                host=connection_details["host"],
                user=connection_details["user"],
                password=connection_details["password"],
                database=connection_details["database"],
                port=connection_details["port"],
                connect_timeout=5
            )
            
            # Test if we can execute a simple query
            with conn.cursor() as cursor:
                cursor.execute("SELECT 1")
                result = cursor.fetchone()
            
            logger.info(f"Successfully connected to ISSABEL database for company {company_id}")
            
            return {
                "status": "success",
                "message": "Successfully connected to ISSABEL database",
                "connection": {
                    "host": connection_details["host"],
                    "user": connection_details["user"],
                    "database": connection_details["database"],
                    "port": connection_details["port"]
                }
            }
            
        except Exception as e:
            logger.error(f"Failed to connect to ISSABEL database: {str(e)}")
            return {
                "status": "error",
                "message": f"Failed to connect to ISSABEL database: {str(e)}",
                "connection": {
                    "host": connection_details["host"],
                    "user": connection_details["user"],
                    "database": connection_details["database"],
                    "port": connection_details["port"]
                }
            }
        finally:
            if conn:
                conn.close()
    
    except HTTPException:
        # Re-raise HTTP exceptions from get_company_db_connection
        raise
    except Exception as e:
        logger.error(f"Error testing database connection: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error testing database connection: {str(e)}"
        )
    

# Add refresh token endpoint
@router.post("/refresh", response_model=TokenResponse)
async def refresh_token(refresh_token: RefreshToken, db = Depends(get_database)):
    """Get a new access token using a refresh token"""
    try:
        # Validate refresh token and get user
        user = await validate_refresh_token(refresh_token.refresh_token, db)
        
        # Create new access token
        access_token, expires_at = create_access_token(
            data={
                "sub": str(user["_id"]),
                "username": user["username"],
                "email": user["email"],
                "company_id": user.get("company_id", "None"),
                "company_code": user.get("company_code", "None"),
                "session_id": str(uuid.uuid4())  # New session for new access token
            }
        )
        
        # Create new refresh token
        new_refresh_token = create_refresh_token(
            data={
                "sub": str(user["_id"]),
                "username": user["username"]
            }
        )
        
        # Calculate expires_in
        expires_in = expires_at - int(datetime.utcnow().timestamp())
        
        return {
            "access_token": access_token,
            "refresh_token": new_refresh_token,
            "token_type": "bearer",
            "expires_in": expires_in
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error refreshing token: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error refreshing token"
        )

    