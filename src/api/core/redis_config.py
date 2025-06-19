import redis
from fastapi import HTTPException, Request, Depends
from fastapi.responses import JSONResponse
from datetime import datetime, timedelta
import json
import os
from dotenv import load_dotenv
import logging
import time
import ipaddress
from typing import Optional, Dict, Any, Union, Callable, List
from functools import wraps
from .redis_retry import redis_client, with_redis_retry, with_async_redis_retry
from fastapi import status

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

# Rate Limiting Configuration
MAX_ATTEMPTS = 5  # Maximum login attempts before lockout
LOCKOUT_TIME = 300  # Lockout time in seconds (5 minutes)
ATTEMPT_EXPIRY = 3600  # Time to keep track of attempts (1 hour)

# Progressive delay configuration (in seconds)
PROGRESSIVE_DELAYS = {
    1: 0,     # First attempt - no delay
    2: 2,     # Second attempt - 2 second delay
    3: 5,     # Third attempt - 5 second delay
    4: 10,    # Fourth attempt - 10 second delay
    5: 30     # Fifth attempt - 30 second delay
}

# Note: We now use the redis_client from redis_retry.py
# No need to initialize Redis client here

class RateLimiter:
    """
    Rate limiting utility that can limit by IP address or username.
    
    Features:
    - IP-based and username-based rate limiting
    - Progressive delays for failed attempts
    - Lockout after max attempts
    - Detailed logging and history tracking
    - Proper TTL handling with first-attempt expiry setting
    - Forwarded IP and proxy support
    """
    
    @staticmethod
    def get_key(identifier: str, key_type: str, by_ip: bool = False) -> str:
        """
        Generate Redis key for rate limiting
        
        Args:
            identifier: Username or IP address
            key_type: Type of key (attempts, log, etc.)
            by_ip: Whether this is an IP-based key
            
        Returns:
            Redis key string
        """
        prefix = "ipratelimit" if by_ip else "ratelimit"
        return f"{prefix}:{key_type}:{identifier}"

    @staticmethod
    def get_client_ip(request: Request) -> str:
        """
        Extract client IP address with support for reverse proxies
        
        Args:
            request: FastAPI request object
            
        Returns:
            Client IP address
        """
        # First check X-Forwarded-For header (for proxies)
        forwarded_for = request.headers.get("X-Forwarded-For")
        if forwarded_for:
            # Get the first IP in case of multiple proxies
            ip = forwarded_for.split(",")[0].strip()
            try:
                # Validate it's a proper IP address
                ipaddress.ip_address(ip)
                return ip
            except ValueError:
                # If not a valid IP, fall back to client.host
                pass
                
        # Fall back to direct client IP
        return request.client.host if request.client else "unknown"

    @staticmethod
    @with_redis_retry()
    def get_attempt_count(identifier: str, by_ip: bool = False) -> int:
        """
        Get the number of attempts for an identifier
        
        Args:
            identifier: Username or IP address
            by_ip: Whether to use IP-based keys
            
        Returns:
            Number of attempts (0 if no attempts)
        """
        key = RateLimiter.get_key(identifier, "attempts", by_ip)
        count = redis_client.get(key)
        return int(count) if count else 0
        
    @staticmethod
    @with_redis_retry()
    def get_time_to_reset(identifier: str, by_ip: bool = False) -> int:
        """
        Get time remaining until attempt counter resets (in seconds)
        
        Args:
            identifier: Username or IP address
            by_ip: Whether to use IP-based keys
            
        Returns:
            Seconds until reset (0 if no TTL or key doesn't exist)
        """
        key = RateLimiter.get_key(identifier, "attempts", by_ip)
        ttl = redis_client.ttl(key)
        return max(0, ttl)  # Return 0 if ttl is -1 (no expiry) or -2 (key doesn't exist)

    @staticmethod
    @with_redis_retry()
    def record_attempt(identifier: str, request: Request, success: bool = False, by_ip: bool = False, username: str = None) -> None:
        """
        Record an authentication attempt
        
        Args:
            identifier: Username or IP address
            request: FastAPI request object
            success: Whether the attempt was successful
            by_ip: Whether to use IP-based keys
            username: Associated username (for IP-based logging)
        """
        # Don't record attempts if Redis is down
        if not redis_client:
            logger.warning(f"⚠️ Redis not available - unable to record login attempt for {identifier}")
            return
        
        # Get the key for this identifier
        attempts_key = RateLimiter.get_key(identifier, "attempts", by_ip)
        log_key = RateLimiter.get_key(identifier, "log", by_ip)
        
        # Record timestamp and user agent
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        user_agent = request.headers.get("User-Agent", "Unknown")
        client_ip = RateLimiter.get_client_ip(request)
        
        # Create log entry with more detailed information
        log_entry = {
            "timestamp": timestamp,
            "ip": client_ip,
            "success": success,
            "user_agent": user_agent,
        }
        
        # Add associated username if provided (for IP tracking)
        if username and by_ip:
            log_entry["username"] = username
        
        # Add to the log list (capped at 10 entries)
        # Use right push to add to the end, then trim to keep only the most recent 10
        log_str = json.dumps(log_entry)
        redis_client.rpush(log_key, log_str)
        redis_client.ltrim(log_key, -10, -1)
        
        # Set log TTL if not already set
        if redis_client.ttl(log_key) < 0:
            redis_client.expire(log_key, ATTEMPT_EXPIRY)
        
        # Record the attempt counter differently based on success
        if success:
            # For successful logins, we just log but don't increment counter
            log_message = f"🔐 LOGIN ATTEMPT [✅ SUCCESS] | "
            log_message += f"User: {username} | " if username else ""
            log_message += f"IP: {client_ip} | Time: {timestamp}"
            logger.info(log_message)
        else:
            # For failed logins, increment the counter
            current = redis_client.incr(attempts_key)
            
            # Set TTL if it's the first attempt
            if current == 1:
                redis_client.expire(attempts_key, ATTEMPT_EXPIRY)
                logger.info(f"⏱️ Set TTL for {attempts_key} to {ATTEMPT_EXPIRY}s")
            
            # Log the failed attempt
            log_message = f"🔐 LOGIN ATTEMPT [❌ FAILED] | "
            if by_ip:
                log_message += f"IP: {identifier} | "
                if username:
                    log_message += f"IP: {client_ip} | " 
            else:
                log_message += f"User: {identifier} | "
                log_message += f"IP: {client_ip} | "
            log_message += f"Attempt #{current} | Time: {timestamp}"
            logger.info(log_message)
            
            # Also warning log
            id_type = "IP" if by_ip else "User"
            logger.warning(f"⚠️ FAILED LOGIN | {id_type}: {identifier} | IP: {client_ip} | Attempt #{current}/{MAX_ATTEMPTS}")
    
    @staticmethod
    @with_redis_retry()
    def reset_attempts(identifier: str, by_ip: bool = False) -> None:
        """
        Reset the attempt counter for an identifier after successful login
        
        Args:
            identifier: Username or IP address
            by_ip: Whether to use IP-based keys
        """
        attempts_key = RateLimiter.get_key(identifier, "attempts", by_ip)
        redis_client.delete(attempts_key)
    
    @staticmethod
    @with_redis_retry()
    def check_rate_limit(identifier: str, request: Request, by_ip: bool = False, username: str = None) -> Dict[str, Any]:
        """
        Check if an identifier has exceeded rate limits and apply appropriate response
        
        Args:
            identifier: Username or IP address to check
            request: FastAPI request object for logging
            by_ip: Whether this is IP-based rate limiting
            username: Optional username for IP-based limiting
            
        Returns:
            Dict: Rate limit info if not blocked, or raises HTTPException if rate limited
        """
        # Get current attempt count
        attempts = RateLimiter.get_attempt_count(identifier, by_ip)
        
        # Get TTL for the attempts key
        ttl = RateLimiter.get_time_to_reset(identifier, by_ip)
        
        # If no attempts yet, allow the request
        if attempts == 0:
            logger.info(f"✅ RATE LIMIT CHECK PASSED | {'IP' if by_ip else 'User'}: {identifier} | IP: {RateLimiter.get_client_ip(request)} | Attempt: 0/{MAX_ATTEMPTS}")
            return {"attempts": 0, "remaining": MAX_ATTEMPTS}
        
        # Check if we're at max attempts
        if attempts >= MAX_ATTEMPTS:
            # If TTL is 0 or negative, reset the counter
            if ttl <= 0:
                RateLimiter.reset_attempts(identifier, by_ip)
                logger.info(f"✅ RATE LIMIT RESET | {'IP' if by_ip else 'User'}: {identifier} | IP: {RateLimiter.get_client_ip(request)}")
                return {"attempts": 0, "remaining": MAX_ATTEMPTS}
            
            # Otherwise, block the request
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail={
                    "error": "Account temporarily locked",
                "status": "locked",
                    "ttl_seconds": ttl,
                    "ttl_minutes": round(ttl / 60, 2),
                    "message": f"Account is locked. Please try again in {round(ttl / 60, 2)} minutes.",
                    "security_event": "lockout",
                "rate_limit_info": {
                    "identifier": identifier,
                    "identifier_type": "ip" if by_ip else "username",
                        "current_attempts": attempts,
                    "max_attempts": MAX_ATTEMPTS,
                    "attempts_remaining": 0,
                        "expiry_seconds": ttl,
                        "expiry_minutes": round(ttl / 60, 2)
                    }
                }
            )
        
        # Apply progressive delay for failed attempts
        if attempts > 1:
            delay = min(2 ** (attempts - 1), 30)  # Exponential backoff, max 30 seconds
            logger.info(f"⏱️ RATE LIMIT DELAY SET | {'IP' if by_ip else 'User'}: {identifier} | IP: {RateLimiter.get_client_ip(request)} | Attempt #{attempts}/{MAX_ATTEMPTS} | Delay: {delay}s | Attempts remaining: {MAX_ATTEMPTS - attempts}")
            
            # Check if we need to wait
            last_attempt_key = RateLimiter.get_key(identifier, "last_attempt", by_ip)
            last_attempt = redis_client.get(last_attempt_key)
            
            if last_attempt:
                last_attempt_time = float(last_attempt)
                time_since_last = time.time() - last_attempt_time
                
                if time_since_last < delay:
                    remaining_delay = delay - time_since_last
                    logger.info(f"⏱️ RATE LIMIT DELAY ACTIVE | {'IP' if by_ip else 'User'}: {identifier} | IP: {RateLimiter.get_client_ip(request)} | Remaining delay: {int(remaining_delay)}s")
                    
                    raise HTTPException(
                        status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                        detail={
                "error": "Please wait before trying again",
                "status": "delayed",
                            "delay": int(remaining_delay),
                            "attempts_remaining": MAX_ATTEMPTS - attempts,
                            "message": f"Please wait {int(remaining_delay)} seconds before trying again. {MAX_ATTEMPTS - attempts} attempts remaining before lockout.",
                "security_event": "progressive_delay",
                "rate_limit_info": {
                    "identifier": identifier,
                    "identifier_type": "ip" if by_ip else "username",
                                "current_attempts": attempts,
                    "max_attempts": MAX_ATTEMPTS,
                                "attempts_remaining": MAX_ATTEMPTS - attempts,
                    "expiry_seconds": ttl
                }
            }
                    )
        
        # Update last attempt timestamp
        redis_client.set(RateLimiter.get_key(identifier, "last_attempt", by_ip), str(time.time()))
        
        # Log partial rate limit info
        logger.info(f"✅ RATE LIMIT PARTIAL | {'IP' if by_ip else 'User'}: {identifier} | IP: {RateLimiter.get_client_ip(request)} | Attempt: {attempts}/{MAX_ATTEMPTS} | Remaining: {MAX_ATTEMPTS - attempts}")
        
        return {
            "attempts": attempts,
            "remaining": MAX_ATTEMPTS - attempts
        }

# Create reusable dependency for FastAPI
def ip_rate_limit(max_attempts: int = MAX_ATTEMPTS) -> Callable:
    """
    FastAPI dependency for IP-based rate limiting
    
    Usage:
        @app.post("/login")
        async def login(request: Request, rate_limit: dict = Depends(ip_rate_limit())):
            # Your login logic here
    
    Args:
        max_attempts: Override the default max attempts
        
    Returns:
        Dependency function for FastAPI
    """
    def dependency(request: Request) -> Dict[str, Any]:
        client_ip = RateLimiter.get_client_ip(request)
        return RateLimiter.check_rate_limit(client_ip, request, by_ip=True)
    
    return dependency

# Create reusable decorator for Flask or other frameworks
def limit_by_ip(func: Callable) -> Callable:
    """
    Decorator for IP-based rate limiting in Flask or other frameworks
    
    Usage:
        @app.route("/login", methods=["POST"])
        @limit_by_ip
        def login():
            # Your login logic here
    
    Args:
        func: The function to decorate
        
    Returns:
        Decorated function
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        request = kwargs.get("request")
        if not request:
            # Try to find request in args
            for arg in args:
                if hasattr(arg, "client") and hasattr(arg, "headers"):
                    request = arg
                    break
        
        if not request:
            logger.error("❌ Request object not found in function arguments")
            return func(*args, **kwargs)
        
        client_ip = RateLimiter.get_client_ip(request)
        
        try:
            RateLimiter.check_rate_limit(client_ip, request, by_ip=True)
            return func(*args, **kwargs)
        except HTTPException as e:
            # Convert to appropriate response for the framework
            response_data = e.detail
            response_data["status_code"] = e.status_code
            # For FastAPI, return the HTTPException
            # For Flask or other frameworks, you might need a different response
            return JSONResponse(
                status_code=e.status_code,
                content=response_data
            )
    
    return wrapper 