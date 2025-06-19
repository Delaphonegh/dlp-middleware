from fastapi import FastAPI, Request, status
import logging
import time
from fastapi.middleware.cors import CORSMiddleware
import os
import asyncio
from dotenv import load_dotenv
from api.routes.auth import router as auth_router
from api.routes.conversations import router as conversations_router
from api.routes.admin import router as admin_router
from api.routes.call_records import router as call_records_router
from api.routes.chat import router as chat_router
from api.routes.ai_generate import router as ai_generate_router
from api.routes.call_recordings import router as call_recordings_router
from api.routes.ai_insights import router as ai_insights_router

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

app = FastAPI(
    title="Delaphone API",
    description="API for Delaphone application",
    version="0.1.0"
)

# CORS configuration
origins = [
    "http://localhost:3000",  # Frontend development server
    "https://app.delaphone.com"  # Production frontend
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Performance monitoring middleware
@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(process_time)
    
    # Log slow requests (over 1 second)
    if process_time > 1:
        logger.warning(f"Slow request: {request.method} {request.url.path} - {process_time:.4f} seconds")
    return response

# Include routers
app.include_router(auth_router)
app.include_router(conversations_router)
app.include_router(admin_router)
app.include_router(call_records_router)
app.include_router(chat_router)
app.include_router(ai_generate_router)
app.include_router(call_recordings_router)
app.include_router(ai_insights_router)

@app.get("/")
async def root():
    return {
        "message": "Welcome to DLP AI API",
        "version": "0.1.0",
        "status": "online",
        "documentation": "/docs"
    }

# Handle Uncaught exceptions
@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):
    logger.error(f"Uncaught exception: {str(exc)}")
    logger.exception(exc)
    return {
        "error": "Internal Server Error",
        "detail": str(exc) if os.getenv("DEBUG", "false").lower() == "true" else "An unexpected error occurred"
    }, status.HTTP_500_INTERNAL_SERVER_ERROR 