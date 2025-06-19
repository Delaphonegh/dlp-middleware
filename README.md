# DLP Middleware

A comprehensive AI-powered middleware solution for call recording processing, transcription, and analysis. This FastAPI-based application provides real-time audio monitoring, AI-driven insights, and a robust API for managing call records and conversations.

## 🌟 Features

### Core Functionality
- **Real-time Audio Monitoring**: Automatically processes new call recordings
- **Multi-Company Support**: Manages multiple companies with individual configurations
- **AI-Powered Transcription**: Uses AssemblyAI for accurate speech-to-text
- **Advanced AI Analysis**: OpenAI-powered topic classification, sentiment analysis, and performance insights
- **Secure Storage**: Google Cloud Storage integration with proper access controls
- **Database Integration**: Supports MySQL (Issabel) and MongoDB
- **Authentication & Authorization**: JWT-based secure access control
- **Rate Limiting**: Built-in API rate limiting with Redis
- **Performance Monitoring**: Request timing and slow query detection
- **Comprehensive Logging**: Structured logging with Langfuse tracing

### API Endpoints
- **Authentication**: User registration, login, password management
- **Conversations**: Chat interface with AI assistants
- **Call Records**: CRUD operations for call data
- **Call Recordings**: Audio file management and processing
- **AI Insights**: Business intelligence and analytics
- **Admin Panel**: Administrative functions and monitoring
- **Health Checks**: System status and diagnostics

## 🏗️ Architecture

```
dlp-middleware/
├── src/
│   ├── api/                    # FastAPI application
│   │   ├── core/              # Core utilities (Redis, tools, agents)
│   │   ├── models/            # Database models
│   │   ├── routes/            # API route handlers
│   │   └── services/          # Business logic services
│   ├── services/              # Background services
│   │   └── audio_monitor_service.py  # Audio processing service
│   ├── utils/                 # Utilities and helpers
│   ├── scripts/               # Utility scripts
│   └── prompts/               # AI prompt templates
├── examples/                  # Usage examples
└── test/                      # Test files
```

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- Redis server
- MongoDB
- MySQL (for Issabel integration)
- Google Cloud Storage account
- AssemblyAI API key
- OpenAI API key
- Groq API key (optional)

### Installation

1. **Clone the repository**
```bash
git clone <repository-url>
cd dlp-middleware
```

2. **Create virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r src/requirements.txt
```

4. **Environment Configuration**
Create a `.env` file in the `src/` directory:

```env
# Database Configuration
MONGODB_URI=mongodb://localhost:27017
MYSQL_HOST=localhost
MYSQL_PORT=3306
MYSQL_USER=""
MYSQL_PASSWORD=""
MYSQL_DATABASE=""

# Redis Configuration
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_PASSWORD=""
REDIS_DB=0

# API Keys
OPENAI_API_KEY=""
ASSEMBLYAI_API_KEY=""
GROQ_API_KEY=""
GOOGLE_APPLICATION_CREDENTIALS=path/to/service-account.json

# Security
JWT_SECRET_KEY=your_jwt_secret
FERNET_KEY=your_fernet_encryption_key

# Langfuse (Optional - for AI tracing)
LANGFUSE_PUBLIC_KEY=""
LANGFUSE_SECRET_KEY=""

# Application Settings
DEBUG=false
LOG_LEVEL=INFO
```

5. **Google Cloud Setup**
   - Create a Google Cloud Storage bucket
   - Download service account credentials JSON
   - Set `GOOGLE_APPLICATION_CREDENTIALS` path

### Running the Application

1. **Start the API server**
```bash
cd src
python main.py
```
or with uvicorn:
```bash
cd src
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

2. **Start the Audio Monitor Service** (optional)
```bash
cd src
python services/audio_monitor_service.py --company-id <your_company_id>
```

3. **Access the API**
   - API Documentation: http://localhost:8000/docs
   - Health Check: http://localhost:8000/

## 📚 API Documentation

### Authentication Endpoints
```bash
POST /auth/register          # User registration
POST /auth/login            # User login
POST /auth/refresh          # Refresh JWT token
POST /auth/forgot-password  # Password reset request
POST /auth/reset-password   # Password reset confirmation
```

### Core Endpoints
```bash
GET  /conversations         # List conversations
POST /conversations         # Create conversation
GET  /conversations/{id}    # Get conversation details

GET  /call-records          # List call records
POST /call-records          # Create call record
GET  /call-records/{id}     # Get call record details

POST /ai-generate           # Generate AI content
POST /ai-insights           # Get AI insights
POST /chat                  # Chat with AI assistant
```

### Admin Endpoints
```bash
GET  /admin/stats           # System statistics
GET  /admin/users           # User management
POST /admin/companies       # Company management
GET  /health               # Health check
```

## 🔧 Configuration

### Redis Configuration
The application uses Redis for:
- Session management
- Rate limiting
- Caching
- Background task queues

### MongoDB Collections
- `companies`: Company configurations
- `users`: User accounts
- `conversations`: Chat conversations
- `call_records`: Call record metadata
- `audio_files`: Audio file information

### Audio Processing Pipeline
1. **Monitor**: Watches for new recording files in Issabel CDR
2. **Download**: Fetches audio files via SFTP
3. **Upload**: Stores files in Google Cloud Storage
4. **Transcribe**: Processes audio with AssemblyAI
5. **Analyze**: Performs AI analysis with OpenAI
6. **Store**: Saves results to MongoDB

## 🛠️ Development

### Running Tests
```bash
cd src
python -m pytest api/tests/
```

### Code Quality
```bash
# Linting
flake8 src/

# Type checking
mypy src/

# Security scanning
bandit -r src/
```

### Database Management
```bash
# Reset MongoDB
python src/reset_mongo.py

# Check Redis connection
python src/check_redis.py

# Flush Redis cache
python src/flush_cache.py
```

## 🔐 Security Features

- **JWT Authentication**: Secure token-based authentication
- **Password Encryption**: Bcrypt hashing with salt
- **Database Encryption**: Fernet encryption for sensitive data
- **CORS Protection**: Configurable cross-origin request handling
- **Rate Limiting**: Redis-based request throttling
- **Input Validation**: Pydantic model validation
- **SQL Injection Prevention**: Parameterized queries

## 📊 Monitoring & Observability

### Logging
- Structured logging with timestamps
- Request/response logging
- Error tracking and alerting
- Performance metrics

### Tracing
- Langfuse integration for AI operation tracing
- Request timing middleware
- Database query monitoring

### Health Checks
```bash
GET /health                 # Basic health check
GET /health/detailed        # Detailed system status
```

## 🚢 Deployment

### Docker Deployment
```bash
# Build image
docker build -t dlp-middleware .

# Run container
docker run -p 8000:8000 --env-file .env dlp-middleware
```

### Production Checklist
- [ ] Set `DEBUG=false` in environment
- [ ] Configure proper CORS origins
- [ ] Set up SSL/TLS certificates
- [ ] Configure log rotation
- [ ] Set up monitoring and alerting
- [ ] Configure backup strategies
- [ ] Review security settings

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🆘 Troubleshooting

### Common Issues

**Connection Issues**
```bash
# Test Redis connection
python src/check_redis.py

# Test MongoDB connection
python -c "from motor.motor_asyncio import AsyncIOMotorClient; import asyncio; asyncio.run(AsyncIOMotorClient('mongodb://localhost:27017').admin.command('ismaster'))"
```

**Audio Processing Issues**
- Verify AssemblyAI API key
- Check Google Cloud Storage permissions
- Ensure SFTP credentials are correct
- Validate audio file formats

**Authentication Issues**
- Verify JWT secret configuration
- Check password encryption settings
- Validate user permissions

### Support
For support, please open an issue on GitHub or contact the development team.

## 🏷️ Version History

- **v0.1.0**: Initial release with core functionality
- Audio monitoring and processing
- AI-powered transcription and analysis
- Multi-company support
- REST API with authentication

---

**Built with ❤️ using FastAPI, Redis, MongoDB, and AI technologies**