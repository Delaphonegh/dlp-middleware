"""
OpenTelemetry configuration for DLP Middleware
"""
import os
import logging
from typing import Optional
from opentelemetry import trace, metrics
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.resources import Resource
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.instrumentation.requests import RequestsInstrumentor
from opentelemetry.instrumentation.redis import RedisInstrumentor
from opentelemetry.instrumentation.pymongo import PymongoInstrumentor
from opentelemetry.instrumentation.logging import LoggingInstrumentor

# Try to import OTLP exporter (for production)
try:
    from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
    OTLP_AVAILABLE = True
except ImportError:
    OTLP_AVAILABLE = False

# Try to import Jaeger exporter (for development)
try:
    from opentelemetry.exporter.jaeger.thrift import JaegerExporter
    JAEGER_AVAILABLE = True
except ImportError:
    JAEGER_AVAILABLE = False

logger = logging.getLogger(__name__)

def get_resource() -> Resource:
    """Create OpenTelemetry resource with service information"""
    return Resource.create({
        "service.name": os.getenv("OTEL_SERVICE_NAME", "dlp-middleware"),
        "service.version": os.getenv("OTEL_SERVICE_VERSION", "0.1.0"),
        "service.environment": os.getenv("ENVIRONMENT", "development"),
        "service.instance.id": os.getenv("HOSTNAME", "localhost"),
    })

def setup_tracing() -> Optional[TracerProvider]:
    """Configure OpenTelemetry tracing"""
    try:
        # Create tracer provider with resource
        resource = get_resource()
        provider = TracerProvider(resource=resource)
        
        # Determine which exporter to use based on environment
        exporter_type = os.getenv("OTEL_EXPORTER_TYPE", "console").lower()
        
        if exporter_type == "otlp" and OTLP_AVAILABLE:
            # Production OTLP exporter
            otlp_endpoint = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4317")
            exporter = OTLPSpanExporter(endpoint=otlp_endpoint)
            logger.info(f"🔍 Using OTLP exporter: {otlp_endpoint}")
            
        elif exporter_type == "jaeger" and JAEGER_AVAILABLE:
            # Development Jaeger exporter  
            jaeger_host = os.getenv("JAEGER_HOST", "localhost")
            jaeger_port = int(os.getenv("JAEGER_PORT", "14268"))
            exporter = JaegerExporter(
                agent_host_name=jaeger_host,
                agent_port=jaeger_port,
            )
            logger.info(f"🔍 Using Jaeger exporter: {jaeger_host}:{jaeger_port}")
            
        else:
            # Fallback to console exporter
            exporter = ConsoleSpanExporter()
            logger.info("🔍 Using Console exporter (fallback)")
        
        # Add span processor
        provider.add_span_processor(BatchSpanProcessor(exporter))
        
        # Set global tracer provider
        trace.set_tracer_provider(provider)
        
        logger.info("✅ OpenTelemetry tracing configured successfully")
        return provider
        
    except Exception as e:
        logger.error(f"❌ Failed to setup OpenTelemetry tracing: {e}")
        return None

def setup_metrics() -> Optional[MeterProvider]:
    """Configure OpenTelemetry metrics"""
    try:
        # Create meter provider with resource
        resource = get_resource()
        provider = MeterProvider(resource=resource)
        
        # Set global meter provider
        metrics.set_meter_provider(provider)
        
        logger.info("✅ OpenTelemetry metrics configured successfully")
        return provider
        
    except Exception as e:
        logger.error(f"❌ Failed to setup OpenTelemetry metrics: {e}")
        return None

def instrument_libraries():
    """Auto-instrument common libraries"""
    try:
        # Instrument logging
        LoggingInstrumentor().instrument(set_logging_format=True)
        logger.info("✅ Logging instrumentation completed")
        
        # Instrument HTTP requests
        RequestsInstrumentor().instrument()
        logger.info("✅ Requests instrumentation completed")
        
        # Instrument Redis
        RedisInstrumentor().instrument()
        logger.info("✅ Redis instrumentation completed")
        
        # Instrument PyMongo
        PymongoInstrumentor().instrument()
        logger.info("✅ PyMongo instrumentation completed")
        
        logger.info("✅ Auto-instrumentation completed for all libraries")
        
    except Exception as e:
        logger.error(f"❌ Failed to auto-instrument libraries: {e}")
        logger.exception(e)

def instrument_fastapi(app):
    """Instrument FastAPI application"""
    try:
        FastAPIInstrumentor.instrument_app(
            app,
            excluded_urls=os.getenv("OTEL_EXCLUDED_URLS", "/health,/metrics"),
            tracer_provider=trace.get_tracer_provider()
        )
        logger.info("✅ FastAPI instrumentation completed")
        
    except Exception as e:
        logger.error(f"❌ Failed to instrument FastAPI: {e}")

def initialize_telemetry(app=None):
    """Initialize complete OpenTelemetry setup"""
    logger.info("🚀 Initializing OpenTelemetry...")
    
    # Setup tracing and metrics
    setup_tracing()
    setup_metrics()
    
    # Auto-instrument libraries
    instrument_libraries()
    
    # Instrument FastAPI if app is provided
    if app:
        instrument_fastapi(app)
    
    logger.info("🎯 OpenTelemetry initialization complete")

def get_tracer(name: str = __name__):
    """Get a tracer instance for custom spans"""
    return trace.get_tracer(name)

def get_meter(name: str = __name__):
    """Get a meter instance for custom metrics"""
    return metrics.get_meter(name) 