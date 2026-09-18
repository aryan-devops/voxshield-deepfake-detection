import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from app.api import health, model, system, analyze
from app.services.model_service import ModelService
from app.config import settings

app = FastAPI(
    title="VoxShield API",
    version="1.0.0",
)


@app.get("/")
def root():
    return {
        "service": "VoxShield",
        "status": "online"
    }


@app.get("/health")
def health():
    return {
        "status": "healthy"
    }
# Setup logging
logging.basicConfig(
    level=getattr(logging, settings.log_level.upper(), logging.INFO),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("voxshield")

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Startup and Shutdown events.
    Loads the VoxShield model once at startup.
    """
    logger.info("Starting VoxShield backend...")
    
    try:
        ModelService.initialize()
    except Exception as e:
        logger.critical(f"Critical failure during model initialization: {str(e)}")
        # We don't exit to allow the API to return degraded status
        
    yield
    
    logger.info("Shutting down VoxShield backend...")

app = FastAPI(
    title="VoxShield API",
    description="Privacy-first AI-generated voice detection API",
    version="1.0.0",
    lifespan=lifespan
)

# CORS
origins = [
    "http://localhost:3000",
    settings.frontend_url
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=list(set(origins)), # Remove duplicates if frontend_url is localhost:3000
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(health.router, prefix="/api", tags=["System"])
app.include_router(system.router, prefix="/api", tags=["System"])
app.include_router(model.router, prefix="/api", tags=["Model"])
app.include_router(analyze.router, prefix="/api", tags=["Analysis"])
