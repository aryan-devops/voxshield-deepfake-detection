from fastapi import APIRouter
from app.schemas.model import SystemStatusResponse
from app.services.model_service import ModelService
from app.config import settings

router = APIRouter()

@router.get("/system/status", response_model=SystemStatusResponse)
async def system_status():
    """
    Returns the system configuration and operational status.
    """
    model_status = ModelService.get_status()
    
    return SystemStatusResponse(
        service="VoxShield",
        status="operational" if model_status["loaded"] else "degraded",
        model_loaded=model_status["loaded"],
        device=model_status["device"],
        max_audio_duration=settings.max_audio_duration,
        window_size=settings.window_size,
        window_overlap=settings.window_overlap,
        supported_formats=settings.supported_formats
    )
