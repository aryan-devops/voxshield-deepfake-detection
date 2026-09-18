from fastapi import APIRouter
from app.services.model_service import ModelService

router = APIRouter()

@router.get("/health")
async def health_check():
    """
    Basic health check.
    """
    model_status = ModelService.get_status()
    if not model_status["loaded"]:
        return {"status": "degraded", "service": "VoxShield", "model_loaded": False}
        
    return {"status": "ok", "service": "VoxShield", "model_loaded": True}
