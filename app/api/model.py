from fastapi import APIRouter
from app.schemas.model import ModelStatusResponse
from app.services.model_service import ModelService

router = APIRouter()

@router.get("/model/status", response_model=ModelStatusResponse)
async def model_status():
    """
    Returns the loaded model status, compute device, and label configuration.
    """
    return ModelStatusResponse(**ModelService.get_status())
