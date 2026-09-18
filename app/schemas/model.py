from pydantic import BaseModel
from typing import Dict, List

class ModelStatusResponse(BaseModel):
    loaded: bool
    device: str
    model_name: str
    processor: str
    labels: Dict[str, str]

class SystemStatusResponse(BaseModel):
    service: str
    status: str
    model_loaded: bool
    device: str
    max_audio_duration: int
    window_size: float
    window_overlap: float
    supported_formats: List[str]
