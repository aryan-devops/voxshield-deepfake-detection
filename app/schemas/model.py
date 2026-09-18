from pydantic import BaseModel, ConfigDict
from typing import Dict, List

class ModelStatusResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    loaded: bool
    device: str
    model_name: str
    processor: str
    labels: Dict[str, str]

class SystemStatusResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    service: str
    status: str
    model_loaded: bool
    device: str
    max_audio_duration: float
    window_size: float
    window_overlap: float
    supported_formats: List[str]
