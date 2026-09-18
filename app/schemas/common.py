from pydantic import BaseModel
from typing import Dict, Optional, Any

class GenericErrorModel(BaseModel):
    code: str
    message: str

class ErrorResponse(BaseModel):
    success: bool = False
    error: GenericErrorModel
