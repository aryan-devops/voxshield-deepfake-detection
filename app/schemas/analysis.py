from pydantic import BaseModel
from typing import List, Dict, Any, Optional

class AudioMetadata(BaseModel):
    filename: str
    duration: float
    sampling_rate: int
    channels: int

class Assessment(BaseModel):
    prediction: str
    fake_probability: float
    real_probability: float
    confidence: float

class Metrics(BaseModel):
    maximum_fake_probability: float
    average_fake_probability: float
    median_fake_probability: float
    total_windows: int
    strong_fake_windows: int
    moderate_fake_windows: int
    strong_window_ratio: float
    moderate_window_ratio: float

class WindowResult(BaseModel):
    index: int
    start_time: float
    end_time: float
    real_probability: float
    fake_probability: float
    prediction: str
    signal: str

class SuspiciousSegment(BaseModel):
    start_time: float
    end_time: float
    fake_probability: float
    severity: str

class TimelinePoint(BaseModel):
    start_time: float
    end_time: float
    real_probability: float
    fake_probability: float

class ExplanationEvidence(BaseModel):
    type: str
    start_time: float
    end_time: float
    fake_probability: float
    description: str

class ExplanationFactor(BaseModel):
    name: str
    value: float
    interpretation: str

class Explanation(BaseModel):
    summary: str
    evidence: List[ExplanationEvidence]
    factors: List[ExplanationFactor]

class ModelInfo(BaseModel):
    name: str
    processor: str
    device: str
    labels: Dict[str, str]

class AnalysisConfigInfo(BaseModel):
    window_size: float
    overlap: float
    method: str

class PrivacyInfo(BaseModel):
    audio_stored: bool
    used_for_training: bool
    retained_after_analysis: bool
    deleted_after_processing: bool

class AnalysisResponse(BaseModel):
    success: bool
    request_id: str
    audio: AudioMetadata
    assessment: Assessment
    metrics: Metrics
    windows: List[WindowResult]
    suspicious_segments: List[SuspiciousSegment]
    timeline: List[TimelinePoint]
    explanation: Explanation
    model: ModelInfo
    analysis: AnalysisConfigInfo
    privacy: PrivacyInfo
