import pytest
import os
import numpy as np
import librosa
import soundfile as sf
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock

# Set up test env vars before importing app
os.environ["VOXSHIELD_MODEL_DIR"] = "./mock_models/voxshield"
os.environ["MAX_AUDIO_DURATION"] = "30"

from app.main import app
from app.services.audio_service import AudioService
from app.services.privacy_service import PrivacyService
from app.services.analysis_service import AnalysisService

client = TestClient(app)

@pytest.fixture
def mock_audio_file(tmp_path):
    # Create a valid 3-second mono wav file
    file_path = tmp_path / "test_audio.wav"
    sr = 16000
    duration = 3.0
    y = np.zeros(int(sr * duration), dtype=np.float32)
    sf.write(str(file_path), y, sr)
    return str(file_path)

@pytest.fixture
def mock_long_audio_file(tmp_path):
    # Create a 31-second mono wav file
    file_path = tmp_path / "test_long_audio.wav"
    sr = 16000
    duration = 31.0
    y = np.zeros(int(sr * duration), dtype=np.float32)
    sf.write(str(file_path), y, sr)
    return str(file_path)

def test_health_check():
    # Will be degraded because mock model doesn't exist
    response = client.get("/api/health")
    assert response.status_code == 200
    assert response.json()["service"] == "VoxShield"

def test_system_status():
    response = client.get("/api/system/status")
    assert response.status_code == 200
    data = response.json()
    assert data["max_audio_duration"] == 30
    assert "wav" in data["supported_formats"]

def test_audio_duration_limit(mock_long_audio_file):
    # Test audio service limits
    with pytest.raises(Exception) as exc_info:
        AudioService.load_audio(mock_long_audio_file, 16000)
    assert "AUDIO_TOO_LONG" in str(exc_info.value)

def test_audio_windows():
    sr = 16000
    duration = 7.5
    y = np.zeros(int(sr * duration), dtype=np.float32)
    
    # 3s window, 1.5s overlap
    windows = AudioService.create_windows(y, sr, duration)
    
    # Expected starts: 0, 1.5, 3.0, 4.5, 6.0
    assert len(windows) == 5
    assert windows[0]["start_time"] == 0.0
    assert windows[0]["end_time"] == 3.0
    
    assert windows[1]["start_time"] == 1.5
    assert windows[1]["end_time"] == 4.5
    
    assert windows[-1]["start_time"] == 6.0
    assert windows[-1]["end_time"] == 7.5

def test_analysis_aggregation():
    windows = [
        {"fake_probability": 0.1},
        {"fake_probability": 0.9},
        {"fake_probability": 0.8},
        {"fake_probability": 0.2},
        {"fake_probability": 0.5}
    ]
    
    metrics = AnalysisService.aggregate_metrics(windows)
    
    assert metrics["maximum_fake_probability"] == 0.9
    assert metrics["average_fake_probability"] == 0.5
    assert metrics["total_windows"] == 5
    assert metrics["strong_fake_windows"] == 2 # 0.9, 0.8
    assert metrics["moderate_fake_windows"] == 1 # 0.5
    
    assessment = AnalysisService.generate_assessment(metrics)
    assert assessment["prediction"] == "LIKELY_AI_GENERATED"

@patch("app.api.analyze.ModelService.get_status")
@patch("app.api.analyze.ModelService.get_sampling_rate")
@patch("app.api.analyze.ModelService.run_inference")
def test_privacy_cleanup_on_success(mock_run, mock_sr, mock_status, mock_audio_file):
    mock_status.return_value = {"loaded": True, "device": "cpu", "labels": {"0": "real", "1": "fake"}}
    mock_sr.return_value = 16000
    mock_run.return_value = (0.2, 0.8, "fake")
    
    # Spy on PrivacyService
    with patch.object(PrivacyService, 'cleanup_temp_file', wraps=PrivacyService.cleanup_temp_file) as spy_cleanup:
        with open(mock_audio_file, "rb") as f:
            response = client.post(
                "/api/analyze",
                files={"file": ("test_audio.wav", f, "audio/wav")}
            )
            
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["privacy"]["deleted_after_processing"] is True
        
        # Ensure cleanup was called
        spy_cleanup.assert_called_once()

@patch("app.api.analyze.ModelService.get_status")
def test_privacy_cleanup_on_exception(mock_status, mock_audio_file):
    mock_status.return_value = {"loaded": True, "device": "cpu", "labels": {"0": "real", "1": "fake"}}
    
    # Force AudioService to throw an unexpected exception
    with patch("app.api.analyze.AudioService.load_audio", side_effect=Exception("Simulated Failure")):
        with patch.object(PrivacyService, 'cleanup_temp_file', wraps=PrivacyService.cleanup_temp_file) as spy_cleanup:
            with open(mock_audio_file, "rb") as f:
                response = client.post(
                    "/api/analyze",
                    files={"file": ("test_audio.wav", f, "audio/wav")}
                )
                
            assert response.status_code == 500
            
            # Ensure cleanup was STILL called despite the 500 error
            spy_cleanup.assert_called_once()
