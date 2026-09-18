from fastapi import APIRouter, UploadFile, File, HTTPException, status
from fastapi.responses import JSONResponse
import uuid
import logging
import os
import shutil

from app.schemas.analysis import AnalysisResponse, AudioMetadata, ModelInfo, AnalysisConfigInfo
from app.services.privacy_service import PrivacyService
from app.services.audio_service import AudioService, AudioError
from app.services.model_service import ModelService
from app.services.analysis_service import AnalysisService
from app.config import settings

logger = logging.getLogger("voxshield")
router = APIRouter()

@router.post("/analyze", response_model=AnalysisResponse)
async def analyze_audio(file: UploadFile = File(...)):
    """
    Receives an audio file, strictly processes it within memory/temporary space,
    runs the VoxShield sliding window inference, and guarantees cleanup before returning.
    """
    request_id = str(uuid.uuid4())
    logger.info(f"[{request_id}] Received audio analysis request.")
    
    # Validation: File presence
    if not file or not file.filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"success": False, "error": {"code": "INVALID_AUDIO", "message": "No file provided."}}
        )

    # Validation: Extension
    ext = file.filename.split('.')[-1].lower()
    if ext not in settings.supported_formats:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "success": False, 
                "error": {
                    "code": "UNSUPPORTED_FORMAT", 
                    "message": f"Format {ext} is not supported. Supported: {settings.supported_formats}"
                }
            }
        )
        
    # Validation: Model loaded
    model_status = ModelService.get_status()
    if not model_status["loaded"]:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={"success": False, "error": {"code": "MODEL_NOT_LOADED", "message": "VoxShield model is currently unavailable."}}
        )

    temp_path = PrivacyService.create_secure_temp_file(suffix=f".{ext}")
    
    try:
        # Check size incrementally (we do this by checking file spooling if large, or just trusting fastAPI upload limits config)
        # We'll save it to the secure temp path to let librosa process it
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        file_size_mb = os.path.getsize(temp_path) / (1024 * 1024)
        if file_size_mb > settings.max_file_size_mb:
            raise AudioError("AUDIO_TOO_LARGE", f"File size exceeds the {settings.max_file_size_mb}MB limit.")

        logger.info(f"[{request_id}] Processing audio at {temp_path}")
        
        # Load and Validate Audio
        target_sr = ModelService.get_sampling_rate()
        audio_array, duration, channels = AudioService.load_audio(temp_path, target_sr)
        
        # Sliding Windows
        windows_data = AudioService.create_windows(audio_array, target_sr, duration)
        
        window_results = []
        suspicious_segments = []
        timeline = []
        
        # Inference
        for w in windows_data:
            real_prob, fake_prob, pred = ModelService.run_inference(w["samples"])
            
            sig_level = "low"
            if fake_prob >= settings.strong_fake_threshold:
                sig_level = "strong"
            elif fake_prob >= settings.moderate_fake_threshold:
                sig_level = "moderate"
                
            window_results.append({
                "index": w["index"],
                "start_time": w["start_time"],
                "end_time": w["end_time"],
                "real_probability": real_prob,
                "fake_probability": fake_prob,
                "prediction": pred,
                "signal": sig_level
            })
            
            timeline.append({
                "start_time": w["start_time"],
                "end_time": w["end_time"],
                "real_probability": real_prob,
                "fake_probability": fake_prob
            })
            
            if sig_level in ["strong", "moderate"]:
                suspicious_segments.append({
                    "start_time": w["start_time"],
                    "end_time": w["end_time"],
                    "fake_probability": fake_prob,
                    "severity": sig_level
                })
                
        # Aggregation & Explanation
        metrics = AnalysisService.aggregate_metrics(window_results)
        assessment = AnalysisService.generate_assessment(metrics)
        explanation = AnalysisService.generate_explanation(metrics, assessment, window_results)
        
        # Construct response
        response = AnalysisResponse(
            success=True,
            request_id=request_id,
            audio=AudioMetadata(
                filename=file.filename,
                duration=duration,
                sampling_rate=target_sr,
                channels=channels
            ),
            assessment=assessment,
            metrics=metrics,
            windows=window_results,
            suspicious_segments=suspicious_segments,
            timeline=timeline,
            explanation=explanation,
            model=ModelInfo(**model_status),
            analysis=AnalysisConfigInfo(
                window_size=settings.window_size,
                overlap=settings.window_overlap,
                method="overlapping_sliding_window"
            ),
            privacy=PrivacyService.get_privacy_metadata()
        )
        
        logger.info(f"[{request_id}] Analysis complete. Result: {assessment['prediction']}")
        return response

    except AudioError as ae:
        logger.warning(f"[{request_id}] Audio validation failed: {ae.message}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"success": False, "error": {"code": ae.code, "message": ae.message}}
        )
    except Exception as e:
        logger.error(f"[{request_id}] Internal analysis error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"success": False, "error": {"code": "INTERNAL_ERROR", "message": "An internal error occurred during processing."}}
        )
    finally:
        # STRICT PRIVACY ENFORCEMENT: Clean up the file no matter what happened
        PrivacyService.cleanup_temp_file(temp_path)
        if not PrivacyService.verify_deleted(temp_path):
            logger.critical(f"[{request_id}] FAILED TO DELETE TEMPORARY FILE: {temp_path}")
