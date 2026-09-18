import librosa
import numpy as np
import logging
from typing import Tuple, List, Dict
from app.config import settings

logger = logging.getLogger("voxshield")

class AudioError(Exception):
    def __init__(self, code: str, message: str):
        self.code = code
        self.message = message
        super().__init__(message)

class AudioService:
    @staticmethod
    def load_audio(file_path: str, target_sr: int) -> Tuple[np.ndarray, float, int]:
        """
        Loads the audio file from the temporary path, resamples it to the target
        sample rate, ensures it is mono, and validates limits.
        """
        try:
            # Load audio: mono=True, sr=target_sr
            audio_array, sr = librosa.load(file_path, sr=target_sr, mono=True)
            duration = librosa.get_duration(y=audio_array, sr=sr)
            
            # Duration limit
            if duration > settings.max_audio_duration:
                raise AudioError(
                    "AUDIO_TOO_LONG", 
                    f"Audio duration ({duration:.2f}s) exceeds the {settings.max_audio_duration} second limit."
                )
                
            if duration == 0 or len(audio_array) == 0:
                raise AudioError("EMPTY_AUDIO", "Audio file contains no data or duration is 0.")
                
            # Number of channels is always 1 because mono=True in librosa.load
            channels = 1
            
            return audio_array, duration, channels
            
        except AudioError:
            raise
        except Exception as e:
            logger.error(f"Failed to process audio file: {str(e)}")
            raise AudioError("AUDIO_PROCESSING_FAILED", "Failed to decode and process the audio file.")

    @staticmethod
    def create_windows(audio_array: np.ndarray, sr: int, duration: float) -> List[Dict]:
        """
        Creates sliding overlapping windows from the audio array.
        Returns a list of dicts: {'index', 'start_time', 'end_time', 'duration', 'samples'}
        """
        window_size_s = settings.window_size
        overlap_s = settings.window_overlap
        
        window_size_samples = int(window_size_s * sr)
        overlap_samples = int(overlap_s * sr)
        step_samples = window_size_samples - overlap_samples
        
        # If the audio is shorter than a single window, return it as one window
        if len(audio_array) <= window_size_samples:
            return [{
                "index": 1,
                "start_time": 0.0,
                "end_time": duration,
                "duration": duration,
                "samples": audio_array
            }]
            
        windows = []
        index = 1
        
        for start_sample in range(0, len(audio_array), step_samples):
            end_sample = start_sample + window_size_samples
            
            window_samples = audio_array[start_sample:end_sample]
            
            # We don't discard the final short window, we just include it
            # Model feature extractor handles padding if required
            start_time = start_sample / sr
            end_time = min((start_sample + len(window_samples)) / sr, duration)
            
            windows.append({
                "index": index,
                "start_time": start_time,
                "end_time": end_time,
                "duration": end_time - start_time,
                "samples": window_samples
            })
            
            index += 1
            
            if end_sample >= len(audio_array):
                break
                
        return windows
