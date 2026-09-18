import tempfile
import os
import logging
from typing import Optional

logger = logging.getLogger("voxshield")

class PrivacyService:
    @staticmethod
    def create_secure_temp_file(suffix: str = "") -> str:
        """
        Creates a secure temporary file that will be strictly managed.
        Returns the absolute path.
        """
        fd, path = tempfile.mkstemp(suffix=suffix)
        os.close(fd) # Close immediately, we just want the secure path
        return path
        
    @staticmethod
    def cleanup_temp_file(path: Optional[str]) -> bool:
        """
        Safely removes the temporary file. 
        Returns True if deleted or didn't exist, False if error.
        """
        if not path:
            return True
            
        try:
            if os.path.exists(path):
                os.remove(path)
                logger.debug(f"Successfully cleaned up temporary file: {path}")
            return True
        except Exception as e:
            logger.error(f"Failed to clean up temporary file {path}: {str(e)}")
            return False
            
    @staticmethod
    def verify_deleted(path: str) -> bool:
        """
        Verifies that a file no longer exists.
        """
        return not os.path.exists(path)

    @staticmethod
    def get_privacy_metadata() -> dict:
        """
        Returns the strict privacy claims enforced by this architecture.
        """
        return {
            "audio_stored": False,
            "used_for_training": False,
            "retained_after_analysis": False,
            "deleted_after_processing": True
        }
