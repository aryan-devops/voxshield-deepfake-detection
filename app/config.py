# pyrefly: ignore [missing-import]
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):

    # VoxShield model
    VOXSHIELD_MODEL_DIR: str = "./models/voxshield"

    # Audio configuration
    MAX_AUDIO_DURATION: float = 30.0
    MAX_FILE_SIZE_MB: float = 25.0

    # Sliding window configuration
    WINDOW_SIZE: float = 3.0
    WINDOW_OVERLAP: float = 0.5

    # Detection thresholds
    STRONG_FAKE_THRESHOLD: float = 0.70
    MODERATE_FAKE_THRESHOLD: float = 0.40

    # Frontend
    FRONTEND_URL: str = "http://localhost:3000"

    # Logging
    LOG_LEVEL: str = "INFO"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )


settings = Settings()