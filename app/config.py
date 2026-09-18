# pyrefly: ignore [missing-import]
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):

    # VoxShield model
    voxshield_model_dir: str = "./models/voxshield"

    # Audio configuration
    max_audio_duration: float = 30.0
    max_file_size_mb: float = 25.0

    # Sliding window configuration
    window_size: float = 3.0
    window_overlap: float = 0.5

    # Detection thresholds
    strong_fake_threshold: float = 0.70
    moderate_fake_threshold: float = 0.40

    # Frontend
    frontend_url: str = "http://localhost:3000"

    # Logging
    log_level: str = "INFO"

    # Supported audio formats
    supported_formats: list[str] = ["wav", "mp3", "flac", "ogg", "m4a"]

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )


settings = Settings()