from pydantic_settings import BaseSettings
from typing import Optional


class Settings(BaseSettings):
    """Application settings."""

    # API Config
    API_PREFIX: str = "/api/v1"
    DEBUG: bool = False
    SECRET_KEY: str

    # Server Config
    HOST: str = "0.0.0.0"
    PORT: int = 8000

    # Hugging Face Config
    HF_API_TOKEN: Optional[str] = None
    DEFAULT_MODEL: str = "gpt2"
    DEVICE: str = "mps"  # "cuda", "mps" (Apple Silicon), or "cpu"

    # Redis Cache Config
    USE_CACHE: bool = True
    REDIS_HOST: str = "localhost"
    REDIS_PORT: int = 6379
    REDIS_PASSWORD: Optional[str] = None
    CACHE_EXPIRATION: int = 3600  # in seconds

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = True
        # Allow device to be updated at runtime
        validate_assignment = True


settings = Settings()
