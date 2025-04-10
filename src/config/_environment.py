from pydantic_settings import BaseSettings
from typing import Optional
from os import getcwd
from os.path import join, exists
from dotenv import load_dotenv


class Environment(BaseSettings):
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
    REDIS_PORT: int = 6303
    REDIS_PASSWORD: Optional[str] = None
    CACHE_EXPIRATION: int = 3600

    APP_ENV: str = "test"

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "case_sensitive": True,
        # Allow device to be updated at runtime
        "validate_assignment": True,
    }

    @classmethod
    def from_env_file(cls, env_file: str | None = None):
        """특정 환경 파일에서 설정을 로드합니다."""
        if env_file is None:
            env_file = join(getcwd(), ".env")

        if exists(env_file):
            load_dotenv(env_file)

        return cls()
