import os
import json
import time
import logging
from typing import Any, Dict, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("llm-api")


def setup_logging(debug: bool = False) -> None:
    """Setup logging level based on debug flag."""
    log_level = logging.DEBUG if debug else logging.INFO
    logger.setLevel(log_level)
    logger.debug("Debug logging enabled")


def timed_execution(func):
    """Decorator to measure function execution time."""

    async def wrapper(*args, **kwargs):
        start_time = time.time()
        result = await func(*args, **kwargs)
        end_time = time.time()
        execution_time = end_time - start_time
        logger.debug(
            f"Function {func.__name__} took {execution_time:.2f} seconds to execute"
        )
        return result

    return wrapper


def safe_json_loads(json_str: str, default: Any = None) -> Any:
    """Safely parse JSON string."""
    try:
        return json.loads(json_str)
    except (json.JSONDecodeError, TypeError) as e:
        logger.error(f"Error parsing JSON: {str(e)}")
        return default


def get_env_bool(name: str, default: bool = False) -> bool:
    """Get boolean value from environment variable."""
    value = os.getenv(name)
    if value is None:
        return default
    return value.lower() in ("1", "true", "yes", "y", "on")


def format_error_response(
    error_message: str, details: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Format error response."""
    response = {"error": error_message}
    if details:
        response["details"] = details
    return response
