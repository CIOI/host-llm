from fastapi import APIRouter
from src.routes.endpoints import text_generation
from src.config.settings import settings

# Create main API router
api_router = APIRouter(prefix=settings.API_PREFIX)

# Include endpoint routers
api_router.include_router(
    text_generation.router, prefix="/text", tags=["text-generation"]
)
