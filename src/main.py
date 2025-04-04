import uvicorn
import torch
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from src.routes.api import api_router
from src.config.settings import settings
from src.utils.helpers import logger, setup_logging

# Set up logging based on debug setting
setup_logging(settings.DEBUG)

# Check for MPS (Apple Silicon GPU) availability
if settings.DEVICE == "mps" and torch.backends.mps.is_available():
    logger.info(
        "Apple Silicon MPS device is available and will be used for acceleration"
    )
elif settings.DEVICE == "mps" and not torch.backends.mps.is_available():
    logger.warning("MPS device requested but not available. Falling back to CPU")
    # Override the device setting
    settings.DEVICE = "cpu"
elif settings.DEVICE == "cuda" and torch.cuda.is_available():
    logger.info(f"CUDA is available with device: {torch.cuda.get_device_name(0)}")
else:
    logger.info("Using CPU for computations")

# Create FastAPI app
app = FastAPI(
    title="LLM API",
    description="API for text generation using Hugging Face models",
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
    debug=settings.DEBUG,
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API router
app.include_router(api_router)


# Add global exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    return JSONResponse(
        status_code=500,
        content={"error": f"Unexpected error: {str(exc)}"},
    )


# Root endpoint
@app.get("/", tags=["status"])
async def root():
    return {"status": "online", "api_version": "0.1.0", "api_docs": "/docs"}


# Health check endpoint
@app.get("/health", tags=["status"])
async def health_check():
    return {"status": "healthy"}


if __name__ == "__main__":
    uvicorn.run(
        "src.main:app", host=settings.HOST, port=settings.PORT, reload=settings.DEBUG
    )
