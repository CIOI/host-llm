from fastapi import APIRouter, HTTPException
from typing import List, Dict, Any

from src.models.schemas import (
    TextGenerationRequest,
    TextGenerationResponse,
    ErrorResponse,
)
from src.services.llm_service import llm_service

router = APIRouter()


@router.post(
    "/generate",
    response_model=TextGenerationResponse,
    responses={400: {"model": ErrorResponse}, 500: {"model": ErrorResponse}},
    summary="Generate text from a prompt",
    description="Generate text using Hugging Face models based on a prompt",
)
async def generate_text(request: TextGenerationRequest) -> Dict[str, Any]:
    """Generate text using a language model."""
    try:
        # Extract params from request
        generation_params = {
            "max_length": request.max_length,
            "num_return_sequences": request.num_return_sequences,
            "temperature": request.temperature,
            "top_p": request.top_p,
            "top_k": request.top_k,
            "do_sample": request.do_sample,
        }

        # Call LLM service
        result = await llm_service.generate_text(
            prompt=request.prompt, model_id=request.model, **generation_params
        )

        # Check for errors
        if "error" in result:
            raise HTTPException(status_code=400, detail=result["error"])

        return result
    except Exception as e:
        # Handle unexpected errors
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")


@router.get(
    "/models",
    response_model=List[Dict[str, Any]],
    summary="Get available models",
    description="Get a list of all available language models",
)
async def list_models() -> List[Dict[str, Any]]:
    """Get a list of all available models."""
    return await llm_service.list_models()


@router.get(
    "/models/{model_id}",
    summary="Get model information",
    description="Get information about a specific model",
)
async def get_model_info(model_id: str) -> Dict[str, Any]:
    """Get information about a specific model."""
    model_info = await llm_service.get_model_info(model_id)
    if not model_info:
        raise HTTPException(status_code=404, detail=f"Model {model_id} not found")
    return model_info


@router.post(
    "/models/{model_id}/load",
    summary="Load a model",
    description="Load a specific model into memory for faster inference",
)
async def load_model(model_id: str) -> Dict[str, Any]:
    """Load a model into memory."""
    result = await llm_service.load_model(model_id)
    if "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])
    return result


@router.post(
    "/models/{model_id}/unload",
    summary="Unload a model",
    description="Unload a model from memory to free resources",
)
async def unload_model(model_id: str) -> Dict[str, Any]:
    """Unload a model from memory."""
    result = await llm_service.unload_model(model_id)
    if "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])
    return result
