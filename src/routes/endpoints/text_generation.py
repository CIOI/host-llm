from fastapi import APIRouter, HTTPException
from typing import List, Dict, Any

from src.models.schemas import (
    TextGenerationRequest,
    TextGenerationResponse,
    ErrorResponse,
)
from src.services.llm_service import LLMService
from src.config._logger import LoggerService


class TextGenerationRouter:
    def __init__(self, llm_service: LLMService, logger: LoggerService):
        self.llm_service = llm_service
        self.logger = logger

    async def generate_text(self, request: TextGenerationRequest) -> Dict[str, Any]:
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
            result = await self.llm_service.generate_text(
                model_id=request.model,
                model_info=model_info,
                prompt=request.prompt,
                **generation_params,
            )

            # Check for errors
            if "error" in result:
                raise HTTPException(status_code=400, detail=result["error"])

            return result
        except Exception as e:
            # Handle unexpected errors
            raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")

    async def list_models(self) -> List[Dict[str, Any]]:
        """Get a list of all available models."""
        return await self.llm_service.list_models()

    async def get_model_info(self, model_id: str) -> Dict[str, Any]:
        """Get information about a specific model."""
        model_info = await self.llm_service.get_model_info(model_id)
        if not model_info:
            raise HTTPException(status_code=404, detail=f"Model {model_id} not found")
        return model_info

    async def load_model(self, model_id: str) -> Dict[str, Any]:
        """Load a model into memory."""
        self.logger.info(f"Loading model {model_id}")
        result = await self.llm_service.load_model(model_id)
        if "error" in result:
            raise HTTPException(status_code=400, detail=result["error"])
        return result

    async def unload_model(self, model_id: str) -> Dict[str, Any]:
        """Unload a model from memory."""
        result = await self.llm_service.unload_model(model_id)
        if "error" in result:
            raise HTTPException(status_code=400, detail=result["error"])
        return result

    def register_routes(self, router: APIRouter) -> APIRouter:
        router.add_api_route(
            "/generate",
            self.generate_text,
            methods=["POST"],
            response_model=TextGenerationResponse,
            responses={400: {"model": ErrorResponse}, 500: {"model": ErrorResponse}},
            summary="Generate text from a prompt",
            description="Generate text using Hugging Face models based on a prompt",
        )
        router.add_api_route(
            "/models",
            self.list_models,
            methods=["GET"],
            response_model=List[Dict[str, Any]],
            summary="Get available models",
            description="Get a list of all available language models",
        )
        router.add_api_route(
            "/models/{model_id:path}",
            self.get_model_info,
            methods=["GET"],
            summary="Get model information",
            description="Get information about a specific model",
        )
        router.add_api_route(
            "/models/{model_id:path}/load",
            self.load_model,
            methods=["POST"],
            summary="Load a model",
            description="Load a specific model into memory for faster inference",
        )
        router.add_api_route(
            "/models/{model_id:path}/unload",
            self.unload_model,
            methods=["POST"],
            summary="Unload a model",
            description="Unload a model from memory to free resources",
        )
        return router
