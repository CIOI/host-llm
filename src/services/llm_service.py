from typing import Dict, Any, List, Optional
from src.controllers.huggingface import HuggingFaceController
from src.models.llm_models import model_registry
from src.services.cache_service import CacheService
from src.config._environment import Environment


class LLMService:
    def __init__(
        self,
        environment: Environment,
        cache_service: CacheService,
        huggingface_controller: HuggingFaceController,
    ):
        self.default_model = environment.DEFAULT_MODEL
        self.huggingface_controller = huggingface_controller
        self.cache_service = cache_service

    async def list_models(self) -> List[Dict[str, Any]]:
        """List all available models."""
        return model_registry.list_models()

    async def get_model_info(self, model_id: str) -> Optional[Dict[str, Any]]:
        """Get information about a specific model."""
        model = model_registry.get_model(model_id)
        if not model:
            return None

        return {
            "id": model_id,
            "name": model.name,
            "description": model.description,
            "loaded": model.loaded,
            "max_length": model.max_length,
        }

    async def load_model(self, model_id: str) -> Dict[str, Any]:
        """Load a model if not already loaded."""
        model_info = model_registry.get_model(model_id)
        if not model_info:
            return {"error": f"Model {model_id} not found"}

        if model_info.loaded:
            return {
                "status": "already_loaded",
                "model_id": model_id,
                "name": model_info.name,
            }

        loaded_model = await self.huggingface_controller.load_model(model_id)
        if not loaded_model:
            return {"error": f"Failed to load model {model_id}"}

        return {"status": "loaded", "model_id": model_id, "name": loaded_model.name}

    async def unload_model(self, model_id: str) -> Dict[str, Any]:
        """Unload a model to free resources."""
        success = await self.huggingface_controller.unload_model(model_id)
        if not success:
            return {"error": f"Failed to unload model {model_id}"}

        return {"status": "unloaded", "model_id": model_id}

    async def generate_text(
        self, prompt: str, model_id: Optional[str] = None, **generation_params
    ) -> Dict[str, Any]:
        """Generate text using the specified model."""
        # Use default model if none specified
        model_id = model_id or self.default_model

        # Validate model existence
        model_info = model_registry.get_model(model_id)
        if not model_info:
            return {"error": f"Model {model_id} not found"}

        # Check cache first
        cached_response = await self.cache_service.get_cached_response(
            model_id, prompt, generation_params
        )

        if cached_response:
            return cached_response

        # Generate text using the controller
        result = await self.huggingface_controller.generate_text(
            model_id, prompt, generation_params
        )

        # Cache the result if successful
        if "error" not in result:
            await self.cache_service.cache_response(
                model_id, prompt, generation_params, result
            )

        return result
