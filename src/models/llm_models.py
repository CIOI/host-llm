from typing import Dict, List, Optional, Any
from pydantic import BaseModel

# Predefined models that are supported by the API
SUPPORTED_MODELS = {
    "gpt2": {
        "name": "gpt2",
        "description": "OpenAI GPT-2 (small version, 124M parameters)",
        "max_length": 1024,
    },
    "gpt2-medium": {
        "name": "gpt2-medium",
        "description": "OpenAI GPT-2 Medium (355M parameters)",
        "max_length": 1024,
    },
    "gpt2-large": {
        "name": "gpt2-large",
        "description": "OpenAI GPT-2 Large (774M parameters)",
        "max_length": 1024,
    },
    "distilgpt2": {
        "name": "distilgpt2",
        "description": "Distilled version of GPT-2 (82M parameters)",
        "max_length": 1024,
    },
    "EleutherAI/gpt-neo-125M": {
        "name": "EleutherAI/gpt-neo-125M",
        "description": "EleutherAI's GPT-Neo (125M parameters)",
        "max_length": 2048,
    },
    "TheBloke/FashionGPT-70B-v1.2-GGUF": {
        "name": "TheBloke/FashionGPT-70B-v1.2-GGUF",
        "description": "TheBloke's FashionGPT model (70 billion parameters)",
        "max_length": 2048,
    },
}


class LLMModel(BaseModel):
    name: str
    description: str
    max_length: int
    loaded: bool = False
    model_instance: Optional[Any] = None
    tokenizer_instance: Optional[Any] = None


class ModelRegistry:
    def __init__(self):
        self.models: Dict[str, LLMModel] = {}
        self._initialize_models()

    def _initialize_models(self):
        """Initialize supported models."""
        for model_id, model_info in SUPPORTED_MODELS.items():
            self.models[model_id] = LLMModel(
                name=model_info["name"],
                description=model_info["description"],
                max_length=model_info["max_length"],
            )

    def get_model(self, model_id: str) -> Optional[LLMModel]:
        """Get a model by ID."""
        return self.models.get(model_id)

    def list_models(self) -> List[Dict[str, Any]]:
        """List all available models."""
        return [
            {
                "id": model_id,
                "name": model.name,
                "description": model.description,
                "loaded": model.loaded,
                "max_length": model.max_length,
            }
            for model_id, model in self.models.items()
        ]

    def add_model(self, model_id: str, model_info: Dict[str, Any]) -> LLMModel:
        """Add a new model to the registry."""
        model = LLMModel(
            name=model_info.get("name", model_id),
            description=model_info.get("description", ""),
            max_length=model_info.get("max_length", 1024),
        )
        self.models[model_id] = model
        return model


# Create global model registry
model_registry = ModelRegistry()
