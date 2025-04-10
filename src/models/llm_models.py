from typing import Dict, List, Optional, Any
from pydantic import BaseModel
from typing import Literal
import json
from findup import glob
from os.path import dirname, join

# Predefined models that are supported by the API


class LLMModel(BaseModel):
    name: str
    task: Literal[
        "image-text-to-text",
        "text-generation",
    ] = "text-generation"
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
        """Initialize models from models.json file."""
        json_path = join(dirname(glob("src/app.py")), "models/models.json")

        try:
            with open(json_path, "r") as f:
                models_data = json.load(f)

            for model_data in models_data:
                model_id = model_data.get("model_id")
                print(model_id)
                if not model_id:
                    continue

                self.models[model_id] = LLMModel(
                    name=model_data.get("name", model_id),
                    description=model_data.get("description", ""),
                    max_length=model_data.get("max_length", 1024),
                    task=model_data.get("task", "text-generation"),
                )

        except Exception as e:
            print(f"Error loading models from JSON: {str(e)}")
            # Fallback to empty registry
            pass

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
                "task": model.task,
            }
            for model_id, model in self.models.items()
        ]

    def add_model(self, model_id: str, model_info: Dict[str, Any]) -> LLMModel:
        """Add a new model to the registry."""
        model = LLMModel(
            name=model_info.get("name", model_id),
            description=model_info.get("description", ""),
            max_length=model_info.get("max_length", 1024),
            task=model_info.get("task", "text-generation"),
        )
        self.models[model_id] = model
        return model


# Create global model registry
model_registry = ModelRegistry()
