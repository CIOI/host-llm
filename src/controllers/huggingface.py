import torch
from typing import Dict, Any, Optional
from transformers import AutoModelForCausalLM, AutoTokenizer
from src.config.settings import settings
from src.models.llm_models import model_registry, LLMModel


class HuggingFaceController:
    def __init__(self):
        self.device = settings.DEVICE
        self.api_token = settings.HF_API_TOKEN

    async def load_model(self, model_id: str) -> Optional[LLMModel]:
        """Load a model from Hugging Face."""
        try:
            model_info = model_registry.get_model(model_id)
            if not model_info:
                return None

            # Skip if model already loaded
            if model_info.loaded:
                return model_info

            # Load tokenizer and model
            tokenizer = AutoTokenizer.from_pretrained(
                model_id, use_auth_token=self.api_token
            )

            # Set torch dtype based on device
            torch_dtype = None
            if self.device == "cuda":
                torch_dtype = torch.float16
            elif self.device == "mps":
                # For Apple Silicon GPU
                torch_dtype = torch.float16

            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                use_auth_token=self.api_token,
                torch_dtype=torch_dtype,
            )

            # Move model to the appropriate device
            model.to(self.device)

            # Update model registry
            model_info.loaded = True
            model_info.model_instance = model
            model_info.tokenizer_instance = tokenizer

            return model_info
        except Exception as e:
            # Log the error
            print(f"Error loading model {model_id}: {str(e)}")
            return None

    async def unload_model(self, model_id: str) -> bool:
        """Unload a model to free memory."""
        try:
            model_info = model_registry.get_model(model_id)
            if not model_info or not model_info.loaded:
                return False

            # Clear CUDA cache if using GPU
            if self.device == "cuda":
                torch.cuda.empty_cache()

            # Reset model instances
            model_info.model_instance = None
            model_info.tokenizer_instance = None
            model_info.loaded = False

            return True
        except Exception as e:
            # Log the error
            print(f"Error unloading model {model_id}: {str(e)}")
            return False

    async def generate_text(
        self, model_id: str, prompt: str, generation_params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate text using a loaded model."""
        try:
            # Load model if not already loaded
            model_info = model_registry.get_model(model_id)
            if not model_info:
                return {"error": f"Model {model_id} not found"}

            if not model_info.loaded:
                model_info = await self.load_model(model_id)
                if not model_info:
                    return {"error": f"Failed to load model {model_id}"}

            model = model_info.model_instance
            tokenizer = model_info.tokenizer_instance

            # Set up generation parameters
            max_length = min(
                generation_params.get("max_length", 50), model_info.max_length
            )

            # Encode the prompt
            inputs = tokenizer(prompt, return_tensors="pt").to(self.device)

            # Generate text
            outputs = model.generate(
                inputs["input_ids"],
                max_length=max_length,
                num_return_sequences=generation_params.get("num_return_sequences", 1),
                temperature=generation_params.get("temperature", 1.0),
                top_p=generation_params.get("top_p", 0.9),
                top_k=generation_params.get("top_k", 50),
                do_sample=generation_params.get("do_sample", True),
                pad_token_id=tokenizer.eos_token_id,
            )

            # Decode the output
            generated_texts = [
                tokenizer.decode(output, skip_special_tokens=True) for output in outputs
            ]

            # Remove prompt from generated texts if present
            if prompt and all(text.startswith(prompt) for text in generated_texts):
                generated_texts = [
                    text[len(prompt) :].strip() for text in generated_texts
                ]

            return {
                "generated_texts": generated_texts,
                "model_used": model_id,
                "prompt": prompt,
                "generation_params": generation_params,
            }
        except Exception as e:
            # Log the error
            print(f"Error generating text with model {model_id}: {str(e)}")
            return {"error": str(e)}


# Create global controller instance
huggingface_controller = HuggingFaceController()
