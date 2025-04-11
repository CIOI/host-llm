import torch
from typing import Dict, Any, Optional
from transformers import AutoModelForCausalLM, AutoTokenizer
from src.models.llm_models import ModelRegistry, LLMModel
from src.config._environment import Environment
from src.config._logger import LoggerService
from transformers import pipeline


class HuggingFaceController:
    def __init__(
        self,
        environment: Environment,
        logger: LoggerService,
        model_registry: ModelRegistry,
    ):
        self.device = environment.DEVICE
        self.api_token = environment.HF_API_TOKEN
        self.logger = logger
        self.model_registry = model_registry
        self.torch_dtype = (
            torch.float16
            if self.device == "cuda" or self.device == "mps"
            else torch.float32
        )

    async def load_model(self, model_id: str) -> Optional[LLMModel]:
        """Load a model from Hugging Face using pipeline API."""
        try:
            model_info = self.model_registry.get_model(model_id)
            if not model_info:
                return None

            # Skip if model already loaded
            if model_info.loaded:
                return model_info

            # Get model's task
            task = getattr(model_info, "task", "text-generation")

            # Create pipeline directly based on task
            model_pipe = pipeline(
                task,
                model=model_id,
                device=self.device,
                torch_dtype=self.torch_dtype,
                token=self.api_token,
            )

            # Store pipeline in model info
            model_info.pipeline = model_pipe
            model_info.loaded = True

            print(f"Successfully loaded {model_id}")
            return model_info

        except Exception as e:
            # Log the error
            print(f"Error loading model {model_id}: {str(e)}")
            import traceback

            traceback.print_exc()
            return None

    async def unload_model(self, model_id: str) -> bool:
        """Unload a model to free memory."""
        try:
            model_info = self.model_registry.get_model(model_id)
            if not model_info or not model_info.loaded:
                return False

            # Clear CUDA cache if using GPU
            if self.device == "cuda":
                torch.cuda.empty_cache()

            # Reset model instances
            model_info.pipeline = None
            model_info.loaded = False

            return True
        except Exception as e:
            # Log the error
            print(f"Error unloading model {model_id}: {str(e)}")
            return False

    async def generate_text(
        self, model_id: str, prompt: str, generation_params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate text using a loaded model with pipeline API."""
        try:
            # Get model info
            model_info = self.model_registry.get_model(model_id)
            if not model_info:
                return {"error": f"Model {model_id} not found"}

            # Load model if not already loaded
            if not model_info.loaded:
                model_info = await self.load_model(model_id)
                if not model_info:
                    return {"error": f"Failed to load model {model_id}"}

            # Get pipeline from model_info
            if not model_info.pipeline:
                return {"error": f"Pipeline not initialized for model {model_id}"}

            # Get model's task and call appropriate method
            task = getattr(model_info, "task", "text-generation")

            if task == "text-generation":
                result = await self._generate_text_with_text_generation(
                    model_info, prompt, generation_params
                )
            elif task == "image-text-to-text":
                result = await self._generate_text_with_image_text_to_text(
                    model_info, prompt, generation_params
                )
            else:
                return {"error": f"Unsupported task type: {task}"}

            # Add common response fields
            result.update(
                {
                    "model_used": model_id,
                    "prompt": prompt,
                    "generation_params": generation_params,
                }
            )

            return result

        except Exception as e:
            # Log the error
            print(f"Error generating text with model {model_id}: {str(e)}")
            import traceback

            traceback.print_exc()
            return {"error": str(e)}

    async def _generate_text_with_text_generation(
        self, model_info: LLMModel, prompt: Any, generation_params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate text using text-generation pipeline."""
        try:
            model_pipe = model_info.pipeline

            # Set up generation parameters
            max_length = min(
                generation_params.get("max_length", 50), model_info.max_length
            )

            pipe_params = {
                "max_length": max_length,
                "num_return_sequences": generation_params.get(
                    "num_return_sequences", 1
                ),
                "temperature": max(generation_params.get("temperature", 1.0), 0.1),
                "top_p": max(generation_params.get("top_p", 0.9), 0.05),
                "top_k": generation_params.get("top_k", 50),
                "do_sample": generation_params.get("do_sample", True),
            }

            # 프롬프트 타입에 따른 처리
            if isinstance(prompt, str):
                # 문자열 프롬프트 - 일반적인 텍스트 생성
                outputs = model_pipe(prompt, **pipe_params)

                # Extract generated texts
                generated_texts = [item["generated_text"] for item in outputs]

                # Remove prompt from generated texts if present
                if prompt and all(text.startswith(prompt) for text in generated_texts):
                    generated_texts = [
                        text[len(prompt) :].strip() for text in generated_texts
                    ]
            else:
                # 구조화된 프롬프트 (리스트, 딕셔너리 등) - 채팅 형식
                # 채팅 모델은 다른 형식으로 출력하기 때문에 후처리가 다릅니다
                outputs = model_pipe(prompt, **pipe_params)

                # 다양한 출력 형식 처리
                if isinstance(outputs, list) and len(outputs) > 0:
                    if isinstance(outputs[0], dict) and "generated_text" in outputs[0]:
                        generated_texts = [item["generated_text"] for item in outputs]
                    else:
                        generated_texts = [str(item) for item in outputs]
                elif isinstance(outputs, dict) and "generated_text" in outputs:
                    generated_texts = [outputs["generated_text"]]
                elif isinstance(outputs, str):
                    generated_texts = [outputs]
                else:
                    generated_texts = [str(outputs)]

            return {"generated_texts": generated_texts}

        except Exception as e:
            print(f"Error in text generation: {str(e)}")
            import traceback

            traceback.print_exc()
            return {"error": str(e)}

    async def _generate_text_with_image_text_to_text(
        self, model_info: LLMModel, prompt: Any, generation_params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate text using image-text-to-text pipeline."""
        try:
            model_pipe = model_info.pipeline

            # Set up generation parameters
            max_length = min(
                generation_params.get("max_length", 50), model_info.max_length
            )

            pipe_params = {
                "max_new_tokens": max_length,
                "do_sample": generation_params.get("do_sample", True),
                "temperature": max(generation_params.get("temperature", 1.0), 0.1),
                "top_p": max(generation_params.get("top_p", 0.9), 0.05),
            }

            # Handle different prompt types
            if isinstance(prompt, str):
                # Simple text prompt
                outputs = model_pipe(prompt, **pipe_params)
            else:
                # Assuming properly formatted multimodal prompt
                outputs = model_pipe(prompt, **pipe_params)

            # Handle various output formats
            if (
                isinstance(outputs, list)
                and len(outputs) > 0
                and "generated_text" in outputs[0]
            ):
                generated_texts = [item["generated_text"] for item in outputs]
            elif isinstance(outputs, dict) and "generated_text" in outputs:
                generated_texts = [outputs["generated_text"]]
            elif isinstance(outputs, str):
                generated_texts = [outputs]
            else:
                # Last resort handling
                generated_texts = [str(outputs)]

            return {"generated_texts": generated_texts}

        except Exception as e:
            print(f"Error in multimodal generation: {str(e)}")
            import traceback

            traceback.print_exc()
            return {"error": str(e)}
