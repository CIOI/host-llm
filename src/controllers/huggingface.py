import torch
from typing import Dict, Any, Optional, List
from transformers import AutoModelForCausalLM, AutoTokenizer
from src.models.llm_models import model_registry, LLMModel
from src.config._environment import Environment


class HuggingFaceController:
    def __init__(self, environment: Environment):
        self.device = environment.DEVICE
        self.api_token = environment.HF_API_TOKEN

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
        self, model_info: LLMModel, prompt: str, **kwargs
    ) -> List[str]:
        """Generate text using a loaded model."""
        try:
            tokenizer = model_info.tokenizer_instance
            model = model_info.model_instance

            # 입력 준비
            inputs = tokenizer(prompt, return_tensors="pt")

            # Gemma 모델용 특별 처리
            if "gemma" in model_info.name.lower():
                # 명시적으로 어텐션 마스크 설정
                input_ids = inputs["input_ids"].to(self.device)
                attention_mask = torch.ones_like(input_ids)

                # 생성 파라미터 조정
                outputs = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_length=kwargs.get("max_length", 100),
                    do_sample=kwargs.get("do_sample", True),
                    temperature=max(kwargs.get("temperature", 0.7), 0.1),  # 최소값 설정
                    top_p=min(max(kwargs.get("top_p", 0.9), 0.1), 0.99),  # 범위 제한
                    num_return_sequences=kwargs.get("num_return_sequences", 1),
                    pad_token_id=tokenizer.eos_token_id,  # 패드 토큰 명시적 설정
                )
            else:
                # 다른 모델용 기존 처리
                inputs = inputs.to(self.device)
                outputs = model.generate(
                    **inputs,
                    max_length=kwargs.get("max_length", 100),
                    do_sample=kwargs.get("do_sample", True),
                    temperature=kwargs.get("temperature", 0.7),
                    top_p=kwargs.get("top_p", 0.9),
                    top_k=kwargs.get("top_k", 50),
                    num_return_sequences=kwargs.get("num_return_sequences", 1),
                )

            # 출력 디코딩
            generated_texts = [
                tokenizer.decode(output, skip_special_tokens=True) for output in outputs
            ]
            return generated_texts

        except Exception as e:
            # 오류 처리
            print(f"Error generating text with model {model_info.name}: {str(e)}")
            return [f"Error: {str(e)}"]
