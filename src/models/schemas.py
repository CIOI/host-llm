from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


class TextGenerationRequest(BaseModel):
    prompt: str = Field(..., description="The input text prompt for generation")
    model: Optional[str] = Field(None, description="Model to use for generation")
    max_length: Optional[int] = Field(
        50, description="Maximum length of generated text"
    )
    num_return_sequences: Optional[int] = Field(
        1, description="Number of sequences to return"
    )
    temperature: Optional[float] = Field(1.0, description="Sampling temperature")
    top_p: Optional[float] = Field(0.9, description="Nucleus sampling parameter")
    top_k: Optional[int] = Field(50, description="Top-k sampling parameter")
    do_sample: Optional[bool] = Field(True, description="Whether to use sampling")

    class Config:
        json_schema_extra = {
            "example": {
                "prompt": "Once upon a time",
                "model": "gpt2",
                "max_length": 100,
                "temperature": 0.8,
            }
        }


class TextGenerationResponse(BaseModel):
    generated_texts: List[str] = Field(
        ..., description="List of generated text sequences"
    )
    model_used: str = Field(..., description="Model that was used for generation")
    prompt: str = Field(..., description="Original prompt")
    generation_params: Dict[str, Any] = Field(
        ..., description="Parameters used for generation"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "generated_texts": [
                    "Once upon a time in a faraway land, there lived a princess..."
                ],
                "model_used": "gpt2",
                "prompt": "Once upon a time",
                "generation_params": {"max_length": 100, "temperature": 0.8},
            }
        }


class ErrorResponse(BaseModel):
    error: str = Field(..., description="Error message")

    class Config:
        json_schema_extra = {"example": {"error": "Model not found or not available"}}
