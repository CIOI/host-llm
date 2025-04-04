# LLM API

FastAPI-based API for text generation using Hugging Face models.

## Features

- Text generation API using Hugging Face models
- Model management (loading/unloading)
- Response caching with Redis
- Asynchronous API endpoints
- Swagger documentation
- Support for GPU acceleration (CUDA and Apple Silicon MPS)

## Requirements

- Python 3.10+
- Poetry
- Redis (optional, for caching)

## Installation

1. Clone the repository
2. Install dependencies with Poetry:

```bash
poetry install
```

3. Copy the `.env.example` file to `.env` and fill in the necessary configuration:

```bash
cp .env.example .env
```

4. Update the environment variables in the `.env` file:
   - Set your Hugging Face API token
   - Configure Redis if using caching
   - Set device to "cuda" if using GPU

## Usage

### Starting the API server

```bash
# Development mode
poetry run python -m src.main

# Production mode
poetry run uvicorn src.main:app --host 0.0.0.0 --port 8000
```

### API Endpoints

The API will be available at `http://localhost:8000/api/v1/`

#### Text Generation

- `POST /api/v1/text/generate` - Generate text from a prompt
  
Example request:
```json
{
  "prompt": "Once upon a time",
  "model": "gpt2",
  "max_length": 100,
  "temperature": 0.8,
  "num_return_sequences": 1
}
```

#### Model Management

- `GET /api/v1/text/models` - List available models
- `GET /api/v1/text/models/{model_id}` - Get information about a model
- `POST /api/v1/text/models/{model_id}/load` - Load a model
- `POST /api/v1/text/models/{model_id}/unload` - Unload a model

### Swagger Documentation

API documentation is available at:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

## Configuration

All configuration is managed through environment variables in the `.env` file:

- `API_PREFIX` - API URL prefix
- `DEBUG` - Enable debug mode
- `SECRET_KEY` - Secret key for security
- `HOST` - Server host
- `PORT` - Server port
- `HF_API_TOKEN` - Hugging Face API token
- `DEFAULT_MODEL` - Default model to use
- `DEVICE` - Device to use ("cpu", "cuda" for NVIDIA GPUs, or "mps" for Apple Silicon M1/M2/M3)
- `USE_CACHE` - Enable Redis caching
- `REDIS_HOST` - Redis host
- `REDIS_PORT` - Redis port
- `REDIS_PASSWORD` - Redis password
- `CACHE_EXPIRATION` - Cache expiration time in seconds

### GPU Acceleration

#### For NVIDIA GPUs:
Set `DEVICE=cuda` in your `.env` file.

#### For Apple Silicon (M1/M2/M3):
Set `DEVICE=mps` in your `.env` file. This requires PyTorch 1.12+ with MPS support.

The system will automatically check for device availability and fall back to CPU if the requested device is not available.

## License

MIT
