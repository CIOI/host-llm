import json
import hashlib
import redis
from typing import Any, Dict, Optional
from src.config.settings import settings


class CacheService:
    def __init__(self):
        self.use_cache = settings.USE_CACHE
        self.expiration = settings.CACHE_EXPIRATION
        self.redis_client = None

        if self.use_cache:
            try:
                self.redis_client = redis.Redis(
                    host=settings.REDIS_HOST,
                    port=settings.REDIS_PORT,
                    password=settings.REDIS_PASSWORD or None,
                    decode_responses=True,
                )
                # Test connection
                self.redis_client.ping()
            except Exception as e:
                print(f"Redis connection failed: {str(e)}")
                self.use_cache = False

    def _generate_cache_key(
        self, model_id: str, prompt: str, params: Dict[str, Any]
    ) -> str:
        """Generate a unique cache key based on model, prompt and parameters."""
        # Sort parameters to ensure consistent cache keys
        sorted_params = {k: params[k] for k in sorted(params.keys())}
        cache_data = {"model_id": model_id, "prompt": prompt, "params": sorted_params}
        # Create a hash of the data
        data_str = json.dumps(cache_data, sort_keys=True)
        return hashlib.md5(data_str.encode()).hexdigest()

    async def get_cached_response(
        self, model_id: str, prompt: str, params: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Retrieve a cached response if available."""
        if not self.use_cache or not self.redis_client:
            return None

        try:
            cache_key = self._generate_cache_key(model_id, prompt, params)
            cached_data = self.redis_client.get(cache_key)

            if cached_data:
                return json.loads(cached_data)

            return None
        except Exception as e:
            print(f"Error retrieving from cache: {str(e)}")
            return None

    async def cache_response(
        self,
        model_id: str,
        prompt: str,
        params: Dict[str, Any],
        response: Dict[str, Any],
    ) -> bool:
        """Store a response in the cache."""
        if not self.use_cache or not self.redis_client:
            return False

        try:
            cache_key = self._generate_cache_key(model_id, prompt, params)
            self.redis_client.setex(cache_key, self.expiration, json.dumps(response))
            return True
        except Exception as e:
            print(f"Error caching response: {str(e)}")
            return False


# Create global cache service instance
cache_service = CacheService()
