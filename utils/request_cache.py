import redis
import json
import hashlib
from typing import Optional, Dict, Any
from datetime import timedelta
from utils.logger import setup_logger

logger = setup_logger('request_cache')

class RequestCache:
    def __init__(self, host: str = 'localhost', port: int = 6379, ttl: int = 3600):
        self.redis = redis.Redis(host=host, port=port, decode_responses=True)
        self.ttl = ttl
        
    def _generate_cache_key(self, request_data: Dict[str, Any]) -> str:
        """Generate a cache key from request data"""
        # Remove request_id and timestamp from cache key generation
        cache_data = request_data.copy()
        cache_data.pop('request_id', None)
        cache_data.pop('timestamp', None)
        
        # Sort dictionary to ensure consistent key generation
        serialized = json.dumps(cache_data, sort_keys=True)
        return hashlib.sha256(serialized.encode()).hexdigest()
        
    async def get_cached_response(self, request_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Get cached response for request"""
        try:
            cache_key = self._generate_cache_key(request_data)
            cached = self.redis.get(cache_key)
            
            if cached:
                logger.info(f"Cache hit for key: {cache_key}")
                return json.loads(cached)
                
            logger.info(f"Cache miss for key: {cache_key}")
            return None
            
        except Exception as e:
            logger.error(f"Cache error: {str(e)}")
            return None
            
    async def cache_response(self, request_data: Dict[str, Any], response: Dict[str, Any]):
        """Cache response for request"""
        try:
            cache_key = self._generate_cache_key(request_data)
            self.redis.setex(
                cache_key,
                timedelta(seconds=self.ttl),
                json.dumps(response)
            )
            logger.info(f"Cached response for key: {cache_key}")
            
        except Exception as e:
            logger.error(f"Cache error: {str(e)}") 