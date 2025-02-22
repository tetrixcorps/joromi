import redis
from datetime import datetime, timedelta
from fastapi import HTTPException
from utils.logger import setup_logger
from utils.permissions import UserRole

logger = setup_logger('rate_limiter')

class RateLimiter:
    def __init__(self, redis_client: redis.Redis):
        self.redis = redis_client
        # Requests per minute by role and service
        self.rate_limits = {
            UserRole.TRIAL: {
                'default': 20,
                'translation': 10,
                'asr': 5,
                'tts': 5,
                'banking': 0
            },
            UserRole.STANDARD: {
                'default': 100,
                'translation': 50,
                'asr': 30,
                'tts': 30,
                'banking': 20
            },
            UserRole.PREMIUM: {
                'default': 500,
                'translation': 200,
                'asr': 100,
                'tts': 100,
                'banking': 50
            },
            UserRole.ADMIN: {
                'default': 1000,
                'translation': 500,
                'asr': 200,
                'tts': 200,
                'banking': 100
            }
        }
        
        # Burst limits (max requests per second)
        self.burst_limits = {
            UserRole.TRIAL: 2,
            UserRole.STANDARD: 5,
            UserRole.PREMIUM: 10,
            UserRole.ADMIN: 20
        }

    async def check_rate_limit(self, client_id: str, service: str, role: UserRole) -> bool:
        """Check if request is within rate limits"""
        try:
            now = datetime.now()
            minute_key = f"rate_limit:{client_id}:{service}:{now.minute}"
            second_key = f"burst_limit:{client_id}:{now.timestamp():.0f}"
            
            # Check burst limit (per second)
            burst_count = self.redis.get(second_key)
            if burst_count and int(burst_count) >= self.burst_limits[role]:
                raise HTTPException(
                    status_code=429,
                    detail="Too many requests per second"
                )
            
            # Check per-minute limit
            count = self.redis.get(minute_key)
            limit = self.rate_limits[role].get(service, self.rate_limits[role]['default'])
            
            if count and int(count) >= limit:
                logger.warning(f"Rate limit exceeded for {client_id} on {service}")
                raise HTTPException(
                    status_code=429,
                    detail=f"Rate limit exceeded. Try again in {60 - now.second} seconds"
                )
            
            # Update counters using pipeline
            pipe = self.redis.pipeline()
            pipe.incr(minute_key)
            pipe.expire(minute_key, 60)  # Expire after 1 minute
            pipe.incr(second_key)
            pipe.expire(second_key, 1)   # Expire after 1 second
            pipe.execute()
            
            return True
            
        except redis.RedisError as e:
            logger.error(f"Redis error in rate limiter: {str(e)}")
            return True  # Allow request if rate limiter fails 