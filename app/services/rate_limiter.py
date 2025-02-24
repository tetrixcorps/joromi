from typing import Dict, Optional
import time
import asyncio
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

@dataclass
class RateLimitConfig:
    requests_per_minute: int
    burst_size: int
    cooldown_period: int  # seconds

class RateLimiter:
    def __init__(self):
        self.user_limits: Dict[str, RateLimitConfig] = {}
        self.request_counts: Dict[str, int] = {}
        self.last_request_time: Dict[str, float] = {}
        
        # Default limits
        self.default_config = RateLimitConfig(
            requests_per_minute=60,
            burst_size=10,
            cooldown_period=60
        )

    async def check_rate_limit(self, user_id: str) -> bool:
        """Check if request should be rate limited"""
        current_time = time.time()
        config = self.user_limits.get(user_id, self.default_config)
        
        # Reset counts if cooldown period has passed
        if (user_id in self.last_request_time and 
            current_time - self.last_request_time[user_id] > config.cooldown_period):
            self.request_counts[user_id] = 0
            
        # Update counts
        self.request_counts[user_id] = self.request_counts.get(user_id, 0) + 1
        self.last_request_time[user_id] = current_time
        
        # Check limits
        return self.request_counts[user_id] <= config.requests_per_minute 