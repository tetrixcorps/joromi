from typing import Optional
import os
import aiofiles
from spaces_sdk import SpacesClient
import logging

logger = logging.getLogger(__name__)

class ModelStorageManager:
    def __init__(self):
        self.spaces_client = SpacesClient(
            region=os.getenv("DO_REGION"),
            access_key=os.getenv("DO_SPACES_KEY"),
            secret_key=os.getenv("DO_SPACES_SECRET")
        )
        self.bucket_name = os.getenv("DO_SPACES_BUCKET")
        self.cache_dir = "/app/model_cache"

    async def get_model(self, model_id: str) -> Optional[str]:
        """Get model from cache or download from Spaces"""
        cache_path = f"{self.cache_dir}/{model_id}"
        
        if os.path.exists(cache_path):
            return cache_path

        try:
            # Download from Spaces
            await self.spaces_client.download_file(
                bucket=self.bucket_name,
                key=f"models/{model_id}",
                destination=cache_path
            )
            return cache_path
        except Exception as e:
            logger.error(f"Failed to download model {model_id}: {e}")
            return None

    async def cache_model(self, model_id: str, model_path: str):
        """Cache model to Spaces"""
        try:
            async with aiofiles.open(model_path, 'rb') as f:
                await self.spaces_client.upload_file(
                    bucket=self.bucket_name,
                    key=f"models/{model_id}",
                    source=await f.read()
                )
        except Exception as e:
            logger.error(f"Failed to cache model {model_id}: {e}") 