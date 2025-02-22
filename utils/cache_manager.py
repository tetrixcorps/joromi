import boto3
import os
from pathlib import Path
import hashlib
import json
from config.spaces_config import SPACES_CONFIG
from datetime import datetime, timedelta
from typing import Dict, Optional, Any
import torch
from botocore.exceptions import ClientError
from utils.logger import setup_logger
from utils.permissions import UserRole
import redis

logger = setup_logger('cache_manager')

class ModelUsageTracker:
    def __init__(self, cache_dir: Path):
        self.usage_file = cache_dir / "model_usage.json"
        self.load_usage()

    def load_usage(self):
        if self.usage_file.exists():
            with open(self.usage_file, 'r') as f:
                self.usage_data = json.load(f)
        else:
            self.usage_data = {}

    def save_usage(self):
        with open(self.usage_file, 'w') as f:
            json.dump(self.usage_data, f)

    def record_usage(self, model_name: str):
        now = datetime.now().isoformat()
        if model_name not in self.usage_data:
            self.usage_data[model_name] = {'access_count': 0, 'last_access': None}
        
        self.usage_data[model_name]['access_count'] += 1
        self.usage_data[model_name]['last_access'] = now
        self.save_usage()

class ModelCacheManager:
    def __init__(self, cache_dir: Path, max_cache_size_gb: int = 100):
        self.cache_dir = cache_dir
        self.max_cache_size_gb = max_cache_size_gb
        self.cache_manifest = self.cache_dir / "manifest.json"
        
        # Initialize Spaces client with retry configuration
        self.spaces_client = self._init_spaces_client()
        self.space_name = SPACES_CONFIG['space_name']
        
        # Initialize usage tracker
        self.usage_tracker = ModelUsageTracker(cache_dir)
        self.load_manifest()

    def _init_spaces_client(self):
        """Initialize Spaces client with retry configuration"""
        try:
            session = boto3.session.Session()
            return session.client('s3',
                endpoint_url=SPACES_CONFIG['endpoint_url'],
                aws_access_key_id=SPACES_CONFIG['access_key'],
                aws_secret_access_key=SPACES_CONFIG['secret_key'],
                region_name=SPACES_CONFIG['region'],
                config=boto3.client.Config(
                    retries=dict(
                        max_attempts=3,
                        mode='adaptive'
                    )
                )
            )
        except Exception as e:
            logger.error(f"Failed to initialize Spaces client: {str(e)}")
            raise

    def check_spaces_connectivity(self) -> bool:
        """Test connection to Digital Ocean Spaces"""
        try:
            self.spaces_client.head_bucket(Bucket=self.space_name)
            return True
        except ClientError as e:
            logger.error(f"Spaces connectivity error: {str(e)}")
            return False

    def get_cache_size(self) -> float:
        """Get current cache size in GB"""
        total_size = 0
        for path in self.cache_dir.rglob('*'):
            if path.is_file():
                total_size += path.stat().st_size
        return total_size / (1024 * 1024 * 1024)  # Convert to GB

    def cleanup_cache(self, required_space_gb: float):
        """Remove least used models to free up space"""
        current_size = self.get_cache_size()
        if current_size + required_space_gb <= self.max_cache_size_gb:
            return

        # Sort models by usage
        sorted_models = sorted(
            self.usage_tracker.usage_data.items(),
            key=lambda x: (x[1]['access_count'], x[1]['last_access'])
        )

        # Remove models until we have enough space
        for model_name, _ in sorted_models:
            model_path = self.cache_dir / "models" / model_name
            if model_path.exists():
                size = sum(f.stat().st_size for f in model_path.rglob('*') if f.is_file())
                size_gb = size / (1024 * 1024 * 1024)
                
                # Remove model files
                for f in model_path.rglob('*'):
                    if f.is_file():
                        f.unlink()
                model_path.rmdir()
                
                # Update manifest
                model_hash = self.get_model_hash(model_name)
                if model_hash in self.manifest:
                    del self.manifest[model_hash]
                self.save_manifest()
                
                current_size -= size_gb
                if current_size + required_space_gb <= self.max_cache_size_gb:
                    break

    def get_model_files(self, model_name: str) -> bool:
        """Download model files from Spaces if not in cache"""
        try:
            # Record model usage
            self.usage_tracker.record_usage(model_name)
            
            model_hash = self.get_model_hash(model_name)
            if model_hash in self.manifest:
                return True

            # Check Spaces connectivity
            if not self.check_spaces_connectivity():
                raise ConnectionError("Cannot connect to Digital Ocean Spaces")

            # Get model size before downloading
            size = self._get_model_size(model_name)
            size_gb = size / (1024 * 1024 * 1024)
            
            # Cleanup cache if needed
            self.cleanup_cache(size_gb)

            # Download files
            prefix = f"models/{model_name}"
            response = self.spaces_client.list_objects_v2(
                Bucket=self.space_name,
                Prefix=prefix
            )

            for obj in response.get('Contents', []):
                target_path = self.cache_dir / obj['Key']
                target_path.parent.mkdir(parents=True, exist_ok=True)
                
                self.spaces_client.download_file(
                    self.space_name,
                    obj['Key'],
                    str(target_path)
                )

            # Update manifest
            self.manifest[model_hash] = {
                'name': model_name,
                'files': [obj['Key'] for obj in response.get('Contents', [])],
                'size': size_gb
            }
            self.save_manifest()
            return True

        except Exception as e:
            logger.error(f"Error downloading model {model_name}: {str(e)}")
            return False

    def _get_model_size(self, model_name: str) -> int:
        """Get total size of model files in bytes"""
        total_size = 0
        prefix = f"models/{model_name}"
        
        try:
            response = self.spaces_client.list_objects_v2(
                Bucket=self.space_name,
                Prefix=prefix
            )
            
            for obj in response.get('Contents', []):
                total_size += obj['Size']
                
            return total_size
        except Exception as e:
            logger.error(f"Error getting model size: {str(e)}")
            return 0

    def load_manifest(self):
        if self.cache_manifest.exists():
            with open(self.cache_manifest, 'r') as f:
                self.manifest = json.load(f)
        else:
            self.manifest = {}

    def save_manifest(self):
        with open(self.cache_manifest, 'w') as f:
            json.dump(self.manifest, f)

    @staticmethod
    def get_model_hash(model_name: str) -> str:
        return hashlib.sha256(model_name.encode()).hexdigest()

class RoleBasedCache:
    def __init__(self, redis_client: redis.Redis):
        self.redis = redis_client
        # TTL in seconds for each role
        self.cache_ttl = {
            UserRole.TRIAL: 300,      # 5 minutes
            UserRole.STANDARD: 600,    # 10 minutes
            UserRole.PREMIUM: 1800,    # 30 minutes
            UserRole.ADMIN: 3600       # 1 hour
        }
        # Cache size limits per role (in number of keys)
        self.cache_limits = {
            UserRole.TRIAL: 100,
            UserRole.STANDARD: 500,
            UserRole.PREMIUM: 2000,
            UserRole.ADMIN: 5000
        }

    def _generate_cache_key(self, request_data: Dict[str, Any], role: UserRole) -> str:
        """Generate a cache key from request data and role"""
        cache_data = request_data.copy()
        cache_data.pop('request_id', None)
        cache_data.pop('timestamp', None)
        cache_data['role'] = role.value
        
        serialized = json.dumps(cache_data, sort_keys=True)
        return hashlib.sha256(serialized.encode()).hexdigest()

    async def get_cached_response(
        self, 
        request_data: Dict[str, Any], 
        role: UserRole,
        client_id: str
    ) -> Optional[Dict[str, Any]]:
        """Get cached response based on role"""
        try:
            cache_key = self._generate_cache_key(request_data, role)
            client_key = f"cache_keys:{client_id}:{role.value}"
            
            # Get cached response
            cached = self.redis.get(cache_key)
            if cached:
                logger.info(f"Cache hit for key: {cache_key}")
                # Update access time
                self.redis.expire(
                    cache_key, 
                    timedelta(seconds=self.cache_ttl[role])
                )
                return json.loads(cached)
            
            logger.info(f"Cache miss for key: {cache_key}")
            return None
            
        except Exception as e:
            logger.error(f"Cache error: {str(e)}")
            return None

    async def cache_response(
        self, 
        request_data: Dict[str, Any], 
        response: Dict[str, Any],
        role: UserRole,
        client_id: str
    ):
        """Cache response with role-specific TTL and limits"""
        try:
            cache_key = self._generate_cache_key(request_data, role)
            client_key = f"cache_keys:{client_id}:{role.value}"
            
            # Check cache size limit for client
            client_keys = self.redis.smembers(client_key)
            if len(client_keys) >= self.cache_limits[role]:
                # Remove oldest key
                oldest_key = self.redis.spop(client_key)
                if oldest_key:
                    self.redis.delete(oldest_key)
            
            # Cache the response
            pipe = self.redis.pipeline()
            pipe.setex(
                cache_key,
                timedelta(seconds=self.cache_ttl[role]),
                json.dumps(response)
            )
            pipe.sadd(client_key, cache_key)
            pipe.execute()
            
            logger.info(f"Cached response for key: {cache_key}")
            
        except Exception as e:
            logger.error(f"Cache error: {str(e)}") 