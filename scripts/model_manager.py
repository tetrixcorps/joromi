import os
import requests
import hashlib
from pathlib import Path
import yaml
import logging

logger = logging.getLogger(__name__)

class ModelManager:
    def __init__(self):
        self.config_file = Path("config/models.yml")
        self.model_dir = Path("models")
        self.model_dir.mkdir(exist_ok=True)
        
    def load_config(self):
        """Load model configuration"""
        with open(self.config_file) as f:
            return yaml.safe_load(f)
            
    def download_model(self, url, filename, expected_hash=None):
        """Download model file if not present"""
        filepath = self.model_dir / filename
        if filepath.exists():
            if expected_hash and self.verify_hash(filepath, expected_hash):
                logger.info(f"Model {filename} already exists and hash matches")
                return filepath
            
        logger.info(f"Downloading {filename}...")
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        with open(filepath, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
                
        if expected_hash and not self.verify_hash(filepath, expected_hash):
            raise ValueError(f"Hash mismatch for {filename}")
            
        return filepath
        
    def verify_hash(self, filepath, expected_hash):
        """Verify file hash"""
        sha256_hash = hashlib.sha256()
        with open(filepath, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest() == expected_hash

    def setup_models(self):
        """Download all required models"""
        config = self.load_config()
        for model_name, model_info in config['models'].items():
            try:
                filepath = self.download_model(
                    model_info['url'],
                    model_info['filename'],
                    model_info.get('hash')
                )
                logger.info(f"Successfully setup {model_name} at {filepath}")
            except Exception as e:
                logger.error(f"Failed to setup {model_name}: {e}") 