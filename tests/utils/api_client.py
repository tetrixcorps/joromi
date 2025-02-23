import requests
import base64
from typing import Dict, Union, Optional

class ModelAPIClient:
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        
    def process_text(self, 
                    text: str, 
                    domain: Optional[str] = None, 
                    confidence_threshold: float = 0.5) -> Dict:
        """
        Send text processing request to the API
        """
        payload = {
            "modality": "text",
            "content": text,
            "domain": domain,
            "confidence_threshold": confidence_threshold
        }
        
        response = requests.post(f"{self.base_url}/process", json=payload)
        response.raise_for_status()
        return response.json()
        
    def process_image(self, 
                     image_path: str, 
                     confidence_threshold: float = 0.5) -> Dict:
        """
        Send image processing request to the API
        """
        with open(image_path, 'rb') as img:
            image_bytes = img.read()
            image_b64 = base64.b64encode(image_bytes).decode()
            
        payload = {
            "modality": "image",
            "content": image_b64,
            "confidence_threshold": confidence_threshold
        }
        
        response = requests.post(f"{self.base_url}/process", json=payload)
        response.raise_for_status()
        return response.json() 