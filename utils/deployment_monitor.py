import time
import torch
import psutil
import requests
from typing import Dict, List
from utils.logger import setup_logger
from utils.gpu_monitor import GPUMonitor

logger = setup_logger('deployment_monitor')

class DeploymentMonitor:
    def __init__(self, host: str = "localhost", port: int = 8000):
        self.host = host
        self.port = port
        self.gpu_monitor = GPUMonitor()
        self.health_endpoint = f"http://{host}:{port}/health"
        
    def check_gpu_setup(self) -> bool:
        """Verify GPU setup"""
        try:
            if not torch.cuda.is_available():
                logger.error("CUDA is not available")
                return False
                
            gpu_count = torch.cuda.device_count()
            logger.info(f"Found {gpu_count} GPU(s)")
            
            for i in range(gpu_count):
                props = torch.cuda.get_device_properties(i)
                logger.info(f"GPU {i}: {props.name}")
                logger.info(f"Memory: {props.total_memory / 1024**3:.1f} GB")
                
            return True
            
        except Exception as e:
            logger.error(f"GPU setup check failed: {str(e)}")
            return False

    def check_service_health(self) -> bool:
        """Check if the service is healthy"""
        try:
            response = requests.get(self.health_endpoint)
            return response.status_code == 200
        except Exception as e:
            logger.error(f"Health check failed: {str(e)}")
            return False

    def monitor_deployment(self, timeout: int = 300) -> bool:
        """Monitor deployment process"""
        start_time = time.time()
        logger.info("Starting deployment monitoring...")
        
        # Check GPU setup
        if not self.check_gpu_setup():
            return False
            
        # Wait for service to be ready
        while time.time() - start_time < timeout:
            if self.check_service_health():
                logger.info("Service is healthy")
                return True
            time.sleep(5)
            
        logger.error("Deployment monitoring timed out")
        return False 