from typing import List, Dict
import asyncio
from utils.logger import setup_logger

logger = setup_logger('service_orchestrator')

class ServicePipeline:
    def __init__(self, services: List[str]):
        self.services = services
        self.results = {}

    async def execute(self, input_data: dict) -> dict:
        """Execute the service pipeline"""
        current_data = input_data.copy()
        
        for service in self.services:
            try:
                # Process through service
                result = await self._call_service(service, current_data)
                
                # Store result
                self.results[service] = result
                
                # Update data for next service
                current_data.update(result)
                
            except Exception as e:
                logger.error(f"Pipeline error at {service}: {str(e)}")
                raise
                
        return current_data

    async def _call_service(self, service: str, data: dict) -> dict:
        """Make service call with retry logic"""
        max_retries = 3
        retry_delay = 1
        
        for attempt in range(max_retries):
            try:
                # Service call implementation
                return await self._make_service_request(service, data)
                
            except Exception as e:
                if attempt == max_retries - 1:
                    raise
                    
                logger.warning(f"Retry {attempt + 1} for {service}: {str(e)}")
                await asyncio.sleep(retry_delay)
                retry_delay *= 2

    async def _make_service_request(self, service: str, data: dict) -> dict:
        """Make actual service request"""
        # Implementation depends on your service client
        pass 