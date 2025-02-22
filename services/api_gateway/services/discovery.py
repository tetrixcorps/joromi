from typing import List
import logging

logger = logging.getLogger(__name__)

class ServiceDiscovery:
    """Service discovery for microservices"""
    
    async def get_service_instances(self, service_name: str) -> List[str]:
        """
        Get list of service instances for a given service name.
        
        Args:
            service_name: Name of the service to discover
            
        Returns:
            List of service URLs
        """
        # For now, return default URLs. In production, this would query service registry
        service_ports = {
            "asr": "8001",
            "tts": "8002",
            "translation": "8003",
            "visual-qa": "8004",
            "banking": "8005"
        }
        
        if service_name not in service_ports:
            logger.warning(f"Unknown service: {service_name}")
            return []
            
        return [f"http://localhost:{service_ports[service_name]}"] 