from typing import Dict, List
import aioredis
import json
import asyncio
from datetime import datetime
import consul
from utils.logger import setup_logger

logger = setup_logger('service_discovery')

class ServiceDiscovery:
    def __init__(self, consul_host: str = "localhost", consul_port: int = 8500):
        self.consul = consul.Consul(host=consul_host, port=consul_port)
        self.service_cache: Dict[str, List[str]] = {}
        self.last_update = {}
        self.cache_ttl = 30  # seconds

    async def register_service(self, service_name: str, host: str, port: int, tags: List[str] = None):
        """Register a service with Consul"""
        try:
            service_id = f"{service_name}-{host}-{port}"
            self.consul.agent.service.register(
                name=service_name,
                service_id=service_id,
                address=host,
                port=port,
                tags=tags or [],
                check=consul.Check().tcp(host, port, "10s")
            )
            logger.info(f"Registered service: {service_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to register service: {str(e)}")
            return False

    async def get_service_instances(self, service_name: str) -> List[str]:
        """Get all healthy instances of a service"""
        # Check cache first
        if self._is_cache_valid(service_name):
            return self.service_cache[service_name]

        try:
            _, services = self.consul.health.service(service_name, passing=True)
            instances = [
                f"http://{svc['Service']['Address']}:{svc['Service']['Port']}"
                for svc in services
            ]
            
            # Update cache
            self.service_cache[service_name] = instances
            self.last_update[service_name] = datetime.now()
            
            return instances
        except Exception as e:
            logger.error(f"Failed to get service instances: {str(e)}")
            return []

    def _is_cache_valid(self, service_name: str) -> bool:
        """Check if cached service data is still valid"""
        if service_name not in self.last_update:
            return False
            
        age = (datetime.now() - self.last_update[service_name]).seconds
        return age < self.cache_ttl 