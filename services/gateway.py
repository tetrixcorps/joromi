from fastapi import FastAPI, HTTPException, Request, Depends, Header
from fastapi.middleware.cors import CORSMiddleware
from schemas.request_schemas import TextRequest, AudioRequest, ImageRequest, MultiModalRequest
from utils.request_cache import RequestCache
from services.request_analyzer import RequestAnalyzer, InputModality, Domain
from services.discovery import ServiceDiscovery
import httpx
from typing import Dict, List, Union, Optional
import asyncio
import random
from middleware.validation import DomainRequestValidator
from utils.rate_limiter import RateLimiter
from utils.permissions import UserRole
from utils.cache_manager import RoleBasedCache

class LoadBalancer:
    def __init__(self, discovery: ServiceDiscovery):
        self.discovery = discovery
        self.current_index: Dict[str, int] = {}

    async def get_next_instance(self, service_name: str) -> str:
        """Get next service instance using round-robin"""
        instances = await self.discovery.get_service_instances(service_name)
        if not instances:
            raise HTTPException(status_code=503, detail=f"No healthy instances of {service_name}")

        if service_name not in self.current_index:
            self.current_index[service_name] = 0
        else:
            self.current_index[service_name] = (self.current_index[service_name] + 1) % len(instances)

        return instances[self.current_index[service_name]]

class APIGateway:
    def __init__(self, discovery: ServiceDiscovery):
        self.app = FastAPI()
        self.setup_middleware()
        self.discovery = discovery
        self.analyzer = RequestAnalyzer()
        self.cache = RoleBasedCache(redis_client)
        self.rate_limiter = RateLimiter(self.cache.redis)
        self.validator = DomainRequestValidator(self.rate_limiter)
        self.setup_routes()

    def setup_middleware(self):
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

    def setup_routes(self):
        @self.app.post("/process/text")
        async def process_text(
            request: TextRequest,
            role: UserRole = Depends(get_user_role)
        ):
            return await self.handle_request(request.dict(), role)

        @self.app.post("/process/audio")
        async def process_audio(request: AudioRequest):
            return await self.handle_request(request.dict())

        @self.app.post("/process/image")
        async def process_image(request: ImageRequest):
            return await self.handle_request(request.dict())

        @self.app.post("/process/multimodal")
        async def process_multimodal(request: MultiModalRequest):
            return await self.handle_request(request.dict())

        @self.app.get("/health")
        async def health():
            return await self.check_services_health()

    async def get_user_role(self, authorization: Optional[str] = Header(None)) -> Optional[UserRole]:
        """Get user role from authorization header"""
        if not authorization:
            return UserRole.TRIAL
            
        try:
            # Implement your token validation logic here
            # This is a simplified example
            token_data = authorization.split()[-1]
            # Decode and validate token
            # Return appropriate role
            return UserRole.STANDARD  # Default to standard for example
        except Exception as e:
            logger.error(f"Error getting user role: {str(e)}")
            return UserRole.TRIAL

    async def handle_request(self, request_data: dict, role: UserRole):
        """Handle incoming request with role-based validation and caching"""
        try:
            client_id = request_data.get('client_id', 'default')
            
            # Analyze request
            modality, domain, required_services = await self.analyzer.analyze_request(request_data)
            
            # Check role-based cache
            cached_response = await self.cache.get_cached_response(
                request_data, 
                role,
                client_id
            )
            if cached_response:
                return cached_response
            
            # Validate with role permissions
            await self.validator.validate_request(
                domain.value,
                request_data,
                client_id,
                role
            )
            
            # Process request with role context
            result = await self.process_service_pipeline(request_data, required_services, role)
            
            response = {
                "result": result,
                "modality": modality.value,
                "domain": domain.value,
                "services_used": required_services,
                "service_level": self.validator.permission_manager.get_service_level(role, domain.value)
            }
            
            # Cache response with role-specific settings
            await self.cache.cache_response(
                request_data,
                response,
                role,
                client_id
            )
            
            return response
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error in request handling: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))

    async def process_service_pipeline(self, request_data: dict, services: List[str], role: UserRole):
        """Process request through multiple services if needed"""
        current_data = request_data
        
        for service in services:
            # Get healthy service instance
            service_url = await self.get_service_url(service)
            if not service_url:
                raise HTTPException(
                    status_code=503,
                    detail=f"Service {service} unavailable"
                )
            
            # Process through service
            try:
                async with httpx.AsyncClient() as client:
                    response = await client.post(
                        f"{service_url}/process",
                        json=current_data,
                        timeout=30.0
                    )
                    response.raise_for_status()
                    current_data = response.json()
                    
            except httpx.TimeoutException:
                raise HTTPException(
                    status_code=504,
                    detail=f"Service {service} timeout"
                )
            except Exception as e:
                raise HTTPException(
                    status_code=500,
                    detail=f"Error in {service}: {str(e)}"
                )
        
        return current_data

    async def get_service_url(self, service_name: str) -> str:
        """Get URL for a healthy service instance"""
        instances = await self.discovery.get_service_instances(service_name)
        if not instances:
            return None
        return instances[0]  # For now, just use the first instance

    async def check_services_health(self):
        """Check health of all services"""
        results = {}
        services = ["asr", "translation", "tts", "banking"]
        
        for service in services:
            instances = await self.discovery.get_service_instances(service)
            results[service] = {
                "status": "healthy" if instances else "unhealthy",
                "instances": len(instances)
            }
            
        return results 