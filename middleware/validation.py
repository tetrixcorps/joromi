from fastapi import Request, HTTPException
from typing import Dict, Any, Optional
import json
from utils.logger import setup_logger
from schemas.domain_schemas import (
    BankingRequest, TranslationRequest, 
    ASRRequest, TTSRequest
)
from utils.rate_limiter import RateLimiter
from utils.permissions import PermissionManager, UserRole

logger = setup_logger('validation_middleware')

class RequestValidationMiddleware:
    async def __call__(self, request: Request, call_next):
        try:
            # Validate request size
            if request.headers.get('content-length', 0) > 10 * 1024 * 1024:  # 10MB limit
                raise HTTPException(status_code=413, detail="Request too large")

            # Validate content type
            if request.headers.get('content-type') != 'application/json':
                raise HTTPException(status_code=415, detail="Only JSON requests are supported")

            # Get request body
            body = await request.body()
            if not body:
                raise HTTPException(status_code=400, detail="Empty request body")

            response = await call_next(request)
            return response

        except json.JSONDecodeError:
            raise HTTPException(status_code=400, detail="Invalid JSON format")
        except Exception as e:
            logger.error(f"Validation error: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))

class DomainRequestValidator:
    def __init__(self, rate_limiter: RateLimiter):
        self.rate_limiter = rate_limiter
        self.permission_manager = PermissionManager()
        self.validators = {
            'banking': BankingRequest,
            'translation': TranslationRequest,
            'asr': ASRRequest,
            'tts': TTSRequest
        }

    async def validate_request(self, domain: str, data: Dict[str, Any], client_id: str, role: Optional[UserRole] = None) -> bool:
        """Validate domain-specific request with role permissions"""
        try:
            # Check rate limit first
            await self.rate_limiter.check_rate_limit(client_id, domain)
            
            # Check role permissions
            if role:
                operation = self._get_operation_type(domain, data)
                if not self.permission_manager.check_permission(role, domain, operation):
                    raise HTTPException(
                        status_code=403,
                        detail=f"Insufficient permissions for {domain}:{operation}"
                    )
            
            # Validate against domain schema
            if domain in self.validators:
                validator = self.validators[domain]
                validator(**data)
                
            # Additional domain-specific validations
            if domain == 'banking':
                await self._validate_banking_request(data, role)
            elif domain == 'translation':
                await self._validate_translation_request(data, role)
                
            return True
            
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Validation error: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))

    def _get_operation_type(self, domain: str, data: Dict[str, Any]) -> str:
        """Determine operation type from request data"""
        if domain == 'banking':
            return data.get('transaction_type', 'balance')
        elif domain == 'translation':
            return 'advanced' if data.get('preserve_formatting') else 'basic'
        elif domain in ['asr', 'tts']:
            return 'full' if data.get('enhance_speech') else 'basic'
        return 'basic'

    async def _validate_banking_request(self, data: Dict[str, Any], role: Optional[UserRole] = None):
        """Validate banking request with role permissions"""
        if data.get('transaction_type') == 'transfer':
            if not data.get('recipient_account'):
                raise ValueError("Recipient account required for transfers")
            if not data.get('amount'):
                raise ValueError("Amount required for transfers")
            
            # Check transfer limits by role
            amount = float(data['amount'])
            max_limits = {
                UserRole.ADMIN: float('inf'),
                UserRole.PREMIUM: 100000,
                UserRole.STANDARD: 10000,
                UserRole.TRIAL: 0
            }
            
            if role and amount > max_limits.get(role, 0):
                raise HTTPException(
                    status_code=403,
                    detail=f"Transfer amount exceeds limit for {role}"
                )

    async def _validate_translation_request(self, data: Dict[str, Any], role: Optional[UserRole] = None):
        """Validate translation request with role permissions"""
        if data.get('source_lang') == data.get('target_lang'):
            raise ValueError("Source and target languages must be different")
            
        # Check advanced features by role
        if data.get('preserve_formatting') and role not in [UserRole.ADMIN, UserRole.PREMIUM]:
            raise HTTPException(
                status_code=403,
                detail="Advanced formatting requires premium subscription"
            ) 