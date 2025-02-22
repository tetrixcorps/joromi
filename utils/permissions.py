from enum import Enum
from typing import Dict, List, Set
from fastapi import HTTPException
from utils.logger import setup_logger

logger = setup_logger('permissions')

class UserRole(str, Enum):
    ADMIN = "admin"
    PREMIUM = "premium"
    STANDARD = "standard"
    TRIAL = "trial"

class Permission(str, Enum):
    BANKING_WRITE = "banking:write"
    BANKING_READ = "banking:read"
    TRANSLATION_FULL = "translation:full"
    TRANSLATION_BASIC = "translation:basic"
    ASR_FULL = "asr:full"
    ASR_BASIC = "asr:basic"
    TTS_FULL = "tts:full"
    TTS_BASIC = "tts:basic"

class PermissionManager:
    def __init__(self):
        self.role_permissions: Dict[UserRole, Set[Permission]] = {
            UserRole.ADMIN: {
                Permission.BANKING_WRITE, Permission.BANKING_READ,
                Permission.TRANSLATION_FULL, Permission.ASR_FULL,
                Permission.TTS_FULL
            },
            UserRole.PREMIUM: {
                Permission.BANKING_READ,
                Permission.TRANSLATION_FULL,
                Permission.ASR_FULL,
                Permission.TTS_FULL
            },
            UserRole.STANDARD: {
                Permission.BANKING_READ,
                Permission.TRANSLATION_BASIC,
                Permission.ASR_BASIC,
                Permission.TTS_BASIC
            },
            UserRole.TRIAL: {
                Permission.TRANSLATION_BASIC,
                Permission.ASR_BASIC
            }
        }

        self.service_permissions: Dict[str, Dict[str, Permission]] = {
            'banking': {
                'transfer': Permission.BANKING_WRITE,
                'balance': Permission.BANKING_READ,
                'payment': Permission.BANKING_WRITE
            },
            'translation': {
                'basic': Permission.TRANSLATION_BASIC,
                'advanced': Permission.TRANSLATION_FULL
            },
            'asr': {
                'basic': Permission.ASR_BASIC,
                'full': Permission.ASR_FULL
            },
            'tts': {
                'basic': Permission.TTS_BASIC,
                'full': Permission.TTS_FULL
            }
        }

    def check_permission(self, role: UserRole, service: str, operation: str = 'basic') -> bool:
        """Check if role has permission for service operation"""
        if service not in self.service_permissions:
            return False

        required_permission = self.service_permissions[service].get(operation)
        if not required_permission:
            return False

        role_perms = self.role_permissions.get(role, set())
        return required_permission in role_perms

    def get_service_level(self, role: UserRole, service: str) -> str:
        """Get service level (basic/full) for role"""
        if service not in self.service_permissions:
            return 'none'

        role_perms = self.role_permissions.get(role, set())
        full_perm = self.service_permissions[service].get('full')
        basic_perm = self.service_permissions[service].get('basic')

        if full_perm in role_perms:
            return 'full'
        elif basic_perm in role_perms:
            return 'basic'
        return 'none' 