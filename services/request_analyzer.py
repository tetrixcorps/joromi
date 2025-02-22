from typing import Dict, List, Tuple
from enum import Enum
import re
from utils.logger import setup_logger

logger = setup_logger('request_analyzer')

class InputModality(Enum):
    TEXT = "text"
    SPEECH = "speech"
    IMAGE = "image"
    MULTIMODAL = "multimodal"

class Domain(Enum):
    GENERAL = "general"
    BANKING = "banking"
    TRANSLATION = "translation"
    VISUAL = "visual"

class RequestAnalyzer:
    def __init__(self):
        self.banking_keywords = {
            'account', 'balance', 'transfer', 'payment', 'transaction',
            'credit', 'debit', 'loan', 'interest', 'bank'
        }
        
        self.translation_indicators = {
            'translate', 'translation', 'convert', 'language',
            'from english to', 'to spanish', 'in french'
        }

    async def analyze_request(self, request_data: dict) -> Tuple[InputModality, Domain, List[str]]:
        """Analyze request to determine input modality, domain, and required services"""
        
        # Determine input modality
        modality = self._detect_modality(request_data)
        
        # Determine domain
        domain = self._detect_domain(request_data)
        
        # Determine required services
        services = self._determine_services(modality, domain, request_data)
        
        logger.info(f"Request analysis: modality={modality}, domain={domain}, services={services}")
        return modality, domain, services

    def _detect_modality(self, request_data: dict) -> InputModality:
        """Detect input modality from request data"""
        if 'audio' in request_data:
            return InputModality.SPEECH
        elif 'image' in request_data:
            return InputModality.IMAGE
        elif all(key in request_data for key in ['text', 'image']):
            return InputModality.MULTIMODAL
        else:
            return InputModality.TEXT

    def _detect_domain(self, request_data: dict) -> Domain:
        """Detect domain from request content"""
        text = request_data.get('text', '').lower()
        
        # Check for banking domain
        if any(keyword in text for keyword in self.banking_keywords):
            return Domain.BANKING
            
        # Check for translation domain
        if any(indicator in text for indicator in self.translation_indicators):
            return Domain.TRANSLATION
            
        # Check for visual domain
        if 'image' in request_data:
            return Domain.VISUAL
            
        return Domain.GENERAL

    def _determine_services(self, modality: InputModality, domain: Domain, request_data: dict) -> List[str]:
        """Determine required services based on modality and domain"""
        services = []
        
        # Add input processing services
        if modality == InputModality.SPEECH:
            services.append("asr")
            
        # Add domain-specific services
        if domain == Domain.BANKING:
            services.append("banking")
        elif domain == Domain.TRANSLATION:
            services.append("translation")
            if request_data.get('target_modality') == 'speech':
                services.append("tts")
                
        # Add output processing services
        if request_data.get('response_format') == 'speech':
            services.append("tts")
            
        return services 