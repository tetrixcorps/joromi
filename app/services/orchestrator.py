from typing import Dict, Any, Optional
from app.models.base import BaseModel
from app.models.general import GeneralPurposeModel
from app.models.visual import VisualQAModel
from app.models.domain import DomainSpecificModel
from app.services.asr import ASRService
from app.services.translation import TranslationService
from app.models.domain import BankingLLMModel, MedicalLLMModel, GeneralLLMModel
from app.models.visual import Pix2StructModel
from app.models.general import MiniCPMModel
import logging

logger = logging.getLogger(__name__)

class ModelOrchestrator:
    def __init__(self):
        self.general_model = MiniCPMModel()
        self.visual_model = Pix2StructModel()
        self.domain_models = {
            'banking': BankingLLMModel(),
            'medical': MedicalLLMModel(),
            'general': GeneralLLMModel()
        }
        self.asr_service = ASRService()
        self.translation_service = TranslationService()

    def _analyze_request(self, metadata: Dict[str, Any]) -> BaseModel:
        """
        Analyze request metadata to determine the appropriate model
        """
        # Check domain specificity
        if metadata.get("domain") and metadata.get("confidence_threshold", 0) > 0.8:
            return self.domain_models[metadata['domain']]
            
        # Check modality
        if metadata.get("modality") == "image":
            return self.visual_model
            
        # Default to general purpose model
        return self.general_model

    async def process_with_language(self, 
                                  content: str, 
                                  metadata: Dict[str, Any],
                                  target_lang: Optional[str] = None) -> Dict[str, Any]:
        """Process request with language handling"""
        try:
            # Detect source language if not specified
            source_lang = metadata.get("source_lang")
            if not source_lang:
                source_lang = await self.translation_service.detect_language(content)
                metadata["source_lang"] = source_lang

            # Translate input to English for model processing if needed
            if source_lang != "eng":
                content = await self.translation_service.translate(content, target_lang="eng", source_lang=source_lang)

            # Process with appropriate model
            model = self._analyze_request(metadata)
            response = await model.process(content, metadata)

            # Translate response if needed
            if target_lang and target_lang != "eng":
                response["text"] = await self.translation_service.translate(
                    response["text"],
                    target_lang=target_lang,
                    source_lang="eng"
                )
                response["language"] = target_lang

            return response

        except Exception as e:
            logger.error(f"Language processing failed: {e}")
            return {
                "text": "An error occurred during language processing.",
                "error": str(e),
                "language": "eng"
            }

    async def process_text(self, content: str, metadata: Dict[str, Any]):
        """Override existing process_text to include language handling"""
        target_lang = metadata.get("target_lang", "eng")
        return await self.process_with_language(content, metadata, target_lang)

    async def process_image(self, image_data: bytes, metadata: Dict[str, Any]):
        model = self._analyze_request(metadata)
        return await model.process(image_data, metadata)

    async def process_audio(self, audio_data: bytes, metadata: Dict[str, Any]):
        # Handle audio input
        transcription = await self.asr_service.transcribe(audio_data)
        translated_text = await self.translation_service.translate(transcription)
        return await self.process_text(translated_text, metadata) 