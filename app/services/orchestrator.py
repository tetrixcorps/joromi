from typing import Dict, Any
from app.models.base import BaseModel
from app.models.general import GeneralPurposeModel
from app.models.visual import VisualQAModel
from app.models.domain import DomainSpecificModel

class ModelOrchestrator:
    def __init__(self):
        self.general_model = GeneralPurposeModel()
        self.visual_model = VisualQAModel()
        self.domain_model = DomainSpecificModel()

    def _analyze_request(self, metadata: Dict[str, Any]) -> BaseModel:
        """
        Analyze request metadata to determine the appropriate model
        """
        # Check domain specificity
        if metadata.get("domain") and metadata.get("confidence_threshold", 0) > 0.8:
            return self.domain_model
            
        # Check modality
        if metadata.get("modality") == "image":
            return self.visual_model
            
        # Default to general purpose model
        return self.general_model

    async def process_text(self, content: str, metadata: Dict[str, Any]):
        model = self._analyze_request(metadata)
        return await model.process(content, metadata)

    async def process_image(self, image_data: bytes, metadata: Dict[str, Any]):
        model = self._analyze_request(metadata)
        return await model.process(image_data, metadata)

    async def process_speech(self, audio_data: bytes, metadata: Dict[str, Any]):
        model = self._analyze_request(metadata)
        return await model.process(audio_data, metadata) 