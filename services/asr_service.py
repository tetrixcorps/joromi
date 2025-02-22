from fastapi import FastAPI, HTTPException
from services.base_service import BaseModelService
from config.model_config import ModelConfigurations
import torch

class ASRService(BaseModelService):
    def __init__(self, port: int = 8001):
        super().__init__("ASR", port)
        self.app = FastAPI()
        self.setup_routes()

    def setup_routes(self):
        @self.app.post("/transcribe")
        async def transcribe(audio_file: bytes):
            return await self.process(audio_file)

        @self.app.get("/health")
        async def health():
            return await self.health_check()

    async def initialize(self):
        self.model, self.processor = await self.model_manager.load_asr_model(
            ModelConfigurations.ASR
        )
        if not self.model:
            raise RuntimeError("Failed to initialize ASR model")

    async def process(self, audio_data: bytes):
        try:
            # Process audio data
            inputs = self.processor(audio_data, return_tensors="pt").to(self.device)
            with torch.no_grad():
                outputs = self.model.generate(**inputs)
            
            transcription = self.processor.decode(outputs[0])
            return {"text": transcription}
            
        except Exception as e:
            self.logger.error(f"ASR processing error: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e)) 