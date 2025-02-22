from fastapi import FastAPI, HTTPException
from services.base_service import BaseModelService
from config.model_config import ModelConfigurations
import torch

class TTSService(BaseModelService):
    def __init__(self, port: int = 8003):
        super().__init__("TTS", port)
        self.app = FastAPI()
        self.setup_routes()

    def setup_routes(self):
        @self.app.post("/synthesize")
        async def synthesize(text: str, language: str):
            return await self.process({"text": text, "language": language})

        @self.app.get("/health")
        async def health():
            return await self.health_check()

    async def initialize(self):
        self.model, self.processor = await self.model_manager.load_tts_model(
            ModelConfigurations.TTS
        )
        if not self.model:
            raise RuntimeError("Failed to initialize TTS model")

    async def process(self, data: dict):
        try:
            inputs = self.processor(
                text=data["text"],
                language=data["language"],
                return_tensors="pt"
            ).to(self.device)
            
            with torch.no_grad():
                speech = self.model.generate_speech(**inputs)
            
            return {"audio": speech.cpu().numpy().tobytes()}
            
        except Exception as e:
            self.logger.error(f"TTS processing error: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e)) 