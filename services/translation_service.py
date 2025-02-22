from fastapi import FastAPI, HTTPException
from services.base_service import BaseModelService
from config.model_config import ModelConfigurations

class TranslationService(BaseModelService):
    def __init__(self, port: int = 8002):
        super().__init__("Translation", port)
        self.app = FastAPI()
        self.setup_routes()

    def setup_routes(self):
        @self.app.post("/translate")
        async def translate(text: str, source_lang: str, target_lang: str):
            return await self.process({
                "text": text,
                "source_lang": source_lang,
                "target_lang": target_lang
            })

        @self.app.get("/health")
        async def health():
            return await self.health_check()

    async def initialize(self):
        self.model, self.processor = await self.model_manager.load_translation_model(
            ModelConfigurations.TRANSLATION
        )
        if not self.model:
            raise RuntimeError("Failed to initialize translation model")

    async def process(self, data: dict):
        try:
            inputs = self.processor(
                text=data["text"],
                src_lang=data["source_lang"],
                tgt_lang=data["target_lang"],
                return_tensors="pt"
            ).to(self.device)
            
            outputs = self.model.generate(**inputs)
            translation = self.processor.decode(outputs[0])
            
            return {"translated_text": translation}
            
        except Exception as e:
            self.logger.error(f"Translation error: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e)) 