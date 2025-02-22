from fastapi import FastAPI, HTTPException
from services.base_service import BaseModelService
from config.model_config import ModelConfigurations
import torch

class BankingService(BaseModelService):
    def __init__(self, port: int = 8004):
        super().__init__("Banking", port)
        self.app = FastAPI()
        self.setup_routes()

    def setup_routes(self):
        @self.app.post("/process_query")
        async def process_query(query: str):
            return await self.process({"query": query})

        @self.app.get("/health")
        async def health():
            return await self.health_check()

    async def initialize(self):
        self.model, self.tokenizer = await self.model_manager.load_banking_model(
            ModelConfigurations.BANKING_LLM
        )
        if not self.model:
            raise RuntimeError("Failed to initialize Banking model")

    async def process(self, data: dict):
        try:
            inputs = self.tokenizer(
                data["query"],
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512
            ).to(self.device)
            
            with torch.no_grad():
                outputs = self.model.generate(**inputs)
            
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            return {"response": response}
            
        except Exception as e:
            self.logger.error(f"Banking query error: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e)) 