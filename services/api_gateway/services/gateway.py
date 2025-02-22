from fastapi import FastAPI, WebSocket, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional
import logging
from .models import (
    ChatRequest, TranslationRequest, BankingRequest,
    ChatResponse, TranslationResponse, BankingResponse
)
from .services import ModelOrchestrator

logger = logging.getLogger(__name__)

class APIGateway:
    def __init__(self, service_discovery):
        self.app = FastAPI(
            title="ML Services API",
            description="API Gateway for ML Services",
            version="1.0.0"
        )
        self.service_discovery = service_discovery
        self.model_orchestrator = ModelOrchestrator()
        self._setup_routes()
        self._setup_middleware()

    def _setup_middleware(self):
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

    def _setup_routes(self):
        # Add your routes here
        pass 