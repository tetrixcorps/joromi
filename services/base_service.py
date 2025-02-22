from abc import ABC, abstractmethod
from models.model_loader import ModelManager
from utils.logger import setup_logger
import torch
from fastapi import FastAPI, HTTPException
from prometheus_client import Counter, Histogram, multiprocess, CollectorRegistry, Gauge
import time
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Dict, Any, Optional

class BaseModelService(ABC):
    def __init__(self, port: int):
        self.logger = setup_logger(f"base_service")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_manager = ModelManager(self.device)
        self.port = port
        self.model = None
        self.tokenizer = None
        self.processor = None
        self.app = FastAPI()
        self.setup_metrics()

    def setup_metrics(self):
        """Setup Prometheus metrics"""
        registry = CollectorRegistry()
        multiprocess.MultiProcessCollector(registry)

        # Request metrics
        self.request_counter = Counter(
            'requests_total',
            'Total requests processed',
            ['service', 'endpoint', 'status']
        )

        # Latency metrics
        self.request_latency = Histogram(
            'request_latency_seconds',
            'Request latency in seconds',
            ['service', 'endpoint']
        )

        # Error metrics
        self.error_counter = Counter(
            'errors_total',
            'Total errors encountered',
            ['service', 'error_type']
        )

        # Model metrics
        self.model_load_time = Histogram(
            'model_load_time_seconds',
            'Time taken to load model',
            ['service', 'model']
        )

        # Resource metrics
        self.gpu_memory_used = Gauge(
            'gpu_memory_used_bytes',
            'GPU memory used in bytes',
            ['service', 'gpu_id']
        )

    async def initialize(self):
        """Initialize model and tokenizer"""
        try:
            self.model = await self.load_model()
            self.tokenizer = await self.load_tokenizer()
        except Exception as e:
            raise RuntimeError(f"Failed to initialize service: {e}")

    @abstractmethod
    async def load_model(self) -> AutoModelForCausalLM:
        """Load the model"""
        pass

    @abstractmethod
    async def load_tokenizer(self) -> AutoTokenizer:
        """Load the tokenizer"""
        pass

    @abstractmethod
    async def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process input data"""
        pass

    async def cleanup(self):
        """Cleanup resources"""
        if hasattr(self, 'model'):
            del self.model
        if hasattr(self, 'tokenizer'):
            del self.tokenizer
        torch.cuda.empty_cache()

    async def health_check(self):
        """Check service health"""
        return {
            "status": "healthy",
            "device": str(self.device),
            "model_loaded": self.model is not None
        }

    async def track_request(self, endpoint: str):
        """Track request metrics"""
        start_time = time.time()
        try:
            self.request_counter.labels(
                service="base_service",
                endpoint=endpoint,
                status='success'
            ).inc()
            
            yield
            
        except Exception as e:
            self.request_counter.labels(
                service="base_service",
                endpoint=endpoint,
                status='error'
            ).inc()
            self.error_counter.labels(
                service="base_service",
                error_type=type(e).__name__
            ).inc()
            raise
        finally:
            self.request_latency.labels(
                service="base_service",
                endpoint=endpoint
            ).observe(time.time() - start_time) 