from abc import ABC, abstractmethod
from models.model_loader import ModelManager
from utils.logger import setup_logger
import torch
from fastapi import FastAPI, HTTPException
from prometheus_client import Counter, Histogram, multiprocess, CollectorRegistry, Gauge
import time

class BaseModelService(ABC):
    def __init__(self, model_name: str, service_port: int):
        self.logger = setup_logger(f"{model_name}_service")
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_manager = ModelManager(self.device)
        self.port = service_port
        self.model_name = model_name
        self.model = None
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

    @abstractmethod
    async def initialize(self):
        """Initialize the model and processor"""
        pass

    @abstractmethod
    async def process(self, input_data):
        """Process input data"""
        pass

    async def health_check(self):
        """Check service health"""
        return {
            "status": "healthy",
            "model_name": self.model_name,
            "device": str(self.device),
            "model_loaded": self.model is not None
        }

    async def track_request(self, endpoint: str):
        """Track request metrics"""
        start_time = time.time()
        try:
            self.request_counter.labels(
                service=self.model_name,
                endpoint=endpoint,
                status='success'
            ).inc()
            
            yield
            
        except Exception as e:
            self.request_counter.labels(
                service=self.model_name,
                endpoint=endpoint,
                status='error'
            ).inc()
            self.error_counter.labels(
                service=self.model_name,
                error_type=type(e).__name__
            ).inc()
            raise
        finally:
            self.request_latency.labels(
                service=self.model_name,
                endpoint=endpoint
            ).observe(time.time() - start_time) 