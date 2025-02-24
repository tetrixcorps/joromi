from typing import Dict, List, Optional
import torch
import psutil
import logging
from prometheus_client import Counter, Gauge, Histogram

logger = logging.getLogger(__name__)

# Monitoring metrics
MODEL_CALLS = Counter(
    'model_calls_total',
    'Number of model calls',
    ['model_name', 'operation']
)

MODEL_ERRORS = Counter(
    'model_errors_total',
    'Number of model errors',
    ['model_name', 'error_type']
)

MODEL_LATENCY = Histogram(
    'model_latency_seconds',
    'Model inference latency',
    ['model_name']
)

GPU_MEMORY_USAGE = Gauge(
    'gpu_memory_usage_bytes',
    'GPU memory usage in bytes',
    ['device']
)

class ModelMonitor:
    def __init__(self):
        self.tracked_models = set()
        self.resource_thresholds = {
            'gpu_memory_percent': 90.0,
            'cpu_percent': 80.0,
            'memory_percent': 80.0
        }

    async def track_model_call(
        self,
        model_name: str,
        operation: str,
        duration: float,
        error: Optional[Exception] = None
    ):
        """Track model usage and performance"""
        MODEL_CALLS.labels(
            model_name=model_name,
            operation=operation
        ).inc()
        
        MODEL_LATENCY.labels(
            model_name=model_name
        ).observe(duration)
        
        if error:
            MODEL_ERRORS.labels(
                model_name=model_name,
                error_type=type(error).__name__
            ).inc()

    async def check_resource_usage(self) -> Dict[str, bool]:
        """Monitor system resource usage"""
        try:
            # Check GPU usage
            if torch.cuda.is_available():
                for i in range(torch.cuda.device_count()):
                    memory_percent = torch.cuda.memory_allocated(i) / torch.cuda.max_memory_allocated(i) * 100
                    GPU_MEMORY_USAGE.labels(device=f"gpu_{i}").set(memory_percent)
                    
            # Check CPU and memory
            cpu_percent = psutil.cpu_percent()
            memory_percent = psutil.virtual_memory().percent
            
            return {
                'gpu_ok': memory_percent < self.resource_thresholds['gpu_memory_percent'],
                'cpu_ok': cpu_percent < self.resource_thresholds['cpu_percent'],
                'memory_ok': memory_percent < self.resource_thresholds['memory_percent']
            }
            
        except Exception as e:
            logger.error(f"Resource check failed: {e}")
            return {'gpu_ok': False, 'cpu_ok': False, 'memory_ok': False} 