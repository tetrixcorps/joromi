from prometheus_client import Counter, Histogram, Info
from functools import wraps
import time

# Metrics
REQUEST_COUNT = Counter(
    'ml_service_requests_total',
    'Total ML service requests',
    ['endpoint', 'model', 'status']
)

LATENCY_HISTOGRAM = Histogram(
    'ml_service_latency_seconds',
    'Request latency in seconds',
    ['endpoint', 'model']
)

MODEL_INFO = Info('ml_model_info', 'Information about loaded models')

def track_request(endpoint: str, model: str):
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = await func(*args, **kwargs)
                REQUEST_COUNT.labels(
                    endpoint=endpoint,
                    model=model,
                    status='success'
                ).inc()
                return result
            except Exception as e:
                REQUEST_COUNT.labels(
                    endpoint=endpoint,
                    model=model,
                    status='error'
                ).inc()
                raise e
            finally:
                LATENCY_HISTOGRAM.labels(
                    endpoint=endpoint,
                    model=model
                ).observe(time.time() - start_time)
        return wrapper
    return decorator 