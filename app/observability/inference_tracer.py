from opentelemetry import trace, metrics
from opentelemetry.trace import Status, StatusCode
import torch
import time
from typing import Dict, Any, Optional
import logging
from dataclasses import dataclass
from prometheus_client import Histogram, Counter, Gauge

logger = logging.getLogger(__name__)

@dataclass
class TraceContext:
    model_name: str
    input_type: str
    batch_size: int
    device: str
    metadata: Dict[str, Any]

class InferenceTracer:
    def __init__(self):
        self.tracer = trace.get_tracer(__name__)
        self.metrics = self._setup_metrics()
        
    def _setup_metrics(self):
        """Initialize metrics collectors"""
        return {
            "inference_duration": Histogram(
                "model_inference_duration_seconds",
                "Time spent on model inference",
                ["model_name", "input_type"]
            ),
            "gpu_memory_usage": Gauge(
                "gpu_memory_usage_bytes",
                "GPU memory usage during inference",
                ["device"]
            ),
            "batch_size": Histogram(
                "inference_batch_size",
                "Batch sizes processed",
                ["model_name"]
            ),
            "feature_extraction_time": Histogram(
                "feature_extraction_duration_seconds",
                "Time spent on feature extraction",
                ["input_type"]
            )
        }

    async def trace_inference(
        self,
        model_input: Any,
        context: TraceContext,
        inference_func: callable
    ) -> Any:
        """Trace model inference with detailed metrics"""
        try:
            with self.tracer.start_as_current_span("model_inference") as span:
                # Set span attributes
                span.set_attributes({
                    "model.name": context.model_name,
                    "input.type": context.input_type,
                    "batch.size": context.batch_size,
                    "device": context.device
                })

                # Record batch size
                self.metrics["batch_size"].labels(
                    model_name=context.model_name
                ).observe(context.batch_size)

                # Feature extraction tracing
                with self.tracer.start_span("feature_extraction") as feat_span:
                    start_time = time.time()
                    features = await self._trace_feature_extraction(model_input, context)
                    self.metrics["feature_extraction_time"].labels(
                        input_type=context.input_type
                    ).observe(time.time() - start_time)
                    feat_span.set_status(Status(StatusCode.OK))

                # GPU memory tracking
                if torch.cuda.is_available():
                    with torch.cuda.nvtx.range("gpu_inference"):
                        self._track_gpu_memory(context.device)

                # Model inference tracing
                start_time = time.time()
                result = await inference_func(features)
                inference_time = time.time() - start_time

                # Record metrics
                self.metrics["inference_duration"].labels(
                    model_name=context.model_name,
                    input_type=context.input_type
                ).observe(inference_time)

                # Add inference events
                span.add_event(
                    "inference_complete",
                    attributes={
                        "duration": inference_time,
                        "output_shape": str(result.shape) if hasattr(result, "shape") else None
                    }
                )

                return result

        except Exception as e:
            logger.error(f"Inference tracing failed: {e}")
            if span:
                span.set_status(Status(StatusCode.ERROR, str(e)))
            raise

    async def _trace_feature_extraction(
        self,
        model_input: Any,
        context: TraceContext
    ) -> Any:
        """Trace feature extraction process"""
        with self.tracer.start_span("feature_processing") as span:
            try:
                if context.input_type == "audio":
                    return await self._process_audio_features(model_input, span)
                elif context.input_type == "text":
                    return await self._process_text_features(model_input, span)
                else:
                    return model_input

            except Exception as e:
                span.set_status(Status(StatusCode.ERROR, str(e)))
                raise

    def _track_gpu_memory(self, device: str):
        """Track GPU memory usage"""
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                memory_allocated = torch.cuda.memory_allocated(i)
                self.metrics["gpu_memory_usage"].labels(
                    device=f"gpu_{i}"
                ).set(memory_allocated)

    async def _process_audio_features(self, audio_input: Any, span: trace.Span) -> Any:
        """Process and trace audio feature extraction"""
        span.set_attribute("feature_type", "audio")
        
        with self.tracer.start_span("audio_preprocessing") as audio_span:
            # Add audio-specific processing events
            audio_span.add_event("normalization")
            audio_span.add_event("feature_extraction")
            
            return audio_input

    async def _process_text_features(self, text_input: Any, span: trace.Span) -> Any:
        """Process and trace text feature extraction"""
        span.set_attribute("feature_type", "text")
        
        with self.tracer.start_span("text_preprocessing") as text_span:
            # Add text-specific processing events
            text_span.add_event("tokenization")
            text_span.add_event("embedding")
            
            return text_input 