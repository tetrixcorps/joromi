from opentelemetry import trace
from opentelemetry.exporter.jaeger.thrift import JaegerExporter
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.sdk.resources import Resource
from opentelemetry.semconv.resource import ResourceAttributes

def setup_tracing():
    """Configure distributed tracing"""
    resource = Resource(attributes={
        ResourceAttributes.SERVICE_NAME: "ml-translation-service",
        ResourceAttributes.DEPLOYMENT_ENVIRONMENT: "production"
    })

    # Create and set tracer provider
    provider = TracerProvider(resource=resource)
    processor = BatchSpanProcessor(JaegerExporter())
    provider.add_span_processor(processor)
    trace.set_tracer_provider(provider) 