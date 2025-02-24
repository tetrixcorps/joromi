from prometheus_client import Histogram, Counter, Gauge

# Audio processing metrics
AUDIO_COMPRESSION_RATIO = Histogram(
    'audio_compression_ratio',
    'Audio compression ratio achieved',
    ['client_id']
)

AUDIO_PROCESSING_LATENCY = Histogram(
    'audio_processing_latency_seconds',
    'Audio processing latency',
    ['operation']
)

AUDIO_ERROR_COUNT = Counter(
    'audio_processing_errors_total',
    'Number of audio processing errors',
    ['error_type']
)

# Stream metrics
ACTIVE_STREAMS = Gauge(
    'active_audio_streams',
    'Number of active audio streams'
)

STREAM_BANDWIDTH = Counter(
    'audio_stream_bytes_total',
    'Total bytes processed in audio streams',
    ['type']  # original/compressed
)

# Dialect analysis metrics
DIALECT_DETECTION_LATENCY = Histogram(
    'dialect_detection_latency_seconds',
    'Dialect detection processing time',
    ['language']
)

DIALECT_CONFIDENCE = Histogram(
    'dialect_confidence',
    'Confidence scores for dialect detection',
    ['language', 'dialect']
)

DIALECT_FEATURE_SCORES = Histogram(
    'dialect_feature_scores',
    'Feature importance scores for dialect analysis',
    ['language', 'feature']
)

# Audio security metrics
AUDIO_SECURITY_METRICS = Counter(
    'audio_security_checks_total',
    'Number of audio security checks',
    ['check_type', 'result']
)

AUDIO_THREATS_DETECTED = Counter(
    'audio_threats_detected_total',
    'Number of audio threats detected',
    ['threat_type']
)

AUDIO_SANITIZATION_TIME = Histogram(
    'audio_sanitization_seconds',
    'Time taken for audio sanitization',
    ['operation']
)

# Model optimization metrics
MODEL_OPTIMIZATION_METRICS = Histogram(
    "model_optimization_metrics",
    "Metrics for model optimization operations",
    ["operation", "metric"]
)

OPTIMIZATION_MEMORY_USAGE = Gauge(
    "optimization_memory_usage_bytes",
    "Memory usage during optimization",
    ["stage"]
)

OPTIMIZATION_TIME = Histogram(
    "optimization_duration_seconds",
    "Time taken for optimization steps",
    ["step"]
) 