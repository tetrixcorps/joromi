import pytest
from app.services.orchestrator import ModelOrchestrator
from app.models.base import BaseModel

@pytest.fixture
def orchestrator():
    return ModelOrchestrator()

async def test_analyze_request_domain_specific(orchestrator):
    metadata = {
        "domain": "medical",
        "confidence_threshold": 0.9,
        "modality": "text"
    }
    model = orchestrator._analyze_request(metadata)
    assert isinstance(model, DomainSpecificModel)

async def test_analyze_request_visual(orchestrator):
    metadata = {
        "modality": "image",
        "confidence_threshold": 0.5
    }
    model = orchestrator._analyze_request(metadata)
    assert isinstance(model, VisualQAModel)

async def test_process_text(orchestrator):
    metadata = {
        "modality": "text",
        "domain": "general"
    }
    result = await orchestrator.process_text("Hello", metadata)
    assert result is not None 