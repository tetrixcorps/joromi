import pytest
import sys
from pathlib import Path

# Add the project root to Python path
root_dir = Path(__file__).parent.parent
sys.path.insert(0, str(root_dir))

# Now we can import from our package
from services.gateway import APIGateway
from services.discovery import ServiceDiscovery

@pytest.fixture
def orchestrator():
    from services.services import ModelOrchestrator
    return ModelOrchestrator()

@pytest.fixture(scope="session")
def event_loop():
    import asyncio
    loop = asyncio.get_event_loop()
    yield loop
    loop.close()

@pytest.fixture
async def mock_discovery():
    class MockDiscovery(ServiceDiscovery):
        async def get_service_instances(self, service_name: str):
            return [f"http://localhost:8001"]
    return MockDiscovery()

@pytest.fixture
async def gateway_app(mock_discovery):
    gateway = APIGateway(mock_discovery)
    return gateway.app

@pytest.fixture
def test_client(gateway_app):
    from fastapi.testclient import TestClient
    return TestClient(gateway_app) 