import pytest
import sys
from pathlib import Path
import asyncio

# Add the project root to Python path
root_dir = Path(__file__).parent.parent
sys.path.insert(0, str(root_dir))

# Now we can import from our package
from services.gateway import APIGateway
from services.discovery import ServiceDiscovery

@pytest.fixture(scope="session")
def event_loop():
    """Create event loop for all async tests"""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
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
async def test_client():
    """Create test client for API testing"""
    from services.api_gateway.app.main import app
    from httpx import AsyncClient
    async with AsyncClient(app=app, base_url="http://test") as client:
        yield client

@pytest.fixture
def orchestrator():
    from services.api_gateway.services.services import ModelOrchestrator
    return ModelOrchestrator() 