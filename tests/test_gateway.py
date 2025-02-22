import pytest
from fastapi.testclient import TestClient

def test_health_check(test_client: TestClient):
    response = test_client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"

@pytest.mark.asyncio
async def test_text_processing(test_client: TestClient):
    request_data = {
        "text": "Hello world",
        "source_lang": "en",
        "target_lang": "fr",
        "request_id": "test-123"
    }
    response = test_client.post("/process/text", json=request_data)
    assert response.status_code == 200
    assert "result" in response.json() 