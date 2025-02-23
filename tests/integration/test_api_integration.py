import pytest
from httpx import AsyncClient

@pytest.mark.asyncio
async def test_chat_endpoint(test_client):
    response = await test_client.post("/chat", json={
        "messages": [{
            "type": "text",
            "content": "Hello, how are you?"
        }]
    })
    
    assert response.status_code == 200
    data = response.json()
    assert "response" in data 