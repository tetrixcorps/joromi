import pytest
from services.dolphin_service import DolphinService
import torch
from unittest.mock import AsyncMock, patch

@pytest.mark.asyncio(loop_scope="function")
async def test_dolphin_service_initialization():
    service = DolphinService(port=8000)
    assert service.port == 8000
    assert service.max_length == 2048

@pytest.mark.asyncio(loop_scope="function")
async def test_dolphin_process():
    service = DolphinService(port=8000)
    await service.initialize()
    
    result = await service.process({
        "prompt": "Hello, how are you?",
        "system_prompt": "You are a helpful assistant"
    })
    
    assert "response" in result
    assert "status" in result
    assert result["status"] == "success" 