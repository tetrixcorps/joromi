import pytest
import asyncio
from httpx import AsyncClient

@pytest.mark.asyncio
async def test_text_to_speech_pipeline():
    async with AsyncClient() as client:
        # Test text-to-speech conversion
        response = await client.post(
            "http://api-gateway/tts",
            json={"text": "Hello, this is a test", "language": "en"}
        )
        assert response.status_code == 200
        assert "audio_url" in response.json()

@pytest.mark.asyncio
async def test_speech_to_text_pipeline():
    async with AsyncClient() as client:
        # Test speech-to-text conversion
        with open("tests/data/test-audio.wav", "rb") as f:
            response = await client.post(
                "http://api-gateway/stt",
                files={"audio": f}
            )
        assert response.status_code == 200
        assert "text" in response.json() 