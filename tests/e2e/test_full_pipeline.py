import pytest
import asyncio
from httpx import AsyncClient

@pytest.mark.asyncio
async def test_complete_pipeline():
    async with AsyncClient() as client:
        # Test the complete pipeline
        # 1. Convert text to speech
        tts_response = await client.post(
            "http://api-gateway/tts",
            json={"text": "Hello, this is a test", "language": "en"}
        )
        assert tts_response.status_code == 200
        audio_url = tts_response.json()["audio_url"]
        
        # 2. Download the audio
        audio_response = await client.get(audio_url)
        assert audio_response.status_code == 200
        
        # 3. Convert speech back to text
        stt_response = await client.post(
            "http://api-gateway/stt",
            files={"audio": audio_response.content}
        )
        assert stt_response.status_code == 200
        text = stt_response.json()["text"]
        
        # 4. Verify the text matches (approximately)
        assert "hello this is a test" in text.lower() 