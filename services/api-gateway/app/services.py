from typing import AsyncGenerator, Optional
import aiohttp
import logging
from .models import ChatMessage, BankingQueryType

logger = logging.getLogger(__name__)

class ModelOrchestrator:
    def __init__(self):
        self.asr_service_url = "http://asr-service:8001"
        self.tts_service_url = "http://tts-service:8002"
        self.translation_service_url = "http://translation-service:8003"
        self.visual_qa_service_url = "http://visual-qa-service:8004"
        self.banking_service_url = "http://banking-service:8005"

    async def process_chat(self, request):
        """
        Process multimodal chat requests by routing to appropriate services.
        """
        processed_messages = []
        for message in request.messages:
            if message.type == "audio":
                # Process audio through ASR
                text = await self.speech_to_text(message.content)
                processed_messages.append(ChatMessage(type="text", content=text))
            elif message.type == "image":
                # Process image through visual QA
                description = await self.process_visual_qa(message.content, "Describe this image")
                processed_messages.append(ChatMessage(type="text", content=description))
            else:
                processed_messages.append(message)

        # Process the final text-based chat
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{self.banking_service_url}/chat",
                json={"messages": [m.dict() for m in processed_messages]}
            ) as response:
                result = await response.json()
                return result

    async def speech_to_text(self, audio_data: bytes, language: str = "en") -> str:
        """
        Convert speech to text using the ASR service.
        """
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{self.asr_service_url}/transcribe",
                data={"audio": audio_data, "language": language}
            ) as response:
                result = await response.json()
                return result["text"]

    async def text_to_speech(
        self,
        text: str,
        language: str = "en",
        voice: str = "default"
    ) -> str:
        """
        Convert text to speech using the TTS service.
        """
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{self.tts_service_url}/synthesize",
                json={"text": text, "language": language, "voice": voice}
            ) as response:
                result = await response.json()
                return result["audio_url"]

    async def stream_chat_response(self, message: str) -> AsyncGenerator[str, None]:
        """
        Stream chat responses for real-time interaction.
        """
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{self.banking_service_url}/stream-chat",
                json={"message": message},
                chunked=True
            ) as response:
                async for chunk in response.content:
                    yield chunk.decode()

    async def check_service_health(self, service_url: str) -> str:
        """
        Check the health of a service.
        """
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{service_url}/health", timeout=5) as response:
                    if response.status == 200:
                        return "healthy"
                    return "degraded"
        except Exception as e:
            logger.error(f"Health check failed for {service_url}: {str(e)}")
            return "unhealthy"

    # Additional methods for translation, visual QA, and banking queries... 