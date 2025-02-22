from typing import AsyncGenerator, Optional
import aiohttp
import logging
from pydantic import BaseModel
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
        """Process multimodal chat requests by routing to appropriate services."""
        processed_messages = []
        for message in request.messages:
            if message.type == "audio":
                text = await self.speech_to_text(message.content)
                processed_messages.append(ChatMessage(type="text", content=text))
            elif message.type == "image":
                description = await self.process_visual_qa(message.content, "Describe this image")
                processed_messages.append(ChatMessage(type="text", content=description))
            else:
                processed_messages.append(message)

        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{self.banking_service_url}/chat",
                json={"messages": [m.model_dump() for m in processed_messages]}
            ) as response:
                return await response.json()

    async def speech_to_text(self, audio_data: bytes, language: str = "en") -> str:
        """Convert speech to text using the ASR service."""
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{self.asr_service_url}/transcribe",
                data={"audio": audio_data, "language": language}
            ) as response:
                result = await response.json()
                return result["text"]

    async def text_to_speech(self, text: str, language: str = "en", voice: str = "default") -> str:
        """Convert text to speech using the TTS service."""
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{self.tts_service_url}/synthesize",
                json={"text": text, "language": language, "voice": voice}
            ) as response:
                result = await response.json()
                return result["audio_url"]

    async def stream_chat_response(self, message: str) -> AsyncGenerator[str, None]:
        """Stream chat responses for real-time interaction."""
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{self.banking_service_url}/stream-chat",
                json={"message": message},  # Simplified - message is always a string
                chunked=True
            ) as response:
                async for chunk in response.content:
                    yield chunk.decode()

    async def process_visual_qa(self, image_data: bytes, question: str) -> str:
        """Process visual question answering."""
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{self.visual_qa_service_url}/analyze",
                data={"image": image_data, "question": question}
            ) as response:
                result = await response.json()
                # Try to get answer, fall back to description if answer not found
                if "answer" not in result:
                    logger.warning("Visual QA service response missing 'answer' key")
                    if "description" in result:
                        return result["description"]
                    raise KeyError("Visual QA service response missing both 'answer' and 'description' keys")
                return result["answer"]

    async def translate(self, text: str, source_lang: Optional[str], target_lang: str) -> str:
        """Translate text between languages."""
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{self.translation_service_url}/translate",
                json={
                    "text": text,
                    "source_lang": source_lang,
                    "target_lang": target_lang
                }
            ) as response:
                result = await response.json()
                return result["translated_text"] 