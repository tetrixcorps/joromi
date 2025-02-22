from fastapi import FastAPI, HTTPException
from .services.models import ChatRequest
from .services.services import ModelOrchestrator
from fastapi.responses import StreamingResponse
import logging

app = FastAPI()
orchestrator = ModelOrchestrator()
logger = logging.getLogger(__name__)

@app.post("/chat")
async def chat(request: ChatRequest):
    """Process a chat request with text, audio, or image messages"""
    try:
        return await orchestrator.process_chat(request)
    except Exception as e:
        logger.error(f"Error processing chat request: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/stream-chat")
async def stream_chat(message: str):
    """Stream chat responses for real-time interaction"""
    try:
        return StreamingResponse(
            orchestrator.stream_chat_response(message),
            media_type="text/plain"
        )
    except Exception as e:
        logger.error(f"Error in streaming chat: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/speech-to-text")
async def speech_to_text(audio: bytes, language: str = "en"):
    """Convert speech to text"""
    try:
        return await orchestrator.speech_to_text(audio, language)
    except Exception as e:
        logger.error(f"Error in speech to text: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/text-to-speech")
async def text_to_speech(text: str, language: str = "en", voice: str = "default"):
    """Convert text to speech"""
    try:
        return await orchestrator.text_to_speech(text, language, voice)
    except Exception as e:
        logger.error(f"Error in text to speech: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/translate")
async def translate(text: str, target_lang: str, source_lang: str = None):
    """Translate text between languages"""
    try:
        return await orchestrator.translate(text, source_lang, target_lang)
    except Exception as e:
        logger.error(f"Error in translation: {e}")
        raise HTTPException(status_code=500, detail=str(e)) 