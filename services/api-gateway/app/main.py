from fastapi import FastAPI, WebSocket, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional
import asyncio
import logging

from .models import (
    ChatRequest,
    TranslationRequest,
    BankingRequest,
    ChatResponse,
    TranslationResponse,
    BankingResponse
)
from .services import ModelOrchestrator

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="ML Services API",
    description="API Gateway for ML Services",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize model orchestrator
model_orchestrator = ModelOrchestrator()

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    """
    Process multimodal chat requests.
    """
    try:
        response = await model_orchestrator.process_chat(request)
        return response
    except Exception as e:
        logger.error(f"Error in chat endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/asr")
async def speech_to_text(
    audio: UploadFile = File(...),
    language: Optional[str] = Form("en")
):
    """
    Convert speech to text using ASR.
    """
    try:
        text = await model_orchestrator.speech_to_text(audio, language)
        return {"text": text}
    except Exception as e:
        logger.error(f"Error in ASR endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/tts")
async def text_to_speech(
    text: str = Form(...),
    language: str = Form("en"),
    voice: Optional[str] = Form("default")
):
    """
    Convert text to speech with optional voice selection.
    """
    try:
        audio_url = await model_orchestrator.text_to_speech(text, language, voice)
        return {"audio_url": audio_url}
    except Exception as e:
        logger.error(f"Error in TTS endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/translation", response_model=TranslationResponse)
async def translate_text(request: TranslationRequest):
    """
    Translate text between languages.
    """
    try:
        translation = await model_orchestrator.translate(
            request.text,
            request.source_lang,
            request.target_lang
        )
        return {"translated_text": translation}
    except Exception as e:
        logger.error(f"Error in translation endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/visual-qa")
async def visual_question_answering(
    image: UploadFile = File(...),
    question: str = Form(...)
):
    """
    Answer questions about uploaded images.
    """
    try:
        answer = await model_orchestrator.process_visual_qa(image, question)
        return {"answer": answer}
    except Exception as e:
        logger.error(f"Error in visual QA endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/banking", response_model=BankingResponse)
async def process_banking_query(request: BankingRequest):
    """
    Process banking-specific queries.
    """
    try:
        response = await model_orchestrator.process_banking_query(request)
        return response
    except Exception as e:
        logger.error(f"Error in banking endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.websocket("/ws/chat")
async def websocket_chat(websocket: WebSocket):
    """
    WebSocket endpoint for real-time chat.
    """
    await websocket.accept()
    try:
        while True:
            message = await websocket.receive_text()
            response_stream = model_orchestrator.stream_chat_response(message)
            async for response_chunk in response_stream:
                await websocket.send_text(response_chunk)
    except Exception as e:
        logger.error(f"Error in WebSocket chat: {str(e)}")
        await websocket.close(code=1000)

@app.post("/text", response_model=ChatResponse, tags=["Basic Endpoints"])
async def text_endpoint(request: ChatRequest):
    """
    Simple text processing endpoint.
    - **text**: The input text from the user
    """
    try:
        # Create a chat message and use existing chat processing
        message = ChatMessage(type=MessageType.TEXT, content=request.messages[0].content)
        response = await model_orchestrator.process_chat(ChatRequest(messages=[message]))
        return response
    except Exception as e:
        logger.error(f"Text processing failed: {str(e)}")
        raise HTTPException(status_code=500, detail="Text processing failed")

@app.post("/speech", tags=["Basic Endpoints"])
async def speech_endpoint(
    file: UploadFile = File(...),
    language: Optional[str] = Form("en")
):
    """
    Simple speech processing endpoint.
    - **file**: Upload audio file (e.g., WAV, MP3)
    - **language**: Optional language code (default: en)
    """
    try:
        # Use existing ASR service
        text = await model_orchestrator.speech_to_text(file, language)
        return {"text": text}
    except Exception as e:
        logger.error(f"Speech processing failed: {str(e)}")
        raise HTTPException(status_code=500, detail="Speech processing failed")

@app.post("/image", tags=["Basic Endpoints"])
async def image_endpoint(
    file: UploadFile = File(...),
    question: str = Form(...)
):
    """
    Simple image processing endpoint.
    - **file**: Upload image file (e.g., PNG, JPEG)
    - **question**: The question about the image
    """
    try:
        # Use existing visual QA service
        answer = await model_orchestrator.process_visual_qa(file, question)
        return {"answer": answer}
    except Exception as e:
        logger.error(f"Image processing failed: {str(e)}")
        raise HTTPException(status_code=500, detail="Image processing failed")

@app.get("/health", tags=["System"])
async def health_check():
    """
    Health check endpoint to verify the API is running.
    Returns status of all connected services.
    """
    try:
        # Check connection to all services
        services_status = {
            "api_gateway": "healthy",
            "asr_service": await check_service_health(model_orchestrator.asr_service_url),
            "tts_service": await check_service_health(model_orchestrator.tts_service_url),
            "translation_service": await check_service_health(model_orchestrator.translation_service_url),
            "visual_qa_service": await check_service_health(model_orchestrator.visual_qa_service_url),
            "banking_service": await check_service_health(model_orchestrator.banking_service_url)
        }
        return {
            "status": "ok",
            "services": services_status
        }
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return {
            "status": "degraded",
            "error": str(e)
        } 