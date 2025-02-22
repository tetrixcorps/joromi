from fastapi import FastAPI, HTTPException, UploadFile, File, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import torch
import uvicorn
from models.orchestrator import ModelOrchestrator
from utils.logger import api_logger, CustomLogger
import io
import aiofiles
import soundfile as sf
from PIL import Image

app = FastAPI(
    title="JoromiGPT API",
    description="Multi-modal AI system with support for text, speech, and vision",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize orchestrator
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
orchestrator = ModelOrchestrator(device)

# Request models
class TextRequest(BaseModel):
    text: str
    language: Optional[str] = "en"
    output_modality: Optional[str] = "text"
    accent: Optional[str] = None
    domain: Optional[str] = None

class SpeechRequest(BaseModel):
    language: Optional[str] = "en"
    output_modality: Optional[str] = "speech"
    accent: Optional[str] = None
    domain: Optional[str] = None

class ImageRequest(BaseModel):
    question: str
    language: Optional[str] = "en"
    output_modality: Optional[str] = "text"
    accent: Optional[str] = None

# Health check endpoint
@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "gpu_available": torch.cuda.is_available(),
        "device": str(device)
    }

# Text endpoint
@app.post("/text")
@CustomLogger.log_execution_time(api_logger)
async def process_text(request: TextRequest):
    try:
        api_logger.info(f"Processing text request in {request.language}")
        response = await orchestrator.process_request({
            "input_type": "text",
            "output_type": request.output_modality,
            "text": request.text,
            "language": request.language,
            "accent": request.accent,
            "domain": request.domain
        })
        return response
    except Exception as e:
        api_logger.error(f"Text processing failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Speech endpoint
@app.post("/speech")
@CustomLogger.log_execution_time(api_logger)
async def process_speech(
    request: SpeechRequest = None,
    file: UploadFile = File(...)
):
    try:
        api_logger.info(f"Processing speech request in {request.language}")
        
        # Read audio file
        contents = await file.read()
        audio_data, sr = sf.read(io.BytesIO(contents))
        
        response = await orchestrator.process_request({
            "input_type": "speech",
            "output_type": request.output_modality,
            "audio": audio_data,
            "sample_rate": sr,
            "language": request.language,
            "accent": request.accent,
            "domain": request.domain
        })
        return response
    except Exception as e:
        api_logger.error(f"Speech processing failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Image endpoint
@app.post("/image")
@CustomLogger.log_execution_time(api_logger)
async def process_image(
    request: ImageRequest,
    file: UploadFile = File(...)
):
    try:
        api_logger.info("Processing image request")
        
        # Read and process image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        
        response = await orchestrator.process_request({
            "input_type": "image",
            "output_type": request.output_modality,
            "image": image,
            "question": request.question,
            "language": request.language,
            "accent": request.accent
        })
        return response
    except Exception as e:
        api_logger.error(f"Image processing failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Get available accents
@app.get("/accents")
async def get_accents():
    """Return list of available African accents"""
    try:
        # This would be populated from your Afro-TTS model
        accents = [
            {"code": "ng", "name": "Nigerian"},
            {"code": "ke", "name": "Kenyan"},
            # ... other accents
        ]
        return {"accents": accents}
    except Exception as e:
        api_logger.error(f"Failed to fetch accents: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Error handler
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    api_logger.error(f"Global error handler: {str(exc)}")
    return {
        "status": "error",
        "message": str(exc),
        "path": request.url.path
    }

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        workers=1
    ) 