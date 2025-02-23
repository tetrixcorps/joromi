from fastapi import APIRouter, HTTPException
from typing import Dict, Any
from app.services.orchestrator import ModelOrchestrator
from app.models.request import ProcessRequest

router = APIRouter()
orchestrator = ModelOrchestrator()

@router.post("/process")
async def process_request(request: ProcessRequest):
    try:
        # Extract metadata and modality from request
        metadata = {
            "modality": request.modality,
            "domain": request.domain,
            "confidence_threshold": request.confidence_threshold
        }
        
        # Route to appropriate handler based on modality
        if request.modality == "text":
            return await orchestrator.process_text(request.content, metadata)
        elif request.modality == "image":
            return await orchestrator.process_image(request.content, metadata)
        elif request.modality == "speech":
            return await orchestrator.process_speech(request.content, metadata)
        else:
            raise HTTPException(status_code=400, message="Unsupported modality")
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) 