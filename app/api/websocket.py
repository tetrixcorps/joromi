from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from typing import Dict, List, Any
import json
from app.services.orchestrator import ModelOrchestrator
from app.services.asr import ASRService
from app.services.translation import TranslationService

router = APIRouter()

class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []
        self.orchestrator = ModelOrchestrator()
        self.asr_service = ASRService()
        self.translation_service = TranslationService()

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def process_audio_stream(self, websocket: WebSocket, audio_chunk: bytes, metadata: Dict[str, Any]):
        try:
            # Notify client that transcription is starting
            await websocket.send_json({
                "type": "transcription_start"
            })

            # Process audio through ASR
            transcription = await self.asr_service.transcribe(audio_chunk)
            
            # Detect language and update metadata
            source_lang = await self.translation_service.detect_language(transcription)
            metadata.update({
                "source_lang": source_lang,
                "target_lang": metadata.get("target_lang", source_lang)
            })

            # Send transcription to client
            await websocket.send_json({
                "type": "transcription",
                "text": transcription,
                "language": source_lang
            })

            # Process through orchestrator with language handling
            response = await self.orchestrator.process_with_language(
                transcription,
                metadata,
                target_lang=metadata["target_lang"]
            )

            # Send model response to client
            await websocket.send_json({
                "type": "response",
                "text": response["text"],
                "language": response.get("language", metadata["target_lang"]),
                "synthesize_speech": True
            })

        except Exception as e:
            await websocket.send_json({
                "type": "error",
                "message": str(e)
            })
        finally:
            await websocket.send_json({
                "type": "transcription_end"
            })

manager = ConnectionManager()

@router.websocket("/ws/chat")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            data = await websocket.receive()
            
            if data["type"] == "bytes":
                # Handle audio data
                await manager.process_audio_stream(websocket, data["bytes"], {})
            elif data["type"] == "text":
                # Handle text messages (JSON)
                message = json.loads(data["text"])
                if message["type"] == "text_input":
                    # Process text directly through orchestrator
                    response = await manager.orchestrator.process_text(
                        message["content"],
                        message.get("metadata", {})
                    )
                    await websocket.send_json({
                        "type": "response",
                        "text": response["text"],
                        "synthesize_speech": message.get("synthesize_speech", False)
                    })

    except WebSocketDisconnect:
        manager.disconnect(websocket) 