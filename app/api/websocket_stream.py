from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from app.utils.websocket import WebSocketManager
from app.services.stream_handler import AudioStreamHandler
import asyncio
import logging

logger = logging.getLogger(__name__)
router = APIRouter()

class StreamingManager:
    def __init__(self):
        self.active_streams = {}
        self.stream_handler = AudioStreamHandler()

    async def handle_stream(self, websocket: WebSocket, client_id: str):
        ws_manager = WebSocketManager(websocket)
        buffer_handler = self.stream_handler.create_buffer_handler(client_id)

        try:
            while True:
                chunk = await ws_manager.receive_audio_stream(buffer_handler)
                if chunk:
                    # Process chunk and send real-time feedback
                    partial_transcription = await self.stream_handler.process_chunk(chunk)
                    if partial_transcription:
                        await ws_manager.safe_send({
                            "type": "partial_transcription",
                            "text": partial_transcription
                        })

        except WebSocketDisconnect:
            await self.stream_handler.cleanup_stream(client_id)
            logger.info(f"Client {client_id} disconnected")

streaming_manager = StreamingManager()

@router.websocket("/ws/stream/{client_id}")
async def stream_endpoint(websocket: WebSocket, client_id: str):
    await websocket.accept()
    await streaming_manager.handle_stream(websocket, client_id) 