from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from app.utils.websocket import WebSocketManager
from app.services.stream_handler import AudioStreamHandler
from app.monitoring.metrics import (
    AUDIO_COMPRESSION_RATIO,
    AUDIO_PROCESSING_LATENCY,
    ACTIVE_STREAMS
)
import asyncio
import logging
import time

logger = logging.getLogger(__name__)
router = APIRouter()

class StreamingManager:
    def __init__(self):
        self.active_streams = {}
        self.stream_handler = AudioStreamHandler()

    async def handle_stream(self, websocket: WebSocket, client_id: str):
        try:
            ACTIVE_STREAMS.inc()
            ws_manager = WebSocketManager(websocket)
            buffer_handler = await self.stream_handler.create_buffer_handler(client_id)

            while True:
                start_time = time.time()
                
                # Receive audio chunk
                chunk = await ws_manager.receive_audio_stream(buffer_handler)
                
                if chunk:
                    # Process with compression
                    processed = await self.stream_handler.process_chunk(chunk, client_id)
                    
                    if processed:
                        # Send processed audio
                        await ws_manager.safe_send(processed)
                        
                        # Record metrics
                        AUDIO_PROCESSING_LATENCY.observe(
                            time.time() - start_time,
                            labels={'operation': 'process_chunk'}
                        )
                        
                        stats = buffer_handler['compression_stats']
                        AUDIO_COMPRESSION_RATIO.observe(
                            stats['original_size'] / stats['compressed_size'],
                            labels={'client_id': client_id}
                        )

        except Exception as e:
            logger.error(f"Stream handling failed: {e}")
        finally:
            ACTIVE_STREAMS.dec()
            await self.stream_handler.cleanup_stream(client_id)

streaming_manager = StreamingManager()

@router.websocket("/ws/stream/{client_id}")
async def stream_endpoint(websocket: WebSocket, client_id: str):
    await websocket.accept()
    await streaming_manager.handle_stream(websocket, client_id) 