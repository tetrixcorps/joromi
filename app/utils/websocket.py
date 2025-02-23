from fastapi import WebSocket
import asyncio
from typing import Optional, Callable
import logging

logger = logging.getLogger(__name__)

class WebSocketManager:
    def __init__(self, websocket: WebSocket):
        self.websocket = websocket
        self.buffer_size = 1024 * 1024  # 1MB buffer
        self.reconnect_attempts = 3
        self.reconnect_delay = 1.0  # seconds

    async def safe_send(self, data: dict) -> bool:
        """Safely send data with reconnection attempts"""
        for attempt in range(self.reconnect_attempts):
            try:
                await self.websocket.send_json(data)
                return True
            except Exception as e:
                logger.error(f"Failed to send WebSocket message (attempt {attempt + 1}): {e}")
                if attempt < self.reconnect_attempts - 1:
                    await asyncio.sleep(self.reconnect_delay)
                    continue
                return False

    async def receive_audio_stream(self, chunk_handler: Callable) -> None:
        """Handle incoming audio stream with chunking"""
        buffer = bytearray()
        
        try:
            while True:
                chunk = await self.websocket.receive_bytes()
                buffer.extend(chunk)

                # Process complete chunks
                while len(buffer) >= self.buffer_size:
                    chunk_to_process = bytes(buffer[:self.buffer_size])
                    buffer = buffer[self.buffer_size:]
                    await chunk_handler(chunk_to_process)

        except Exception as e:
            logger.error(f"Error in audio stream processing: {e}")
            await self.safe_send({
                "type": "error",
                "message": "Audio stream processing failed"
            }) 