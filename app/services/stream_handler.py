from app.services.audio_processor import AudioCompressor
from app.services.audio_sanitizer import AudioSanitizer
import asyncio
from typing import Dict, Optional
import logging
from app.exceptions import InvalidAudioError, MaliciousAudioError
from app.monitoring.metrics import AUDIO_THREATS_DETECTED

logger = logging.getLogger(__name__)

class AudioStreamHandler:
    def __init__(self):
        self.active_streams: Dict[str, Dict] = {}
        self.compressor = AudioCompressor()
        self.sanitizer = AudioSanitizer()
        self.chunk_size = 1024 * 16  # 16KB chunks
        self.sample_rate = 16000

    async def create_buffer_handler(self, client_id: str) -> Dict:
        """Create a new stream buffer for a client"""
        self.active_streams[client_id] = {
            'buffer': bytearray(),
            'last_processed': 0,
            'compression_stats': {
                'original_size': 0,
                'compressed_size': 0
            }
        }
        return self.active_streams[client_id]

    async def process_chunk(self, chunk: bytes, client_id: str) -> Optional[bytes]:
        """Process an audio chunk with security checks"""
        try:
            stream = self.active_streams.get(client_id)
            if not stream:
                return None

            # Add chunk to buffer
            stream['buffer'].extend(chunk)
            
            # Process complete chunks
            if len(stream['buffer']) >= self.chunk_size:
                # Extract chunk for processing
                process_chunk = bytes(stream['buffer'][:self.chunk_size])
                stream['buffer'] = stream['buffer'][self.chunk_size:]

                # Validate and sanitize audio
                sanitized_chunk, security_metadata = await self.sanitizer.validate(
                    process_chunk
                )

                # Apply compression if audio is safe
                compressed = await self.compressor.process_chunk(
                    sanitized_chunk, 
                    self.sample_rate
                )

                # Update security stats
                stream['security_stats'] = {
                    'checks_passed': security_metadata['checks_passed'],
                    'security_level': security_metadata['security_level']
                }

                return compressed

            return None

        except InvalidAudioError as e:
            logger.warning(f"Invalid audio detected: {e}")
            return None
        except MaliciousAudioError as e:
            logger.error(f"Malicious audio detected: {e}")
            AUDIO_THREATS_DETECTED.labels(
                threat_type=str(e)
            ).inc()
            return None
        except Exception as e:
            logger.error(f"Stream processing failed: {e}")
            return None

    async def cleanup_stream(self, client_id: str):
        """Clean up stream resources"""
        if client_id in self.active_streams:
            stats = self.active_streams[client_id]['compression_stats']
            compression_ratio = (
                stats['original_size'] / stats['compressed_size'] 
                if stats['compressed_size'] > 0 else 0
            )
            logger.info(
                f"Stream {client_id} closed. "
                f"Compression ratio: {compression_ratio:.2f}x"
            )
            del self.active_streams[client_id] 