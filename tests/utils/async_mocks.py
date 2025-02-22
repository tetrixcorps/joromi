import asyncio
from typing import List, Union, Any
from unittest.mock import AsyncMock

class FakeContent:
    """Mock aiohttp response content with async iteration support"""
    def __init__(self, chunks: List[Union[str, bytes, Any]]):
        self._chunks = chunks
        self._iter = None

    def __aiter__(self):
        """Initialize async iteration"""
        self._iter = iter(self._chunks)
        return self

    async def __anext__(self):
        """Get next chunk asynchronously"""
        try:
            chunk = next(self._iter)
            await asyncio.sleep(0)  # Simulate async behavior
            
            # Convert chunk to bytes if it's a string
            if isinstance(chunk, str):
                return chunk.encode('utf-8')
            # Pass through if already bytes
            elif isinstance(chunk, bytes):
                return chunk
            # Convert other types to string then bytes
            else:
                return str(chunk).encode('utf-8')
        except StopIteration:
            raise StopAsyncIteration

    async def iter_any(self):
        """Alternative iteration method used by aiohttp"""
        async for chunk in self:
            yield chunk

    async def iter_chunked(self):
        """Chunked iteration method"""
        async for chunk in self:
            yield chunk

    async def read(self):
        """Read entire content as bytes"""
        chunks = []
        async for chunk in self:
            chunks.append(chunk)
        return b''.join(chunks)

class FakeResponseCM:
    """Mock aiohttp ClientResponse with context manager support"""
    def __init__(self, response_data, status=200):
        self.mock_response = AsyncMock()
        self.mock_response.status = status
        
        if isinstance(response_data, dict):
            self.mock_response.json = AsyncMock(return_value=response_data)
        elif isinstance(response_data, (bytes, str)):
            self.mock_response.content = FakeContent([response_data])
        elif isinstance(response_data, list):
            self.mock_response.content = FakeContent(response_data)
        elif isinstance(response_data, Exception):
            self.mock_response.json = AsyncMock(side_effect=response_data)
            self.mock_response.read = AsyncMock(side_effect=response_data)
        
        # Add common response methods
        self.mock_response.read = getattr(
            self.mock_response, 'read',
            AsyncMock(return_value=b'')
        )
        self.mock_response.close = AsyncMock()

    async def __aenter__(self):
        return self.mock_response

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.mock_response.close()
        return False  # Don't suppress exceptions

def create_mock_response(data, status=200):
    """Helper function to create mock responses"""
    return FakeResponseCM(data, status) 