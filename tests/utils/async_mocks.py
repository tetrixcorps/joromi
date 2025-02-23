import asyncio
from typing import List, Union, Any
import aiohttp
from unittest.mock import AsyncMock

class StreamError(Exception):
    """Custom error for stream failures"""
    pass

class BaseContent:
    """Base class for mock content with async iteration"""
    def __init__(self, chunks):
        self._chunks = list(chunks)
        self._iter = None

    async def iter_any(self):
        """Implementation of aiohttp's StreamReader.iter_any()"""
        for chunk in self._chunks:
            if isinstance(chunk, str):
                yield chunk.encode('utf-8')
            elif isinstance(chunk, bytes):
                yield chunk
            else:
                yield str(chunk).encode('utf-8')

    async def iter_chunked(self):
        """Implementation of aiohttp's StreamReader.iter_chunked()"""
        async for chunk in self.iter_any():
            yield chunk

    async def read(self):
        """Implementation of aiohttp's StreamReader.read()"""
        chunks = []
        async for chunk in self.iter_any():
            chunks.append(chunk)
        return b''.join(chunks)

class FakeContent(BaseContent):
    """Standard mock content"""
    pass

class ErrorContent(BaseContent):
    """Mock content that raises errors"""
    def __init__(self, error):
        super().__init__([])
        self.error = error

    async def iter_any(self):
        raise self.error

class SlowContent(BaseContent):
    """Mock content with delayed responses"""
    def __init__(self, chunks, delay=0.1):
        super().__init__(chunks)
        self.delay = delay

    async def iter_any(self):
        for chunk in self._chunks:
            await asyncio.sleep(self.delay)
            if isinstance(chunk, str):
                yield chunk.encode('utf-8')
            elif isinstance(chunk, bytes):
                yield chunk
            else:
                yield str(chunk).encode('utf-8')

class FakeResponseCM:
    """Context manager for fake responses"""
    def __init__(
        self, 
        chunks: List[Union[str, bytes]], 
        status: int = 200,
        content_class: type = FakeContent,
        **content_kwargs
    ):
        self.mock_response = AsyncMock()
        self.mock_response.status = status
        self.mock_response.content = content_class(chunks, **content_kwargs)
        
        # Handle JSON response
        if isinstance(chunks, (list, tuple)):
            self.mock_response.json = AsyncMock(return_value=chunks[0] if chunks else {})
        else:
            self.mock_response.json = AsyncMock(return_value=chunks)

    async def __aenter__(self):
        return self.mock_response

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.mock_response.close()
        return False

def create_error_response(error: Exception = None):
    """Create a response that raises an error"""
    if error is None:
        error = aiohttp.ClientError("Connection lost")
    return FakeResponseCM(
        chunks=[],
        content_class=ErrorContent,
        error=error
    )

def create_slow_response(chunks: List[str], delay: float = 1.0):
    """Create a response with delayed chunks"""
    return FakeResponseCM(
        chunks=chunks,
        content_class=SlowContent,
        delay=delay
    )

def create_timeout_response():
    """Create a response that times out"""
    return create_slow_response(["Timeout"], delay=31.0)  # Default timeout is 30s 