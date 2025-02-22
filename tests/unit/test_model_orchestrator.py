import pytest
from services.services import ModelOrchestrator
from services.models import ChatMessage, ChatRequest, MessageType
from unittest.mock import AsyncMock, patch
import asyncio
from tests.utils.async_mocks import create_mock_response
import aiohttp

class FakeContent:
    """Helper class to simulate aiohttp response content"""
    def __init__(self, chunks):
        self._chunks = chunks

    async def iter_any(self):
        """Simulate async chunk iteration"""
        for chunk in self._chunks:
            # Convert chunk to bytes if it's a string
            if isinstance(chunk, str):
                yield chunk.encode('utf-8')
            # Pass through if already bytes
            elif isinstance(chunk, bytes):
                yield chunk
            # Convert other types to string then bytes
            else:
                yield str(chunk).encode('utf-8')

    async def iter_chunked(self):
        """Alternative chunk iteration method"""
        async for chunk in self.iter_any():
            yield chunk

    async def read(self):
        """Read entire content as bytes"""
        chunks = []
        async for chunk in self.iter_any():
            chunks.append(chunk)
        return b''.join(chunks)

class FakeResponseCM:
    """Helper class to mock async context manager responses"""
    def __init__(self, response_data):
        self.mock_response = AsyncMock()
        if isinstance(response_data, dict):
            self.mock_response.json = AsyncMock(return_value=response_data)
        elif isinstance(response_data, (bytes, str)):
            self.mock_response.content = FakeContent([response_data])
        elif isinstance(response_data, list):
            self.mock_response.content = FakeContent(response_data)
        elif isinstance(response_data, Exception):
            self.mock_response.json = AsyncMock(side_effect=response_data)

    async def __aenter__(self):
        return self.mock_response

    async def __aexit__(self, *exc):
        pass

@pytest.fixture
def orchestrator():
    return ModelOrchestrator()

@pytest.mark.asyncio(scope="function")
async def test_process_chat_text_message(orchestrator):
    """Test processing of simple text messages"""
    text_message = ChatMessage(
        type=MessageType.TEXT,
        content="What's my account balance?"
    )
    request = ChatRequest(messages=[text_message])
    
    expected_response = {
        "response": "Your account balance is $1000",
        "confidence": 0.95,
        "metadata": {"query_type": "ACCOUNT_INQUIRY"}
    }
    
    def fake_post(url, **kwargs):
        return FakeResponseCM(expected_response)
    
    with patch('aiohttp.ClientSession.post', side_effect=fake_post):
        result = await orchestrator.process_chat(request)
        assert result == expected_response

@pytest.mark.asyncio(scope="function")
async def test_process_chat_audio_message(orchestrator):
    """Test processing of audio messages with ASR"""
    audio_message = ChatMessage(
        type=MessageType.AUDIO,
        content="audio_bytes_here"
    )
    request = ChatRequest(messages=[audio_message])
    
    responses = {
        'transcribe': {"text": "What's my account balance?"},
        'banking': {
            "response": "Your account balance is $1000",
            "confidence": 0.95
        }
    }
    
    def fake_post(url, **kwargs):
        if url.endswith('/transcribe'):
            return FakeResponseCM(responses['transcribe'])
        return FakeResponseCM(responses['banking'])
    
    with patch('aiohttp.ClientSession.post', side_effect=fake_post):
        result = await orchestrator.process_chat(request)
        assert result == responses['banking']

@pytest.mark.asyncio(scope="function")
async def test_process_chat_image_message(orchestrator):
    """Test processing of image messages with Visual QA"""
    image_message = ChatMessage(
        type=MessageType.IMAGE,
        content="image_bytes_here"
    )
    request = ChatRequest(messages=[image_message])
    
    responses = {
        'visual-qa': {"answer": "An image of a bank statement"},
        'banking': {
            "response": "I can help you understand your bank statement",
            "confidence": 0.9
        }
    }
    
    def fake_post(url, **kwargs):
        if 'visual-qa' in url:
            assert kwargs['data'] == {"image": "image_bytes_here", "question": "Describe this image"}
            return FakeResponseCM(responses['visual-qa'])
        return FakeResponseCM(responses['banking'])
    
    with patch('aiohttp.ClientSession.post', side_effect=fake_post):
        result = await orchestrator.process_chat(request)
        assert result == responses['banking']

@pytest.mark.asyncio(scope="function")
async def test_process_chat_mixed_messages(orchestrator):
    """Test processing of multiple messages of different types"""
    messages = [
        ChatMessage(type=MessageType.TEXT, content="Hello"),
        ChatMessage(type=MessageType.IMAGE, content="image_bytes"),
        ChatMessage(type=MessageType.AUDIO, content="audio_bytes")
    ]
    request = ChatRequest(messages=messages)
    
    responses = {
        'visual-qa': {"answer": "An image"},
        'transcribe': {"text": "How are you?"},
        'banking': {
            "response": "I can help you with that",
            "confidence": 0.95
        }
    }
    
    def fake_post(url, **kwargs):
        if 'visual-qa' in url:
            assert kwargs['data'] == {"image": "image_bytes", "question": "Describe this image"}
            return FakeResponseCM(responses['visual-qa'])
        elif url.endswith('/transcribe'):
            assert kwargs['data'] == {"audio": "audio_bytes", "language": "en"}
            return FakeResponseCM(responses['transcribe'])
        return FakeResponseCM(responses['banking'])
    
    with patch('aiohttp.ClientSession.post', side_effect=fake_post):
        result = await orchestrator.process_chat(request)
        assert result == responses['banking']

@pytest.mark.asyncio(scope="function")
async def test_error_handling(orchestrator):
    """Test error handling in the orchestrator"""
    text_message = ChatMessage(
        type=MessageType.TEXT,
        content="Test message"
    )
    request = ChatRequest(messages=[text_message])
    
    def fake_post(url, **kwargs):
        raise Exception("Service unavailable")
    
    with patch('aiohttp.ClientSession.post', side_effect=fake_post):
        with pytest.raises(Exception) as exc_info:
            await orchestrator.process_chat(request)
        assert str(exc_info.value) == "Service unavailable"

@pytest.mark.asyncio(scope="function")
async def test_speech_to_text(orchestrator):
    """Test the speech-to-text service method"""
    audio_data = b"fake_audio_bytes"
    language = "en"
    expected_response = {"text": "Hello, this is a test"}
    
    def fake_post(url, **kwargs):
        assert url.endswith('/transcribe')
        assert kwargs['data'] == {"audio": audio_data, "language": language}
        return FakeResponseCM(expected_response)
    
    with patch('aiohttp.ClientSession.post', side_effect=fake_post):
        result = await orchestrator.speech_to_text(audio_data, language)
        assert result == expected_response["text"]

@pytest.mark.asyncio(scope="function")
async def test_text_to_speech(orchestrator):
    """Test the text-to-speech service method"""
    text = "Convert this to speech"
    language = "en"
    voice = "default"
    expected_response = {"audio_url": "https://storage.com/audio/123.mp3"}
    
    def fake_post(url, **kwargs):
        assert url.endswith('/synthesize')
        assert kwargs['json'] == {"text": text, "language": language, "voice": voice}
        return FakeResponseCM(expected_response)
    
    with patch('aiohttp.ClientSession.post', side_effect=fake_post):
        result = await orchestrator.text_to_speech(text, language, voice)
        assert result == expected_response["audio_url"]

@pytest.mark.asyncio(scope="function")
async def test_translate(orchestrator):
    """Test the translation service method"""
    text = "Hello, world!"
    source_lang = "en"
    target_lang = "es"
    expected_response = {"translated_text": "¡Hola, mundo!"}
    
    def fake_post(url, **kwargs):
        assert url.endswith('/translate')
        assert kwargs['json'] == {
            "text": text,
            "source_lang": source_lang,
            "target_lang": target_lang
        }
        return FakeResponseCM(expected_response)
    
    with patch('aiohttp.ClientSession.post', side_effect=fake_post):
        result = await orchestrator.translate(text, source_lang, target_lang)
        assert result == expected_response["translated_text"]

@pytest.mark.asyncio(scope="function")
async def test_process_visual_qa_with_answer(orchestrator):
    """Test visual QA with answer key"""
    image_data = b"fake_image_bytes"
    question = "What is shown in this image?"
    expected_response = {"answer": "The image shows a bank statement"}
    
    def fake_post(url, **kwargs):
        assert url.endswith('/analyze')
        assert kwargs['data'] == {"image": image_data, "question": question}
        return FakeResponseCM(expected_response)
    
    with patch('aiohttp.ClientSession.post', side_effect=fake_post):
        result = await orchestrator.process_visual_qa(image_data, question)
        assert result == expected_response["answer"]

@pytest.mark.asyncio
async def test_stream_chat_response(orchestrator):
    """Test streaming chat response with various chunk types"""
    message = "Tell me a story"
    chunks = [
        "Once upon",
        b" a time",
        123,
        {"key": "value"}
    ]
    
    def fake_post(url, **kwargs):
        assert url.endswith('/stream-chat')
        assert kwargs['json'] == {"message": message}
        return create_mock_response(chunks)
    
    with patch('aiohttp.ClientSession.post', side_effect=fake_post):
        received = []
        async for chunk in orchestrator.stream_chat_response(message):
            received.append(chunk)
        
        expected = [
            "Once upon",
            " a time",
            "123",
            "{'key': 'value'}"
        ]
        assert received == expected

@pytest.mark.asyncio
async def test_stream_chat_response_error(orchestrator):
    """Test streaming chat response error handling"""
    message = "Error test"
    error = Exception("Stream error")
    
    def fake_post(url, **kwargs):
        return create_mock_response(error)
    
    with patch('aiohttp.ClientSession.post', side_effect=fake_post):
        with pytest.raises(Exception) as exc_info:
            async for _ in orchestrator.stream_chat_response(message):
                pass
        assert "Stream error" in str(exc_info.value)

@pytest.mark.asyncio
async def test_stream_chat_empty_response(orchestrator):
    """Test streaming chat with empty response"""
    message = "Empty test"
    
    def fake_post(url, **kwargs):
        return create_mock_response([])
    
    with patch('aiohttp.ClientSession.post', side_effect=fake_post):
        chunks = []
        async for chunk in orchestrator.stream_chat_response(message):
            chunks.append(chunk)
        assert chunks == []

@pytest.mark.asyncio(scope="function")
async def test_service_timeout_handling():
    """Test handling of service timeouts"""
    orchestrator = ModelOrchestrator()
    text = "Test message"
    
    def fake_post(url, **kwargs):
        return FakeResponseCM(asyncio.TimeoutError("Service timeout"))
    
    with patch('aiohttp.ClientSession.post', side_effect=fake_post):
        with pytest.raises(Exception) as exc_info:
            await orchestrator.text_to_speech(text)
        assert "Service timeout" in str(exc_info.value)

@pytest.mark.asyncio(scope="function")
async def test_invalid_response_handling():
    """Test handling of invalid service responses"""
    orchestrator = ModelOrchestrator()
    text = "Test message"
    
    def fake_post(url, **kwargs):
        return FakeResponseCM({"invalid_key": "value"})
    
    with patch('aiohttp.ClientSession.post', side_effect=fake_post):
        with pytest.raises(KeyError) as exc_info:
            await orchestrator.text_to_speech(text)
        assert "audio_url" in str(exc_info.value)

@pytest.mark.asyncio(scope="function")
async def test_visual_qa_fallback_to_description(orchestrator):
    """Test visual QA falls back to description if answer not found"""
    image_data = b"fake_image_bytes"
    question = "What is shown in this image?"
    response_with_description = {"description": "An image of a bank statement"}
    
    def fake_post(url, **kwargs):
        assert url.endswith('/analyze')
        assert kwargs['data'] == {"image": image_data, "question": question}
        return FakeResponseCM(response_with_description)
    
    with patch('aiohttp.ClientSession.post', side_effect=fake_post):
        result = await orchestrator.process_visual_qa(image_data, question)
        assert result == response_with_description["description"]

@pytest.mark.asyncio(scope="function")
async def test_visual_qa_missing_keys(orchestrator):
    """Test visual QA handles missing keys"""
    image_data = b"fake_image_bytes"
    question = "What is shown in this image?"
    invalid_response = {"invalid_key": "value"}
    
    def fake_post(url, **kwargs):
        assert url.endswith('/analyze')
        assert kwargs['data'] == {"image": image_data, "question": question}
        return FakeResponseCM(invalid_response)
    
    with patch('aiohttp.ClientSession.post', side_effect=fake_post):
        with pytest.raises(KeyError) as exc_info:
            await orchestrator.process_visual_qa(image_data, question)
        assert "missing both 'answer' and 'description' keys" in str(exc_info.value)

@pytest.mark.asyncio(scope="function")
async def test_stream_chat_response_mixed_types(orchestrator):
    """Test streaming response with mixed chunk types"""
    message = "Mixed data"
    expected_chunks = [
        "text chunk",
        b"binary chunk",
        123,  # number will be converted to string
        {"key": "value"}  # dict will be converted to string
    ]
    
    def fake_post(url, **kwargs):
        assert url.endswith('/stream-chat')
        return FakeResponseCM(expected_chunks)
    
    with patch('aiohttp.ClientSession.post', side_effect=fake_post):
        chunks = []
        async for chunk in orchestrator.stream_chat_response(message):
            chunks.append(chunk)
        
        # Convert expected chunks to strings for comparison
        expected_str_chunks = [
            "text chunk",
            "binary chunk",
            "123",
            "{'key': 'value'}"
        ]
        assert chunks == expected_str_chunks

@pytest.mark.asyncio
async def test_stream_chat_response_large_chunks(orchestrator):
    """Test streaming chat with large data chunks"""
    message = "Generate long response"
    large_chunks = ["x" * 1024 for _ in range(5)]  # 5 chunks of 1KB each
    
    def fake_post(url, **kwargs):
        assert url.endswith('/stream-chat')
        return create_mock_response(large_chunks)
    
    with patch('aiohttp.ClientSession.post', side_effect=fake_post):
        chunks = []
        async for chunk in orchestrator.stream_chat_response(message):
            chunks.append(chunk)
        assert all(len(c) == 1024 for c in chunks)
        assert len(chunks) == 5

@pytest.mark.asyncio
async def test_stream_chat_response_unicode(orchestrator):
    """Test streaming chat with unicode characters"""
    message = "Translate to Chinese"
    unicode_chunks = ["你好", "世界", "！"]
    
    def fake_post(url, **kwargs):
        assert url.endswith('/stream-chat')
        return create_mock_response(unicode_chunks)
    
    with patch('aiohttp.ClientSession.post', side_effect=fake_post):
        chunks = []
        async for chunk in orchestrator.stream_chat_response(message):
            chunks.append(chunk)
        assert chunks == unicode_chunks

@pytest.mark.asyncio
async def test_stream_chat_response_slow(orchestrator):
    """Test streaming chat with slow responses"""
    message = "Slow response"
    chunks = ["chunk1", "chunk2", "chunk3"]
    
    class SlowFakeContent(FakeContent):
        async def __anext__(self):
            try:
                chunk = next(self._iter)
                await asyncio.sleep(0.1)  # Simulate network delay
                return chunk.encode() if isinstance(chunk, str) else chunk
            except StopIteration:
                raise StopAsyncIteration
    
    class SlowResponseCM(FakeResponseCM):
        def __init__(self, chunks):
            super().__init__(chunks)
            self.mock_response.content = SlowFakeContent(chunks)
    
    def fake_post(url, **kwargs):
        return SlowResponseCM(chunks)
    
    with patch('aiohttp.ClientSession.post', side_effect=fake_post):
        start_time = asyncio.get_event_loop().time()
        received = []
        async for chunk in orchestrator.stream_chat_response(message):
            received.append(chunk)
        elapsed = asyncio.get_event_loop().time() - start_time
        
        assert received == chunks
        assert elapsed >= 0.3  # At least 0.1s per chunk

@pytest.mark.asyncio
async def test_stream_chat_response_connection_error(orchestrator):
    """Test streaming chat with connection errors"""
    message = "Connection test"
    
    class ConnectionErrorContent(FakeContent):
        async def __anext__(self):
            await asyncio.sleep(0.1)
            raise aiohttp.ClientError("Connection lost")
    
    class ErrorResponseCM(FakeResponseCM):
        def __init__(self):
            super().__init__([])
            self.mock_response.content = ConnectionErrorContent([])
    
    def fake_post(url, **kwargs):
        return ErrorResponseCM()
    
    with patch('aiohttp.ClientSession.post', side_effect=fake_post):
        with pytest.raises(aiohttp.ClientError) as exc_info:
            async for _ in orchestrator.stream_chat_response(message):
                pass
        assert "Connection lost" in str(exc_info.value) 