import pytest
from services.services import ModelOrchestrator
from services.models import ChatMessage, ChatRequest, MessageType
from unittest.mock import AsyncMock, patch
import asyncio
from tests.utils.async_mocks import (
    FakeResponseCM, 
    create_error_response,
    create_slow_response,
    create_timeout_response,
    StreamError
)
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

@pytest.mark.asyncio(loop_scope="function")
async def test_stream_chat_success():
    """Test successful streaming chat response"""
    orchestrator = ModelOrchestrator()
    test_message = "Hello"
    expected_chunks = ["Hello", " world", "!"]
    
    with patch('aiohttp.ClientSession.post', return_value=FakeResponseCM(expected_chunks)):
        chunks = []
        async for chunk in orchestrator.stream_chat_response(test_message):
            chunks.append(chunk.decode())
        assert chunks == expected_chunks

@pytest.mark.asyncio(loop_scope="function")
async def test_stream_chat_connection_error():
    """Test connection error handling"""
    orchestrator = ModelOrchestrator()
    
    with patch('aiohttp.ClientSession.post', side_effect=aiohttp.ClientError("Connection lost")):
        with pytest.raises(StreamError):
            async for _ in orchestrator.stream_chat_response("Hello"):
                pass

@pytest.mark.asyncio(loop_scope="function")
async def test_stream_chat_timeout():
    """Test timeout handling"""
    orchestrator = ModelOrchestrator()
    
    with patch('aiohttp.ClientSession.post', side_effect=asyncio.TimeoutError()):
        with pytest.raises(asyncio.TimeoutError):
            async for _ in orchestrator.stream_chat_response("Hello", timeout=1.0):
                pass

@pytest.mark.asyncio(loop_scope="function")
async def test_stream_chat_slow_response():
    """Test slow response handling"""
    orchestrator = ModelOrchestrator()
    chunks = ["Slow", " response"]
    
    with patch('aiohttp.ClientSession.post', return_value=create_slow_response(chunks, delay=0.1)):
        received = []
        async for chunk in orchestrator.stream_chat_response("Hello"):
            received.append(chunk.decode())
        assert received == chunks 