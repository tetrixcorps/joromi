from pydantic import BaseModel, Field, validator
from typing import Optional, Dict, Any, Union
from enum import Enum
import base64

class ResponseFormat(str, Enum):
    TEXT = "text"
    SPEECH = "speech"
    JSON = "json"

class BaseRequest(BaseModel):
    request_id: str = Field(..., description="Unique request identifier")
    response_format: ResponseFormat = Field(default=ResponseFormat.TEXT)

class TextRequest(BaseRequest):
    text: str = Field(..., min_length=1, max_length=2000)
    source_lang: Optional[str] = Field(default="en")
    target_lang: Optional[str] = Field(default="en")

class AudioRequest(BaseRequest):
    audio: str = Field(..., description="Base64 encoded audio data")
    sample_rate: int = Field(default=16000)
    
    @validator('audio')
    def validate_audio(cls, v):
        try:
            base64.b64decode(v)
            return v
        except:
            raise ValueError("Invalid base64 encoded audio data")

class ImageRequest(BaseRequest):
    image: str = Field(..., description="Base64 encoded image data")
    text: Optional[str] = None
    
    @validator('image')
    def validate_image(cls, v):
        try:
            base64.b64decode(v)
            return v
        except:
            raise ValueError("Invalid base64 encoded image data")

class MultiModalRequest(BaseRequest):
    text: str = Field(..., min_length=1, max_length=2000)
    image: Optional[str] = None
    audio: Optional[str] = None 