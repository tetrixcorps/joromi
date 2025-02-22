from pydantic import BaseModel, Field
from typing import Optional, List, Union
from enum import Enum

class MessageType(str, Enum):
    TEXT = "text"
    AUDIO = "audio"
    IMAGE = "image"

class ChatMessage(BaseModel):
    type: MessageType
    content: str
    metadata: Optional[dict] = None

class ChatRequest(BaseModel):
    messages: List[ChatMessage]
    context: Optional[dict] = None
    stream: bool = False

class ChatResponse(BaseModel):
    response: str
    confidence: float = Field(..., ge=0.0, le=1.0)
    metadata: Optional[dict] = None

class TranslationRequest(BaseModel):
    text: str
    source_lang: Optional[str] = None
    target_lang: str

class TranslationResponse(BaseModel):
    translated_text: str
    detected_language: Optional[str] = None

class BankingQueryType(str, Enum):
    ACCOUNT_INQUIRY = "account_inquiry"
    CARD_SERVICES = "card_services"
    SECURITY = "security"
    GENERAL = "general"

class BankingRequest(BaseModel):
    query: str
    query_type: Optional[BankingQueryType] = None
    context: Optional[dict] = None

class BankingResponse(BaseModel):
    response: str
    query_type: BankingQueryType
    requires_auth: bool = False
    confidence: float = Field(..., ge=0.0, le=1.0) 