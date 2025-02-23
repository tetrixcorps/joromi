from pydantic import BaseModel, Field, validator, constr
from typing import Optional, List
from decimal import Decimal
import re

class BankingRequest(BaseModel):
    account_number: Optional[constr(regex=r'^\d{10}$')] = None
    transaction_type: Optional[str] = Field(None, enum=['transfer', 'balance', 'payment'])
    amount: Optional[Decimal] = Field(None, ge=0, le=1000000)
    recipient_account: Optional[constr(regex=r'^\d{10}$')] = None
    description: Optional[str] = Field(None, max_length=200)

    @validator('amount')
    def validate_amount(cls, v):
        if v and v.as_tuple().exponent < -2:
            raise ValueError('Amount cannot have more than 2 decimal places')
        return v

class TranslationRequest(BaseModel):
    source_lang: str = Field(..., min_length=2, max_length=5)
    target_lang: str = Field(..., min_length=2, max_length=5)
    text: str = Field(..., min_length=1, max_length=5000)
    preserve_formatting: Optional[bool] = False
    glossary_terms: Optional[dict] = None

    @validator('source_lang', 'target_lang')
    def validate_lang_code(cls, v):
        if not re.match(r'^[a-z]{2}(-[A-Z]{2})?$', v):
            raise ValueError('Invalid language code format')
        return v

class ASRRequest(BaseModel):
    audio_format: str = Field(..., enum=['wav', 'mp3', 'ogg'])
    sample_rate: int = Field(..., ge=8000, le=48000)
    language: str = Field(..., min_length=2, max_length=5)
    enhance_speech: Optional[bool] = False
    speaker_diarization: Optional[bool] = False

class TTSRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=1000)
    language: str = Field(..., min_length=2, max_length=5)
    voice_id: Optional[str] = None
    speed: Optional[float] = Field(None, ge=0.5, le=2.0)
    pitch: Optional[float] = Field(None, ge=-20, le=20) 