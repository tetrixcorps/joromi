from pydantic import BaseModel
from typing import Optional, Union
from enum import Enum

class Modality(str, Enum):
    TEXT = "text"
    IMAGE = "image"
    SPEECH = "speech"

class ProcessRequest(BaseModel):
    modality: Modality
    content: Union[str, bytes]
    domain: Optional[str] = None
    confidence_threshold: float = 0.5 