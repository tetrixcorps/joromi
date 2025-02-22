from dataclasses import dataclass
from typing import Dict, Optional
import torch
from enum import Enum

@dataclass
class ModelSpecs:
    name: str
    type: str
    device_requirements: Dict
    trust_remote_code: bool = True
    quantization: Optional[str] = None
    processor: Optional[str] = None
    model_class: Optional[str] = None

@dataclass
class ModelConfig:
    name: str
    path: str
    type: str
    device: str = "cuda"

class ModelConfigurations:
    GENERAL_PURPOSE = ModelSpecs(
        name='openbmb/MiniCPM-o-2_6',
        type='general',
        device_requirements={
            'min_vram': 8,
            'dtype': torch.bfloat16
        }
    )
    
    VISUAL_QA = ModelSpecs(
        name='google/pix2struct-ai2d-base',
        type='vision',
        device_requirements={
            'min_vram': 8,
            'dtype': torch.float16
        },
        processor='Pix2StructProcessor',
        model_class='Pix2StructForConditionalGeneration'
    )
    
    MERGED_LLM = ModelSpecs(
        name='merged_llm',
        type='specialized',
        device_requirements={
            'min_vram': 8,
            'dtype': torch.float16
        },
        models=[
            'sethanimesh/Meta-Llama-3.1-8B-Banking-GGUF',
            'Llama3.1_medicine_fine-tuned_24-09_16bit_gguf',
            'cognitivecomputations/Dolphin3.0-Llama3.1-8B-GGUF'
        ]
    )
    
    TTS = ModelSpecs(
        name='intronhealth/afro-tts',
        type='tts',
        device_requirements={
            'min_vram': 4,
            'dtype': torch.float32
        },
        model_class='AfroTTSModel',
        processor='AfroTTSProcessor'
    )
    
    ASR = ModelSpecs(
        name='Intel/whisper-large-int8-static-inc',
        type='asr',
        device_requirements={
            'min_vram': 4,
            'dtype': torch.int8
        }
    )
    
    TRANSLATION = ModelSpecs(
        name='facebook/Seamless-M4T',
        type='translation',
        device_requirements={
            'min_vram': 6,
            'dtype': torch.float16
        },
        model_class='SeamlessM4TModel',
        processor='SeamlessM4TProcessor'
    )
    
    BANKING_LLM = ModelSpecs(
        name='unsloth/Meta-Llama-3.1-8B',
        type='banking',
        device_requirements={
            'min_vram': 8,
            'dtype': None,  # Will be handled by FastLanguageModel
            'load_in_4bit': True,
            'max_seq_length': 2048
        },
        model_class='FastLanguageModel',
        quantization='4bit'
    )
    
    DOLPHIN = ModelSpecs(
        name='cognitivecomputations/Dolphin3.0-Llama3.1-8B-GGUF',
        type='llm',
        device_requirements={
            'min_vram': 8,
            'dtype': torch.bfloat16,
            'load_in_8bit': True,
            'max_seq_length': 4096
        },
        trust_remote_code=True
    )
    
    PIX2STRUCT = ModelConfig(
        name="google/pix2struct-base",
        path="models/pix2struct",
        type="visual_qa"
    )
    
    DOLPHIN_CHAT = ModelConfig(
        name="cognitivecomputations/dolphin-2.6-mixtral-8x7b",
        path="models/dolphin",
        type="chat"
    )
    
    SPEECH_MODEL = ModelConfig(
        name="speech_model",
        path="models/speech",
        type="speech"
    )
    
    ASR_MODEL = ModelConfig(
        name="asr_model",
        path="models/asr",
        type="asr"
    )
    
    TRANSLATION_MODEL = ModelConfig(
        name="translation_model",
        path="models/translation",
        type="translation"
    ) 