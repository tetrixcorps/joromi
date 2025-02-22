from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import numpy as np
from enum import Enum

class LanguageCode(Enum):
    ENGLISH = "eng"
    FRENCH = "fra"
    SPANISH = "spa"
    GERMAN = "deu"
    CHINESE = "zho"
    JAPANESE = "jpn"
    ARABIC = "ara"
    RUSSIAN = "rus"
    PORTUGUESE = "por"
    HINDI = "hin"
    SWAHILI = "swa"
    YORUBA = "yor"
    IGBO = "ibo"
    HAUSA = "hau"
    SHONA = "sna"
    XHOSA = "xho"
    TWI = "twi"

@dataclass
class TranslationPair:
    source: LanguageCode
    target: LanguageCode
    quality_score: float = 1.0
    supports_audio: bool = False

class TranslationConfig:
    # Language pairs with quality scores
    LANGUAGE_PAIRS = {
        (LanguageCode.ENGLISH, LanguageCode.FRENCH): TranslationPair(
            LanguageCode.ENGLISH, LanguageCode.FRENCH, 0.95, True
        ),
        (LanguageCode.ENGLISH, LanguageCode.SPANISH): TranslationPair(
            LanguageCode.ENGLISH, LanguageCode.SPANISH, 0.94, True
        ),
        # African languages
        (LanguageCode.ENGLISH, LanguageCode.SWAHILI): TranslationPair(
            LanguageCode.ENGLISH, LanguageCode.SWAHILI, 0.85, True
        ),
        (LanguageCode.ENGLISH, LanguageCode.YORUBA): TranslationPair(
            LanguageCode.ENGLISH, LanguageCode.YORUBA, 0.83, True
        ),
        (LanguageCode.ENGLISH, LanguageCode.IGBO): TranslationPair(
            LanguageCode.ENGLISH, LanguageCode.IGBO, 0.82, True
        ),
        (LanguageCode.ENGLISH, LanguageCode.HAUSA): TranslationPair(
            LanguageCode.ENGLISH, LanguageCode.HAUSA, 0.84, True
        ),
        (LanguageCode.ENGLISH, LanguageCode.TWI): TranslationPair(
            LanguageCode.ENGLISH, LanguageCode.TWI, 0.80, True
        ),
        # West African languages
        (LanguageCode.ENGLISH, LanguageCode.YORUBA): TranslationPair(
            LanguageCode.ENGLISH, LanguageCode.YORUBA, 0.83, True
        ),
        (LanguageCode.ENGLISH, LanguageCode.IGBO): TranslationPair(
            LanguageCode.ENGLISH, LanguageCode.IGBO, 0.82, True
        ),
        (LanguageCode.ENGLISH, LanguageCode.HAUSA): TranslationPair(
            LanguageCode.ENGLISH, LanguageCode.HAUSA, 0.84, True
        ),
        (LanguageCode.ENGLISH, LanguageCode.TWI): TranslationPair(
            LanguageCode.ENGLISH, LanguageCode.TWI, 0.80, True
        ),
        # East African languages
        (LanguageCode.ENGLISH, LanguageCode.SWAHILI): TranslationPair(
            LanguageCode.ENGLISH, LanguageCode.SWAHILI, 0.85, True
        ),
        # Southern African languages
        (LanguageCode.ENGLISH, LanguageCode.SHONA): TranslationPair(
            LanguageCode.ENGLISH, LanguageCode.SHONA, 0.81, True
        ),
        (LanguageCode.ENGLISH, LanguageCode.XHOSA): TranslationPair(
            LanguageCode.ENGLISH, LanguageCode.XHOSA, 0.82, True
        ),
        # Add cross-language translation between African languages
        (LanguageCode.YORUBA, LanguageCode.HAUSA): TranslationPair(
            LanguageCode.YORUBA, LanguageCode.HAUSA, 0.78, True
        ),
        (LanguageCode.SWAHILI, LanguageCode.YORUBA): TranslationPair(
            LanguageCode.SWAHILI, LanguageCode.YORUBA, 0.77, True
        ),
        # Add more language pairs...
    }
    
    # Add language-specific configurations
    LANGUAGE_CONFIGS = {
        LanguageCode.YORUBA: {
            "tone_markers": True,
            "dialect_aware": True,
            "default_dialect": "standard"
        },
        LanguageCode.IGBO: {
            "tone_markers": True,
            "dialect_aware": True,
            "default_dialect": "central"
        },
        LanguageCode.HAUSA: {
            "tone_markers": True,
            "dialect_aware": True,
            "default_dialect": "kano"
        },
        LanguageCode.SWAHILI: {
            "dialect_aware": True,
            "default_dialect": "coastal"
        },
        LanguageCode.SHONA: {
            "dialect_aware": True,
            "default_dialect": "zezuru"
        },
        LanguageCode.XHOSA: {
            "tone_markers": True,
            "click_sounds": True
        },
        LanguageCode.TWI: {
            "tone_markers": True,
            "dialect_aware": True,
            "default_dialect": "akuapem"
        }
    }
    
    @staticmethod
    def get_supported_languages() -> List[str]:
        """Get list of supported languages"""
        languages = set()
        for src, tgt in TranslationConfig.LANGUAGE_PAIRS.keys():
            languages.add(src.value)
            languages.add(tgt.value)
        return sorted(list(languages))
    
    @staticmethod
    def is_pair_supported(source: str, target: str) -> bool:
        """Check if language pair is supported"""
        try:
            src = LanguageCode(source)
            tgt = LanguageCode(target)
            return (src, tgt) in TranslationConfig.LANGUAGE_PAIRS
        except ValueError:
            return False

class BatchTranslator:
    def __init__(self, model, processor, device):
        self.model = model
        self.processor = processor
        self.device = device
        self.batch_size = 8
        
    async def translate_batch(
        self,
        texts: List[str],
        source_lang: str,
        target_lang: str
    ) -> List[Dict]:
        """Translate a batch of texts"""
        results = []
        
        # Process in batches
        for i in range(0, len(texts), self.batch_size):
            batch_texts = texts[i:i + self.batch_size]
            
            # Prepare batch inputs
            inputs = self.processor(
                text=batch_texts,
                src_lang=source_lang,
                tgt_lang=target_lang,
                return_tensors="pt",
                padding=True,
                truncation=True
            ).to(self.device)
            
            # Generate translations
            outputs = self.model.generate(
                **inputs,
                tgt_lang=target_lang,
                max_length=256
            )
            
            # Decode translations
            translations = self.processor.batch_decode(
                outputs,
                skip_special_tokens=True
            )
            
            # Add results
            for text, translation in zip(batch_texts, translations):
                results.append({
                    "original": text,
                    "translated": translation,
                    "source_lang": source_lang,
                    "target_lang": target_lang
                })
        
        return results

class LanguageDetector:
    def __init__(self, device: torch.device):
        self.device = device
        self.model = AutoModelForSequenceClassification.from_pretrained(
            "papluca/xlm-roberta-base-language-detection"
        ).to(device)
        self.tokenizer = AutoTokenizer.from_pretrained(
            "papluca/xlm-roberta-base-language-detection"
        )
        
    def detect_language(self, text: str) -> Tuple[str, float]:
        """Detect language of text"""
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
            
        predicted_id = predictions.argmax().item()
        confidence = predictions.max().item()
        
        predicted_lang = self.model.config.id2label[predicted_id]
        return predicted_lang, confidence
    
    def detect_batch(self, texts: List[str]) -> List[Tuple[str, float]]:
        """Detect languages for a batch of texts"""
        inputs = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
            
        results = []
        for pred in predictions:
            lang_id = pred.argmax().item()
            confidence = pred.max().item()
            lang = self.model.config.id2label[lang_id]
            results.append((lang, confidence))
            
        return results 