from typing import List, Dict, Optional
import re
import unicodedata
from enum import Enum
import numpy as np
from sacrebleu import BLEU
from nltk.translate.meteor_score import meteor_score
import torch
import random

class ToneMarker(Enum):
    HIGH = "́"
    LOW = "̀"
    MID = "̄"
    RISING = "̌"
    FALLING = "̂"

class DialectRegion(Enum):
    # Yoruba dialects
    YORUBA_OYO = "oyo"
    YORUBA_LAGOS = "lagos"
    YORUBA_EKITI = "ekiti"
    
    # Igbo dialects
    IGBO_CENTRAL = "central"
    IGBO_NORTHERN = "northern"
    IGBO_SOUTHERN = "southern"
    
    # Hausa dialects
    HAUSA_KANO = "kano"
    HAUSA_SOKOTO = "sokoto"
    HAUSA_ZARIA = "zaria"

class AfricanLanguagePreprocessor:
    def __init__(self, language: str):
        self.language = language
        self.tone_markers = {m.value for m in ToneMarker}
        
    def normalize_text(self, text: str) -> str:
        """Normalize text while preserving tone markers and special characters"""
        # Decompose characters to handle combined characters
        text = unicodedata.normalize('NFD', text)
        
        # Preserve tone markers and special characters
        normalized = []
        for char in text:
            if char in self.tone_markers:
                normalized.append(char)
            elif unicodedata.category(char).startswith('L'):  # Letters
                normalized.append(char)
            elif char in '.,!?':  # Basic punctuation
                normalized.append(char)
            elif char == ' ':
                normalized.append(char)
        
        return ''.join(normalized)
    
    def apply_dialect_rules(self, text: str, dialect: str) -> str:
        """Apply dialect-specific transformations"""
        if self.language == "yor":
            if dialect == DialectRegion.YORUBA_OYO.value:
                # Oyo dialect transformations
                text = self._apply_oyo_rules(text)
            elif dialect == DialectRegion.YORUBA_EKITI.value:
                # Ekiti dialect transformations
                text = self._apply_ekiti_rules(text)
        # Add rules for other languages...
        return text
    
    def _apply_oyo_rules(self, text: str) -> str:
        """Apply Oyo dialect-specific rules"""
        # Example transformations
        replacements = {
            'ʃ': 's',  # Replace ʃ with s
            'ɔ': 'o',  # Replace ɔ with o
            # Add more replacements
        }
        for old, new in replacements.items():
            text = text.replace(old, new)
        return text

class DataAugmenter:
    def __init__(self, language: str):
        self.language = language
        self.preprocessor = AfricanLanguagePreprocessor(language)
    
    def augment_text(self, text: str, techniques: List[str] = None) -> List[str]:
        """Apply various augmentation techniques"""
        if techniques is None:
            techniques = ['tone_variation', 'dialect_variation']
        
        augmented = []
        for technique in techniques:
            if technique == 'tone_variation':
                augmented.extend(self._apply_tone_variations(text))
            elif technique == 'dialect_variation':
                augmented.extend(self._apply_dialect_variations(text))
        
        return augmented
    
    def _apply_tone_variations(self, text: str) -> List[str]:
        """Create variations with different tone patterns"""
        variations = []
        # Create tone pattern variations while maintaining meaning
        # This requires language-specific rules
        return variations
    
    def _apply_dialect_variations(self, text: str) -> List[str]:
        """Create dialect variations"""
        variations = []
        if self.language == "yor":
            dialects = [d.value for d in DialectRegion if d.name.startswith("YORUBA")]
            for dialect in dialects:
                variation = self.preprocessor.apply_dialect_rules(text, dialect)
                variations.append(variation)
        return variations

class AfricanLanguageEvaluator:
    def __init__(self, language: str):
        self.language = language
        self.bleu = BLEU()
        
    def evaluate(
        self,
        predictions: List[str],
        references: List[str],
        metrics: List[str] = None
    ) -> Dict[str, float]:
        """Evaluate translations using multiple metrics"""
        if metrics is None:
            metrics = ['bleu', 'tone_accuracy', 'dialect_consistency']
        
        results = {}
        for metric in metrics:
            if metric == 'bleu':
                results['bleu'] = self._calculate_bleu(predictions, references)
            elif metric == 'tone_accuracy':
                results['tone_accuracy'] = self._calculate_tone_accuracy(
                    predictions,
                    references
                )
            elif metric == 'dialect_consistency':
                results['dialect_consistency'] = self._calculate_dialect_consistency(
                    predictions,
                    references
                )
        
        return results
    
    def _calculate_bleu(self, predictions: List[str], references: List[str]) -> float:
        """Calculate BLEU score with African language considerations"""
        return self.bleu.corpus_score(predictions, [references]).score
    
    def _calculate_tone_accuracy(
        self,
        predictions: List[str],
        references: List[str]
    ) -> float:
        """Calculate accuracy of tone marker preservation"""
        total_accuracy = 0
        for pred, ref in zip(predictions, references):
            pred_tones = self._extract_tone_patterns(pred)
            ref_tones = self._extract_tone_patterns(ref)
            accuracy = len(set(pred_tones) & set(ref_tones)) / len(set(ref_tones))
            total_accuracy += accuracy
        return total_accuracy / len(predictions)
    
    def _extract_tone_patterns(self, text: str) -> List[str]:
        """Extract tone patterns from text"""
        patterns = []
        text = unicodedata.normalize('NFD', text)
        for char in text:
            if char in {m.value for m in ToneMarker}:
                patterns.append(char)
        return patterns

class TranslationCache:
    def __init__(self, max_size: int = 10000):
        self.cache = {}
        self.max_size = max_size
        
    def get(self, text: str, source_lang: str, target_lang: str) -> Optional[str]:
        """Get cached translation"""
        key = (text, source_lang, target_lang)
        return self.cache.get(key)
    
    def add(self, text: str, translation: str, source_lang: str, target_lang: str):
        """Add translation to cache"""
        if len(self.cache) >= self.max_size:
            # Remove random entry if cache is full
            key_to_remove = random.choice(list(self.cache.keys()))
            del self.cache[key_to_remove]
        
        key = (text, source_lang, target_lang)
        self.cache[key] = translation 