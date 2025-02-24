import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel
from typing import Dict, List, Optional
import logging
from dataclasses import dataclass
from app.utils.tensor_utils import batch_encode

logger = logging.getLogger(__name__)

@dataclass
class DialectInfo:
    name: str
    confidence: float
    region: str
    features: Dict[str, float]

class DialectAnalyzer:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load models and tokenizers
        self.tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-large")
        self.encoder = AutoModel.from_pretrained("xlm-roberta-large").to(self.device)
        
        # Load dialect classifiers for different languages
        self.dialect_classifiers = {
            "swh": self._load_classifier("swahili_dialects_v2.pt"),
            "yor": self._load_classifier("yoruba_dialects_v1.pt"),
            "hau": self._load_classifier("hausa_dialects_v1.pt"),
        }
        
        # Dialect features and regions
        self.dialect_metadata = {
            "swh": {
                "regions": ["coastal", "inland", "urban", "rural"],
                "features": ["tone_patterns", "vocabulary", "grammar_structures"]
            },
            "yor": {
                "regions": ["oyo", "ekiti", "ijebu", "lagos"],
                "features": ["intonation", "lexical_choice", "phonological_patterns"]
            },
            "hau": {
                "regions": ["kano", "sokoto", "bauchi", "kaduna"],
                "features": ["consonant_variation", "vowel_length", "tonal_patterns"]
            }
        }

    def _load_classifier(self, model_path: str) -> torch.nn.Module:
        """Load dialect classification model"""
        try:
            model = torch.load(f"models/dialects/{model_path}", map_location=self.device)
            model.eval()
            return model
        except Exception as e:
            logger.error(f"Failed to load dialect classifier {model_path}: {e}")
            return None

    async def _generate_embedding(self, text: str) -> torch.Tensor:
        """Generate text embeddings for dialect analysis"""
        try:
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512
            ).to(self.device)

            with torch.no_grad():
                outputs = self.encoder(**inputs)
                # Use CLS token embedding
                embedding = outputs.last_hidden_state[:, 0, :]
                
            return embedding

        except Exception as e:
            logger.error(f"Embedding generation failed: {e}")
            return None

    async def detect_dialect(self, text: str, language_code: str) -> Optional[DialectInfo]:
        """Identify regional variations within supported languages"""
        try:
            if language_code not in self.dialect_classifiers:
                return None

            classifier = self.dialect_classifiers[language_code]
            if not classifier:
                return None

            # Generate embeddings
            embedding = await self._generate_embedding(text)
            if embedding is None:
                return None

            # Get dialect predictions
            with torch.no_grad():
                outputs = classifier(embedding)
                probs = torch.softmax(outputs, dim=1)
                
                # Get the most likely dialect
                confidence, dialect_idx = torch.max(probs, dim=1)
                
                # Extract feature importance scores
                feature_scores = self._extract_feature_scores(
                    classifier,
                    embedding,
                    self.dialect_metadata[language_code]["features"]
                )

            # Map to dialect information
            dialect_info = DialectInfo(
                name=self.dialect_metadata[language_code]["regions"][dialect_idx],
                confidence=confidence.item(),
                region=self.dialect_metadata[language_code]["regions"][dialect_idx],
                features=feature_scores
            )

            return dialect_info

        except Exception as e:
            logger.error(f"Dialect detection failed: {e}")
            return None

    async def analyze_dialect_features(
        self,
        text: str,
        language_code: str
    ) -> Dict[str, List[Dict[str, float]]]:
        """Analyze specific linguistic features of the dialect"""
        try:
            dialect_info = await self.detect_dialect(text, language_code)
            if not dialect_info:
                return {}

            # Analyze specific features based on language
            if language_code == "swh":
                return await self._analyze_swahili_features(text, dialect_info)
            elif language_code == "yor":
                return await self._analyze_yoruba_features(text, dialect_info)
            elif language_code == "hau":
                return await self._analyze_hausa_features(text, dialect_info)
            
            return {}

        except Exception as e:
            logger.error(f"Feature analysis failed: {e}")
            return {}

    def _extract_feature_scores(
        self,
        model: torch.nn.Module,
        embedding: torch.Tensor,
        features: List[str]
    ) -> Dict[str, float]:
        """Extract importance scores for linguistic features"""
        try:
            # Get attention weights from the model
            attention_weights = model.get_attention_weights(embedding)
            
            # Map attention to features
            feature_scores = {}
            for idx, feature in enumerate(features):
                score = attention_weights[0, idx].mean().item()
                feature_scores[feature] = score
                
            return feature_scores

        except Exception as e:
            logger.error(f"Feature score extraction failed: {e}")
            return {feature: 0.0 for feature in features}

    async def _analyze_swahili_features(
        self,
        text: str,
        dialect_info: DialectInfo
    ) -> Dict[str, List[Dict[str, float]]]:
        """Analyze Swahili-specific dialect features"""
        return {
            "tone_patterns": [
                {"pattern": "rising", "confidence": dialect_info.features["tone_patterns"]},
                {"pattern": "falling", "confidence": 1 - dialect_info.features["tone_patterns"]}
            ],
            "vocabulary": [
                {"type": "coastal", "confidence": dialect_info.features["vocabulary"]},
                {"type": "inland", "confidence": 1 - dialect_info.features["vocabulary"]}
            ],
            "grammar": [
                {"structure": "standard", "confidence": dialect_info.features["grammar_structures"]},
                {"structure": "regional", "confidence": 1 - dialect_info.features["grammar_structures"]}
            ]
        } 