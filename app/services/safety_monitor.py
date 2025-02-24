from typing import Dict, List, Optional, Tuple
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import logging
from dataclasses import dataclass
from app.monitoring.metrics import SAFETY_CHECK_METRICS

logger = logging.getLogger(__name__)

@dataclass
class SafetyCheckResult:
    is_safe: bool
    risk_scores: Dict[str, float]
    flagged_categories: List[str]
    mitigation_applied: Optional[str] = None

class SafetyMonitor:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load safety models
        self.toxicity_model = AutoModelForSequenceClassification.from_pretrained(
            "facebook/roberta-hate-speech-dynabench-r4-target"
        ).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(
            "facebook/roberta-hate-speech-dynabench-r4-target"
        )
        
        # Safety thresholds
        self.safety_thresholds = {
            "hate_speech": 0.7,
            "bias": 0.6,
            "personal_info": 0.8,
            "harmful_content": 0.7
        }
        
        # Content policies
        self.content_policies = {
            "max_toxicity_score": 0.7,
            "max_bias_score": 0.6,
            "allowed_topics": set(["education", "technology", "culture", "general"]),
            "blocked_patterns": set([
                r"\b(?:password|credit_card|ssn)\b",
                r"\b(?:address|phone_number)\b"
            ])
        }

    async def check_content_safety(self, text: str) -> SafetyCheckResult:
        """Check content for safety concerns"""
        try:
            # Tokenize and get model predictions
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=512
            ).to(self.device)
            
            with torch.no_grad():
                outputs = self.toxicity_model(**inputs)
                scores = torch.sigmoid(outputs.logits)

            # Analyze different safety aspects
            risk_scores = {
                "hate_speech": scores[0][0].item(),
                "bias": await self._check_bias(text),
                "personal_info": await self._check_personal_info(text),
                "harmful_content": await self._check_harmful_content(text)
            }

            # Identify flagged categories
            flagged = [
                category for category, score in risk_scores.items()
                if score > self.safety_thresholds[category]
            ]

            # Record metrics
            SAFETY_CHECK_METRICS.labels(
                check_type="content_safety",
                result="flagged" if flagged else "safe"
            ).inc()

            return SafetyCheckResult(
                is_safe=len(flagged) == 0,
                risk_scores=risk_scores,
                flagged_categories=flagged
            )

        except Exception as e:
            logger.error(f"Safety check failed: {e}")
            return SafetyCheckResult(
                is_safe=False,
                risk_scores={},
                flagged_categories=["check_failed"]
            )

    async def apply_content_policy(
        self,
        text: str,
        context: Dict[str, any]
    ) -> Tuple[str, SafetyCheckResult]:
        """Apply content policy and return sanitized text"""
        safety_result = await self.check_content_safety(text)
        
        if not safety_result.is_safe:
            sanitized_text = await self._sanitize_content(
                text,
                safety_result.flagged_categories
            )
            safety_result.mitigation_applied = "content_sanitization"
            return sanitized_text, safety_result
            
        return text, safety_result

    async def _sanitize_content(
        self,
        text: str,
        flagged_categories: List[str]
    ) -> str:
        """Sanitize content based on flagged categories"""
        sanitized = text
        
        for category in flagged_categories:
            if category == "personal_info":
                sanitized = self._redact_personal_info(sanitized)
            elif category == "hate_speech":
                sanitized = self._remove_harmful_language(sanitized)
            elif category == "bias":
                sanitized = self._neutralize_bias(sanitized)
                
        return sanitized 