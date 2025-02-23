from transformers import M4T2ForAllTasks, AutoProcessor
import torch
from typing import Dict, Optional
import logging

logger = logging.getLogger(__name__)

class TranslationService:
    def __init__(self):
        self.model = M4T2ForAllTasks.from_pretrained("facebook/seamless-m4t-v2-large")
        self.processor = AutoProcessor.from_pretrained("facebook/seamless-m4t-v2-large")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        
        # Supported language codes
        self.supported_languages = {
            "en": "eng",  # English
            "es": "spa",  # Spanish
            "fr": "fra",  # French
            "de": "deu",  # German
            "zh": "cmn",  # Chinese
            "ja": "jpn",  # Japanese
            "ko": "kor",  # Korean
            "ar": "ara",  # Arabic
            # Add more languages as needed
        }

    async def detect_language(self, text: str) -> str:
        """Detect the language of the input text"""
        try:
            inputs = self.processor(text=text, return_tensors="pt").to(self.device)
            with torch.no_grad():
                output = self.model.generate(**inputs, tgt_lang="eng", return_dict_in_generate=True)
                detected_lang = self.model.config.lang_to_code[output.sequences[0][0].item()]
                return detected_lang
        except Exception as e:
            logger.error(f"Language detection failed: {e}")
            return "eng"  # Default to English

    async def translate(self, 
                       text: str, 
                       target_lang: str = "eng", 
                       source_lang: Optional[str] = None) -> str:
        """Translate text to target language"""
        try:
            if not source_lang:
                source_lang = await self.detect_language(text)

            inputs = self.processor(
                text=text,
                src_lang=source_lang,
                return_tensors="pt"
            ).to(self.device)

            with torch.no_grad():
                output = self.model.generate(
                    **inputs,
                    tgt_lang=target_lang,
                    max_length=1024,
                    num_beams=5
                )

            translated_text = self.processor.decode(output[0], skip_special_tokens=True)
            return translated_text

        except Exception as e:
            logger.error(f"Translation failed: {e}")
            return text  # Return original text if translation fails 