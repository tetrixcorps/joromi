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
        
        # Supported language codes including African languages
        self.supported_languages = {
            # African Languages
            "af": "afr",  # Afrikaans
            "am": "amh",  # Amharic
            "ha": "hau",  # Hausa
            "ig": "ibo",  # Igbo
            "ln": "lin",  # Lingala
            "mg": "mlg",  # Malagasy
            "ny": "nya",  # Nyanja/Chichewa
            "om": "orm",  # Oromo
            "sn": "sna",  # Shona
            "so": "som",  # Somali
            "sw": "swh",  # Swahili
            "wo": "wol",  # Wolof
            "xh": "xho",  # Xhosa
            "yo": "yor",  # Yoruba
            "zu": "zul",  # Zulu
            
            # Existing languages
            "en": "eng",  # English
            "es": "spa",  # Spanish
            "fr": "fra",  # French
            "de": "deu",  # German
            "zh": "cmn",  # Chinese
            "ja": "jpn",  # Japanese
            "ko": "kor",  # Korean
            "ar": "ara",  # Arabic
        }

        # Add language names for UI display
        self.language_names = {
            # African Languages
            "afr": "Afrikaans",
            "amh": "Amharic (አማርኛ)",
            "hau": "Hausa (Hausa)",
            "ibo": "Igbo (Igbo)",
            "lin": "Lingala (Lingála)",
            "mlg": "Malagasy",
            "nya": "Nyanja/Chichewa (Chichewa)",
            "orm": "Oromo (Oromoo)",
            "sna": "Shona (chiShona)",
            "som": "Somali (Soomaali)",
            "swh": "Swahili (Kiswahili)",
            "wol": "Wolof (Wolof)",
            "xho": "Xhosa (isiXhosa)",
            "yor": "Yoruba (Yorùbá)",
            "zul": "Zulu (isiZulu)",
            
            # Existing languages
            "eng": "English",
            "spa": "Spanish (Español)",
            "fra": "French (Français)",
            "deu": "German (Deutsch)",
            "cmn": "Chinese (中文)",
            "jpn": "Japanese (日本語)",
            "kor": "Korean (한국어)",
            "ara": "Arabic (العربية)",
        }

    async def get_language_name(self, lang_code: str) -> str:
        """Get the display name of a language"""
        return self.language_names.get(lang_code, lang_code)

    async def is_african_language(self, lang_code: str) -> bool:
        """Check if a language code represents an African language"""
        african_codes = {
            "afr", "amh", "hau", "ibo", "lin", "mlg", "nya", 
            "orm", "sna", "som", "swh", "wol", "xho", "yor", "zul"
        }
        return lang_code in african_codes

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