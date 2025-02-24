from transformers import M4T2ForAllTasks, AutoProcessor
import torch
from typing import Dict, Optional, Any
import logging
from app.services.dialect_analyzer import DialectAnalyzer
from app.services.safety_monitor import SafetyMonitor
from app.services.rate_limiter import RateLimiter
from app.monitoring.model_monitor import ModelMonitor
import time
from app.observability.inference_tracer import InferenceTracer, TraceContext
from app.services.model_optimizer import ModelOptimizer, OptimizationConfig

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

        self.dialect_analyzer = DialectAnalyzer()
        self.safety_monitor = SafetyMonitor()
        self.rate_limiter = RateLimiter()
        self.model_monitor = ModelMonitor()
        self.inference_tracer = InferenceTracer()
        self.model_optimizer = ModelOptimizer(
            OptimizationConfig(
                pruning_method="magnitude",
                target_sparsity=0.3,
                distillation_temperature=2.0,
                quantization_bits=8
            )
        )

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

    async def translate(
        self,
        text: str,
        target_lang: str,
        source_lang: Optional[str] = None
    ) -> str:
        """Translate text with tracing"""
        try:
            # Create trace context
            context = TraceContext(
                model_name="seamless-m4t-v2",
                input_type="text",
                batch_size=1,
                device=self.device,
                metadata={
                    "source_lang": source_lang,
                    "target_lang": target_lang
                }
            )

            # Wrap inference in tracer
            async def inference_func(text_input):
                inputs = self.processor(
                    text=text_input,
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

                return self.processor.decode(output[0], skip_special_tokens=True)

            return await self.inference_tracer.trace_inference(
                text,
                context,
                inference_func
            )

        except Exception as e:
            logger.error(f"Translation failed: {e}")
            return text

    async def analyze_text(
        self,
        text: str,
        source_lang: Optional[str] = None
    ) -> Dict[str, Any]:
        """Analyze text for language and dialect information"""
        try:
            # Detect language if not provided
            if not source_lang:
                source_lang = await self.detect_language(text)

            # Get dialect information if available
            dialect_info = None
            if source_lang in self.dialect_analyzer.dialect_classifiers:
                dialect_info = await self.dialect_analyzer.detect_dialect(
                    text,
                    source_lang
                )

            # Get detailed feature analysis
            dialect_features = {}
            if dialect_info:
                dialect_features = await self.dialect_analyzer.analyze_dialect_features(
                    text,
                    source_lang
                )

            return {
                "language": source_lang,
                "language_name": await self.get_language_name(source_lang),
                "is_african": await self.is_african_language(source_lang),
                "dialect": {
                    "name": dialect_info.name if dialect_info else None,
                    "confidence": dialect_info.confidence if dialect_info else 0.0,
                    "region": dialect_info.region if dialect_info else None,
                    "features": dialect_features
                }
            }

        except Exception as e:
            logger.error(f"Text analysis failed: {e}")
            return {
                "language": source_lang,
                "language_name": await self.get_language_name(source_lang),
                "is_african": await self.is_african_language(source_lang),
                "dialect": None
            } 

    async def translate_with_dialect(
        self,
        text: str,
        target_lang: str,
        preserve_dialect: bool = True
    ) -> Dict[str, Any]:
        """Translate text while preserving dialect features"""
        try:
            # Analyze source text
            analysis = await self.analyze_text(text)
            source_lang = analysis["language"]
            
            # Perform base translation
            translated_text = await self.translate(
                text,
                target_lang,
                source_lang
            )
            
            # Adjust translation for dialect if requested
            if preserve_dialect and analysis["dialect"]:
                translated_text = await self._adjust_translation_for_dialect(
                    translated_text,
                    target_lang,
                    analysis["dialect"]
                )

            return {
                "text": translated_text,
                "source_analysis": analysis,
                "target_lang": target_lang
            }

        except Exception as e:
            logger.error(f"Dialect-aware translation failed: {e}")
            # Fallback to regular translation
            return {
                "text": await self.translate(text, target_lang),
                "source_analysis": None,
                "target_lang": target_lang
            } 

    async def translate_with_safety(
        self,
        text: str,
        target_lang: str,
        user_id: str,
        context: Dict[str, any]
    ) -> Dict[str, Any]:
        """Translate text with safety checks"""
        try:
            # Check rate limit
            if not await self.rate_limiter.check_rate_limit(user_id):
                raise Exception("Rate limit exceeded")

            # Check content safety
            sanitized_text, safety_result = await self.safety_monitor.apply_content_policy(
                text,
                context
            )

            # Track model usage
            start_time = time.time()
            try:
                translation = await self.translate(sanitized_text, target_lang)
                await self.model_monitor.track_model_call(
                    "translation",
                    "translate",
                    time.time() - start_time
                )
            except Exception as e:
                await self.model_monitor.track_model_call(
                    "translation",
                    "translate",
                    time.time() - start_time,
                    error=e
                )
                raise

            # Check translation safety
            translated_safety = await self.safety_monitor.check_content_safety(translation)

            return {
                "text": translation,
                "source_safety": safety_result,
                "target_safety": translated_safety,
                "mitigation_applied": safety_result.mitigation_applied
            }

        except Exception as e:
            logger.error(f"Safe translation failed: {e}")
            return {
                "error": str(e),
                "text": text
            } 

    async def optimize_translation_model(self):
        """Optimize the translation model"""
        try:
            # Create evaluation dataloader
            eval_dataloader = self._create_eval_dataloader()
            
            # Optimize model
            optimized_model, metrics = await self.model_optimizer.optimize_model(
                self.model,
                eval_dataloader
            )
            
            # Log optimization results
            logger.info(
                "Model optimization complete",
                extra={
                    "size_reduction": metrics["size_reduction"],
                    "inference_speedup": metrics["inference_speedup"],
                    "accuracy_change": metrics["accuracy_change"]
                }
            )
            
            # Update model if optimization was successful
            if metrics["accuracy_change"] > -0.05:  # Allow 5% accuracy drop
                self.model = optimized_model
                logger.info("Updated to optimized model")
            else:
                logger.warning("Optimization resulted in too much accuracy loss")
                
        except Exception as e:
            logger.error(f"Model optimization failed: {e}") 