from typing import Dict, Any, Optional, Union, List
import torch
from dataclasses import dataclass
from config.model_config import ModelConfigurations
from models.model_loader import ModelManager
from utils.logger import model_logger, CustomLogger
import json
from transformers import Pix2StructForConditionalGeneration, Pix2StructProcessor
from utils.banking_utils import BankingQueryParser, BankingQueryType, BankingErrorHandler
import numpy as np
from utils.audio_processor import AudioPreprocessor, AudioConfig
from utils.tts_utils import TTSProcessor, TTSConfigurations, EmotionType
from utils.translation_utils import TranslationConfig, BatchTranslator, LanguageDetector
from services.dolphin_service import DolphinService
from services.asr_service import ASRService
from services.translation_service import TranslationService
import logging

logger = logging.getLogger(__name__)

@dataclass
class RequestMetadata:
    request_type: str
    input_modality: str
    output_modality: str
    language: Optional[str] = None
    accent: Optional[str] = None

class ModelOrchestrator:
    def __init__(self, device: torch.device):
        self.device = device
        self.model_manager = ModelManager(device)
        self.logger = model_logger
        
        # Initialize all models
        self._initialize_models()
        
        self.audio_processor = AudioPreprocessor(AudioConfig(
            sample_rate=16000,
            remove_silence=True,
            noise_reduction=True,
            target_db=-20
        ))
        
        self.tts_processor = TTSProcessor()
        
        self.language_detector = LanguageDetector(device)
        self.batch_translator = None  # Will be initialized when needed
        
        # Initialize services
        self.services = {
            "dolphin": DolphinService(port=8001),
            "asr": ASRService(port=8002),
            "translation": TranslationService(port=8003)
        }
        
        # Set default model
        self.default_model = "pix2struct"  # Use Pix2Struct as default
        
    @CustomLogger.log_execution_time(model_logger)
    def _initialize_models(self):
        """Initialize all required models"""
        try:
            # Load Pix2Struct as default model
            self.model_manager.load_model(ModelConfigurations.PIX2STRUCT)
            
            # Load Dolphin as secondary model
            self.model_manager.load_model(ModelConfigurations.DOLPHIN)
            
            # Load other models
            self.model_manager.load_model(ModelConfigurations.SPEECH.name)
            self.model_manager.load_model(ModelConfigurations.ASR.name)
            self.model_manager.load_model(ModelConfigurations.TRANSLATION.name)
            
            self.logger.info("All models initialized successfully")
        except Exception as e:
            self.logger.error(f"Failed to initialize models: {str(e)}")
            raise

    def _analyze_request(self, request: Dict[str, Any]) -> RequestMetadata:
        """Analyze request to determine required models"""
        input_type = request.get('input_type', 'text')
        output_type = request.get('output_type', 'text')
        language = request.get('language', 'en')
        accent = request.get('accent')
        
        # Determine request type based on content and modalities
        if 'image' in request:
            request_type = 'visual_qa'
        elif request.get('domain') in ['banking', 'medical']:
            request_type = 'specialized'
        else:
            request_type = 'general'
            
        return RequestMetadata(
            request_type=request_type,
            input_modality=input_type,
            output_modality=output_type,
            language=language,
            accent=accent
        )

    @CustomLogger.log_execution_time(model_logger)
    async def process_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Process incoming request and route to appropriate models"""
        try:
            metadata = self._analyze_request(request)
            self.logger.info(f"Processing request type: {metadata.request_type}")
            
            # Speech-to-text if needed
            if metadata.input_modality == 'speech':
                asr_model, asr_tokenizer = self.model_manager.get_model(ModelConfigurations.ASR.name)
                request['text'] = await self._process_speech(request['audio'], asr_model, asr_tokenizer)
            
            # Translation if needed
            if metadata.language != 'en':
                translation_model, translation_tokenizer = self.model_manager.get_model(
                    ModelConfigurations.TRANSLATION.name
                )
                request['text'] = await self._translate(
                    request['text'], 
                    translation_model, 
                    translation_tokenizer,
                    source_lang=metadata.language
                )
            
            # Route to appropriate model for main processing
            response = await self._route_request(request, metadata)
            
            # Text-to-speech if needed
            if metadata.output_modality == 'speech':
                tts_model, tts_tokenizer = self.model_manager.get_model(ModelConfigurations.SPEECH.name)
                response['audio'] = await self._generate_speech(
                    response['text'],
                    tts_model,
                    tts_tokenizer,
                    accent=metadata.accent,
                    style=metadata.accent,
                    rate=1.0,
                    pitch_shift=0.0,
                    emotion=None,
                    emotion_intensity=1.0
                )
            
            return response
            
        except Exception as e:
            self.logger.error(f"Error processing request: {str(e)}")
            return {
                'error': str(e),
                'status': 'failed'
            }

    async def _route_request(
        self, 
        request: Dict[str, Any], 
        metadata: RequestMetadata
    ) -> Dict[str, Any]:
        """Route request to appropriate model"""
        try:
            # Check if model is explicitly specified
            model_name = request.get('model', self.default_model)
            
            if model_name == "pix2struct":
                model = self.model_manager.get_model(ModelConfigurations.PIX2STRUCT.name)
                processor = Pix2StructProcessor.from_pretrained(ModelConfigurations.PIX2STRUCT.name)
                return await self._process_visual_qa(request, model, processor)
                
            elif model_name == "dolphin":
                return await self.services["dolphin"].process(request)
                
            elif metadata.request_type == 'visual_qa':
                # Default to Pix2Struct for visual QA
                model = self.model_manager.get_model(ModelConfigurations.PIX2STRUCT.name)
                processor = Pix2StructProcessor.from_pretrained(ModelConfigurations.PIX2STRUCT.name)
                return await self._process_visual_qa(request, model, processor)
                
            elif metadata.request_type == 'chat':
                # Use Dolphin for chat
                return await self.services["dolphin"].process(request)
                
            else:
                # Default to Pix2Struct for other cases
                model = self.model_manager.get_model(ModelConfigurations.PIX2STRUCT.name)
                processor = Pix2StructProcessor.from_pretrained(ModelConfigurations.PIX2STRUCT.name)
                return await self._process_visual_qa(request, model, processor)
                
        except Exception as e:
            self.logger.error(f"Error routing request: {str(e)}")
            raise

    # Implement specific processing methods
    async def _process_speech(self, audio_input, model, processor):
        """Process speech input to text with enhanced audio preprocessing and segmentation"""
        try:
            self.logger.info("Processing speech input")
            
            # Preprocess and potentially segment audio
            processed_segments, processing_info = self.audio_processor.process_long_audio(
                audio_input
            )
            self.logger.info(f"Audio preprocessing completed: {processing_info}")
            
            # Process each segment
            transcriptions = []
            segment_info = processing_info.get('segmentation', {}).get('segment_info', [{}])
            
            for segment, info in zip(processed_segments, segment_info):
                # Process the enhanced audio segment
                input_features = processor(
                    segment,
                    sampling_rate=16000,
                    return_tensors="pt"
                ).input_features.to(self.device)
                
                # Generate transcription for segment
                predicted_ids = model.generate(input_features)
                transcription = processor.batch_decode(
                    predicted_ids,
                    skip_special_tokens=True
                )[0]
                
                transcriptions.append(transcription)
                self.logger.info(f"Processed segment: {info.get('start_time', 0)}-{info.get('end_time', 0)} seconds")
            
            # Merge transcriptions if needed
            if len(transcriptions) > 1:
                final_transcription = self.audio_processor.segmenter.merge_transcriptions(
                    transcriptions,
                    segment_info
                )
            else:
                final_transcription = transcriptions[0]
            
            self.logger.info("Speech processing completed")
            return {
                "status": "success",
                "text": final_transcription,
                "model_type": "asr",
                "processing_info": processing_info,
                "segments": len(processed_segments)
            }
            
        except Exception as e:
            self.logger.error(f"Speech processing failed: {str(e)}")
            raise

    async def _translate(
        self,
        text: Union[str, List[str]],
        model,
        processor,
        source_lang: Optional[str] = None,
        target_lang: str = "eng",
        detect_language: bool = True
    ) -> Dict[str, Any]:
        """Enhanced translation with language detection and batch support"""
        try:
            # Initialize batch translator if needed
            if self.batch_translator is None:
                self.batch_translator = BatchTranslator(model, processor, self.device)
            
            # Handle batch translation
            if isinstance(text, list):
                if detect_language:
                    # Detect languages for batch
                    detected = self.language_detector.detect_batch(text)
                    source_langs = [lang for lang, _ in detected]
                    confidences = [conf for _, conf in detected]
                else:
                    source_langs = [source_lang] * len(text)
                    confidences = [1.0] * len(text)
                
                # Translate batch
                translations = await self.batch_translator.translate_batch(
                    text,
                    source_langs[0],  # Use first detected language
                    target_lang
                )
                
                return {
                    "status": "success",
                    "translations": translations,
                    "detected_languages": list(zip(source_langs, confidences)),
                    "model_type": "translation"
                }
            
            # Single text translation
            if detect_language and not source_lang:
                detected_lang, confidence = self.language_detector.detect_language(text)
                source_lang = detected_lang
                self.logger.info(f"Detected language: {detected_lang} (confidence: {confidence:.2f})")
            
            # Verify language pair support
            if not TranslationConfig.is_pair_supported(source_lang, target_lang):
                raise ValueError(f"Translation pair not supported: {source_lang} -> {target_lang}")
            
            # Translate
            inputs = processor(
                text=text,
                src_lang=source_lang,
                tgt_lang=target_lang,
                return_tensors="pt"
            ).to(self.device)
            
            output_tokens = model.generate(
                **inputs,
                tgt_lang=target_lang,
                max_length=256
            )
            
            translation = processor.decode(
                output_tokens[0].tolist(),
                skip_special_tokens=True
            )
            
            return {
                "status": "success",
                "text": translation,
                "source_lang": source_lang,
                "target_lang": target_lang,
                "model_type": "translation",
                "supported_languages": TranslationConfig.get_supported_languages()
            }
            
        except Exception as e:
            self.logger.error(f"Translation failed: {str(e)}")
            raise

    async def _process_visual_qa(self, request, model, processor):
        """Process visual question answering using Pix2Struct"""
        try:
            self.logger.info("Processing visual QA request")
            
            # Prepare inputs
            inputs = processor(
                images=request['image'],
                text=request['question'],
                return_tensors="pt"
            ).to(self.device)
            
            # Generate prediction
            predictions = model.generate(**inputs)
            
            # Decode response
            answer = processor.decode(predictions[0], skip_special_tokens=True)
            
            self.logger.info("Visual QA processing completed")
            return {
                "status": "success",
                "text": answer,
                "confidence": float(predictions[0].max().item())
            }
            
        except Exception as e:
            self.logger.error(f"Visual QA processing failed: {str(e)}")
            raise

    async def _process_specialized(self, request, model, tokenizer):
        """Process specialized domain requests"""
        # Implementation for specialized domains
        pass

    async def _process_general(self, request, model, tokenizer):
        """Process general requests"""
        # Implementation for general purpose
        pass

    async def _generate_speech(
        self,
        text: str,
        model,
        processor,
        accent: Optional[str] = None,
        style: Optional[str] = None,
        rate: float = 1.0,
        pitch_shift: float = 0.0,
        emotion: Optional[str] = None,
        emotion_intensity: float = 1.0
    ) -> Dict[str, Any]:
        """Generate speech with enhanced control over accent, style, and emotion"""
        try:
            self.logger.info(
                f"Generating speech with accent: {accent}, "
                f"style: {style}, emotion: {emotion}"
            )
            
            # Get configurations
            accent_config = self.tts_processor.configs.ACCENT_CONFIGS.get(
                accent,
                self.tts_processor.configs.ACCENT_CONFIGS['nigerian']
            )
            
            # Generate base speech
            inputs = processor(
                text=text,
                accent=accent_config.name,
                return_tensors="pt"
            ).to(self.device)
            
            speech_ids = model.generate(
                **inputs,
                do_sample=True,
                temperature=0.7
            )
            
            # Convert to audio
            speech = processor.batch_decode(speech_ids)[0].cpu().numpy()
            
            # Apply emotion if specified
            if emotion:
                speech = self.tts_processor.process_with_emotion(
                    speech,
                    emotion,
                    emotion_intensity,
                    accent_config
                )
            else:
                # Apply standard processing
                speech = self.tts_processor.adjust_speech_parameters(
                    speech,
                    accent_config,
                    rate=rate,
                    pitch_shift=pitch_shift,
                    style=style
                )
            
            return {
                "status": "success",
                "audio": speech,
                "sampling_rate": processor.sampling_rate,
                "accent_used": accent,
                "style_used": style,
                "emotion_used": emotion,
                "rate": rate,
                "pitch_shift": pitch_shift,
                "available_accents": list(self.tts_processor.configs.ACCENT_CONFIGS.keys()),
                "available_styles": list(self.tts_processor.configs.VOICE_STYLES.keys()),
                "available_emotions": [e.value for e in EmotionType]
            }
            
        except Exception as e:
            self.logger.error(f"Speech generation failed: {str(e)}")
            raise

    async def _process_banking(self, request: Dict[str, Any], model, tokenizer) -> Dict[str, Any]:
        """Process banking-specific requests with enhanced error handling"""
        try:
            query = request['text']
            
            # Validate query for sensitive information
            if not BankingQueryParser.validate_query(query):
                raise BankingErrorHandler.BankingError(
                    "Query contains sensitive information that should not be shared",
                    "security_violation",
                    BankingQueryType.SECURITY
                )
            
            # Classify query type
            query_type = BankingQueryParser.classify_query(query)
            self.logger.info(f"Processing banking query type: {query_type.value}")
            
            # Format input using Alpaca template with query type context
            input_text = f'''Below is an instruction that describes a task, paired with an appropriate response.

## Instruction:
User Query Type: {query_type.value}
User Query: {query}

### Input:
None

### Response:
Answer:'''
            
            # Prepare inputs
            inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
            
            # Adjust generation parameters based on query type
            generation_params = self._get_generation_params(query_type)
            
            # Generate response
            outputs = model.generate(
                **inputs,
                **generation_params
            )
            
            # Decode and process response
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            processed_response = self._process_banking_response(response, query_type)
            
            return {
                "status": "success",
                "text": processed_response["answer"],
                "model_type": "banking",
                "query_type": query_type.value,
                "confidence": processed_response.get("confidence", 1.0),
                "requires_action": processed_response.get("requires_action", False)
            }
            
        except BankingErrorHandler.BankingError as e:
            self.logger.error(f"Banking error: {str(e)}")
            return BankingErrorHandler.handle_error(e, query_type)
        except Exception as e:
            self.logger.error(f"Banking query processing failed: {str(e)}")
            return BankingErrorHandler.handle_error(e, BankingQueryType.GENERAL)

    def _get_generation_params(self, query_type: BankingQueryType) -> Dict[str, Any]:
        """Get generation parameters based on query type"""
        params = {
            BankingQueryType.SECURITY: {
                "max_new_tokens": 100,
                "temperature": 0.3,
                "top_p": 0.9
            },
            BankingQueryType.CARD_OPERATIONS: {
                "max_new_tokens": 75,
                "temperature": 0.5,
                "top_p": 0.95
            },
            BankingQueryType.TRANSACTION: {
                "max_new_tokens": 50,
                "temperature": 0.7,
                "top_p": 0.9
            }
        }
        return params.get(query_type, {
            "max_new_tokens": 50,
            "temperature": 0.7,
            "top_p": 0.95
        })

    def _process_banking_response(self, response: str, query_type: BankingQueryType) -> Dict[str, Any]:
        """Process and structure banking response"""
        parts = response.split('Answer:')
        if len(parts) > 1:
            answer = parts[1].strip()
        else:
            answer = response.strip()
            
        # Check for action requirements
        requires_action = any(keyword in answer.lower() 
                            for keyword in ["contact", "visit", "call", "verify"])
        
        return {
            "answer": answer,
            "requires_action": requires_action,
            "confidence": 0.95 if requires_action else 0.85
        }

    async def generate_text(
        self,
        prompt: str,
        model,
        tokenizer,
        max_length: int = 1024,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50
    ) -> Dict[str, Any]:
        """Generate text using Dolphin model"""
        try:
            # Prepare inputs
            inputs = tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=max_length
            ).to(self.device)
            
            # Generate
            outputs = model.generate(
                **inputs,
                max_length=max_length,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
            
            # Decode output
            generated_text = tokenizer.decode(
                outputs[0],
                skip_special_tokens=True
            )
            
            return {
                "status": "success",
                "generated_text": generated_text,
                "model_type": "dolphin"
            }
            
        except Exception as e:
            self.logger.error(f"Text generation failed: {str(e)}")
            return {
                "status": "error",
                "error": str(e),
                "model_type": "dolphin"
            }

    async def initialize(self):
        """Initialize all services"""
        for service_name, service in self.services.items():
            try:
                await service.initialize()
                logger.info(f"Initialized {service_name} service")
            except Exception as e:
                logger.error(f"Failed to initialize {service_name}: {e}")
                raise

    async def process_request(self, service_name: str, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process request using specified service"""
        if service_name not in self.services:
            return {"error": f"Service {service_name} not found", "status": "error"}
        
        return await self.services[service_name].process(input_data)

    async def cleanup(self):
        """Cleanup all services"""
        for service in self.services.values():
            await service.cleanup() 