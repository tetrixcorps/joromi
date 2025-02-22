import logging
from typing import Dict, Tuple, Optional
import torch
from transformers import AutoModel, AutoTokenizer, WhisperForConditionalGeneration, WhisperProcessor, AutoModelForSpeechSeq2Seq, AutoProcessor, SeamlessM4TModel, SeamlessM4TProcessor, AutoModelForCausalLM
from config.model_config import ModelSpecs
from config.paths import ModelPaths, MODEL_CACHE
import os
from pathlib import Path
from unsloth import FastLanguageModel
from utils.cache_manager import ModelCacheManager
from utils.gpu_monitor import GPUMonitor

class ModelManager:
    def __init__(self, device: torch.device):
        self.device = device
        self.models: Dict[str, torch.nn.Module] = {}
        self.tokenizers: Dict[str, AutoTokenizer] = {}
        self.logger = logging.getLogger(__name__)
        
        # Create model directories
        ModelPaths.create_dirs()
        
        # Set environment variable for Hugging Face cache
        os.environ['TRANSFORMERS_CACHE'] = str(MODEL_CACHE)
        
        self.cache_manager = ModelCacheManager(
            cache_dir=Path("/app/model_cache")
        )
        
        self.gpu_monitor = GPUMonitor()
        self.gpu_monitor.start()
        
    def check_device_requirements(self, specs: ModelSpecs) -> bool:
        """Check if device meets model requirements"""
        if self.device.type == 'cuda':
            vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
            return vram >= specs.device_requirements['min_vram']
        return False
        
    def load_model(self, specs: ModelSpecs):
        try:
            # Check GPU memory availability
            memory_usage = self.gpu_monitor.get_memory_usage()
            if any(usage > 90 for usage in memory_usage.values()):
                self.logger.warning("High GPU memory usage detected")
            
            # Ensure model files are in cache
            if not self.cache_manager.get_model_files(specs.name):
                raise ValueError(f"Failed to load model files for {specs.name}")
            
            # Load model to GPU
            return self._load_to_gpu(specs)
            
        except Exception as e:
            self.logger.error(f"Error loading model: {str(e)}")
            return None, None
            
    def get_model(self, name: str) -> Tuple[Optional[torch.nn.Module], Optional[AutoTokenizer]]:
        """Get a loaded model and its tokenizer"""
        return self.models.get(name), self.tokenizers.get(name)
    
    def _get_model_path(self, model_name: str) -> Path:
        """Get the appropriate path for a model"""
        if 'pix2struct' in model_name:
            return ModelPaths.PIX2STRUCT
        elif 'minicpm' in model_name:
            return ModelPaths.MINICPM
        elif 'afro-tts' in model_name:
            return ModelPaths.AFRO_TTS
        elif 'whisper' in model_name:
            return ModelPaths.WHISPER
        else:
            return MODEL_CACHE 

    def load_banking_model(self, specs: ModelSpecs) -> Tuple[Optional[torch.nn.Module], Optional[AutoTokenizer]]:
        """Load banking-specific model"""
        try:
            self.logger.info(f"Loading banking model: {specs.name}")
            
            model_path = self._get_model_path(specs.name)
            
            model, tokenizer = FastLanguageModel.from_pretrained(
                model_name=specs.name,
                max_seq_length=specs.device_requirements['max_seq_length'],
                dtype=specs.device_requirements['dtype'],
                load_in_4bit=specs.device_requirements['load_in_4bit'],
                device_map="auto",
                cache_dir=model_path
            )
            
            self.models[specs.name] = model
            self.tokenizers[specs.name] = tokenizer
            
            return model, tokenizer
            
        except Exception as e:
            self.logger.error(f"Failed to load banking model: {str(e)}")
            return None, None 

    def load_asr_model(self, specs: ModelSpecs) -> Tuple[Optional[torch.nn.Module], Optional[WhisperProcessor]]:
        """Load ASR-specific model"""
        try:
            self.logger.info(f"Loading ASR model: {specs.name}")
            
            model_path = self._get_model_path(specs.name)
            
            # Load model and processor
            model = WhisperForConditionalGeneration.from_pretrained(
                specs.name,
                cache_dir=model_path,
                torch_dtype=specs.device_requirements['dtype']
            ).to(self.device)
            
            processor = WhisperProcessor.from_pretrained(
                specs.name,
                cache_dir=model_path
            )
            
            self.models[specs.name] = model
            self.tokenizers[specs.name] = processor
            
            return model, processor
            
        except Exception as e:
            self.logger.error(f"Failed to load ASR model: {str(e)}")
            return None, None 

    def load_tts_model(self, specs: ModelSpecs) -> Tuple[Optional[torch.nn.Module], Optional[AutoProcessor]]:
        """Load TTS-specific model"""
        try:
            self.logger.info(f"Loading TTS model: {specs.name}")
            
            model_path = self._get_model_path(specs.name)
            
            # Load model and processor
            model = AutoModelForSpeechSeq2Seq.from_pretrained(
                specs.name,
                cache_dir=model_path,
                torch_dtype=specs.device_requirements['dtype']
            ).to(self.device)
            
            processor = AutoProcessor.from_pretrained(
                specs.name,
                cache_dir=model_path
            )
            
            self.models[specs.name] = model
            self.tokenizers[specs.name] = processor
            
            return model, processor
            
        except Exception as e:
            self.logger.error(f"Failed to load TTS model: {str(e)}")
            return None, None 

    def load_translation_model(self, specs: ModelSpecs) -> Tuple[Optional[torch.nn.Module], Optional[AutoProcessor]]:
        """Load translation-specific model"""
        try:
            self.logger.info(f"Loading translation model: {specs.name}")
            
            model_path = self._get_model_path(specs.name)
            
            # Load model and processor
            model = SeamlessM4TModel.from_pretrained(
                specs.name,
                cache_dir=model_path,
                torch_dtype=specs.device_requirements['dtype']
            ).to(self.device)
            
            processor = SeamlessM4TProcessor.from_pretrained(
                specs.name,
                cache_dir=model_path
            )
            
            self.models[specs.name] = model
            self.tokenizers[specs.name] = processor
            
            return model, processor
            
        except Exception as e:
            self.logger.error(f"Failed to load translation model: {str(e)}")
            return None, None 

    def load_dolphin_model(self, model_name: str = "cognitivecomputations/Dolphin3.0-Llama3.1-8B-GGUF") -> Tuple[Optional[torch.nn.Module], Optional[AutoTokenizer]]:
        """Load Dolphin 3.0 model"""
        try:
            self.logger.info(f"Loading Dolphin model: {model_name}")
            
            # Configure model loading
            model_config = {
                "device_map": "auto",
                "torch_dtype": torch.bfloat16,
                "load_in_8bit": True,
                "trust_remote_code": True
            }
            
            # Load model and tokenizer
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                **model_config
            )
            
            tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                trust_remote_code=True
            )
            
            # Register the model
            self.models[model_name] = model
            self.tokenizers[model_name] = tokenizer
            
            return model, tokenizer
            
        except Exception as e:
            self.logger.error(f"Failed to load Dolphin model: {str(e)}")
            return None, None 

    def _load_to_gpu(self, specs: ModelSpecs) -> Tuple[Optional[torch.nn.Module], Optional[AutoTokenizer]]:
        """Load a model to GPU"""
        try:
            if not self.check_device_requirements(specs):
                raise ValueError(f"Insufficient VRAM for {specs.name}")
            
            # Ensure model files are in cache
            self.cache_manager.get_model_files(specs.name)
            
            # Determine model path
            model_path = self._get_model_path(specs.name)
            self.logger.info(f"Loading model from: {model_path}")
            
            # Download or load model
            model = AutoModel.from_pretrained(
                specs.name,
                cache_dir=model_path,
                trust_remote_code=specs.trust_remote_code,
                torch_dtype=specs.device_requirements['dtype']
            )
            
            model = model.eval().to(self.device)
            tokenizer = AutoTokenizer.from_pretrained(specs.name, trust_remote_code=True)
            
            # Special handling for TTS models
            if specs.type == 'tts' and hasattr(model, 'init_tts'):
                model.init_tts()
                model.tts.float()
            
            self.models[specs.name] = model
            self.tokenizers[specs.name] = tokenizer
            
            self.logger.info(f"Successfully loaded {specs.name}")
            return model, tokenizer
            
        except Exception as e:
            self.logger.error(f"Error loading {specs.name}: {str(e)}")
            return None, None 

    def __del__(self):
        if hasattr(self, 'gpu_monitor'):
            self.gpu_monitor.stop()