import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
import streamlit as st
from PIL import Image
import logging
import gc
from pathlib import Path
import yaml

logger = logging.getLogger(__name__)

class MemoryManager:
    @staticmethod
    def clear_gpu_memory():
        """Clear GPU memory"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()
    
    @staticmethod
    def log_memory_usage():
        """Log current memory usage"""
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated()/1024**2
            cached = torch.cuda.memory_reserved()/1024**2
            logger.info(f"GPU Memory - Used: {allocated:.2f}MB, Cached: {cached:.2f}MB")

class ModelHandler:
    @st.cache_resource
    def load_models():
        """Load all required models"""
        try:
            # Load configuration
            config_path = Path("config/models.yml")
            with open(config_path) as f:
                config = yaml.safe_load(f)
            
            models = {}
            for model_name, model_info in config['models'].items():
                logger.info(f"Loading {model_name} model...")
                
                # Load model and tokenizer from HuggingFace
                model = AutoModelForCausalLM.from_pretrained(
                    model_info['model_id'],
                    torch_dtype=torch.float16 if model_info['dtype'] == 'float16' else torch.float32,
                    device_map="auto",
                    trust_remote_code=True,
                    low_cpu_mem_usage=True
                )
                
                tokenizer = AutoTokenizer.from_pretrained(
                    model_info['model_id']
                )
                
                models[model_name] = {
                    'model': model,
                    'tokenizer': tokenizer
                }
                
            return models
            
        except Exception as e:
            logger.error(f"Error loading models: {e}")
            raise

class DolphinHandler:
    def __init__(self):
        self.models = ModelHandler.load_models()
        self.model = self.models["dolphin"]["model"]
        self.tokenizer = self.models["dolphin"]["tokenizer"]
        
        # Set default parameters
        self.max_length = 2048
        self.temperature = 0.7
        self.top_p = 0.9
        self.repetition_penalty = 1.1
    
    async def generate_response(self, prompt: str, system_prompt: str = None) -> str:
        """Generate response using Dolphin model"""
        try:
            MemoryManager.clear_gpu_memory()
            
            # Prepare conversation format
            if system_prompt:
                full_prompt = f"<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant"
            else:
                full_prompt = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant"
            
            # Tokenize input
            inputs = self.tokenizer(
                full_prompt,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.max_length
            )
            
            # Move to GPU if available
            if torch.cuda.is_available():
                inputs = {k: v.to("cuda") for k, v in inputs.items()}
            
            # Generate response
            with torch.inference_mode():
                outputs = self.model.generate(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    max_length=self.max_length,
                    temperature=self.temperature,
                    top_p=self.top_p,
                    repetition_penalty=self.repetition_penalty,
                    do_sample=True,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id
                )
            
            # Decode response
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Clean up response
            response = response.split("<|im_start|>assistant")[-1].strip()
            response = response.split("<|im_end|>")[0].strip()
            
            MemoryManager.log_memory_usage()
            return response
            
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            raise
        finally:
            MemoryManager.clear_gpu_memory() 