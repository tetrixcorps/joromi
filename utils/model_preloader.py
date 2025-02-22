from typing import List, Dict
import asyncio
from models.model_loader import ModelManager
from config.model_config import ModelConfigurations
from utils.logger import setup_logger
from utils.gpu_monitor import GPUMonitor

logger = setup_logger('model_preloader')

class ModelPreloader:
    def __init__(self, model_manager: ModelManager):
        self.model_manager = model_manager
        self.gpu_monitor = GPUMonitor()
        self.loaded_models: Dict[str, bool] = {}
        
    async def preload_models(self):
        """Preload models in order of priority"""
        priority_models = [
            ModelConfigurations.TRANSLATION,  # Load translation model first
            ModelConfigurations.BANKING_LLM,  # Then banking model
            ModelConfigurations.ASR,         # Speech recognition
            ModelConfigurations.TTS,         # Text-to-speech
            ModelConfigurations.DOLPHIN      # General purpose LLM
        ]
        
        for model_spec in priority_models:
            # Check GPU memory before loading
            memory_usage = self.gpu_monitor.get_memory_usage()
            if any(usage > 90 for usage in memory_usage.values()):
                logger.warning("High GPU memory usage, pausing preload")
                await asyncio.sleep(30)  # Wait for memory to free up
                continue
                
            try:
                logger.info(f"Preloading model: {model_spec.name}")
                model, processor = self.model_manager.load_model(model_spec)
                
                if model is not None:
                    self.loaded_models[model_spec.name] = True
                    logger.info(f"Successfully preloaded {model_spec.name}")
                else:
                    logger.error(f"Failed to preload {model_spec.name}")
                    
            except Exception as e:
                logger.error(f"Error preloading {model_spec.name}: {str(e)}")
                
            # Brief pause between loads to prevent GPU overload
            await asyncio.sleep(5)
            
    def get_loading_status(self) -> Dict[str, bool]:
        """Get status of model loading"""
        return self.loaded_models 