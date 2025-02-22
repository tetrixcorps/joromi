from frontend.model_handler import ModelHandler
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    """Initialize and verify models"""
    try:
        logger.info("Loading models...")
        models = ModelHandler.load_models()
        
        for model_name, model_dict in models.items():
            logger.info(f"Successfully loaded {model_name}")
            logger.info(f"Model device: {next(model_dict['model'].parameters()).device}")
            
    except Exception as e:
        logger.error(f"Error initializing models: {e}")
        raise

if __name__ == "__main__":
    main() 