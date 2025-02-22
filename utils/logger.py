import logging
import sys
from pathlib import Path
from logging.handlers import RotatingFileHandler
import os
from datetime import datetime
from functools import wraps
import time

def setup_logger(service_name: str):
    """Setup logger with file and console handlers"""
    logger = logging.getLogger(service_name)
    logger.setLevel(logging.INFO)

    # Create logs directory if it doesn't exist
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)

    # Create formatters
    file_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    console_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # File handler with rotation
    log_file = log_dir / f"{service_name}_{datetime.now().strftime('%Y%m%d')}.log"
    file_handler = RotatingFileHandler(
        log_file,
        maxBytes=10*1024*1024,  # 10MB
        backupCount=5
    )
    file_handler.setFormatter(file_formatter)
    file_handler.setLevel(logging.INFO)

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(console_formatter)
    console_handler.setLevel(logging.INFO)

    # Add handlers to logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger

class CustomLogger:
    @staticmethod
    def log_execution_time(logger):
        """Decorator to log function execution time"""
        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                start_time = time.time()
                try:
                    result = func(*args, **kwargs)
                    execution_time = time.time() - start_time
                    logger.info(f"{func.__name__} executed in {execution_time:.2f} seconds")
                    return result
                except Exception as e:
                    execution_time = time.time() - start_time
                    logger.error(f"{func.__name__} failed after {execution_time:.2f} seconds: {str(e)}")
                    raise
            return wrapper
        return decorator

# Create different loggers for different components
model_logger = setup_logger('model')
api_logger = setup_logger('api')
performance_logger = setup_logger('performance')

# Example usage in model_loader.py:
"""
from utils.logger import model_logger, CustomLogger

class ModelManager:
    @CustomLogger.log_execution_time(model_logger)
    def load_model(self, specs: ModelSpecs):
        model_logger.info(f"Starting to load model: {specs.name}")
        try:
            # ... model loading code ...
            model_logger.info(f"Successfully loaded {specs.name}")
        except Exception as e:
            model_logger.error(f"Failed to load {specs.name}: {str(e)}")
            raise
""" 