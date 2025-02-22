from pathlib import Path

# Base paths
ROOT_DIR = Path(__file__).parent.parent
MODELS_DIR = ROOT_DIR / "models"
DOWNLOADED_MODELS = MODELS_DIR / "downloaded"
MODEL_CACHE = MODELS_DIR / "cache"

# Create directories if they don't exist
DOWNLOADED_MODELS.mkdir(parents=True, exist_ok=True)
MODEL_CACHE.mkdir(parents=True, exist_ok=True)

# Model-specific paths
class ModelPaths:
    PIX2STRUCT = DOWNLOADED_MODELS / "pix2struct"
    MINICPM = DOWNLOADED_MODELS / "minicpm"
    AFRO_TTS = DOWNLOADED_MODELS / "afro_tts"
    WHISPER = DOWNLOADED_MODELS / "whisper"
    
    @classmethod
    def create_dirs(cls):
        """Create all model directories"""
        for path in [cls.PIX2STRUCT, cls.MINICPM, cls.AFRO_TTS, cls.WHISPER]:
            path.mkdir(exist_ok=True) 