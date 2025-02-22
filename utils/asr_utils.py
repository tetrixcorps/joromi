from enum import Enum
from typing import Dict, Any

class ASRError(Exception):
    def __init__(self, message: str, error_type: str):
        self.message = message
        self.error_type = error_type
        super().__init__(self.message)

class ASRErrorHandler:
    @staticmethod
    def handle_error(error: Exception) -> Dict[str, Any]:
        if isinstance(error, ASRError):
            return {
                "status": "error",
                "error_type": error.error_type,
                "message": str(error)
            }
        return {
            "status": "error",
            "error_type": "general",
            "message": f"ASR processing failed: {str(error)}"
        } 