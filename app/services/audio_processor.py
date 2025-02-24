import librosa
import numpy as np
import torch
from typing import Optional
import logging

logger = logging.getLogger(__name__)

class AudioCompressor:
    def __init__(self, 
                 threshold_db: float = -20,
                 ratio: float = 4.0,
                 attack_ms: float = 5.0,
                 release_ms: float = 50.0):
        self.threshold = threshold_db
        self.ratio = ratio
        self.attack_time = attack_ms / 1000.0
        self.release_time = release_ms / 1000.0
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # State variables
        self.prev_gain = None
        self.prev_level = None

    async def process_chunk(self, chunk: bytes, sample_rate: int = 16000) -> bytes:
        """Apply adaptive dynamic range compression"""
        try:
            # Convert bytes to numpy array
            audio_data = np.frombuffer(chunk, dtype=np.float32)
            
            # Calculate RMS level in dB
            rms = librosa.feature.rms(y=audio_data)[0]
            level_db = 20 * np.log10(rms + 1e-9)

            # Calculate gain reduction
            gain_db = np.minimum(0, (level_db - self.threshold) * (1 - 1/self.ratio))
            
            # Apply smoothing
            if self.prev_gain is not None:
                coeff = np.exp(-1/(sample_rate * (
                    self.attack_time if gain_db < self.prev_gain else self.release_time
                )))
                gain_db = coeff * self.prev_gain + (1 - coeff) * gain_db

            # Convert to linear gain
            gain = np.power(10, gain_db/20)
            
            # Apply compression
            compressed = audio_data * gain

            # Update state
            self.prev_gain = gain_db
            self.prev_level = level_db

            # GPU acceleration if available
            if torch.cuda.is_available():
                compressed = await self._gpu_accelerated_process(compressed)

            return compressed.tobytes()

        except Exception as e:
            logger.error(f"Audio compression failed: {e}")
            return chunk  # Return original chunk if processing fails

    async def _gpu_accelerated_process(self, audio_data: np.ndarray) -> np.ndarray:
        """Apply additional GPU-accelerated processing"""
        try:
            # Convert to tensor and move to GPU
            audio_tensor = torch.from_numpy(audio_data).to(self.device)
            
            # Apply additional processing (e.g., noise reduction)
            with torch.no_grad():
                # High-pass filter
                audio_tensor = self._apply_highpass_filter(audio_tensor)
                
                # Noise gate
                audio_tensor = self._apply_noise_gate(audio_tensor)

            return audio_tensor.cpu().numpy()

        except Exception as e:
            logger.error(f"GPU processing failed: {e}")
            return audio_data

    def _apply_highpass_filter(self, audio_tensor: torch.Tensor) -> torch.Tensor:
        """Apply high-pass filter to remove low frequency noise"""
        cutoff_freq = 80  # Hz
        nyquist = 16000 / 2
        normalized_cutoff = cutoff_freq / nyquist
        
        # Simple IIR filter
        alpha = 0.95 * normalized_cutoff
        filtered = torch.zeros_like(audio_tensor)
        filtered[0] = audio_tensor[0]
        
        for i in range(1, len(audio_tensor)):
            filtered[i] = alpha * (filtered[i-1] + audio_tensor[i] - audio_tensor[i-1])
            
        return filtered

    def _apply_noise_gate(self, audio_tensor: torch.Tensor) -> torch.Tensor:
        """Apply noise gate to reduce background noise"""
        threshold = 0.01
        mask = torch.abs(audio_tensor) > threshold
        return audio_tensor * mask.float() 