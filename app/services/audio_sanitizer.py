import numpy as np
from typing import Optional, Tuple, Dict, List
import librosa
import hashlib
import logging
from dataclasses import dataclass
from app.monitoring.metrics import AUDIO_SECURITY_METRICS
from app.exceptions import InvalidAudioError, MaliciousAudioError

logger = logging.getLogger(__name__)

@dataclass
class AudioSecurityConfig:
    max_file_size: int = 10_000_000  # 10MB
    max_duration: float = 300.0  # 5 minutes
    allowed_sample_rates = {8000, 16000, 22050, 44100, 48000}
    allowed_formats = {'wav', 'mp3', 'ogg', 'flac'}
    fingerprint_threshold: float = 0.85

class AudioSanitizer:
    def __init__(self):
        self.config = AudioSecurityConfig()
        self.known_fingerprints = set()  # Could be replaced with Redis/DB
        self.malicious_patterns = self._load_malicious_patterns()
        
    async def validate(self, audio: bytes) -> Tuple[bytes, Dict[str, any]]:
        """Comprehensive audio validation and sanitization"""
        try:
            # Basic checks
            await self._check_size(audio)
            
            # Convert to numpy array for analysis
            audio_data, sample_rate = await self._load_audio(audio)
            
            # Security checks
            await self._validate_format(audio)
            await self._check_duration(audio_data, sample_rate)
            await self._check_audio_fingerprint(audio_data)
            await self._scan_for_malicious_patterns(audio_data)
            
            # Audio quality checks
            sanitized_audio = await self._sanitize_audio(audio_data, sample_rate)
            
            # Generate security metadata
            metadata = await self._generate_security_metadata(audio_data, sample_rate)
            
            return sanitized_audio, metadata

        except Exception as e:
            logger.error(f"Audio validation failed: {e}")
            AUDIO_SECURITY_METRICS.labels(
                check_type="validation",
                result="failed"
            ).inc()
            raise

    async def _check_size(self, audio: bytes):
        """Check file size limits"""
        if len(audio) > self.config.max_file_size:
            raise InvalidAudioError(
                f"File size {len(audio)} exceeds limit {self.config.max_file_size}"
            )

    async def _load_audio(self, audio: bytes) -> Tuple[np.ndarray, int]:
        """Safely load audio data"""
        try:
            # Load audio using librosa with safety checks
            audio_data, sample_rate = librosa.load(
                audio,
                sr=None,
                duration=self.config.max_duration
            )
            
            if sample_rate not in self.config.allowed_sample_rates:
                raise InvalidAudioError(f"Invalid sample rate: {sample_rate}")
                
            return audio_data, sample_rate
            
        except Exception as e:
            raise InvalidAudioError(f"Failed to load audio: {e}")

    async def _check_audio_fingerprint(self, audio_data: np.ndarray):
        """Generate and check audio fingerprint"""
        try:
            # Generate fingerprint using chromagram
            chroma = librosa.feature.chroma_cqt(y=audio_data)
            fingerprint = hashlib.sha256(chroma.tobytes()).hexdigest()
            
            # Check against known fingerprints
            if fingerprint in self.known_fingerprints:
                similarity = await self._compute_fingerprint_similarity(
                    chroma,
                    self.known_fingerprints[fingerprint]
                )
                if similarity > self.config.fingerprint_threshold:
                    raise MaliciousAudioError("Suspicious audio fingerprint detected")
                    
            # Store new fingerprint
            self.known_fingerprints.add(fingerprint)
            
        except Exception as e:
            logger.error(f"Fingerprint check failed: {e}")
            raise

    async def _scan_for_malicious_patterns(self, audio_data: np.ndarray):
        """Scan for known malicious patterns"""
        try:
            # Convert to frequency domain
            stft = librosa.stft(audio_data)
            mag = np.abs(stft)
            
            # Check for suspicious patterns
            for pattern in self.malicious_patterns:
                if await self._pattern_match(mag, pattern):
                    raise MaliciousAudioError(
                        f"Detected malicious pattern: {pattern['name']}"
                    )
                    
        except Exception as e:
            logger.error(f"Pattern scan failed: {e}")
            raise

    async def _sanitize_audio(
        self,
        audio_data: np.ndarray,
        sample_rate: int
    ) -> bytes:
        """Clean and normalize audio"""
        try:
            # Remove DC offset
            audio_data = librosa.util.normalize(audio_data)
            
            # Apply high-pass filter to remove sub-audible content
            audio_data = librosa.effects.hpss(audio_data)[0]
            
            # Clip extreme values
            audio_data = np.clip(audio_data, -1.0, 1.0)
            
            # Convert back to bytes
            return librosa.util.buf_to_float(
                audio_data.astype(np.float32).tobytes(),
                dtype=np.float32
            )
            
        except Exception as e:
            logger.error(f"Audio sanitization failed: {e}")
            raise

    async def _generate_security_metadata(
        self,
        audio_data: np.ndarray,
        sample_rate: int
    ) -> Dict[str, any]:
        """Generate security-related metadata"""
        return {
            "duration": len(audio_data) / sample_rate,
            "sample_rate": sample_rate,
            "fingerprint": hashlib.sha256(audio_data.tobytes()).hexdigest(),
            "checks_passed": [
                "size_validation",
                "format_validation",
                "fingerprint_check",
                "pattern_scan",
                "sanitization"
            ],
            "security_level": "high"
        }

    def _load_malicious_patterns(self) -> List[Dict]:
        """Load known malicious audio patterns"""
        return [
            {
                "name": "ultrasonic_payload",
                "frequency_range": (20000, 48000),
                "threshold": 0.8
            },
            {
                "name": "hidden_message",
                "frequency_range": (18000, 20000),
                "threshold": 0.7
            },
            {
                "name": "steganography",
                "pattern_type": "spectral_dips",
                "threshold": 0.9
            }
        ]

    async def _pattern_match(
        self,
        spectrogram: np.ndarray,
        pattern: Dict
    ) -> bool:
        """Match spectrogram against malicious pattern"""
        if pattern["name"] == "ultrasonic_payload":
            return await self._check_ultrasonic_content(
                spectrogram,
                pattern["frequency_range"],
                pattern["threshold"]
            )
        elif pattern["name"] == "hidden_message":
            return await self._check_hidden_message(
                spectrogram,
                pattern["frequency_range"],
                pattern["threshold"]
            )
        elif pattern["name"] == "steganography":
            return await self._check_steganography(
                spectrogram,
                pattern["threshold"]
            )
        return False 