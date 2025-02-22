import librosa
import numpy as np
from typing import Dict, Any, Tuple, Optional, List
import torch
from dataclasses import dataclass
import math

@dataclass
class AudioConfig:
    sample_rate: int = 16000
    normalize: bool = True
    remove_silence: bool = True
    noise_reduction: bool = True
    target_db: float = -20
    min_silence_duration: float = 0.3
    silence_threshold: float = 0.05
    chunk_size: int = 1024

class AudioSegmenter:
    def __init__(self, config: AudioConfig):
        self.config = config
        self.max_segment_length = 30  # Maximum segment length in seconds
        self.overlap_duration = 0.5    # Overlap between segments in seconds
        
    def segment_audio(
        self, 
        audio: np.ndarray,
        sample_rate: int
    ) -> Tuple[List[np.ndarray], Dict[str, Any]]:
        """Segment long audio into overlapping chunks"""
        # Calculate sizes in samples
        max_samples = int(self.max_segment_length * sample_rate)
        overlap_samples = int(self.overlap_duration * sample_rate)
        
        # Calculate number of segments
        audio_length = len(audio)
        n_segments = math.ceil(audio_length / (max_samples - overlap_samples))
        
        segments = []
        segment_info = []
        
        for i in range(n_segments):
            start = i * (max_samples - overlap_samples)
            end = min(start + max_samples, audio_length)
            
            segment = audio[start:end]
            
            # Pad last segment if needed
            if len(segment) < max_samples:
                segment = np.pad(segment, (0, max_samples - len(segment)))
            
            segments.append(segment)
            segment_info.append({
                'start_time': start / sample_rate,
                'end_time': end / sample_rate,
                'duration': len(segment) / sample_rate
            })
        
        return segments, {
            'n_segments': n_segments,
            'segment_info': segment_info,
            'total_duration': audio_length / sample_rate
        }

    def merge_transcriptions(
        self, 
        transcriptions: List[str],
        segment_info: List[Dict[str, Any]]
    ) -> str:
        """Merge transcriptions from segments with overlap handling"""
        if not transcriptions:
            return ""
            
        if len(transcriptions) == 1:
            return transcriptions[0]
        
        merged = []
        overlap_tokens = 5  # Number of tokens to check for overlap
        
        for i in range(len(transcriptions)):
            current_trans = transcriptions[i].split()
            
            if i == 0:
                merged.extend(current_trans[:-overlap_tokens])
            elif i == len(transcriptions) - 1:
                merged.extend(current_trans)
            else:
                # Find best overlap point
                next_trans = transcriptions[i + 1].split()
                overlap_found = False
                
                for j in range(len(current_trans) - overlap_tokens):
                    overlap_segment = ' '.join(current_trans[j:j + overlap_tokens])
                    if overlap_segment in ' '.join(next_trans):
                        merged.extend(current_trans[:-overlap_tokens])
                        overlap_found = True
                        break
                
                if not overlap_found:
                    merged.extend(current_trans[:-overlap_tokens])
        
        return ' '.join(merged)

class AudioPreprocessor:
    def __init__(self, config: AudioConfig = AudioConfig()):
        self.config = config
        self.segmenter = AudioSegmenter(config)
        
    def process_audio(
        self, 
        audio_input: np.ndarray,
        original_sr: Optional[int] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Process audio with multiple enhancement options"""
        try:
            # Track processing steps
            processing_info = {}
            
            # Resample if needed
            if original_sr and original_sr != self.config.sample_rate:
                audio_input = librosa.resample(
                    audio_input,
                    orig_sr=original_sr,
                    target_sr=self.config.sample_rate
                )
                processing_info['resampled'] = True
            
            # Remove silence
            if self.config.remove_silence:
                audio_input, silence_info = self._remove_silence(audio_input)
                processing_info['silence_removed'] = silence_info
            
            # Apply noise reduction
            if self.config.noise_reduction:
                audio_input = self._reduce_noise(audio_input)
                processing_info['noise_reduced'] = True
            
            # Normalize audio
            if self.config.normalize:
                audio_input = self._normalize_audio(audio_input)
                processing_info['normalized'] = True
            
            return audio_input, processing_info
            
        except Exception as e:
            raise RuntimeError(f"Audio preprocessing failed: {str(e)}")
    
    def _remove_silence(
        self, 
        audio: np.ndarray
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Remove silence from audio"""
        # Calculate energy of signal
        energy = librosa.feature.rms(y=audio, frame_length=self.config.chunk_size)[0]
        
        # Find non-silent parts
        threshold = self.config.silence_threshold * np.max(energy)
        non_silent = energy > threshold
        
        # Convert frame-wise decisions to sample-wise
        non_silent = np.repeat(non_silent, self.config.chunk_size)[:len(audio)]
        
        # Only keep non-silent parts
        audio_cleaned = audio[non_silent]
        
        return audio_cleaned, {
            'original_duration': len(audio) / self.config.sample_rate,
            'processed_duration': len(audio_cleaned) / self.config.sample_rate,
            'reduction_percent': (1 - len(audio_cleaned) / len(audio)) * 100
        }
    
    def _reduce_noise(self, audio: np.ndarray) -> np.ndarray:
        """Apply noise reduction"""
        # Estimate noise from the first 1000ms
        noise_sample = audio[:int(self.config.sample_rate)]
        noise_profile = np.mean(np.abs(librosa.stft(noise_sample)), axis=1)
        
        # Apply spectral subtraction
        S = librosa.stft(audio)
        S_mag = np.abs(S)
        S_phase = np.angle(S)
        
        # Subtract noise profile
        S_cleaned = np.maximum(S_mag - noise_profile.reshape(-1, 1), 0)
        
        # Reconstruct signal
        audio_cleaned = librosa.istft(S_cleaned * np.exp(1j * S_phase))
        
        return audio_cleaned
    
    def _normalize_audio(self, audio: np.ndarray) -> np.ndarray:
        """Normalize audio to target dB"""
        # Calculate current dB
        current_db = 20 * np.log10(np.maximum(np.max(np.abs(audio)), 1e-10))
        
        # Calculate required gain
        gain = 10 ** ((self.config.target_db - current_db) / 20)
        
        return audio * gain

    def process_long_audio(
        self,
        audio_input: np.ndarray,
        original_sr: Optional[int] = None
    ) -> Tuple[List[np.ndarray], Dict[str, Any]]:
        """Process and segment long audio files"""
        try:
            # Initial processing
            processed_audio, processing_info = self.process_audio(audio_input, original_sr)
            
            # Segment if longer than max_segment_length
            duration = len(processed_audio) / self.config.sample_rate
            if duration > self.segmenter.max_segment_length:
                segments, segment_info = self.segmenter.segment_audio(
                    processed_audio,
                    self.config.sample_rate
                )
                
                # Process each segment
                processed_segments = []
                segment_processing_info = []
                
                for i, segment in enumerate(segments):
                    processed_segment, segment_proc_info = self.process_audio(
                        segment,
                        self.config.sample_rate
                    )
                    processed_segments.append(processed_segment)
                    segment_processing_info.append(segment_proc_info)
                
                return processed_segments, {
                    **processing_info,
                    'segmentation': segment_info,
                    'segment_processing': segment_processing_info
                }
            
            return [processed_audio], {
                **processing_info,
                'segmentation': None
            }
            
        except Exception as e:
            raise RuntimeError(f"Long audio processing failed: {str(e)}")

class AudioBatchProcessor:
    def __init__(self, config: AudioConfig = AudioConfig()):
        self.processor = AudioPreprocessor(config)
    
    def process_batch(
        self, 
        audio_files: list,
        original_sr: Optional[int] = None
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Process a batch of audio files"""
        processed_audio = []
        batch_info = []
        
        for audio in audio_files:
            audio_processed, info = self.processor.process_audio(audio, original_sr)
            processed_audio.append(audio_processed)
            batch_info.append(info)
        
        # Pad to same length
        max_len = max(len(audio) for audio in processed_audio)
        padded_audio = [
            np.pad(audio, (0, max_len - len(audio))) 
            for audio in processed_audio
        ]
        
        # Convert to tensor
        audio_tensor = torch.tensor(padded_audio)
        
        return audio_tensor, {
            'batch_size': len(audio_files),
            'max_length': max_len,
            'processing_info': batch_info
        } 