from transformers import WhisperProcessor, WhisperForConditionalGeneration
import torch

class ASRService:
    def __init__(self):
        self.processor = WhisperProcessor.from_pretrained("openai/whisper-large-v2")
        self.model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-large-v2")
        
    async def transcribe(self, audio_data: bytes) -> str:
        # Process audio and return transcription
        inputs = self.processor(audio_data, return_tensors="pt")
        generated_ids = self.model.generate(**inputs)
        transcription = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return transcription 