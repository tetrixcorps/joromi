from services.base_service import BaseModelService
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)

class DolphinService(BaseModelService):
    def __init__(self, port: int):
        super().__init__(port)
        self.max_length = 2048
        self.temperature = 0.7
        self.top_p = 0.9
        self.repetition_penalty = 1.1

    async def load_model(self) -> AutoModelForCausalLM:
        """Load Dolphin model"""
        return AutoModelForCausalLM.from_pretrained(
            "cognitivecomputations/dolphin-2.6-mixtral-8x7b",
            torch_dtype=torch.float16,
            device_map="auto"
        )

    async def load_tokenizer(self) -> AutoTokenizer:
        """Load Dolphin tokenizer"""
        return AutoTokenizer.from_pretrained(
            "cognitivecomputations/dolphin-2.6-mixtral-8x7b"
        )

    async def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process input with Dolphin model"""
        try:
            prompt = input_data.get("prompt", "")
            system_prompt = input_data.get("system_prompt")

            # Format prompt
            if system_prompt:
                full_prompt = f"<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant"
            else:
                full_prompt = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant"

            # Tokenize
            inputs = self.tokenizer(
                full_prompt,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.max_length
            ).to(self.device)

            # Generate
            with torch.inference_mode():
                outputs = self.model.generate(
                    **inputs,
                    max_length=self.max_length,
                    temperature=self.temperature,
                    top_p=self.top_p,
                    repetition_penalty=self.repetition_penalty,
                    do_sample=True
                )

            # Decode and clean response
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            response = response.split("<|im_start|>assistant")[-1].strip()
            response = response.split("<|im_end|>")[0].strip()

            return {
                "response": response,
                "status": "success"
            }

        except Exception as e:
            logger.error(f"Error in Dolphin service: {e}")
            return {
                "error": str(e),
                "status": "error"
            } 