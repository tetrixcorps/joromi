import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple, List
import logging
from dataclasses import dataclass
from transformers import PreTrainedModel
import numpy as np
from app.monitoring.metrics import MODEL_OPTIMIZATION_METRICS

logger = logging.getLogger(__name__)

@dataclass
class OptimizationConfig:
    pruning_method: str = "magnitude"  # magnitude, structured, or gradual
    target_sparsity: float = 0.3  # 30% sparsity
    pruning_schedule: str = "cubic"
    distillation_temperature: float = 2.0
    quantization_bits: int = 8
    eval_metric: str = "bleu"

class ModelOptimizer:
    def __init__(self, config: OptimizationConfig):
        self.config = config
        self.teacher_model = None
        self.student_model = None
        
    async def optimize_model(
        self,
        model: PreTrainedModel,
        eval_dataloader: torch.utils.data.DataLoader
    ) -> Tuple[PreTrainedModel, Dict[str, float]]:
        """Apply optimization techniques to reduce model size"""
        try:
            # Initial model statistics
            original_size = self._get_model_size(model)
            
            # 1. Pruning
            pruned_model = await self._apply_pruning(model)
            
            # 2. Knowledge Distillation
            distilled_model = await self._apply_distillation(pruned_model)
            
            # 3. Quantization
            optimized_model = await self._apply_quantization(distilled_model)
            
            # Evaluate and collect metrics
            metrics = await self._evaluate_optimization(
                original_model=model,
                optimized_model=optimized_model,
                eval_dataloader=eval_dataloader
            )
            
            return optimized_model, metrics

        except Exception as e:
            logger.error(f"Model optimization failed: {e}")
            raise

    async def _apply_pruning(self, model: PreTrainedModel) -> PreTrainedModel:
        """Apply model pruning based on configuration"""
        try:
            if self.config.pruning_method == "magnitude":
                return await self._magnitude_pruning(model)
            elif self.config.pruning_method == "structured":
                return await self._structured_pruning(model)
            elif self.config.pruning_method == "gradual":
                return await self._gradual_pruning(model)
            else:
                raise ValueError(f"Unknown pruning method: {self.config.pruning_method}")

        except Exception as e:
            logger.error(f"Pruning failed: {e}")
            raise

    async def _magnitude_pruning(self, model: PreTrainedModel) -> PreTrainedModel:
        """Magnitude-based weight pruning"""
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                # Calculate weight importance
                importance = torch.abs(module.weight.data)
                threshold = torch.quantile(
                    importance,
                    self.config.target_sparsity
                )
                
                # Create pruning mask
                mask = importance > threshold
                
                # Apply mask
                module.weight.data *= mask
                
                # Record metrics
                sparsity = 1.0 - (mask.sum() / mask.numel())
                MODEL_OPTIMIZATION_METRICS.labels(
                    operation="pruning",
                    layer=name
                ).observe(sparsity)
                
        return model

    async def _apply_distillation(
        self,
        student_model: PreTrainedModel
    ) -> PreTrainedModel:
        """Apply knowledge distillation"""
        if self.teacher_model is None:
            logger.warning("Teacher model not set, skipping distillation")
            return student_model

        try:
            class DistillationLoss(nn.Module):
                def __init__(self, temperature):
                    super().__init__()
                    self.temperature = temperature
                    self.kl_div = nn.KLDivLoss(reduction="batchmean")

                def forward(self, student_logits, teacher_logits):
                    soft_targets = nn.functional.softmax(
                        teacher_logits / self.temperature,
                        dim=-1
                    )
                    student_log_probs = nn.functional.log_softmax(
                        student_logits / self.temperature,
                        dim=-1
                    )
                    return self.kl_div(student_log_probs, soft_targets)

            distillation_criterion = DistillationLoss(
                self.config.distillation_temperature
            )
            
            # Training loop would go here
            # This is a simplified version
            
            return student_model

        except Exception as e:
            logger.error(f"Distillation failed: {e}")
            raise

    async def _apply_quantization(self, model: PreTrainedModel) -> PreTrainedModel:
        """Apply quantization to reduce model size"""
        try:
            # Dynamic quantization
            quantized_model = torch.quantization.quantize_dynamic(
                model,
                {nn.Linear},  # Quantize linear layers
                dtype=torch.qint8
            )
            
            # Record metrics
            original_size = self._get_model_size(model)
            quantized_size = self._get_model_size(quantized_model)
            compression_ratio = original_size / quantized_size
            
            MODEL_OPTIMIZATION_METRICS.labels(
                operation="quantization",
                metric="compression_ratio"
            ).observe(compression_ratio)
            
            return quantized_model

        except Exception as e:
            logger.error(f"Quantization failed: {e}")
            raise

    async def _evaluate_optimization(
        self,
        original_model: PreTrainedModel,
        optimized_model: PreTrainedModel,
        eval_dataloader: torch.utils.data.DataLoader
    ) -> Dict[str, float]:
        """Evaluate optimization results"""
        metrics = {
            "size_reduction": self._get_size_reduction(
                original_model,
                optimized_model
            ),
            "inference_speedup": await self._measure_inference_speedup(
                original_model,
                optimized_model,
                eval_dataloader
            ),
            "accuracy_change": await self._measure_accuracy_change(
                original_model,
                optimized_model,
                eval_dataloader
            )
        }
        
        # Record metrics
        for metric_name, value in metrics.items():
            MODEL_OPTIMIZATION_METRICS.labels(
                operation="evaluation",
                metric=metric_name
            ).observe(value)
            
        return metrics

    def _get_model_size(self, model: PreTrainedModel) -> int:
        """Calculate model size in bytes"""
        return sum(
            param.nelement() * param.element_size()
            for param in model.parameters()
        )

    async def _measure_inference_speedup(
        self,
        original_model: PreTrainedModel,
        optimized_model: PreTrainedModel,
        eval_dataloader: torch.utils.data.DataLoader
    ) -> float:
        """Measure inference speed improvement"""
        try:
            # Measure original model speed
            start_time = time.time()
            with torch.no_grad():
                for batch in eval_dataloader:
                    _ = original_model(**batch)
            original_time = time.time() - start_time

            # Measure optimized model speed
            start_time = time.time()
            with torch.no_grad():
                for batch in eval_dataloader:
                    _ = optimized_model(**batch)
            optimized_time = time.time() - start_time

            return original_time / optimized_time

        except Exception as e:
            logger.error(f"Speed measurement failed: {e}")
            return 1.0 