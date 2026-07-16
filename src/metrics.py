"""
metrics.py
================================================================================
Unified profiling and evaluation-metric utilities.

Contains:
    - LatencyProfiler: wall-clock ms/image measurement across batch sizes,
      with CUDA-safe synchronization and warmup.
    - HardwareProfiler: peak VRAM tracking via torch.cuda.max_memory_allocated
      and trainable-vs-total parameter accounting.
    - MetricsWrapper: safe wrappers around scikit-learn F1 (micro/macro) and
      AUROC, tolerant to degenerate single-class batches.
================================================================================
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

logger = logging.getLogger("defect_bench.metrics")


  
# Latency profiling
  
@dataclass
class LatencyResult:
    batch_size: int
    ms_per_batch: float
    ms_per_image: float
    throughput_img_per_sec: float


class LatencyProfiler:
    """Measures wall-clock inference latency across a list of batch sizes."""

    def __init__(self, device: torch.device, warmup_iters: int = 10, measured_iters: int = 50) -> None:
        self.device = device
        self.warmup_iters = warmup_iters
        self.measured_iters = measured_iters

    @torch.no_grad()
    def profile(
        self,
        model: nn.Module,
        input_shape: Tuple[int, int, int],
        batch_sizes: List[int],
    ) -> Dict[int, LatencyResult]:
        model.eval().to(self.device)
        results: Dict[int, LatencyResult] = {}

        for bs in batch_sizes:
            dummy = torch.randn(bs, *input_shape, device=self.device)

            for _ in range(self.warmup_iters):
                _ = model(dummy)
            if self.device.type == "cuda":
                torch.cuda.synchronize()

            timings: List[float] = []
            for _ in range(self.measured_iters):
                start = time.perf_counter()
                _ = model(dummy)
                if self.device.type == "cuda":
                    torch.cuda.synchronize()
                timings.append((time.perf_counter() - start) * 1000.0)

            ms_per_batch = float(np.mean(timings))
            ms_per_image = ms_per_batch / bs
            throughput = 1000.0 / ms_per_image

            results[bs] = LatencyResult(
                batch_size=bs,
                ms_per_batch=ms_per_batch,
                ms_per_image=ms_per_image,
                throughput_img_per_sec=throughput,
            )
            logger.info(
                "Latency[bs=%d]: %.3f ms/batch | %.3f ms/img | %.1f img/s",
                bs,
                ms_per_batch,
                ms_per_image,
                throughput,
            )
        return results


  
# Hardware profiling (VRAM + parameter accounting)
  
@dataclass
class HardwareProfile:
    peak_vram_mb: Optional[float]
    total_params: int
    trainable_params: int
    trainable_ratio: float
    model_size_mb: float


class HardwareProfiler:
    """Tracks peak VRAM allocation and parameter counts for a given model/device."""

    def __init__(self, device: torch.device) -> None:
        self.device = device

    @torch.no_grad()
    def profile(
        self, model: nn.Module, input_shape: Tuple[int, int, int], batch_size: int = 16
    ) -> HardwareProfile:
        peak_vram_mb: Optional[float] = None

        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)
            torch.cuda.empty_cache()
            model.eval().to(self.device)
            dummy = torch.randn(batch_size, *input_shape, device=self.device)
            _ = model(dummy)
            torch.cuda.synchronize()
            peak_vram_mb = torch.cuda.max_memory_allocated(self.device) / (1024 ** 2)

        total_params, trainable_params = count_parameters(model)
        model_size_mb = total_params * 4 / (1024 ** 2)  # fp32 assumption

        profile = HardwareProfile(
            peak_vram_mb=peak_vram_mb,
            total_params=total_params,
            trainable_params=trainable_params,
            trainable_ratio=trainable_params / max(total_params, 1),
            model_size_mb=model_size_mb,
        )
        logger.info(
            "Hardware profile: VRAM=%s MB | total_params=%d | trainable=%d (%.3f%%) | size=%.2f MB",
            f"{peak_vram_mb:.1f}" if peak_vram_mb is not None else "N/A",
            total_params,
            trainable_params,
            profile.trainable_ratio * 100,
            model_size_mb,
        )
        return profile


def count_parameters(model: nn.Module) -> Tuple[int, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


  
# Classification metric wrappers
  
@dataclass
class ClassificationMetrics:
    accuracy: float
    f1_micro: float
    f1_macro: float
    auroc: Optional[float] = None


class MetricsWrapper:
    """Safe scikit-learn wrappers tolerant of degenerate single-class inputs."""

    @staticmethod
    def compute(
        y_true: np.ndarray, y_pred: np.ndarray, y_score: Optional[np.ndarray] = None
    ) -> ClassificationMetrics:
        y_true = np.asarray(y_true).ravel()
        y_pred = np.asarray(y_pred).ravel()

        acc = float(accuracy_score(y_true, y_pred))
        f1_micro = float(f1_score(y_true, y_pred, average="micro", zero_division=0))
        f1_macro = float(f1_score(y_true, y_pred, average="macro", zero_division=0))

        auroc: Optional[float] = None
        if y_score is not None and len(np.unique(y_true)) > 1:
            try:
                auroc = float(roc_auc_score(y_true, y_score))
            except ValueError as exc:
                logger.warning("AUROC computation failed: %s", exc)
                auroc = None
        elif y_score is not None:
            logger.warning("AUROC skipped: y_true contains a single class in this batch.")

        return ClassificationMetrics(accuracy=acc, f1_micro=f1_micro, f1_macro=f1_macro, auroc=auroc)
