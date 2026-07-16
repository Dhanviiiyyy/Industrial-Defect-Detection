"""
visualizer.py
================================================================================
Visualization utilities for patch-level anomaly localization and latent
embedding inspection.

Contains:
    - PatchFeatureExtractor: forward-hook-based extractor that pulls
      intermediate token/patch maps from ViT-style transformer blocks or
      spatial feature maps from CNN backbones.
    - AnomalyHeatmapVisualizer: builds a nominal-only patch feature bank and
      computes a localized anomaly heatmap via patch-to-nominal cosine
      similarity (1 - max_similarity = anomaly score per patch).
    - EmbeddingProjector: PCA / t-SNE projection of high-dimensional
      image-level embeddings into a 2D scatter plot separating normal vs
      anomalous samples.
================================================================================
"""

from __future__ import annotations

import logging
import os
from typing import Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")  # headless-safe backend
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

logger = logging.getLogger("defect_bench.visualizer")


  
# Patch / token feature extraction via forward hooks
  
class PatchFeatureExtractor:
    """
    Registers a forward hook on a target sub-module to capture intermediate
    activations. For transformers this is typically the output of the final
    (or second-to-last) encoder block, shape (B, N_tokens, D). For CNNs this
    is the spatial feature map of the last conv stage, shape (B, C, H, W).
    """

    def __init__(self, model: nn.Module, layer_name: str) -> None:
        self.model = model
        self.layer_name = layer_name
        self._features: Optional[torch.Tensor] = None
        self._hook_handle = None
        self._register_hook()

    def _register_hook(self) -> None:
        module = dict(self.model.named_modules()).get(self.layer_name)
        if module is None:
            raise ValueError(f"Layer '{self.layer_name}' not found in model.")

        def hook(_module: nn.Module, _inp: Tuple, out: torch.Tensor) -> None:
            self._features = out.detach()

        self._hook_handle = module.register_forward_hook(hook)

    def get_features(self) -> Optional[torch.Tensor]:
        return self._features

    def remove(self) -> None:
        if self._hook_handle is not None:
            self._hook_handle.remove()


def tokens_to_patch_grid(tokens: torch.Tensor, has_cls_token: bool = True) -> torch.Tensor:
    """
    Converts a transformer token sequence (B, N, D) into a square spatial
    patch grid (B, D, h, w), dropping the CLS token if present.
    """
    b, n, d = tokens.shape
    if has_cls_token:
        tokens = tokens[:, 1:, :]
        n -= 1
    side = int(round(n ** 0.5))
    if side * side != n:
        raise ValueError(f"Token count {n} is not a perfect square; cannot form a grid.")
    grid = tokens.reshape(b, side, side, d).permute(0, 3, 1, 2)
    return grid  # (B, D, h, w)


  
# Patch-level anomaly heatmap via cosine similarity to nominal feature bank
  
class AnomalyHeatmapVisualizer:
    """
    Builds a memory bank of patch-level embeddings from *nominal-only* images,
    then scores a query image's patches by (1 - max cosine similarity to the
    nearest nominal patch), producing a spatial anomaly heatmap.
    """

    def __init__(self, feature_bank_max_size: int = 20000) -> None:
        self.feature_bank: Optional[torch.Tensor] = None  # (M, D), L2-normalized
        self.feature_bank_max_size = feature_bank_max_size

    def build_feature_bank(self, patch_features_list: List[torch.Tensor]) -> None:
        """
        Args:
            patch_features_list: list of (D, h, w) or (N, D) per-image patch
                feature tensors extracted from *normal* training images.
        """
        flat_feats = []
        for feat in patch_features_list:
            if feat.dim() == 3:  # (D, h, w) -> (h*w, D)
                d, h, w = feat.shape
                feat = feat.reshape(d, h * w).permute(1, 0)
            flat_feats.append(feat)
        bank = torch.cat(flat_feats, dim=0)
        bank = F.normalize(bank, dim=-1)

        if bank.shape[0] > self.feature_bank_max_size:
            idx = torch.randperm(bank.shape[0])[: self.feature_bank_max_size]
            bank = bank[idx]

        self.feature_bank = bank
        logger.info("Nominal feature bank built: %d patch vectors (dim=%d)", *bank.shape)

    def compute_heatmap(self, query_patch_feat: torch.Tensor, output_size: Tuple[int, int]) -> np.ndarray:
        """
        Args:
            query_patch_feat: (D, h, w) patch feature map for one query image.
            output_size: (H, W) target resolution for the upsampled heatmap.
        Returns:
            2D numpy array in [0, 1], higher = more anomalous.
        """
        if self.feature_bank is None:
            raise RuntimeError("Feature bank not built. Call build_feature_bank first.")

        d, h, w = query_patch_feat.shape
        query = query_patch_feat.reshape(d, h * w).permute(1, 0)  # (h*w, D)
        query = F.normalize(query, dim=-1)

        sims = query @ self.feature_bank.T  # (h*w, M)
        max_sim, _ = sims.max(dim=-1)  # (h*w,)
        anomaly_score = 1.0 - max_sim  # higher = more anomalous
        anomaly_map = anomaly_score.reshape(1, 1, h, w)

        upsampled = F.interpolate(anomaly_map, size=output_size, mode="bilinear", align_corners=False)
        arr = upsampled.squeeze().cpu().numpy()
        arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-8)
        return arr

    @staticmethod
    def overlay_heatmap(image_rgb: np.ndarray, heatmap: np.ndarray, alpha: float = 0.5, save_path: Optional[str] = None):
        """Renders and optionally saves an RGB image + heatmap overlay via matplotlib."""
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        axes[0].imshow(image_rgb)
        axes[0].set_title("Input")
        axes[0].axis("off")

        axes[1].imshow(heatmap, cmap="jet")
        axes[1].set_title("Anomaly Heatmap")
        axes[1].axis("off")

        axes[2].imshow(image_rgb)
        axes[2].imshow(heatmap, cmap="jet", alpha=alpha)
        axes[2].set_title("Overlay")
        axes[2].axis("off")

        plt.tight_layout()
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            fig.savefig(save_path, dpi=150)
            logger.info("Saved heatmap visualization to %s", save_path)
        plt.close(fig)


  
# 2D embedding projection (PCA / t-SNE)
  
class EmbeddingProjector:
    """Projects high-dimensional image-level embeddings into 2D for inspection."""

    def __init__(self, method: str = "pca", random_state: int = 42) -> None:
        assert method in ("pca", "tsne"), "method must be 'pca' or 'tsne'"
        self.method = method
        self.random_state = random_state

    def project(self, embeddings: np.ndarray) -> np.ndarray:
        if self.method == "pca":
            reducer = PCA(n_components=2, random_state=self.random_state)
        else:
            perplexity = min(30, max(5, embeddings.shape[0] // 3))
            reducer = TSNE(n_components=2, random_state=self.random_state, perplexity=perplexity, init="pca")
        return reducer.fit_transform(embeddings)

    def plot(
        self,
        embeddings: np.ndarray,
        labels: np.ndarray,
        title: str = "Latent Space Projection",
        save_path: Optional[str] = None,
    ) -> None:
        coords = self.project(embeddings)
        fig, ax = plt.subplots(figsize=(6, 6))

        normal_mask = labels == 0
        anomaly_mask = labels == 1

        ax.scatter(
            coords[normal_mask, 0], coords[normal_mask, 1],
            c="#2E86AB", label="Normal", alpha=0.7, s=25, edgecolors="none",
        )
        ax.scatter(
            coords[anomaly_mask, 0], coords[anomaly_mask, 1],
            c="#C73E1D", label="Anomalous", alpha=0.7, s=25, edgecolors="none",
        )
        ax.set_title(f"{title} ({self.method.upper()})")
        ax.set_xlabel("Component 1")
        ax.set_ylabel("Component 2")
        ax.legend()
        plt.tight_layout()

        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            fig.savefig(save_path, dpi=150)
            logger.info("Saved embedding projection to %s", save_path)
        plt.close(fig)
