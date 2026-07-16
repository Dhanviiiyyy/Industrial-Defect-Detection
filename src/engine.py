"""
engine.py
================================================================================
Core training and evaluation routines.

Contains:
    - FeatureExtractionEngine: runs a frozen/LoRA-adapted backbone over a
      dataloader and caches pooled embeddings + labels.
    - train_linear_probe / evaluate_linear_probe: baseline classifier
      training/eval atop cached embeddings.
    - run_few_shot_ablation: trains linear probes using only N samples per
      class (few-shot shot counts from config) and reports metrics per shot
      count.
    - run_layer_selection_ablation: compares downstream accuracy/F1/AUROC
      using (a) the final-layer CLS token only vs (b) concatenated
      representations from the final two transformer encoder layers.
================================================================================
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .metrics import ClassificationMetrics, MetricsWrapper
from .models import LinearProbeHead
from .visualizer import PatchFeatureExtractor

logger = logging.getLogger("defect_bench.engine")


 
# Feature extraction
 
@dataclass
class CachedFeatures:
    embeddings: torch.Tensor  # (N, D)
    labels: torch.Tensor      # (N,)


class FeatureExtractionEngine:
    """Runs a backbone in eval mode over a dataloader, caching pooled embeddings."""

    def __init__(self, backbone: nn.Module, device: torch.device, model_type: str) -> None:
        self.backbone = backbone.to(device).eval()
        self.device = device
        self.model_type = model_type

    @torch.no_grad()
    def extract(self, loader: DataLoader) -> CachedFeatures:
        all_embeddings: List[torch.Tensor] = []
        all_labels: List[torch.Tensor] = []

        for images, labels in loader:
            images = images.to(self.device)
            out = self.backbone(images)
            pooled = self._pool(out)
            all_embeddings.append(pooled.cpu())
            all_labels.append(labels)

        embeddings = torch.cat(all_embeddings, dim=0)
        labels_t = torch.cat(all_labels, dim=0)
        logger.info("Extracted features: %s, labels: %s", tuple(embeddings.shape), tuple(labels_t.shape))
        return CachedFeatures(embeddings=embeddings, labels=labels_t)

    @staticmethod
    def _pool(out: torch.Tensor) -> torch.Tensor:
        """Handles both CNN (B, D) and transformer (B, N, D) style outputs."""
        if out.dim() == 4:  # (B, C, H, W)
            return out.mean(dim=[2, 3])
        if out.dim() == 3:  # (B, N, D) -> take CLS token (index 0)
            return out[:, 0, :]
        return out  # already (B, D)


 
# Two-layer feature extraction for layer-selection ablation
 
class DualLayerFeatureExtractionEngine:
    """
    Captures token outputs of the last two transformer encoder blocks via
    forward hooks, enabling comparison between:
      (a) final CLS token only
      (b) concatenation of CLS tokens from the last two blocks
    """

    def __init__(
        self,
        backbone: nn.Module,
        device: torch.device,
        second_to_last_layer_name: str,
        last_layer_name: str,
    ) -> None:
        self.backbone = backbone.to(device).eval()
        self.device = device
        self.hook_penultimate = PatchFeatureExtractor(backbone, second_to_last_layer_name)
        self.hook_final = PatchFeatureExtractor(backbone, last_layer_name)

    @torch.no_grad()
    def extract(self, loader: DataLoader) -> Tuple[CachedFeatures, CachedFeatures]:
        """Returns (final_cls_only_features, concat_last_two_features)."""
        final_embeds, concat_embeds, labels_all = [], [], []

        for images, labels in loader:
            images = images.to(self.device)
            _ = self.backbone(images)

            final_tok = self.hook_final.get_features()
            penult_tok = self.hook_penultimate.get_features()
            if final_tok is None or penult_tok is None:
                raise RuntimeError("Hooks failed to capture activations; check layer names.")

            final_cls = final_tok[:, 0, :]
            penult_cls = penult_tok[:, 0, :]
            concat_cls = torch.cat([penult_cls, final_cls], dim=-1)

            final_embeds.append(final_cls.cpu())
            concat_embeds.append(concat_cls.cpu())
            labels_all.append(labels)

        labels_t = torch.cat(labels_all, dim=0)
        return (
            CachedFeatures(embeddings=torch.cat(final_embeds, dim=0), labels=labels_t),
            CachedFeatures(embeddings=torch.cat(concat_embeds, dim=0), labels=labels_t),
        )

    def close(self) -> None:
        self.hook_penultimate.remove()
        self.hook_final.remove()


 
# Linear probe train / eval
 
def train_linear_probe(
    features: torch.Tensor,
    labels: torch.Tensor,
    num_classes: int = 2,
    epochs: int = 50,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    device: Optional[torch.device] = None,
) -> LinearProbeHead:
    device = device or torch.device("cpu")
    head = LinearProbeHead(in_features=features.shape[1], num_classes=num_classes).to(device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()

    features = features.to(device)
    labels = labels.to(device)

    head.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        logits = head(features)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        if (epoch + 1) % max(1, epochs // 5) == 0:
            logger.debug("Linear probe epoch %d/%d - loss=%.4f", epoch + 1, epochs, loss.item())

    return head


@torch.no_grad()
def evaluate_linear_probe(
    head: LinearProbeHead, features: torch.Tensor, labels: torch.Tensor, device: Optional[torch.device] = None
) -> ClassificationMetrics:
    device = device or torch.device("cpu")
    head.eval().to(device)
    features = features.to(device)

    logits = head(features)
    probs = torch.softmax(logits, dim=-1)
    preds = torch.argmax(probs, dim=-1).cpu().numpy()
    scores = probs[:, 1].cpu().numpy()  # anomaly-class probability
    y_true = labels.cpu().numpy()

    return MetricsWrapper.compute(y_true, preds, scores)


 
# Few-shot shot-count ablation
 
def _stratified_subsample(
    embeddings: torch.Tensor, labels: torch.Tensor, shots: int, seed: int = 42
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Selects up to `shots` samples per class (class-balanced few-shot subset)."""
    rng = np.random.default_rng(seed)
    selected_idx: List[int] = []
    for cls in torch.unique(labels).tolist():
        cls_idx = torch.nonzero(labels == cls, as_tuple=True)[0].numpy()
        n = min(shots, len(cls_idx))
        chosen = rng.choice(cls_idx, size=n, replace=False)
        selected_idx.extend(chosen.tolist())
    selected_idx = np.array(selected_idx)
    return embeddings[selected_idx], labels[selected_idx]


def run_few_shot_ablation(
    train_features: CachedFeatures,
    test_features: CachedFeatures,
    shot_counts: List[int],
    device: torch.device,
    epochs: int = 50,
    lr: float = 1e-3,
    seed: int = 42,
) -> Dict[int, ClassificationMetrics]:
    """
    For each shot count N in shot_counts, trains a linear probe on N samples
    per class drawn from train_features and evaluates on the full test set.
    """
    results: Dict[int, ClassificationMetrics] = {}
    num_classes = int(torch.unique(train_features.labels).numel())

    for shots in shot_counts:
        few_x, few_y = _stratified_subsample(train_features.embeddings, train_features.labels, shots, seed)
        head = train_linear_probe(
            few_x, few_y, num_classes=num_classes, epochs=epochs, lr=lr, device=device
        )
        metrics = evaluate_linear_probe(head, test_features.embeddings, test_features.labels, device)
        results[shots] = metrics
        logger.info(
            "Few-shot[N=%d]: acc=%.4f f1_micro=%.4f f1_macro=%.4f auroc=%s",
            shots, metrics.accuracy, metrics.f1_micro, metrics.f1_macro,
            f"{metrics.auroc:.4f}" if metrics.auroc is not None else "N/A",
        )
    return results


 
# Layer-selection ablation: final CLS token vs concat(last two layers)
 
def run_layer_selection_ablation(
    dual_engine: DualLayerFeatureExtractionEngine,
    train_loader: DataLoader,
    test_loader: DataLoader,
    device: torch.device,
    epochs: int = 50,
    lr: float = 1e-3,
) -> Dict[str, ClassificationMetrics]:
    """
    Compares two feature representations for downstream linear-probe accuracy:
      'cls_token_final'         -> CLS token from the final encoder layer only
      'concat_last_two_layers'  -> concat(CLS_{L-1}, CLS_{L})
    """
    train_final, train_concat = dual_engine.extract(train_loader)
    test_final, test_concat = dual_engine.extract(test_loader)
    num_classes = int(torch.unique(train_final.labels).numel())

    results: Dict[str, ClassificationMetrics] = {}

    head_final = train_linear_probe(
        train_final.embeddings, train_final.labels, num_classes=num_classes, epochs=epochs, lr=lr, device=device
    )
    results["cls_token_final"] = evaluate_linear_probe(
        head_final, test_final.embeddings, test_final.labels, device
    )

    head_concat = train_linear_probe(
        train_concat.embeddings, train_concat.labels, num_classes=num_classes, epochs=epochs, lr=lr, device=device
    )
    results["concat_last_two_layers"] = evaluate_linear_probe(
        head_concat, test_concat.embeddings, test_concat.labels, device
    )

    for mode, metrics in results.items():
        logger.info(
            "LayerSelection[%s]: acc=%.4f f1_micro=%.4f f1_macro=%.4f auroc=%s",
            mode, metrics.accuracy, metrics.f1_micro, metrics.f1_macro,
            f"{metrics.auroc:.4f}" if metrics.auroc is not None else "N/A",
        )
    return results


 
# Baseline end-to-end train + evaluate (full-shot linear probe)
 
def run_baseline(
    backbone: nn.Module,
    model_type: str,
    train_loader: DataLoader,
    test_loader: DataLoader,
    device: torch.device,
    epochs: int = 50,
    lr: float = 1e-3,
) -> ClassificationMetrics:
    extractor = FeatureExtractionEngine(backbone, device, model_type)
    train_feats = extractor.extract(train_loader)
    test_feats = extractor.extract(test_loader)

    # Guard: if training split is single-class (typical MVTec 'good'-only train),
    # synthesize a minimal balanced probe set by borrowing a few test anomalies
    # is avoided here—baseline probe instead uses test-time k-fold style split
    # when train labels are single-class, falling back to test_feats itself.
    if int(torch.unique(train_feats.labels).numel()) < 2:
        logger.warning(
            "Training split contains a single class; using test-set split for "
            "linear-probe supervision (baseline diagnostic mode)."
        )
        n = test_feats.embeddings.shape[0]
        split = max(1, n // 2)
        head = train_linear_probe(
            test_feats.embeddings[:split], test_feats.labels[:split],
            num_classes=2, epochs=epochs, lr=lr, device=device,
        )
        return evaluate_linear_probe(head, test_feats.embeddings[split:], test_feats.labels[split:], device)

    num_classes = int(torch.unique(train_feats.labels).numel())
    head = train_linear_probe(
        train_feats.embeddings, train_feats.labels, num_classes=num_classes, epochs=epochs, lr=lr, device=device
    )
    return evaluate_linear_probe(head, test_feats.embeddings, test_feats.labels, device)
