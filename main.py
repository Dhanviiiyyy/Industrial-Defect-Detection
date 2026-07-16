"""
main.py
================================================================================
System orchestration pipeline for the Industrial Defect Detection Benchmark.

Parses config/baseline_config.yaml, builds dataloaders (real MVTec-AD or the
automatic synthetic fallback), iterates across model types and adaptation
configurations, runs baseline evaluation + few-shot + layer-selection
ablations, profiles latency/VRAM/parameter footprint, and prints a final
Markdown leaderboard comparing Accuracy, F1, AUROC, Latency, VRAM, and Model
Size.

Usage:
    python -m src.main --config config/baseline_config.yaml
================================================================================
"""
from __future__ import annotations

import argparse
import copy
import logging
import os
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import torch
import yaml

# Ensure correct repository tracking imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_loader import build_dataloaders
from src.engine import (
    DualLayerFeatureExtractionEngine,
    FeatureExtractionEngine,
    run_baseline,
    run_few_shot_ablation,
    run_layer_selection_ablation,
)
from src.metrics import HardwareProfiler, LatencyProfiler
from src.models import build_model, count_lora_parameters
from src.visualizer import AnomalyHeatmapVisualizer, EmbeddingProjector, PatchFeatureExtractor


# Logging setup
def setup_logging(cfg: Dict) -> logging.Logger:
    log_cfg = cfg["logging"]
    level = getattr(logging, log_cfg.get("level", "INFO").upper(), logging.INFO)
    handlers: List[logging.Handler] = []

    if log_cfg.get("console", True):
        handlers.append(logging.StreamHandler(sys.stdout))

    out_dir = cfg["experiment"]["output_dir"]
    os.makedirs(out_dir, exist_ok=True)
    log_path = os.path.join(out_dir, log_cfg.get("log_file", "benchmark.log"))
    handlers.append(logging.FileHandler(log_path))

    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        handlers=handlers,
        force=True,
    )
    return logging.getLogger("defect_bench.main")


def resolve_device(device_cfg: str) -> torch.device:
    if device_cfg == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(device_cfg)


def load_config(path: str) -> Dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


# Layer-name resolution for hook-based dual-layer extraction (per model)
TRANSFORMER_LAYER_HOOKS = {
    "vit_b_16": {
        "penultimate": "encoder.layers.encoder_layer_10",
        "final": "encoder.layers.encoder_layer_11",
    },
    "dinov2_vits14": {
        "penultimate": "blocks.10",
        "final": "blocks.11",
    },
}


@dataclass
class LeaderboardRow:
    model_name: str
    adaptation: str
    accuracy: float
    f1_micro: float
    f1_macro: float
    auroc: Optional[float]
    latency_bs1_ms: float
    latency_bs16_ms: float
    peak_vram_mb: Optional[float]
    model_size_mb: float
    trainable_params: int
    total_params: int


def format_markdown_leaderboard(rows: List[LeaderboardRow]) -> str:
    header = (
        "| Model | Adaptation | Accuracy | F1 (micro) | F1 (macro) | AUROC | "
        "Latency@1 (ms) | Latency@16 (ms) | Peak VRAM (MB) | Size (MB) | "
        "Trainable / Total Params |\n"
        "|---|---|---|---|---|---|---|---|---|---|---|\n"
    )
    lines = []
    for r in rows:
        auroc_str = f"{r.auroc:.4f}" if r.auroc is not None else "N/A"
        vram_str = f"{r.peak_vram_mb:.1f}" if r.peak_vram_mb is not None else "N/A (CPU)"
        lines.append(
            f"| {r.model_name} | {r.adaptation} | {r.accuracy:.4f} | {r.f1_micro:.4f} | "
            f"{r.f1_macro:.4f} | {auroc_str} | {r.latency_bs1_ms:.3f} | {r.latency_bs16_ms:.3f} | "
            f"{vram_str} | {r.model_size_mb:.2f} | {r.trainable_params:,} / {r.total_params:,} |"
        )
    return header + "\n".join(lines)


# Per-model benchmark run
def benchmark_model(
    model_name: str,
    cfg: Dict,
    device: torch.device,
    logger: logging.Logger,
) -> LeaderboardRow:
    model_cfg = cfg["models"][model_name]
    adaptation_cfg = cfg["adaptation"]
    profiling_cfg = cfg["profiling"]
    training_cfg = cfg["training"]
    image_size = cfg["data"]["image_size"]

    logger.info("=" * 88)
    logger.info("Benchmarking model: %s", model_name)
    logger.info("=" * 88)

    backbone, feature_dim, model_type = build_model(model_name, model_cfg, adaptation_cfg, pretrained=True)
    backbone = backbone.to(device)

    lora_params = count_lora_parameters(backbone) if model_type == "transformer" else 0
    adaptation_label = f"LoRA(r={adaptation_cfg['lora_rank']},a={adaptation_cfg['lora_alpha']})" if lora_params > 0 else "frozen"

    # ---- Dataloaders (batch size swapped per profiling pass; base loaders use bs=16) ----
    data_cfg = copy.deepcopy(cfg)
    data_cfg["data"]["_batch_size"] = 16
    train_loader, test_loader, used_synthetic = build_dataloaders(data_cfg)
    if used_synthetic:
        logger.info("Using SyntheticIndustrialDataset fallback for '%s' benchmark.", model_name)

    # ---- Baseline linear-probe evaluation ----
    baseline_metrics = run_baseline(
        backbone, model_type, train_loader, test_loader, device,
        epochs=training_cfg["linear_probe_epochs"], lr=training_cfg["learning_rate"],
    )

    # ---- Extract Base Structural Reference Features (For Visuals & Ablations) ----
    extractor = FeatureExtractionEngine(backbone, device, model_type)
    train_feats = extractor.extract(train_loader)
    test_feats = extractor.extract(test_loader)

    # ---- Diagnostic Visualization Plot Generation ----
    output_dir = cfg["experiment"]["output_dir"]
    
    # 1. Generate Latent Space Projection Plot (PCA/t-SNE)
    if cfg["evaluation"]["visualize"]["embedding_projection"]:
        proj_method = cfg["evaluation"]["visualize"]["embedding_projection"]
        logger.info("Generating 2D %s latent space visualization...", proj_method.upper())
        
        projector = EmbeddingProjector(method=proj_method, random_state=cfg["experiment"]["seed"])
        
        test_embeddings_np = test_feats.embeddings.numpy()
        test_labels_np = test_feats.labels.numpy()
        
        plot_save_path = os.path.join(output_dir, f"{model_name}_latent_projection.png")
        projector.plot(
            embeddings=test_embeddings_np,
            labels=test_labels_np,
            title=f"{model_name} Latent Embedding Space",
            save_path=plot_save_path
        )

    # 2. Generate Spatial Anomaly Localization Heatmap (For Transformers)
    if cfg["evaluation"]["visualize"]["heatmap"] and model_type == "transformer":
        logger.info("Generating spatial anomaly localization map for %s...", model_name)
        try:
            heatmap_viz = AnomalyHeatmapVisualizer()
            raw_dataset = test_loader.dataset
            
            # Find the first anomalous test image to visually overlay
            anom_idx = 0
            for idx in range(len(raw_dataset)):
                _, label = raw_dataset[idx]
                if label > 0:  # Any defect class
                    anom_idx = idx
                    break
            
            img_tensor, _ = raw_dataset[anom_idx]
            x_in = img_tensor.unsqueeze(0).to(device)
            
            # Leverage our registered hooks to cleanly capture patch activations
            hooks = TRANSFORMER_LAYER_HOOKS.get(model_name)
            if hooks:
                hook_extractor = PatchFeatureExtractor(backbone, hooks["final"])
                with torch.no_grad():
                    _ = backbone(x_in)
                
                features_raw = hook_extractor.get_features()
                hook_extractor.remove()
                
                if features_raw is not None:
                    # Drop CLS token if present in transformer dimensions [B, N, D]
                    if features_raw.dim() == 3:
                        features_raw = features_raw[:, 1:, :]
                    
                    B, N, D = features_raw.shape
                    grid_side = int(N ** 0.5)
                    spatial_patches = features_raw.reshape(B, grid_side, grid_side, D).permute(0, 3, 1, 2).squeeze(0)
                    
                    # Align nominal context features to calculate anomaly vector bounds
                    normal_indices = torch.nonzero(test_feats.labels == 0).squeeze(-1)
                    nominal_vectors = test_feats.embeddings[normal_indices]
                    
                    heatmap_viz.feature_bank = torch.nn.functional.normalize(nominal_vectors, dim=-1).to(device)
                    heatmap_map = heatmap_viz.compute_heatmap(spatial_patches.to(device), output_size=(image_size, image_size))
                    
                    # Reconstruct original image format for visualization
                    mock_rgb = img_tensor.permute(1, 2, 0).cpu().numpy()
                    mock_rgb = (mock_rgb - mock_rgb.min()) / (mock_rgb.max() - mock_rgb.min() + 1e-8)
                    mock_rgb = (mock_rgb * 255).astype(np.uint8)
                    
                    heatmap_save_path = os.path.join(output_dir, f"{model_name}_defect_heatmap.png")
                    heatmap_viz.overlay_heatmap(mock_rgb, heatmap_map, alpha=0.4, save_path=heatmap_save_path)
                else:
                    logger.warning("Hook returned empty features for heatmap generation.")
            else:
                logger.warning("No transformer layer hooks defined for model variant: %s", model_name)
        except Exception as e:
            logger.exception("Spatial heatmap visualization failed for %s", model_name)

    # ---- Few-shot ablation ----
    if int(torch.unique(train_feats.labels).numel()) >= 2:
        few_shot_results = run_few_shot_ablation(
            train_feats, test_feats, training_cfg["few_shot_shots"], device,
            epochs=training_cfg["linear_probe_epochs"], lr=training_cfg["learning_rate"],
        )
        for shots, m in few_shot_results.items():
            logger.info("  [few-shot N=%d] acc=%.4f f1_macro=%.4f", shots, m.accuracy, m.f1_macro)
    else:
        logger.info("  Skipping few-shot ablation: training split is single-class.")

    # ---- Layer-selection ablation (transformers only) ----
    if model_type == "transformer" and model_name in TRANSFORMER_LAYER_HOOKS:
        hooks = TRANSFORMER_LAYER_HOOKS[model_name]
        try:
            dual_engine = DualLayerFeatureExtractionEngine(
                backbone, device, hooks["penultimate"], hooks["final"]
            )
            layer_results = run_layer_selection_ablation(
                dual_engine, train_loader, test_loader, device,
                epochs=training_cfg["linear_probe_epochs"], lr=training_cfg["learning_rate"],
            )
            dual_engine.close()
            for mode, m in layer_results.items():
                logger.info("  [layer-select %s] acc=%.4f f1_macro=%.4f", mode, m.accuracy, m.f1_macro)
        except Exception as exc:
            logger.warning("  Layer-selection ablation skipped for %s: %s", model_name, exc)

    # ---- Latency profiling ----
    latency_profiler = LatencyProfiler(
        device, warmup_iters=profiling_cfg["warmup_iters"], measured_iters=profiling_cfg["measured_iters"]
    )
    latency_results = latency_profiler.profile(
        backbone, input_shape=(3, image_size, image_size), batch_sizes=profiling_cfg["batch_sizes"]
    )

    # ---- Hardware / parameter profiling ----
    hw_profiler = HardwareProfiler(device)
    hw_profile = hw_profiler.profile(backbone, input_shape=(3, image_size, image_size), batch_size=16)

    return LeaderboardRow(
        model_name=model_name,
        adaptation=adaptation_label,
        accuracy=baseline_metrics.accuracy,
        f1_micro=baseline_metrics.f1_micro,
        f1_macro=baseline_metrics.f1_macro,
        auroc=baseline_metrics.auroc,
        latency_bs1_ms=latency_results[1].ms_per_image if 1 in latency_results else float("nan"),
        latency_bs16_ms=latency_results[16].ms_per_image if 16 in latency_results else float("nan"),
        peak_vram_mb=hw_profile.peak_vram_mb,
        model_size_mb=hw_profile.model_size_mb,
        trainable_params=hw_profile.trainable_params,
        total_params=hw_profile.total_params,
    )


# Entry point
def main() -> None:
    parser = argparse.ArgumentParser(description="Industrial Defect Detection Benchmark Suite")
    parser.add_argument("--config", type=str, default="config/baseline_config.yaml")
    parser.add_argument(
        "--models", type=str, nargs="+", default=None,
        help="Subset of model names to benchmark (default: all in config).",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    logger = setup_logging(cfg)

    torch.manual_seed(cfg["experiment"]["seed"])
    device = resolve_device(cfg["experiment"]["device"])
    logger.info("Resolved device: %s", device)

    model_names = args.models or list(cfg["models"].keys())
    rows: List[LeaderboardRow] = []

    for model_name in model_names:
        try:
            row = benchmark_model(model_name, cfg, device, logger)
            rows.append(row)
        except Exception as exc:
            logger.exception("Benchmark failed for model '%s': %s", model_name, exc)

    leaderboard_md = format_markdown_leaderboard(rows)
    logger.info("\n" + "=" * 40 + " FINAL LEADERBOARD " + "=" * 40)
    print("\n" + leaderboard_md + "\n")

    out_path = os.path.join(cfg["experiment"]["output_dir"], "leaderboard.md")
    os.makedirs(cfg["experiment"]["output_dir"], exist_ok=True)
    with open(out_path, "w") as f:
        f.write(leaderboard_md)
    logger.info("Leaderboard written to %s", out_path)


if __name__ == "__main__":
    main()