# Visual Quality Control: Benchmarking Vision Foundation Models for Low-Latency Industrial Defect Detection

A modular computer vision benchmarking framework for evaluating the trade-offs between traditional CNNs and Vision Foundation Models (VFMs) under industrial deployment constraints.

The project simulates semiconductor and nano-electronics visual inspection pipelines, emphasizing:

- Few-shot adaptation
- Parameter-efficient fine-tuning through a custom LoRA implementation
- Inference latency and memory profiling
- Industrial anomaly detection on the MVTec AD benchmark

---

## Overview

This benchmark compares conventional convolutional architectures against Vision Transformers adapted using manually implemented Low-Rank Adaptation (LoRA). Beyond predictive performance, the framework measures deployment-oriented metrics such as latency, memory usage, model size, and parameter efficiency.

The repository is designed as a modular experimentation framework, allowing rapid evaluation of different backbones, adaptation methods, and visualization techniques.

---

## Features

- **Synthetic defect generation** for dependency-free pipeline validation
- **Manual LoRA implementation** by directly patching Transformer attention projection matrices (Q and V) without external PEFT libraries
- **Few-shot fine-tuning** for industrial inspection scenarios
- **Latency benchmarking** across multiple batch sizes
- **Memory profiling** on CPU/GPU
- **Embedding visualization** using PCA, t-SNE, and cosine similarity analysis
- **Config-driven experiments** through centralized YAML configuration

---

## Benchmark Results

Performance on the **MVTec AD** industrial anomaly detection dataset.

| Model | Adaptation | Accuracy | F1 (Micro) | F1 (Macro) | AUROC | Latency (BS=1) | Latency (BS=16) | Peak VRAM | Model Size | Trainable Parameters |
|-------|------------|----------|------------|------------|--------|----------------|-----------------|------------|-------------|-----------------------|
| **ResNet-50** | Frozen Backbone | 0.5641 | 0.5641 | 0.3607 | 0.3930 | 87.48 ms | 64.41 ms | CPU | 89.68 MB | 0 / 23,508,032 (0.00%) |
| **DINOv2 (ViT-S/14)** | **Manual LoRA (r=8, α=16)** | **0.5897** | **0.5897** | **0.4222** | **0.9225** | **114.85 ms** | **78.24 ms** | CPU | **84.70 MB** | **147,456 / 22,204,032 (0.66%)** |

---

## Repository Structure

```text
industrial-defect-vfm/
│
├── config/
│   └── baseline_config.yaml      # Experiment configuration
│
├── src/
│   ├── data_loader.py            # MVTec loader + synthetic texture generator
│   ├── models.py                 # Backbone loading + manual LoRA injection
│   ├── metrics.py                # Evaluation metrics and latency profiling
│   ├── engine.py                 # Few-shot training pipeline
│   └── visualizer.py             # PCA, t-SNE, cosine similarity visualization
│
└── main.py                       # End-to-end experiment runner
```

---

# Technical Analysis

## 1. Accuracy vs. AUROC

One of the primary observations is the significant discrepancy between DINOv2's classification accuracy (0.5897) and its AUROC (0.9225).

### Threshold Calibration Bias

MVTec AD samples are organized alphabetically by defect category (e.g., *bent*, *broken*, *glue*, *good*). During evaluation, the pipeline performs a fixed downstream split without shuffling. This introduces class imbalance during threshold calibration because healthy and defective samples are not uniformly distributed.

As a result, threshold-dependent metrics such as Accuracy and F1 decrease despite the underlying feature representations remaining highly separable.

### Feature Representation Quality

The AUROC of **0.9225** demonstrates that DINOv2 learns highly discriminative latent representations capable of separating defective from non-defective samples across a wide range of thresholds.

In contrast, the frozen ResNet-50 backbone achieves only **0.3930 AUROC**, indicating that generic ImageNet semantic features fail to capture the subtle geometric irregularities characteristic of industrial surface defects.

---

## 2. Latent Space Topology

Principal Component Analysis (PCA) reveals distinct differences in feature organization between the two architectures.

### Texture-Level Clustering

Both models naturally separate different industrial textures into distinct global clusters, including:

- Grid patterns
- Brushed metal
- Carpet textures

This indicates that both backbones capture coarse texture semantics.

### Defect Separation

Within each texture cluster, the learned representations diverge substantially.

- **ResNet-50** exhibits heavy overlap between defective and pristine samples.
- **DINOv2 + Manual LoRA** produces clear intra-cluster separation, demonstrating that low-rank adaptation successfully specializes the pretrained feature space toward anomaly discrimination while updating fewer than 1% of the model parameters.

---

## 3. Computational Scaling

Latency measurements across different batch sizes highlight the trade-offs between convolutional and transformer architectures.

### Batch Size = 1

When processing individual components in a continuous manufacturing pipeline, DINOv2 provides competitive inference while preserving substantially stronger feature quality.

### Batch Size = 16

As throughput increases, the convolutional operations of ResNet-50 benefit from improved CPU cache locality and localized receptive fields, allowing the CNN to scale more efficiently than the global attention mechanism used by Vision Transformers.

This reflects a practical deployment trade-off between representation quality and throughput efficiency.

---

# Installation

Install the project dependencies:

```bash
pip install -r requirements.txt
```

---

# Running the Benchmark

## Synthetic Benchmark

Run the complete benchmarking pipeline using the built-in synthetic texture generator:

```bash
python main.py
```

---

## MVTec AD Evaluation

Download the MVTec AD dataset and place it inside the project directory.

Update the dataset configuration:

```yaml
data:
  dataset_name: "mvtec_ad"
  root_dir: "./data/mvtec-ad"
  category: "grid"
```

Then execute:

```bash
python main.py
```

---
