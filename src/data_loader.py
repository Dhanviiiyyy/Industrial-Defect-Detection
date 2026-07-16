"""
data_loader.py
================================================================================
Dataset utilities for the Industrial Defect Detection Benchmark.

Contains:
    - MVTecStyleDataset: reads a real MVTec-AD-formatted directory tree.
    - SyntheticIndustrialDataset: zero-dependency, numpy/opencv-driven
      synthetic texture + anomaly generator used automatically whenever the
      configured real dataset path does not exist on disk. This guarantees
      the benchmark suite is immediately executable without any external
      data download.
    - build_dataloaders: factory that inspects the config and transparently
      returns real or synthetic loaders behind an identical interface.
================================================================================
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T

logger = logging.getLogger("defect_bench.data_loader")

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def build_transform(image_size: int) -> T.Compose:
    """Standard ImageNet-normalized transform pipeline shared by all backbones."""
    return T.Compose(
        [
            T.ToPILImage(),
            T.Resize((image_size, image_size)),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


  
# Real MVTec-AD-style dataset
  
@dataclass
class Sample:
    path: str
    label: int          # 0 = normal, 1 = anomalous
    mask_path: Optional[str] = None


class MVTecStyleDataset(Dataset):
    """
    Expects the canonical MVTec-AD layout:

        root/
          <category>/
            train/good/*.png
            test/good/*.png
            test/<defect_type>/*.png
            ground_truth/<defect_type>/*_mask.png

    Args:
        root_dir: dataset root.
        category: object/texture category subfolder.
        split: "train" or "test".
        image_size: resize target (square).
        transform: optional override transform; defaults to ImageNet normalize.
    """

    VALID_EXT = (".png", ".jpg", ".jpeg", ".bmp")

    def __init__(
        self,
        root_dir: str,
        category: str,
        split: str = "train",
        image_size: int = 224,
        transform: Optional[Callable] = None,
    ) -> None:
        self.root_dir = root_dir
        self.category = category
        self.split = split
        self.image_size = image_size
        self.transform = transform or build_transform(image_size)
        self.samples: List[Sample] = self._index_samples()
        logger.info(
            "MVTecStyleDataset[%s/%s]: indexed %d samples",
            category,
            split,
            len(self.samples),
        )

    def _index_samples(self) -> List[Sample]:
        samples: List[Sample] = []
        split_dir = os.path.join(self.root_dir, self.category, self.split)
        if not os.path.isdir(split_dir):
            raise FileNotFoundError(f"Split directory not found: {split_dir}")

        for defect_type in sorted(os.listdir(split_dir)):
            defect_dir = os.path.join(split_dir, defect_type)
            if not os.path.isdir(defect_dir):
                continue
            
            # Pure Binary Map: 0 for pristine, 1 for ANY anomaly modality
            label = 0 if defect_type == "good" else 1
            
            for fname in sorted(os.listdir(defect_dir)):
                if fname.lower().endswith(self.VALID_EXT):
                    img_path = os.path.join(defect_dir, fname)
                    mask_path = None
                    if label == 1:
                        candidate = os.path.join(
                            self.root_dir,
                            self.category,
                            "ground_truth",
                            defect_type,
                            fname.rsplit(".", 1)[0] + "_mask.png",
                        )
                        mask_path = candidate if os.path.isfile(candidate) else None
                    samples.append(Sample(path=img_path, label=label, mask_path=mask_path))
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        sample = self.samples[idx]
        image = cv2.imread(sample.path, cv2.IMREAD_COLOR)
        if image is None:
            raise IOError(f"Failed to read image: {sample.path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        tensor = self.transform(image)
        return tensor, sample.label


  
# Synthetic fallback dataset (zero-dependency, generated on the fly)
  
class SyntheticIndustrialDataset(Dataset):
    """
    Procedurally generates industrial surface textures with numpy/opencv and
    injects synthetic structural anomalies (scratches, cuts, dent blobs).
    Used automatically when the real dataset root does not exist, so the
    entire pipeline remains runnable with zero external downloads.
    """

    TEXTURE_TYPES = ("grid_lines", "brushed_metal", "carpet_weave")
    ANOMALY_TYPES = ("scratch", "cut", "dent_blob")

    def __init__(
        self,
        num_normal: int,
        num_anomalous: int,
        image_size: int = 224,
        texture_types: Optional[List[str]] = None,
        anomaly_types: Optional[List[str]] = None,
        transform: Optional[Callable] = None,
        seed: int = 42,
    ) -> None:
        self.image_size = image_size
        self.texture_types = texture_types or list(self.TEXTURE_TYPES)
        self.anomaly_types = anomaly_types or list(self.ANOMALY_TYPES)
        self.transform = transform or build_transform(image_size)
        self.rng = np.random.default_rng(seed)

        self.records: List[Tuple[int, int]] = []  # (label, local_seed)
        for i in range(num_normal):
            self.records.append((0, int(self.rng.integers(0, 1_000_000))))
        for i in range(num_anomalous):
            self.records.append((1, int(self.rng.integers(0, 1_000_000))))
        self.rng.shuffle(self.records)  # type: ignore[arg-type]

        logger.info(
            "SyntheticIndustrialDataset: %d normal / %d anomalous samples (size=%d)",
            num_normal,
            num_anomalous,
            image_size,
        )

    def __len__(self) -> int:
        return len(self.records)

    # ---------------------------------------------------------------- textures
    def _gen_texture(self, local_rng: np.random.Generator) -> np.ndarray:
        kind = local_rng.choice(self.texture_types)
        h = w = self.image_size
        base = np.full((h, w), 180, dtype=np.uint8)
        noise = local_rng.normal(0, 8, size=(h, w)).astype(np.int16)

        if kind == "grid_lines":
            spacing = local_rng.integers(12, 24)
            for y in range(0, h, spacing):
                base[y : y + 1, :] = 90
            for x in range(0, w, spacing):
                base[:, x : x + 1] = 90
        elif kind == "brushed_metal":
            for _ in range(h // 2):
                y = local_rng.integers(0, h)
                x0 = local_rng.integers(0, w // 2)
                length = local_rng.integers(20, w)
                thickness = 1
                cv2.line(
                    base,
                    (x0, y),
                    (min(x0 + length, w - 1), y),
                    color=int(local_rng.integers(140, 210)),
                    thickness=thickness,
                )
        elif kind == "carpet_weave":
            spacing = local_rng.integers(6, 12)
            for y in range(0, h, spacing):
                offset = spacing // 2 if (y // spacing) % 2 == 0 else 0
                for x in range(offset, w, spacing * 2):
                    cv2.rectangle(
                        base,
                        (x, y),
                        (min(x + spacing, w - 1), min(y + spacing, h - 1)),
                        color=int(local_rng.integers(150, 200)),
                        thickness=-1,
                    )

        img = np.clip(base.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        return img_rgb

    # ---------------------------------------------------------------- anomalies
    def _inject_anomaly(
        self, img: np.ndarray, local_rng: np.random.Generator
    ) -> np.ndarray:
        h, w, _ = img.shape
        kind = local_rng.choice(self.anomaly_types)
        img = img.copy()

        if kind == "scratch":
            pt1 = (int(local_rng.integers(0, w)), int(local_rng.integers(0, h)))
            angle = local_rng.uniform(0, 2 * np.pi)
            length = local_rng.integers(30, min(h, w) // 2)
            pt2 = (
                int(np.clip(pt1[0] + length * np.cos(angle), 0, w - 1)),
                int(np.clip(pt1[1] + length * np.sin(angle), 0, h - 1)),
            )
            cv2.line(img, pt1, pt2, color=(20, 20, 20), thickness=2)

        elif kind == "cut":
            x0, y0 = int(local_rng.integers(0, w - 20)), int(local_rng.integers(0, h - 20))
            cut_w, cut_h = local_rng.integers(8, 25), local_rng.integers(8, 25)
            pts = np.array(
                [
                    [x0, y0],
                    [x0 + cut_w, y0 + local_rng.integers(-5, 5)],
                    [x0 + cut_w // 2, y0 + cut_h],
                ],
                dtype=np.int32,
            )
            cv2.fillPoly(img, [pts], color=(10, 10, 10))

        elif kind == "dent_blob":
            cx, cy = int(local_rng.integers(20, w - 20)), int(local_rng.integers(20, h - 20))
            radius = int(local_rng.integers(6, 18))
            overlay = img.copy()
            cv2.circle(overlay, (cx, cy), radius, color=(60, 60, 60), thickness=-1)
            img = cv2.addWeighted(overlay, 0.6, img, 0.4, 0)

        return img

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        label, local_seed = self.records[idx]
        local_rng = np.random.default_rng(local_seed)
        img = self._gen_texture(local_rng)
        if label == 1:
            img = self._inject_anomaly(img, local_rng)
        tensor = self.transform(img)
        return tensor, label


  
# Loader factory
  
def build_dataloaders(cfg: dict) -> Tuple[DataLoader, DataLoader, bool]:
    """
    Builds (train_loader, test_loader, used_synthetic_flag).

    Falls back automatically to SyntheticIndustrialDataset when
    cfg['data']['root_dir'] / category does not exist on disk.
    """
    data_cfg = cfg["data"]
    root_dir = data_cfg["root_dir"]
    category = data_cfg["category"]
    image_size = data_cfg["image_size"]

    real_path = os.path.join(root_dir, category)
    use_synthetic = not os.path.isdir(real_path)

    if use_synthetic:
        logger.warning(
            "Dataset path '%s' not found. Falling back to SyntheticIndustrialDataset.",
            real_path,
        )
        syn_cfg = data_cfg["synthetic"]
        train_ds: Dataset = SyntheticIndustrialDataset(
            num_normal=syn_cfg["num_train_normal"],
            num_anomalous=0,
            image_size=image_size,
            texture_types=syn_cfg["texture_types"],
            anomaly_types=syn_cfg["anomaly_types"],
            seed=cfg["experiment"]["seed"],
        )
        test_ds: Dataset = SyntheticIndustrialDataset(
            num_normal=syn_cfg["num_test_normal"],
            num_anomalous=syn_cfg["num_test_anomalous"],
            image_size=image_size,
            texture_types=syn_cfg["texture_types"],
            anomaly_types=syn_cfg["anomaly_types"],
            seed=cfg["experiment"]["seed"] + 1,
        )
    else:
        train_ds = MVTecStyleDataset(root_dir, category, split="train", image_size=image_size)
        test_ds = MVTecStyleDataset(root_dir, category, split="test", image_size=image_size)

    train_loader = DataLoader(
        train_ds,
        batch_size=data_cfg.get("_batch_size", 16),
        shuffle=True,
        num_workers=data_cfg["num_workers"],
        pin_memory=data_cfg["pin_memory"],
        drop_last=False,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=data_cfg.get("_batch_size", 16),
        shuffle=False,
        num_workers=data_cfg["num_workers"],
        pin_memory=data_cfg["pin_memory"],
        drop_last=False,
    )
    return train_loader, test_loader, use_synthetic
