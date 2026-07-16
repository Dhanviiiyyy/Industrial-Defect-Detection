"""
models.py
================================================================================
Model factory + manual Low-Rank Adaptation (LoRA) implementation.

Contains:
    - LoRALinear: a manual nn.Module wrapping an existing nn.Linear with
      trainable low-rank A/B matrices, scaled by alpha/rank. The original
      linear weights remain frozen; only A, B (and optionally bias) train.
    - inject_lora_into_attention: dynamically walks a transformer module
      tree (torchvision ViT or torch.hub DINOv2) and replaces the query/value
      projection layers inside multi-head attention blocks with LoRALinear
      wrappers, without any external PEFT dependency.
    - build_model: factory returning (backbone, feature_dim, model_type) for
      resnet50 / vit_b_16 / dinov2_vits14, with LoRA optionally injected.
================================================================================
"""

from __future__ import annotations

import logging
import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torchvision

logger = logging.getLogger("defect_bench.models")


  
# Manual LoRA layer
  
class LoRALinear(nn.Module):
    """
    Wraps a frozen nn.Linear with a trainable low-rank decomposition:

        h = W0 x + (alpha / rank) * B (A x)

    W0 (the original weight) and its bias are frozen. Only A ∈ R^{rank x in},
    B ∈ R^{out x rank} are trainable. B is zero-initialized so the adapted
    layer is numerically identical to the frozen base at initialization.
    """

    def __init__(
        self,
        base_linear: nn.Linear,
        rank: int = 8,
        alpha: int = 16,
        dropout: float = 0.05,
    ) -> None:
        super().__init__()
        self.in_features = base_linear.in_features
        self.out_features = base_linear.out_features
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / float(rank)

        # Freeze the original projection.
        self.base = base_linear
        for p in self.base.parameters():
            p.requires_grad = False

        self.lora_A = nn.Parameter(torch.zeros(rank, self.in_features))
        self.lora_B = nn.Parameter(torch.zeros(self.out_features, rank))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)  # ensures identity behavior at init

        self.dropout = nn.Dropout(p=dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base(x)
        lora_out = self.dropout(x) @ self.lora_A.T @ self.lora_B.T
        return base_out + self.scaling * lora_out

    def extra_repr(self) -> str:
        return (
            f"in={self.in_features}, out={self.out_features}, "
            f"rank={self.rank}, alpha={self.alpha}, scaling={self.scaling:.3f}"
        )


def count_lora_parameters(module: nn.Module) -> int:
    return sum(
        p.numel()
        for name, p in module.named_parameters()
        if "lora_" in name and p.requires_grad
    )


  
# LoRA injection into attention Q/V projections
  
def _split_qkv_and_wrap(
    attn_module: nn.Module, rank: int, alpha: int, dropout: float
) -> bool:
    """
    Handles the common torchvision MultiheadAttention-style module where
    q/k/v are packed into a single `in_proj_weight`/`in_proj_bias` (as in
    nn.MultiheadAttention) by splitting it into separate q/k/v nn.Linear
    layers, then wrapping the q and v sub-layers with LoRALinear.
    Returns True if a patch was applied.
    """
    if not hasattr(attn_module, "in_proj_weight") or attn_module.in_proj_weight is None:
        return False

    embed_dim = attn_module.embed_dim
    in_proj_weight = attn_module.in_proj_weight.data
    in_proj_bias = (
        attn_module.in_proj_bias.data if attn_module.in_proj_bias is not None else None
    )

    q_w, k_w, v_w = in_proj_weight.split(embed_dim, dim=0)
    if in_proj_bias is not None:
        q_b, k_b, v_b = in_proj_bias.split(embed_dim, dim=0)
    else:
        q_b = k_b = v_b = None

    def make_linear(w: torch.Tensor, b: Optional[torch.Tensor]) -> nn.Linear:
        lin = nn.Linear(embed_dim, embed_dim, bias=b is not None)
        lin.weight.data.copy_(w)
        if b is not None:
            lin.bias.data.copy_(b)
        return lin

    q_proj = make_linear(q_w, q_b)
    k_proj = make_linear(k_w, k_b)
    v_proj = make_linear(v_w, v_b)

    attn_module.q_proj = LoRALinear(q_proj, rank=rank, alpha=alpha, dropout=dropout)
    attn_module.k_proj = k_proj
    for p in attn_module.k_proj.parameters():
        p.requires_grad = False
    attn_module.v_proj = LoRALinear(v_proj, rank=rank, alpha=alpha, dropout=dropout)

    # Disable the packed projection path; downstream custom forward (if any)
    # should reference q_proj/k_proj/v_proj. We keep in_proj frozen as a
    # fallback reference but zero its grad requirement.
    attn_module.in_proj_weight.requires_grad = False
    if attn_module.in_proj_bias is not None:
        attn_module.in_proj_bias.requires_grad = False
    return True


def inject_lora_into_attention(
    model: nn.Module,
    rank: int = 8,
    alpha: int = 16,
    dropout: float = 0.05,
    target_modules: Tuple[str, ...] = ("query", "value", "qkv", "in_proj"),
) -> int:
    """
    Walks the module tree and patches attention projections with LoRALinear.
    Supports three common attention implementations:
      1. Separate `query`/`key`/`value` nn.Linear attributes (custom impls).
      2. A fused `qkv` nn.Linear (timm/DINOv2-style ViT blocks).
      3. `nn.MultiheadAttention` with packed `in_proj_weight` (torchvision ViT).

    Returns the number of attention blocks patched.
    """
    patched_count = 0

    for name, module in model.named_modules():
        # Case 1: explicit query / value nn.Linear attributes.
        has_q = hasattr(module, "query") and isinstance(module.query, nn.Linear)
        has_v = hasattr(module, "value") and isinstance(module.value, nn.Linear)
        if has_q and has_v and "query" in target_modules:
            module.query = LoRALinear(module.query, rank, alpha, dropout)
            module.value = LoRALinear(module.value, rank, alpha, dropout)
            patched_count += 1
            continue

        # Case 2: fused qkv linear (DINOv2 / timm ViT block.attn.qkv).
        if hasattr(module, "qkv") and isinstance(module.qkv, nn.Linear) and "qkv" in target_modules:
            fused = module.qkv
            dim = fused.out_features // 3
            wrapped = _FusedQKVLoRA(fused, dim, rank=rank, alpha=alpha, dropout=dropout)
            module.qkv = wrapped
            patched_count += 1
            continue

        # Case 3: torchvision nn.MultiheadAttention packed in_proj.
        if isinstance(module, nn.MultiheadAttention) and "in_proj" in target_modules:
            if _split_qkv_and_wrap(module, rank, alpha, dropout):
                patched_count += 1

    logger.info("LoRA injected into %d attention blocks (rank=%d, alpha=%d)", patched_count, rank, alpha)
    return patched_count


class _FusedQKVLoRA(nn.Module):
    """
    Wraps a fused `qkv` nn.Linear (shape: in -> 3*dim) so that only the Q and
    V slices receive a LoRA update, while K remains frozen/base-only. Applied
    to timm/DINOv2-style ViT blocks that project Q, K, V in a single matmul.
    """

    def __init__(self, fused: nn.Linear, dim: int, rank: int, alpha: int, dropout: float) -> None:
        super().__init__()
        self.dim = dim
        self.base = fused
        for p in self.base.parameters():
            p.requires_grad = False

        in_features = fused.in_features
        self.scaling = alpha / float(rank)
        self.lora_A_q = nn.Parameter(torch.zeros(rank, in_features))
        self.lora_B_q = nn.Parameter(torch.zeros(dim, rank))
        self.lora_A_v = nn.Parameter(torch.zeros(rank, in_features))
        self.lora_B_v = nn.Parameter(torch.zeros(dim, rank))
        nn.init.kaiming_uniform_(self.lora_A_q, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.lora_A_v, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B_q)
        nn.init.zeros_(self.lora_B_v)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        qkv = self.base(x)
        q, k, v = qkv.split(self.dim, dim=-1)
        x_d = self.dropout(x)
        q = q + self.scaling * (x_d @ self.lora_A_q.T @ self.lora_B_q.T)
        v = v + self.scaling * (x_d @ self.lora_A_v.T @ self.lora_B_v.T)
        return torch.cat([q, k, v], dim=-1)


  
# Backbone factory
  
def build_model(
    model_name: str,
    model_cfg: Dict,
    adaptation_cfg: Optional[Dict] = None,
    pretrained: bool = True,
) -> Tuple[nn.Module, int, str]:
    """
    Returns (backbone_module, feature_dim, model_type).
    Optionally injects LoRA into attention Q/V projections per adaptation_cfg.
    """
    model_type = model_cfg["type"]

    if model_name == "resnet50":
        weights = torchvision.models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        backbone = torchvision.models.resnet50(weights=weights)
        backbone.fc = nn.Identity()
        feature_dim = model_cfg["feature_dim"]

    elif model_name == "vit_b_16":
        weights = torchvision.models.ViT_B_16_Weights.IMAGENET1K_SWAG_E2E_V1 if pretrained else None
        backbone = torchvision.models.vit_b_16(weights=weights)
        backbone.heads = nn.Identity()
        feature_dim = model_cfg["feature_dim"]

    elif model_name == "dinov2_vits14":
        try:
            backbone = torch.hub.load(
                model_cfg["hub_repo"], model_cfg["hub_entry"], pretrained=pretrained
            )
        except Exception as exc:  # pragma: no cover - network-dependent
            logger.warning(
                "torch.hub load for %s failed (%s). Falling back to randomly "
                "initialized ViT-S/14-equivalent placeholder.",
                model_name,
                exc,
            )
            backbone = torchvision.models.vit_b_16(weights=None)
            backbone.heads = nn.Identity()
        feature_dim = model_cfg["feature_dim"]

    else:
        raise ValueError(f"Unknown model name: {model_name}")

    if adaptation_cfg and adaptation_cfg.get("freeze_backbone", True):
        for p in backbone.parameters():
            p.requires_grad = False

    if adaptation_cfg and model_type == "transformer" and adaptation_cfg.get("method") == "lora":
        inject_lora_into_attention(
            backbone,
            rank=adaptation_cfg["lora_rank"],
            alpha=adaptation_cfg["lora_alpha"],
            dropout=adaptation_cfg.get("lora_dropout", 0.05),
            target_modules=tuple(adaptation_cfg.get("target_modules", ["query", "value"]))
            + ("qkv", "in_proj"),
        )

    return backbone, feature_dim, model_type


class LinearProbeHead(nn.Module):
    """Single trainable linear classification head atop frozen/LoRA features."""

    def __init__(self, in_features: int, num_classes: int = 2) -> None:
        super().__init__()
        self.fc = nn.Linear(in_features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


def get_trainable_parameter_groups(
    backbone: nn.Module, head: nn.Module, backbone_lr: float, head_lr: float
) -> List[Dict]:
    """Splits parameter groups: LoRA/backbone params at backbone_lr, head at head_lr."""
    backbone_params = [p for p in backbone.parameters() if p.requires_grad]
    head_params = [p for p in head.parameters() if p.requires_grad]
    groups = []
    if backbone_params:
        groups.append({"params": backbone_params, "lr": backbone_lr})
    if head_params:
        groups.append({"params": head_params, "lr": head_lr})
    return groups
