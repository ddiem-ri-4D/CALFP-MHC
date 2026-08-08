"""
encoder.py
----------
Shared encoder backbone (Fig.1c–d), used identically by both CALFP_PS
(presentation) and CALFP_BA (affinity) — matches Fig.1e "Shared Encoder
(pre-trained) → split heads": one encoder architecture/hyperparameter set,
two independent task heads on top.

Pipeline (per residue-level fingerprint input)
------------------------------------------------
FingerprintResidueEncoder  → peptide_fp (B, 25, model_dim)
                            → mhc_fp     (B, 34, model_dim)
      ↓
CrossAttentionBlock  (Fig.1c: "Q = Peptide, K = V = MHC", scaled
                      dot-product cross-attention — NOT self-attention)
      ↓  → interaction features (B, 25, model_dim), same length as peptide
LayerNorm + residual (against the peptide query stream)
      ↓
ResidueInteractionBlock  (Fig.1c: "... followed by convolutional layers
                          to model element-wise physical/chemical
                          interactions")
      ↓
LayerNorm + residual
      ↓
MultiQueryAttentionBlock  (Fig.1d bottleneck transformer, self-attention,
                           9 heads — "Bottleneck Transformers / MHSA")
      ↓
LayerNorm + residual
      ↓
  ├─ mean-pool over sequence → g (B, model_dim)   [Stage-1 SupCon input]
  └─ flatten → (B, 25*model_dim)                  [Stage-2 task-head input]

Hyperparameters (identical for both CALFP_PS and CALFP_BA — the
manuscript's "Training hyperparameters" section states one set of
numbers and does not distinguish EL vs BA):
    model_dim (bottleneck)   256
    conv_channels           3200
    kernel_size                9
    num_heads                  9
    dropout                  0.2
"""

import math
import torch
import torch.nn as nn
from calfp.encoding.fingerprint_encoder import FingerprintResidueEncoder


# ── Attention primitives ────────────────────────────────────────────────────

class ScaledDotAttention(nn.Module):
    """Scaled dot-product attention with dropout."""

    def __init__(self, dropout: float):
        super().__init__()
        self.drop = nn.Dropout(dropout)

    def forward(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
        scale = math.sqrt(Q.shape[-1])
        scores = torch.bmm(Q, K.transpose(1, 2)) / scale
        weights = nn.functional.softmax(scores, dim=-1)
        self.attention = weights          # stored for interpretability (Fig.4f,i)
        return torch.bmm(self.drop(weights), V)


def _split_heads(X: torch.Tensor, h: int) -> torch.Tensor:
    B, L, _ = X.shape
    X = X.reshape(B, L, h, -1).permute(0, 2, 1, 3)
    return X.reshape(B * h, L, -1)


def _merge_heads(X: torch.Tensor, h: int) -> torch.Tensor:
    _, L, D = X.shape
    B = X.shape[0] // h
    X = X.reshape(B, h, L, D).permute(0, 2, 1, 3)
    return X.reshape(B, L, h * D)


class CrossAttentionBlock(nn.Module):
    """
    Fig.1c interaction layer: Q = Peptide, K = V = MHC.

    This is genuine scaled dot-product cross-attention (softmax(QK^T/sqrt(d))V)
    between two DIFFERENT sequences — not self-attention on a concatenated
    peptide+MHC sequence. Output has the peptide's sequence length (queries
    determine output length), matching "peptide fragment interacts with a
    kernel generated from the MHC pseudo-sequence" framing in Methods.
    """

    def __init__(self, dim: int, num_heads: int, dropout: float):
        super().__init__()
        self.num_heads = num_heads
        self.attention = ScaledDotAttention(dropout)
        self.wq = nn.Linear(dim, dim * num_heads, bias=False)
        self.kw = nn.Linear(dim, dim * num_heads, bias=False)
        self.wv = nn.Linear(dim, dim * num_heads, bias=False)
        self.wo = nn.Linear(dim * num_heads, dim, bias=False)

    def forward(self, q_src: torch.Tensor, kv_src: torch.Tensor) -> torch.Tensor:
        """
        Args:
            q_src:  (B, L_pep, dim)  — peptide representation (queries)
            kv_src: (B, L_mhc, dim)  — MHC representation (keys & values)
        Returns:
            (B, L_pep, dim)
        """
        Q = _split_heads(self.wq(q_src), self.num_heads)
        K = _split_heads(self.kw(kv_src), self.num_heads)
        V = _split_heads(self.wv(kv_src), self.num_heads)
        out = _merge_heads(self.attention(Q, K, V), self.num_heads)
        return self.wo(out)


class MultiQueryAttentionBlock(nn.Module):
    """
    Multi-head SELF-attention block (Fig.1d bottleneck transformer / MHSA).
    Unlike CrossAttentionBlock, Q/K/V all come from the same sequence.
    """

    def __init__(self, dim: int, num_heads: int, dropout: float):
        super().__init__()
        self.num_heads = num_heads
        self.attention = ScaledDotAttention(dropout)
        self.wq = nn.Linear(dim, dim * num_heads, bias=False)
        self.kw = nn.Linear(dim, dim * num_heads, bias=False)
        self.wv = nn.Linear(dim, dim * num_heads, bias=False)
        self.wo = nn.Linear(dim * num_heads, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        Q = _split_heads(self.wq(x), self.num_heads)
        K = _split_heads(self.kw(x), self.num_heads)
        V = _split_heads(self.wv(x), self.num_heads)
        out = _merge_heads(self.attention(Q, K, V), self.num_heads)
        return self.wo(out)


# ── Convolution block ────────────────────────────────────────────────────────

class ResidueInteractionBlock(nn.Module):
    """
    Depthwise-separable convolution block ("... followed by convolutional
    layers to model element-wise physical and chemical interactions",
    Fig.1c; ResNet-style refinement, Fig.1d).

    Expand (×2) → GLU gating → depthwise conv → BN → SiLU → compress → dropout.
    """

    def __init__(self, dim: int, conv_channels: int, kernel_size: int,
                 dropout: float, use_group_norm: bool = False):
        super().__init__()
        pad = (kernel_size - 1) // 2
        norm_layer = (
            nn.GroupNorm(num_groups=1, num_channels=conv_channels)
            if use_group_norm else nn.BatchNorm1d(conv_channels)
        )
        self.sequential = nn.Sequential(
            nn.Conv1d(dim, 2 * conv_channels, 1, stride=1, padding=0, bias=True),
            nn.GLU(dim=1),
            nn.Conv1d(conv_channels, conv_channels, kernel_size,
                      stride=1, padding=pad, groups=conv_channels, bias=True),
            norm_layer,
            nn.SiLU(),
            nn.Conv1d(conv_channels, dim, 1, stride=1, padding=0, bias=True),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L, D) — transpose to (B, D, L) for Conv1d, then back
        return self.sequential(x.transpose(1, 2)).transpose(1, 2)


# ── Shared encoder ───────────────────────────────────────────────────────────

class CALFPEncoder(nn.Module):
    """
    Shared encoder backbone (Fig.1e "Shared Encoder (pre-trained)").
    Used identically (same hyperparameters, same module structure) by both
    CALFP_PS and CALFP_BA, so a Stage-1-pretrained encoder checkpoint can
    in principle be loaded into either downstream head.

    Output sequence length equals the peptide length (PEP_MAX_LEN), since
    the interaction layer (CrossAttentionBlock) queries from the peptide
    stream and keys/values from the MHC stream.
    """

    def __init__(
        self,
        model_dim: int = 256,          # bottleneck dim (Methods: d_b=256)
        num_heads: int = 9,
        num_channels: int = 3200,      # matches Training hyperparameters (both heads)
        depthwise_kernel_size: int = 9,
        dropout: float = 0.2,
        pep_max_len: int = 25,
        mhc_len: int = 34,
    ):
        super().__init__()
        self.model_dim = model_dim
        self.pep_max_len = pep_max_len

        # Shared fingerprint+positional encoder for both peptide and MHC,
        # projecting FP_DIM(4263) -> model_dim(256) (Methods "Dimensionality
        # reduction": U0 = F*W_red).
        self.fp_encoder = FingerprintResidueEncoder(
            project_dim=model_dim, max_len=max(pep_max_len, mhc_len))

        # Fig.1c: Q=Peptide, K=V=MHC cross-attention interaction layer
        self.cross_attention = CrossAttentionBlock(model_dim, num_heads, dropout)
        self.norm_interact = nn.LayerNorm(model_dim)

        # Fig.1c/1d: convolutional refinement + bottleneck self-attention
        self.conv = ResidueInteractionBlock(
            dim=model_dim, conv_channels=num_channels,
            kernel_size=depthwise_kernel_size, dropout=dropout,
        )
        self.selfattention = MultiQueryAttentionBlock(model_dim, num_heads, dropout)
        self.norm = nn.LayerNorm(model_dim)   # reused after conv and after self-attn

    def forward(self, pep: torch.Tensor, mhc: torch.Tensor):
        """
        Args:
            pep: (B, pep_max_len)  LongTensor amino acid indices
            mhc: (B, mhc_len)      LongTensor amino acid indices
        Returns:
            pooled:   (B, model_dim)             — mean-pooled g, for SupCon
            flattened: (B, pep_max_len*model_dim) — for task heads
        """
        pep_fp = self.fp_encoder(pep)   # (B, 25, model_dim)
        mhc_fp = self.fp_encoder(mhc)   # (B, 34, model_dim)

        # Fig.1c: Q=Peptide, K=V=MHC
        x = self.cross_attention(pep_fp, mhc_fp)      # (B, 25, model_dim)
        x = self.norm_interact(pep_fp + x)             # residual vs. peptide query stream

        residual = x
        x = self.conv(x)
        x = self.norm(residual + x)

        residual = x
        x = self.selfattention(x)
        x = self.norm(residual + x)

        pooled = x.mean(dim=1)                          # (B, model_dim)
        flattened = x.reshape(x.shape[0], -1)            # (B, 25*model_dim)
        return pooled, flattened
