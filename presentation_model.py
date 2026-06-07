"""
presentation_model.py
---------------------
CALFP-MHC Presentation Score Model (CALFP_PS).

Predicts the probability that a peptide is eluted and presented on the
cell surface (MHC class I or II), given the peptide sequence and the
MHC pseudo-sequence.

Architecture overview
---------------------
Input
  ├─ Peptide tokens  (batch, pep_len)
  └─ MHC tokens      (batch, 34)
        ↓
FingerprintResidueEncoder          [novel: cheminformatics fingerprints]
  → (batch, pep_len+34, fp_dim)
        ↓
DynamicInteractionConv             [MHC-conditioned depthwise conv block]
  → (batch, pep_len+34, fp_dim)
        ↓
Bottleneck Transformer blocks      [Conv-Norm → MHSA-Norm with residuals]
  → (batch, pep_len+34, fp_dim)
        ↓
Global average pooling             → (batch, fp_dim)
        ↓
PresentationHead (MLP)             → (batch, 2)   [logits for CE loss]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from fingerprint_encoder import FingerprintResidueEncoder, FP_DIM


# ---------------------------------------------------------------------------
# Sub-modules
# ---------------------------------------------------------------------------

class DynamicInteractionConv(nn.Module):
    """
    Depthwise-separable convolution block that models local physicochemical
    interactions along the concatenated peptide–MHC sequence.

    Structure:
        Pointwise expand (×2) → GLU → Depthwise conv → BN → SiLU
        → Pointwise compress → Dropout
    """

    def __init__(self, in_dim: int, expand_channels: int,
                 kernel_size: int = 9, dropout: float = 0.2):
        super().__init__()
        assert (kernel_size - 1) % 2 == 0, "kernel_size must be odd"
        pad = (kernel_size - 1) // 2
        self.net = nn.Sequential(
            # Pointwise expand
            nn.Conv1d(in_dim, 2 * expand_channels, 1, bias=True),
            nn.GLU(dim=1),
            # Depthwise conv over sequence positions
            nn.Conv1d(expand_channels, expand_channels, kernel_size,
                      padding=pad, groups=expand_channels, bias=True),
            nn.BatchNorm1d(expand_channels),
            nn.SiLU(),
            # Pointwise compress back to in_dim
            nn.Conv1d(expand_channels, in_dim, 1, bias=True),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L, D) → conv expects (B, D, L)
        return self.net(x.transpose(1, 2)).transpose(1, 2)


class ScaledDotProductAttention(nn.Module):
    """Standard scaled dot-product attention with dropout."""

    def __init__(self, dropout: float = 0.0):
        super().__init__()
        self.drop = nn.Dropout(dropout)

    def forward(self, Q: torch.Tensor,
                K: torch.Tensor,
                V: torch.Tensor) -> torch.Tensor:
        scale = Q.shape[-1] ** 0.5
        attn = F.softmax(torch.bmm(Q, K.transpose(1, 2)) / scale, dim=-1)
        return torch.bmm(self.drop(attn), V)


class MultiHeadSelfAttention(nn.Module):
    """
    Multi-head self-attention.  Query, key, and value projections are
    computed jointly from the same input tensor.
    """

    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.h = num_heads
        self.attn = ScaledDotProductAttention(dropout)
        self.Wq = nn.Linear(embed_dim, embed_dim * num_heads, bias=False)
        self.Wk = nn.Linear(embed_dim, embed_dim * num_heads, bias=False)
        self.Wv = nn.Linear(embed_dim, embed_dim * num_heads, bias=False)
        self.Wo = nn.Linear(embed_dim * num_heads, embed_dim, bias=False)

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        B, L, _ = x.shape
        x = x.reshape(B, L, self.h, -1).permute(0, 2, 1, 3)
        return x.reshape(B * self.h, L, -1)

    def _merge_heads(self, x: torch.Tensor, B: int) -> torch.Tensor:
        _, L, D = x.shape
        x = x.reshape(B, self.h, L, D).permute(0, 2, 1, 3)
        return x.reshape(B, L, -1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        Q = self._split_heads(self.Wq(x))
        K = self._split_heads(self.Wk(x))
        V = self._split_heads(self.Wv(x))
        out = self._merge_heads(self.attn(Q, K, V), B)
        return self.Wo(out)


class BottleneckTransformerBlock(nn.Module):
    """
    A single bottleneck transformer block:
        ConvModule → LayerNorm (pre-norm residual)
        MHSA       → LayerNorm (pre-norm residual)
    """

    def __init__(self, embed_dim: int, num_heads: int,
                 expand_channels: int, kernel_size: int = 9,
                 dropout: float = 0.2):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.conv_block = DynamicInteractionConv(
            embed_dim, expand_channels, kernel_size, dropout)
        self.mhsa = MultiHeadSelfAttention(embed_dim, num_heads, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Conv sub-layer with pre-norm residual
        x = self.norm1(x + self.conv_block(x))
        # Attention sub-layer with pre-norm residual
        x = self.norm2(x + self.mhsa(x))
        return x


class PresentationHead(nn.Module):
    """
    Two-class output head: produces logits for binary cross-entropy loss.
    Presentation probability = softmax(logits)[:, 1].
    """

    def __init__(self, in_dim: int, hidden_dim: int, dropout: float = 0.2):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim, bias=True),
            nn.SiLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 64, bias=True),
            nn.ReLU(),
            nn.Linear(64, 2, bias=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)


# ---------------------------------------------------------------------------
# Top-level model
# ---------------------------------------------------------------------------

class CALFP_PS(nn.Module):
    """
    CALFP-MHC Presentation Score model.

    Outputs raw logits (2-dim); apply softmax[:, 1] to get presentation
    probability during inference.

    Args:
        fp_dim:          fingerprint dimension per residue (default 4263)
        num_heads:       MHSA heads (default 9)
        num_tf_blocks:   number of BottleneckTransformerBlocks (default 1)
        expand_channels: inner channels of DynamicInteractionConv (default 3200)
        kernel_size:     depthwise conv kernel (default 9, must be odd)
        hidden_dim:      MLP hidden units in PresentationHead (default 800)
        dropout:         dropout rate (default 0.2)
        seq_len:         total sequence length = pep_max_len + mhc_len
                         (default 59 = 25 + 34)
    """

    def __init__(
        self,
        fp_dim: int = FP_DIM,
        num_heads: int = 9,
        num_tf_blocks: int = 1,
        expand_channels: int = 3200,
        kernel_size: int = 9,
        hidden_dim: int = 800,
        dropout: float = 0.2,
        seq_len: int = 59,        # 25 (pep padded) + 34 (MHC pseudo-seq)
    ):
        super().__init__()
        self.encoder = FingerprintResidueEncoder()   # maps tokens → fp vectors
        self.tf_blocks = nn.ModuleList([
            BottleneckTransformerBlock(
                fp_dim, num_heads, expand_channels, kernel_size, dropout)
            for _ in range(num_tf_blocks)
        ])
        self.head = PresentationHead(fp_dim, hidden_dim, dropout)

    def forward(self, pep_ids: torch.Tensor,
                mhc_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pep_ids: (B, pep_len)  — padded peptide token indices
            mhc_ids: (B, 34)       — MHC pseudo-seq token indices
        Returns:
            logits:  (B, 2)
        """
        # Encode both sequences to fingerprint space
        pep_enc = self.encoder(pep_ids)   # (B, pep_len, fp_dim)
        mhc_enc = self.encoder(mhc_ids)   # (B, 34, fp_dim)
        # Concatenate along sequence dimension
        x = torch.cat([pep_enc, mhc_enc], dim=1)   # (B, L, fp_dim)
        # Transformer blocks
        for block in self.tf_blocks:
            x = block(x)
        # Global average pooling over sequence → (B, fp_dim)
        x = x.mean(dim=1)
        return self.head(x)