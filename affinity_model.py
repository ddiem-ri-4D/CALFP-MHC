"""
affinity_model.py
-----------------
CALFP-MHC  —  Binding Affinity Network (CALFP_BA)

Predicts a continuous binding affinity score in [0, 1] (rescaled IC50:
1 − log(IC50) / log(50 000)) for peptide–MHC class I and class II pairs.

Pipeline
--------
Input tokens  (amino acid index tensors: peptide + MHC pseudo-sequence)
      ↓
FingerprintResidueEncoder  (index → 4,263-dim fingerprint + positional
                            encoding + normalization; same encoder class
                            as CALFP_PS, separate learned instance)
      ↓
ResidueInteractionBlock   (same design as CALFP_PS, smaller channel width)
      ↓
LayerNorm  +  residual
      ↓
MultiQueryAttentionBlock  (9 heads)
      ↓
LayerNorm  +  residual
      ↓
Flatten  →  RegressionMLP  →  scalar output

Key differences from CALFP_PS
------------------------------
- conv_channels = 1600  (vs 3200 for PS) — BA datasets are typically
  smaller than EL datasets; the lighter conv block reduces overfitting.
- mlp_hidden = 600  (vs 800 for PS)
- Output is a single scalar (not 2-class logits).

Hyperparameters
----------------------------------------------------
fp_dim          4263   (fingerprint dimension — replaces vocab_size=21)
sequence_length   59
conv_channels   1600
kernel_size        9
num_heads          9
mlp_hidden       600
dropout          0.2
"""

import math
import torch
import torch.nn as nn
from presentation_model import (
    ResidueInteractionBlock,
    MultiQueryAttentionBlock,
)
from fingerprint_encoder import FingerprintResidueEncoder, FP_DIM


class RegressionMLP(nn.Module):
    """
    Single-value regression head for binding affinity prediction.

    Structure: Linear → SiLU → BN → Dropout → Linear → ReLU → Linear(1)
    Output is a raw scalar; apply sigmoid at inference if a [0,1] value
    is needed for downstream scoring.
    """

    def __init__(self, seq_flat_dim: int, hidden_dim: int, dropout: float):
        super().__init__()
        self.sequential = nn.Sequential(
            nn.Linear(seq_flat_dim, hidden_dim, bias=True),
            nn.SiLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 64, bias=True),
            nn.ReLU(),
            nn.Linear(64, 1, bias=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.sequential(x).squeeze(-1)   # (B,)


class CALFP_BA(nn.Module):
    """
    CALFP-MHC Binding Affinity model.

    Accepts one-hot encoded peptide and MHC tensors and produces a
    continuous affinity score per peptide–MHC pair.  Training uses MSE
    loss on rescaled IC50 values.

    This class preserves the internal attribute names (norm, selfattention,
    conv, flatten, feature_selection) required to load pre-trained weight
    files without modification.
    """

    def __init__(
        self,
        model_dim: int = 256,        # bottleneck dim per Methods (d_b=256)
        num_hiddens: int = 600,
        num_heads: int = 9,
        num_step: int = 59,
        num_channels: int = 1600,
        depthwise_kernel_size: int = 9,
        dropout: float = 0.2,
        max_len: int = 34,
        bias: bool = False,
    ):
        super().__init__()
        self.model_dim = model_dim
        self.fp_encoder = FingerprintResidueEncoder(project_dim=model_dim, max_len=max_len)
        self.norm = nn.LayerNorm(model_dim)
        self.selfattention = MultiQueryAttentionBlock(
            model_dim, num_heads, dropout)
        self.conv = ResidueInteractionBlock(
            vocab_size=model_dim,
            conv_channels=num_channels,
            kernel_size=depthwise_kernel_size,
            dropout=dropout,
            use_group_norm=False,
        )
        self.flatten = nn.Flatten(start_dim=1, end_dim=-1)
        self.feature_selection = RegressionMLP(
            seq_flat_dim=model_dim * num_step,
            hidden_dim=num_hiddens,
            dropout=dropout,
        )

    def encode(self, pep: torch.Tensor, mhc: torch.Tensor) -> torch.Tensor:
        """Pooled encoder output — see CALFP_PS.encode() docstring."""
        pep_fp = self.fp_encoder(pep)
        mhc_fp = self.fp_encoder(mhc)
        x = torch.cat([pep_fp, mhc_fp], dim=1)
        residual = x
        x = self.conv(x)
        x = self.norm(residual + x)
        residual = x
        x = self.selfattention(x)
        x = self.norm(residual + x)
        return x.mean(dim=1)

    def forward(self, pep: torch.Tensor,
                mhc: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pep: (B, pep_len)  LongTensor of amino acid indices
            mhc: (B, 34)       LongTensor of amino acid indices
        Returns:
            affinity: (B,)  predicted rescaled IC50
        """
        pep_fp = self.fp_encoder(pep)
        mhc_fp = self.fp_encoder(mhc)
        x = torch.cat([pep_fp, mhc_fp], dim=1)
        residual = x
        x = self.conv(x)
        x = self.norm(residual + x)
        residual = x
        x = self.selfattention(x)
        x = self.norm(residual + x)
        x = self.flatten(x)
        return self.feature_selection(x)
