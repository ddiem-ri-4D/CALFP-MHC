"""
affinity_model.py
-----------------
CALFP-MHC Binding Affinity Model (CALFP_BA).

Predicts a continuous binding affinity score in [0, 1]
(rescaled IC50: 1 - log(IC50) / log(50000)) for peptide–MHC pairs,
for both HLA class I and class II.

Architecture overview
---------------------
Input
  ├─ Peptide tokens  (batch, pep_len)
  └─ MHC tokens      (batch, 34)
        ↓
FingerprintResidueEncoder     [same encoder as presentation model]
  → (batch, pep_len+34, fp_dim)
        ↓
DynamicInteractionConv        [slightly smaller channel width vs PS model]
  → (batch, pep_len+34, fp_dim)
        ↓
Bottleneck Transformer blocks
  → (batch, pep_len+34, fp_dim)
        ↓
Global average pooling        → (batch, fp_dim)
        ↓
AffinityHead (MLP)            → (batch,)   [single scalar, no sigmoid here]

Loss: MSE on rescaled IC50 values; apply sigmoid at inference if a
probability-like [0,1] output is needed.
"""

import torch
import torch.nn as nn
from fingerprint_encoder import FingerprintResidueEncoder, FP_DIM
from presentation_model import (
    DynamicInteractionConv,
    BottleneckTransformerBlock,
)


# ---------------------------------------------------------------------------
# Affinity-specific output head
# ---------------------------------------------------------------------------

class AffinityRegressionHead(nn.Module):
    """
    Single-value regression head for binding affinity prediction.

    Compared to PresentationHead:
      - Smaller hidden dimension (600 vs 800), reflecting lower label noise
        in IC50 measurements.
      - Outputs a single scalar (no softmax needed).
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
            nn.Linear(64, 1, bias=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x).squeeze(-1)   # (B,)


# ---------------------------------------------------------------------------
# Top-level model
# ---------------------------------------------------------------------------

class CALFP_BA(nn.Module):
    """
    CALFP-MHC Binding Affinity model.

    Shares the FingerprintResidueEncoder and BottleneckTransformerBlock
    building blocks with CALFP_PS, but uses:
      - Smaller expand_channels (1600 vs 3200) — BA data has fewer samples
        than EL data, so a lighter convolutional block reduces overfitting.
      - Smaller hidden_dim in the regression head (600 vs 800).
      - Single scalar output instead of 2-class logits.

    Args:
        fp_dim:          fingerprint dimension per residue (default 4263)
        num_heads:       MHSA heads (default 9)
        num_tf_blocks:   number of BottleneckTransformerBlocks (default 1)
        expand_channels: inner channels of DynamicInteractionConv (default 1600)
        kernel_size:     depthwise conv kernel (default 9, must be odd)
        hidden_dim:      MLP hidden units in AffinityRegressionHead (default 600)
        dropout:         dropout rate (default 0.2)
    """

    def __init__(
        self,
        fp_dim: int = FP_DIM,
        num_heads: int = 9,
        num_tf_blocks: int = 1,
        expand_channels: int = 1600,
        kernel_size: int = 9,
        hidden_dim: int = 600,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.encoder = FingerprintResidueEncoder()
        self.tf_blocks = nn.ModuleList([
            BottleneckTransformerBlock(
                fp_dim, num_heads, expand_channels, kernel_size, dropout)
            for _ in range(num_tf_blocks)
        ])
        self.head = AffinityRegressionHead(fp_dim, hidden_dim, dropout)

    def forward(self, pep_ids: torch.Tensor,
                mhc_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pep_ids: (B, pep_len)  — padded peptide token indices
            mhc_ids: (B, 34)       — MHC pseudo-seq token indices
        Returns:
            affinity: (B,)         — predicted rescaled IC50 in [0, 1]
        """
        pep_enc = self.encoder(pep_ids)
        mhc_enc = self.encoder(mhc_ids)
        x = torch.cat([pep_enc, mhc_enc], dim=1)
        for block in self.tf_blocks:
            x = block(x)
        x = x.mean(dim=1)
        return self.head(x)