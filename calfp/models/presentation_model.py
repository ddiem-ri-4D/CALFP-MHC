"""
presentation_model.py
---------------------
CALFP-MHC  —  Presentation Score Network (CALFP_PS)

Predicts the probability that a peptide is processed and displayed on the
MHC surface (eluted-ligand task, binary classification).

Architecture (Fig.1e "Shared Encoder (pre-trained) -> split heads"):
    CALFPEncoder (shared architecture with CALFP_BA -- see encoder.py)
      -> PresentationHead (EL head: 2-class logits -> softmax -> P(presented))

Hyperparameters (identical to CALFP_BA -- the manuscript's "Training
hyperparameters" section gives one set of numbers, not separate ones
per head)
----------------------------------------------------
fp_dim (raw)        4263   (fingerprint dim, projected down to model_dim)
model_dim (d_b)       256   (bottleneck dim)
conv_channels        3200
kernel_size              9
num_heads                 9
mlp_hidden             800
dropout                0.2
"""

import os as _os
import sys as _sys
_sys.path.insert(0, _os.path.abspath(_os.path.join(_os.path.dirname(__file__), '..', '..')))

import torch
import torch.nn as nn
from calfp.models.encoder import CALFPEncoder


class PresentationHead(nn.Module):
    """
    EL head (Fig.1e "EL Head: Binary CE loss -> P(presented)").
    Structure: Linear -> SiLU -> BN -> Dropout -> Linear -> ReLU -> Linear(2)
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
            nn.Linear(64, 2, bias=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.sequential(x)


# Backward-compatible alias (older code/scripts may still import this name)
ClassificationMLP = PresentationHead


class CALFP_PS(nn.Module):
    """
    CALFP-MHC Presentation Score model = shared CALFPEncoder + PresentationHead.

    encode() exposes the pooled encoder output (for Stage-1 SupCon
    pretraining via ContrastiveProjectionHead); forward() runs the full
    encoder + EL head pipeline for Stage-2 fine-tuning / inference.
    """

    def __init__(
        self,
        model_dim: int = 256,
        num_hiddens: int = 800,
        num_heads: int = 9,
        num_channels: int = 3200,
        depthwise_kernel_size: int = 9,
        dropout: float = 0.2,
        pep_max_len: int = 25,
        mhc_len: int = 34,
    ):
        super().__init__()
        self.model_dim = model_dim
        self.encoder = CALFPEncoder(
            model_dim=model_dim, num_heads=num_heads, num_channels=num_channels,
            depthwise_kernel_size=depthwise_kernel_size, dropout=dropout,
            pep_max_len=pep_max_len, mhc_len=mhc_len,
        )
        self.head = PresentationHead(
            seq_flat_dim=model_dim * pep_max_len,
            hidden_dim=num_hiddens, dropout=dropout,
        )

    def encode(self, pep: torch.Tensor, mhc: torch.Tensor) -> torch.Tensor:
        """Pooled encoder output g (B, model_dim) -- for Stage-1 SupCon pretraining."""
        pooled, _ = self.encoder(pep, mhc)
        return pooled

    def forward(self, pep: torch.Tensor, mhc: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pep: (B, pep_max_len)  LongTensor of amino acid indices
            mhc: (B, mhc_len)      LongTensor of amino acid indices
        Returns:
            logits: (B, 2)
        """
        _, flattened = self.encoder(pep, mhc)
        return self.head(flattened)


class ContrastiveProjectionHead(nn.Module):
    """
    Stage-1 SupCon projection head (Methods: "two-layer projection head
    ... 2-layer MLP -> L2 norm unit-norm embeddings on hypersphere",
    "two fully connected layers (800 -> 64) with SiLU activation").

    Shared design -- instantiate one per model (CALFP_PS / CALFP_BA both
    use this same class during their respective Stage-1 pretraining).
    """

    def __init__(self, in_dim: int = 256, hidden_dim: int = 800, out_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, g: torch.Tensor) -> torch.Tensor:
        z = self.net(g)
        return nn.functional.normalize(z, p=2, dim=-1)   # unit hypersphere
