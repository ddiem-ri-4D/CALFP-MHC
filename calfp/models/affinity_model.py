"""
affinity_model.py
------------------
CALFP-MHC  --  Binding Affinity Network (CALFP_BA)

Predicts a continuous binding-affinity score (rescaled IC50 / %Rank).

Architecture (Fig.1e "Shared Encoder (pre-trained) -> split heads"):
    CALFPEncoder (SAME architecture and hyperparameters as CALFP_PS --
                  see encoder.py; the manuscript's "Training
                  hyperparameters" section states one set of numbers
                  and does not distinguish EL vs BA encoders)
      -> AffinityHead (BA head: MSE + Pearson loss -> IC50 & %Rank)

Hyperparameters (identical to CALFP_PS)
----------------------------------------------------
model_dim (d_b)       256
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


class AffinityHead(nn.Module):
    """
    BA head (Fig.1e "BA Head: MSE + Pearson loss -> IC50 & %Rank").
    Structure: Linear -> SiLU -> BN -> Dropout -> Linear -> ReLU -> Linear(1)

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


# Backward-compatible alias
RegressionMLP = AffinityHead


class CALFP_BA(nn.Module):
    """
    CALFP-MHC Binding Affinity model = shared CALFPEncoder + AffinityHead.

    Uses the SAME CALFPEncoder hyperparameters as CALFP_PS (model_dim=256,
    conv_channels=3200, num_heads=9, mlp_hidden=800) -- previously this
    model used a lighter 1600/600 configuration with no basis in the
    manuscript; that discrepancy has been removed.
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
        self.head = AffinityHead(
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
            affinity: (B,)  predicted rescaled IC50
        """
        _, flattened = self.encoder(pep, mhc)
        return self.head(flattened)
