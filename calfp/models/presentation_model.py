"""
presentation_model.py
---------------------
CALFP-MHC  —  Presentation Score Network (CALFP_PS)

Predicts the probability that a peptide is processed and displayed on the
MHC surface (eluted-ligand task, binary classification).

Pipeline
--------
Input tokens  (amino acid index tensors: peptide + MHC pseudo-sequence)
      ↓
FingerprintResidueEncoder  (index → 4,263-dim MACCS+ECFP4+ECFP6+RDKit
                            fingerprint, + additive sinusoidal positional
                            encoding, + row-wise max normalization)
      ↓
ResidueInteractionBlock   (pointwise expand → GLU → depthwise conv →
                           BatchNorm → SiLU → pointwise compress → Dropout)
      ↓
LayerNorm  +  residual connection
      ↓
MultiQueryAttentionBlock  (scaled dot-product, 9 heads)
      ↓
LayerNorm  +  residual connection
      ↓
Flatten  →  ClassificationMLP  →  2-dim logits
      ↓
softmax[:, 1]  =  presentation probability

Hyperparameters
----------------------------------------------------
fp_dim             4263   (fingerprint dimension)
sequence_length       59   (25 peptide + 34 MHC pseudo-seq)
conv_channels       3200
kernel_size            9
num_heads               9
mlp_hidden            800
dropout               0.2
"""

import math
import torch
import torch.nn as nn
from calfp.encoding.fingerprint_encoder import FingerprintResidueEncoder, FP_DIM


# ── Building blocks ──────────────────────────────────────────────────────────

class ResidueInteractionBlock(nn.Module):
    """
    Depthwise-separable convolution block for local residue-residue
    interaction modelling along the concatenated peptide–MHC sequence.

    Expand (×2) → GLU gating → depthwise conv → BN → SiLU
    → compress → dropout.

    The GLU halves the channel count after expansion, so the residual
    connection can be added without a projection.
    """

    def __init__(
        self,
        vocab_size: int,
        conv_channels: int,
        kernel_size: int,
        dropout: float,
        use_group_norm: bool = False,
    ):
        super().__init__()
        pad = (kernel_size - 1) // 2
        norm_layer = (
            nn.GroupNorm(num_groups=1, num_channels=conv_channels)
            if use_group_norm
            else nn.BatchNorm1d(conv_channels)
        )
        self.sequential = nn.Sequential(
            # pointwise: vocab_size → 2*conv_channels
            nn.Conv1d(vocab_size, 2 * conv_channels, 1,
                      stride=1, padding=0, bias=True),
            nn.GLU(dim=1),
            # depthwise: conv_channels → conv_channels
            nn.Conv1d(conv_channels, conv_channels, kernel_size,
                      stride=1, padding=pad,
                      groups=conv_channels, bias=True),
            norm_layer,
            nn.SiLU(),
            # pointwise compress: conv_channels → vocab_size
            nn.Conv1d(conv_channels, vocab_size, 1,
                      stride=1, padding=0, bias=True),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L, D) — transpose to (B, D, L) for Conv1d, then back
        return self.sequential(x.transpose(1, 2)).transpose(1, 2)


class ScaledDotAttention(nn.Module):
    """Scaled dot-product attention with dropout."""

    def __init__(self, dropout: float):
        super().__init__()
        self.drop = nn.Dropout(dropout)

    def forward(self, Q: torch.Tensor,
                K: torch.Tensor,
                V: torch.Tensor) -> torch.Tensor:
        scale = math.sqrt(Q.shape[-1])
        scores = torch.bmm(Q, K.transpose(1, 2)) / scale
        weights = nn.functional.softmax(scores, dim=-1)
        self.attention = weights          # stored for interpretability
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


class MultiQueryAttentionBlock(nn.Module):
    """
    Multi-head self-attention block.

    Layer names (wq, kw, wv, wo) are preserved from the original CALFP
    training to ensure pre-trained weight compatibility.
    """

    def __init__(self, vocab_size: int, num_heads: int, dropout: float):
        super().__init__()
        self.num_heads = num_heads
        self.attention = ScaledDotAttention(dropout)
        # Layer names must match saved state_dict keys exactly
        self.wq = nn.Linear(vocab_size, vocab_size * num_heads, bias=False)
        self.kw = nn.Linear(vocab_size, vocab_size * num_heads, bias=False)
        self.wv = nn.Linear(vocab_size, vocab_size * num_heads, bias=False)
        self.wo = nn.Linear(vocab_size * num_heads, vocab_size,  bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        Q = _split_heads(self.wq(x), self.num_heads)
        K = _split_heads(self.kw(x), self.num_heads)
        V = _split_heads(self.wv(x), self.num_heads)
        out = _merge_heads(self.attention(Q, K, V), self.num_heads)
        return self.wo(out)


class ClassificationMLP(nn.Module):
    """
    Feature selection MLP: flattened sequence → 2-class logits.

    Structure: Linear → SiLU → BN → Dropout → Linear → ReLU → Linear(2)
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


# ── Top-level model ───────────────────────────────────────────────────────────

class CALFP_PS(nn.Module):
    """
    CALFP-MHC Presentation Score model.

    Accepts one-hot encoded peptide and MHC tensors (shapes: B×pep_len×21
    and B×34×21 respectively), processes them through a residue interaction
    convolution block and a multi-query attention block, then produces
    2-dimensional logits for binary cross-entropy training.

    Inference: apply softmax and take index-1 probability as the
    presentation score.  Threshold > 0.5 indicates likely presentation.

    This class preserves the internal layer names (norm, selfattention,
    conv, flatten, feature_selection) required to load pre-trained weight
    files without modification.
    """

    def __init__(
        self,
        model_dim: int = 256,        # bottleneck dim per Methods (d_b=256)
        num_hiddens: int = 800,
        num_heads: int = 9,
        num_step: int = 59,          # seq_len = 25 (pep) + 34 (MHC)
        num_channels: int = 3200,
        depthwise_kernel_size: int = 9,
        dropout: float = 0.2,
        max_len: int = 34,           # longest of peptide(<=25)/MHC(34) branch
        bias: bool = False,
    ):
        super().__init__()
        self.model_dim = model_dim
        # Shared fingerprint+positional encoder for both peptide and MHC,
        # with a linear projection FP_DIM(4263) -> model_dim(256), matching
        # the paper's "Dimensionality reduction" step (U0 = F*W_red).
        self.fp_encoder = FingerprintResidueEncoder(project_dim=model_dim, max_len=max_len)
        # Must use these attribute names to match saved state_dict keys
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
        self.feature_selection = ClassificationMLP(
            seq_flat_dim=model_dim * num_step,
            hidden_dim=num_hiddens,
            dropout=dropout,
        )

    def encode(self, pep: torch.Tensor, mhc: torch.Tensor) -> torch.Tensor:
        """
        Backbone up through the encoder, average-pooled across sequence
        positions (Methods: "average pooling across sequence positions
        produces a global representation g"). Used for Stage-1 SupCon
        pretraining via ContrastiveProjectionHead; NOT used by forward()
        (forward() flattens instead, matching the existing fine-tuning
        head design already in this file).

        Returns:
            g: (B, model_dim)
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
        return x.mean(dim=1)   # (B, model_dim)

    def forward(self, pep: torch.Tensor,
                mhc: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pep: (B, pep_len)  LongTensor of amino acid indices
            mhc: (B, 34)       LongTensor of amino acid indices
        Returns:
            logits: (B, 2)
        """
        pep_fp = self.fp_encoder(pep)   # (B, pep_len, 4263)
        mhc_fp = self.fp_encoder(mhc)   # (B, 34, 4263)
        x = torch.cat([pep_fp, mhc_fp], dim=1)   # (B, 59, 4263)
        residual = x
        x = self.conv(x)
        x = self.norm(residual + x)
        residual = x
        x = self.selfattention(x)
        x = self.norm(residual + x)
        x = self.flatten(x)
        return self.feature_selection(x)


class ContrastiveProjectionHead(nn.Module):
    """
    Stage-1 SupCon projection head (Methods: "two-layer projection head
    ... 2-layer MLP → L2 norm unit-norm embeddings on hypersphere",
    "two fully connected layers (800 → 64) with SiLU activation").

    Takes the pooled encoder output (model_dim, default 256) and projects
    to a 64-dim unit-norm embedding for the SupCon loss. Discarded after
    Stage 1 — Stage 2 fine-tunes the existing classification/regression
    head on top of the (frozen or unfrozen, per your training script)
    encoder instead.
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
