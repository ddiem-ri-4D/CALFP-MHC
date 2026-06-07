"""
fingerprint_encoder.py
----------------------
Amino-acid-level cheminformatics fingerprint encoding for CALFP-MHC.

Each of the 20 standard amino acids (plus padding token X) is represented as a
concatenated vector of four molecular fingerprint descriptors computed from
its side-chain SMILES:
    - MACCS keys          (166-bit)
    - ECFP radius-2       (1024-bit, Morgan r=2)
    - ECFP radius-3       (2048-bit, Morgan r=3)
    - RDKit path          (1024-bit)

Concatenated fingerprint dimension per residue: 166 + 1024 + 2048 + 1024 = 4263

The padding token X receives an all-zero fingerprint vector.

Trigonometric positional modulation T(p) is applied element-wise:
    T(p) = 0.5 * (1 + sin(2*pi*p / L + phi))
where p is the residue position, L is the sequence length, and phi is a
learned phase offset initialised to zero.  This scales each fingerprint
dimension by a smooth sinusoidal envelope that varies with position,
providing a lightweight positional signal without altering the fingerprint
dimensionality.

Reference for the fingerprint repurposing idea:
    Lee & Min, AmorProt, Biochemistry 2023.
    Adamczyk et al., Bioinformatics 2025.
"""

import math
import torch
import torch.nn as nn
import numpy as np

# ---------------------------------------------------------------------------
# Precomputed fingerprint table (shape: 21 x 4263)
# Generated once at import time via RDKit; stored as a float32 numpy array.
# ---------------------------------------------------------------------------

# Side-chain SMILES for the 20 standard amino acids.
# Source: PubChem / standard biochemistry references.
_AA_SMILES = {
    'A': 'CC(N)C(=O)O',
    'R': 'NC(CCCNC(=N)N)C(=O)O',
    'N': 'NC(CC(=O)N)C(=O)O',
    'D': 'NC(CC(=O)O)C(=O)O',
    'C': 'NC(CS)C(=O)O',
    'E': 'NC(CCC(=O)O)C(=O)O',
    'Q': 'NC(CCC(=O)N)C(=O)O',
    'G': 'NCC(=O)O',
    'H': 'NC(Cc1cnc[nH]1)C(=O)O',
    'I': 'CC[C@H](C)[C@@H](N)C(=O)O',
    'L': 'CC(C)C[C@@H](N)C(=O)O',
    'K': 'NCCCCC(N)C(=O)O',
    'M': 'CSCCC(N)C(=O)O',
    'F': 'NC(Cc1ccccc1)C(=O)O',
    'P': 'OC(=O)[C@@H]1CCCN1',
    'S': 'NC(CO)C(=O)O',
    'T': 'CC(O)[C@@H](N)C(=O)O',
    'W': 'NC(Cc1c[nH]c2ccccc12)C(=O)O',
    'Y': 'NC(Cc1ccc(O)cc1)C(=O)O',
    'V': 'CC(C)[C@@H](N)C(=O)O',
    'X': None,   # padding token → zero vector
}

# Ordered token list (must match aa_to_idx below)
_TOKENS = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L',
           'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y', 'X']

AA_TO_IDX = {aa: i for i, aa in enumerate(_TOKENS)}

FP_DIM = 166 + 1024 + 2048 + 1024   # = 4263


def _build_fingerprint_table() -> np.ndarray:
    """
    Build the (21, 4263) fingerprint lookup table.
    Falls back to a random table if RDKit is not installed, so that the
    module can be imported without RDKit for unit testing purposes.
    """
    try:
        from rdkit import Chem
        from rdkit.Chem import MACCSkeys, AllChem, RDKFingerprint
        from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator

        mg2 = GetMorganGenerator(radius=2, fpSize=1024)
        mg3 = GetMorganGenerator(radius=3, fpSize=2048)

        table = np.zeros((21, FP_DIM), dtype=np.float32)
        for aa, smi in _AA_SMILES.items():
            if smi is None:
                continue   # X → zero row
            idx = AA_TO_IDX[aa]
            mol = Chem.MolFromSmiles(smi)
            maccs = np.array(MACCSkeys.GenMACCSKeys(mol), dtype=np.float32)
            ecfp4 = np.array(mg2.GetFingerprintAsNumPy(mol), dtype=np.float32)
            ecfp6 = np.array(mg3.GetFingerprintAsNumPy(mol), dtype=np.float32)
            rdk  = np.array(RDKFingerprint(mol, fpSize=1024), dtype=np.float32)
            table[idx] = np.concatenate([maccs, ecfp4, ecfp6, rdk])
        return table

    except ImportError:
        # RDKit unavailable: use a fixed random seed table for reproducibility
        rng = np.random.default_rng(seed=42)
        table = rng.random((21, FP_DIM), dtype=np.float32)
        table[-1] = 0.0   # padding row stays zero
        return table


# Module-level constant: built once, shared across all instances
_FP_TABLE = torch.from_numpy(_build_fingerprint_table())   # (21, 4263)


# ---------------------------------------------------------------------------
# Positional modulation
# ---------------------------------------------------------------------------

class TrigonometricPositionalModulation(nn.Module):
    """
    Applies a learned sinusoidal amplitude envelope to each sequence position.

    For position p in a sequence of length L:
        T(p) = 0.5 * (1 + sin(2*pi*p/L + phi))

    phi is a scalar learned parameter (initialised to 0).  T(p) is broadcast
    across the fingerprint dimension, scaling all bits uniformly per position.

    This is a deliberate lightweight design: the fingerprint already encodes
    chemical identity; positional modulation only needs to break symmetry
    between identical residues at different positions.
    """

    def __init__(self):
        super().__init__()
        self.phi = nn.Parameter(torch.zeros(1))   # learnable phase offset

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, fp_dim)
        Returns:
            x_modulated: (batch, seq_len, fp_dim)
        """
        B, L, D = x.shape
        positions = torch.arange(L, dtype=torch.float32, device=x.device)
        # T shape: (1, L, 1) → broadcasts over batch and fp_dim
        T = 0.5 * (1.0 + torch.sin(
            2.0 * math.pi * positions / max(L, 1) + self.phi
        )).view(1, L, 1)
        return x * T


# ---------------------------------------------------------------------------
# Main encoder
# ---------------------------------------------------------------------------

class FingerprintResidueEncoder(nn.Module):
    """
    Converts a batch of integer-encoded sequences to fingerprint embeddings
    with trigonometric positional modulation.

    Input:  (batch, seq_len)  — integer token indices in [0, 20]
    Output: (batch, seq_len, FP_DIM)
    """

    def __init__(self, project_dim: int = 0):
        """
        Args:
            project_dim: if > 0, add a linear projection from FP_DIM to
                         project_dim after modulation (reduces memory for
                         very deep stacks).  Default 0 = no projection.
        """
        super().__init__()
        # Non-trainable lookup table
        self.register_buffer('fp_table', _FP_TABLE)   # (21, 4263)
        self.positional_mod = TrigonometricPositionalModulation()
        self.project = (
            nn.Linear(FP_DIM, project_dim, bias=False)
            if project_dim > 0 else nn.Identity()
        )
        self.out_dim = project_dim if project_dim > 0 else FP_DIM

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            token_ids: (batch, seq_len)  LongTensor
        Returns:
            encoded:   (batch, seq_len, out_dim)
        """
        # Lookup: (batch, seq_len, 4263)
        x = self.fp_table[token_ids]
        # Positional modulation
        x = self.positional_mod(x)
        # Optional projection
        x = self.project(x)
        return x