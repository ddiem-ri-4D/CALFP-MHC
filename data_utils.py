"""
data_utils.py
-------------
Data loading, validation, and inference utilities for CALFP-MHC.

Encoding strategy
-----------------
CALFP-MHC uses cheminformatics molecular fingerprints (MACCS, ECFP4,
ECFP6, RDKit; 4,263-dim per residue, see fingerprint_encoder.py) as the
network's actual input, per the manuscript's Methods.

This module converts each peptide/MHC string to a LongTensor of amino
acid indices (0..20).  The lookup from index → 4,263-dim fingerprint
(plus additive sinusoidal positional encoding and normalization) happens
inside the model itself (FingerprintResidueEncoder, shared between the
peptide and MHC branches), not here — this keeps the fingerprint table
and positional-encoding parameters bundled with the model checkpoint.

Input file format
-----------------
CSV (preferred), TSV, or Parquet with columns:
    peptide  — amino acid string, length 7–25, standard residues only
    allele   — HLA allele name matching an entry in HLA_library.csv

Output columns added
--------------------
    presentation_score  — softmax probability of MHC surface display
    affinity_score      — rescaled IC50 score  (if --BA True)
"""

import logging
import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# ── Constants ────────────────────────────────────────────────────────────────

VALID_RESIDUES = set('ACDEFGHIKLMNPQRSTVWY')
PADDING_TOKEN  = 'X'
PEP_MAX_LEN    = 25    # peptides padded to this length
MHC_PSEUDO_LEN = 34    # fixed MHC pseudo-sequence length

# Vocabulary: 20 standard amino acids + padding X
# Must match the token order in fingerprint_encoder._TOKENS exactly, since
# these indices are used to look up rows in the fingerprint table.
from fingerprint_encoder import AA_TO_IDX, FP_DIM

VOCAB_SIZE = len(AA_TO_IDX)   # 21 (20 aa + padding X)


# ── Logging ───────────────────────────────────────────────────────────────────

class RunLogger:
    """Minimal file logger for validation errors and run metadata."""

    def __init__(self, log_path: str):
        self._log = logging.getLogger(log_path)
        self._log.setLevel(logging.INFO)
        if not self._log.handlers:
            fh = logging.FileHandler(log_path, mode='w')
            fh.setFormatter(logging.Formatter(
                '%(asctime)s  [%(levelname)s]  %(message)s'))
            self._log.addHandler(fh)

    def info(self, msg: str):     self._log.info(msg)
    def critical(self, msg: str): self._log.critical(msg)


# ── Encoding helpers ──────────────────────────────────────────────────────────

def encode_sequence(seq: str, fixed_len: int) -> torch.Tensor:
    """
    Pad *seq* with 'X' to *fixed_len* and convert to amino acid index tensor.
    The model's FingerprintResidueEncoder looks these indices up against the
    4,263-dim fingerprint table (see fingerprint_encoder.py).

    Args:
        seq:       raw amino acid string
        fixed_len: target length (right-pad with X; truncate if longer)
    Returns:
        tensor: (fixed_len,)  LongTensor of indices in [0, 20]
    """
    padded = (seq + PADDING_TOKEN * fixed_len)[:fixed_len]
    return torch.tensor(
        [AA_TO_IDX.get(aa, AA_TO_IDX['X']) for aa in padded],
        dtype=torch.long,
    )


# ── Dataset ───────────────────────────────────────────────────────────────────

class PepMHCDataset(Dataset):
    """
    PyTorch Dataset for peptide–MHC pairs, encoded as amino acid index
    tensors (LongTensor). The model converts indices → 4,263-dim
    fingerprint vectors internally via FingerprintResidueEncoder.

    Storing indices (not the expanded fingerprints) keeps memory usage
    the same as the old one-hot version — expansion to 4,263 dims happens
    lazily per-batch inside the model's forward pass.
    """

    def __init__(self, peptides: list[str], mhc_seqs: list[str]):
        self.pep = torch.stack(
            [encode_sequence(p, PEP_MAX_LEN) for p in peptides])   # (N,25,21)
        self.mhc = torch.stack(
            [encode_sequence(m, MHC_PSEUDO_LEN) for m in mhc_seqs])# (N,34,21)

    def __len__(self) -> int:
        return len(self.pep)

    def __getitem__(self, idx: int):
        return self.pep[idx], self.mhc[idx]


# ── Input loading & validation ────────────────────────────────────────────────

def load_input(
    filepath: str,
    hla_library_path: str,
    log_path: str = 'error.log',
    batch_size: int = 128,
) -> tuple[pd.DataFrame, DataLoader]:
    """
    Load, validate, and return an input file as a DataFrame + DataLoader.

    Accepted formats: .csv, .tsv / .txt, .parquet
    Required columns: 'peptide', 'allele'

    Exits with a critical log message if validation fails.
    """
    log = RunLogger(log_path)
    ext = os.path.splitext(filepath)[1].lower()

    # -- Read file ----------------------------------------------------------
    try:
        if ext == '.parquet':
            df = pd.read_parquet(filepath)
            # Normalise legacy column names
            rename = {}
            for col in df.columns:
                cl = col.lower().strip()
                if cl in ('peptide', 'pep', 'sequence'):
                    rename[col] = 'peptide'
                elif cl in ('allele', 'allele name', 'hla', 'mhc'):
                    rename[col] = 'allele'
            df = df.rename(columns=rename)
        elif ext in ('.tsv', '.txt'):
            df = pd.read_csv(filepath, sep='\t')
        else:
            df = pd.read_csv(filepath)
    except Exception as exc:
        log.critical(f'Cannot read input file "{filepath}": {exc}')
        sys.exit(1)

    # -- Required columns ---------------------------------------------------
    missing = {'peptide', 'allele'} - set(df.columns)
    if missing:
        log.critical(
            f'Missing required columns: {missing}. '
            f'Found: {list(df.columns)}')
        sys.exit(1)

    df = df[['peptide', 'allele']].copy()
    df['peptide'] = df['peptide'].astype(str).str.strip()
    df['allele']  = df['allele'].astype(str).str.strip()

    # -- HLA library --------------------------------------------------------
    try:
        hla_df  = pd.read_csv(hla_library_path)
        hla_lib = dict(zip(hla_df['Allele Name'], hla_df['MHC pseudo-seq']))
    except Exception as exc:
        log.critical(f'Cannot read HLA library "{hla_library_path}": {exc}')
        sys.exit(1)

    unknown = set(df['allele']) - set(hla_lib)
    if unknown:
        log.critical(
            f'Unrecognised allele(s): {unknown}. '
            f'Check HLA_library.csv for the full supported list.')
        sys.exit(1)

    df['mhc_seq'] = df['allele'].map(hla_lib)

    # -- Peptide validation -------------------------------------------------
    for pep in df['peptide']:
        if not (7 <= len(pep) <= 25):
            log.critical(
                f'Peptide "{pep}" has length {len(pep)}; '
                f'accepted range is 7–25.')
            sys.exit(1)
        bad_chars = set(pep) - VALID_RESIDUES
        if bad_chars:
            log.critical(
                f'Peptide "{pep}" contains non-standard character(s): '
                f'{bad_chars}.')
            sys.exit(1)

    print(f'Input OK: {len(df)} peptide–MHC pair(s) loaded.')
    log.info(f'Loaded {len(df)} pairs from {filepath}')

    # -- DataLoader ---------------------------------------------------------
    dataset = PepMHCDataset(df['peptide'].tolist(), df['mhc_seq'].tolist())
    loader  = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    return df, loader


# ── Inference ─────────────────────────────────────────────────────────────────

@torch.no_grad()
def run_presentation_inference(
    net: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> np.ndarray:
    """
    Run a CALFP_PS model over *loader* and return presentation probabilities.

    Returns:
        np.ndarray, shape (N,), dtype float32, values in [0, 1]
    """
    net.eval()
    probs_list = []
    for pep_oh, mhc_oh in loader:
        pep_oh = pep_oh.to(device)
        mhc_oh = mhc_oh.to(device)
        logits = net(pep_oh, mhc_oh)                      # (B, 2)
        p = F.softmax(logits, dim=1)[:, 1]                # (B,)
        probs_list.append(p.cpu())
    return torch.cat(probs_list).numpy().astype(np.float32)


@torch.no_grad()
def run_affinity_inference(
    net: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> np.ndarray:
    """
    Run a CALFP_BA model over *loader* and return affinity scores.

    Returns:
        np.ndarray, shape (N,), dtype float32
    """
    net.eval()
    scores_list = []
    for pep_oh, mhc_oh in loader:
        pep_oh  = pep_oh.to(device)
        mhc_oh  = mhc_oh.to(device)
        scores  = net(pep_oh, mhc_oh)                     # (B,)
        scores_list.append(scores.cpu())
    return torch.cat(scores_list).numpy().astype(np.float32)
