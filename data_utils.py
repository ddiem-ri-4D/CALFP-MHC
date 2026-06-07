"""
data_utils.py
-------------
Data loading, validation, and inference utilities for CALFP-MHC.

Key differences from one-hot-based tools (e.g. CapHLA):
  - Sequences are tokenised to integer indices (not one-hot vectors).
    The FingerprintResidueEncoder performs the lookup to fingerprint space
    inside the model forward pass, keeping the DataLoader lightweight.
  - Input format: CSV with columns 'peptide' and 'allele'
    (tab-separated and parquet are also supported).
  - Peptide validation: length 7–25, standard amino acids only.
  - MHC pseudo-sequences are looked up from HLA_library.csv (34 aa each).
"""

import logging
import sys
import os
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from fingerprint_encoder import AA_TO_IDX

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

VALID_AAS = set('ACDEFGHIKLMNPQRSTVWY')   # 20 standard amino acids
PAD_TOKEN = 'X'
PEP_MAX_LEN = 25   # peptides padded / truncated to this length
MHC_LEN = 34       # fixed pseudo-sequence length

# ---------------------------------------------------------------------------
# Logging helper
# ---------------------------------------------------------------------------

class RunLogger:
    """Thin wrapper around Python's logging module for error reporting."""

    def __init__(self, log_path: str, level: int = logging.INFO):
        self.logger = logging.getLogger(log_path)
        self.logger.setLevel(level)
        if not self.logger.handlers:
            fh = logging.FileHandler(log_path, mode='w')
            fh.setFormatter(logging.Formatter('%(asctime)s  %(levelname)s  %(message)s'))
            self.logger.addHandler(fh)

    def critical(self, msg: str):
        self.logger.critical(msg)

    def info(self, msg: str):
        self.logger.info(msg)


# ---------------------------------------------------------------------------
# Tokenisation
# ---------------------------------------------------------------------------

def tokenise_sequence(seq: str, fixed_len: int) -> list[int]:
    """
    Pad a sequence with 'X' to fixed_len and convert each character to its
    integer index in AA_TO_IDX.

    Args:
        seq:       amino acid string (no spaces)
        fixed_len: target length (pad with X if shorter, truncate if longer)
    Returns:
        List of integer token indices, length == fixed_len
    """
    padded = (seq + PAD_TOKEN * fixed_len)[:fixed_len]
    return [AA_TO_IDX.get(aa, AA_TO_IDX['X']) for aa in padded]


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class PepMHCDataset(Dataset):
    """
    PyTorch Dataset for peptide–MHC pairs.

    Stores pre-tokenised integer tensors (not one-hot matrices), keeping
    memory usage proportional to sequence count rather than sequence length
    × vocabulary size.
    """

    def __init__(self, peptides: list[str], mhc_seqs: list[str]):
        """
        Args:
            peptides:  list of raw peptide strings
            mhc_seqs:  list of 34-aa MHC pseudo-sequences (one per peptide)
        """
        pep_tokens = [tokenise_sequence(p, PEP_MAX_LEN) for p in peptides]
        mhc_tokens = [tokenise_sequence(m, MHC_LEN)    for m in mhc_seqs]
        self.pep = torch.tensor(pep_tokens, dtype=torch.long)   # (N, 25)
        self.mhc = torch.tensor(mhc_tokens, dtype=torch.long)   # (N, 34)

    def __len__(self) -> int:
        return len(self.pep)

    def __getitem__(self, idx: int):
        return self.pep[idx], self.mhc[idx]


# ---------------------------------------------------------------------------
# Input loading and validation
# ---------------------------------------------------------------------------

def load_input(
    filepath: str,
    hla_library_path: str,
    log_path: str = 'error.log',
) -> tuple[pd.DataFrame, DataLoader]:
    """
    Load and validate an input file, returning a DataFrame and a DataLoader.

    Supported formats: .csv, .tsv, .parquet
    Required columns: 'peptide', 'allele'

    Args:
        filepath:         path to the input file
        hla_library_path: path to HLA_library.csv
        log_path:         path for the error log
    Returns:
        (validated_df, data_loader)
    """
    log = RunLogger(log_path)

    # -- Load input file ----------------------------------------------------
    ext = os.path.splitext(filepath)[1].lower()
    try:
        if ext == '.parquet':
            df = pd.read_parquet(filepath)
            # Normalise column names from legacy parquet format
            col_map = {}
            for c in df.columns:
                if c.lower() in ('peptide', 'pep'):
                    col_map[c] = 'peptide'
                elif c.lower() in ('allele', 'hla', 'allele name'):
                    col_map[c] = 'allele'
            df = df.rename(columns=col_map)
        elif ext in ('.tsv', '.txt'):
            df = pd.read_csv(filepath, sep='\t')
        else:
            df = pd.read_csv(filepath)
    except Exception as exc:
        log.critical(f'Failed to read input file: {exc}')
        sys.exit(1)

    # -- Column validation --------------------------------------------------
    required = {'peptide', 'allele'}
    missing = required - set(df.columns)
    if missing:
        log.critical(
            f"Input file is missing required columns: {missing}. "
            f"Found columns: {list(df.columns)}"
        )
        sys.exit(1)

    df = df[['peptide', 'allele']].copy()
    df['peptide'] = df['peptide'].astype(str).str.strip()
    df['allele']  = df['allele'].astype(str).str.strip()

    # -- HLA library lookup ------------------------------------------------
    hla_df = pd.read_csv(hla_library_path)
    hla_lib: dict[str, str] = dict(
        zip(hla_df['Allele Name'], hla_df['MHC pseudo-seq'])
    )
    unknown_alleles = set(df['allele']) - set(hla_lib.keys())
    if unknown_alleles:
        log.critical(
            f"Unknown HLA allele(s): {unknown_alleles}. "
            f"Check HLA_library.csv for the full list of supported alleles."
        )
        sys.exit(1)
    df['mhc_seq'] = df['allele'].map(hla_lib)

    # -- Peptide validation ------------------------------------------------
    for _, row in df.iterrows():
        pep = row['peptide']
        if not (7 <= len(pep) <= 25):
            log.critical(
                f"Peptide '{pep}' has length {len(pep)}; "
                f"valid range is 7–25 amino acids."
            )
            sys.exit(1)
        invalid_chars = set(pep) - VALID_AAS
        if invalid_chars:
            log.critical(
                f"Peptide '{pep}' contains non-standard amino acid(s): "
                f"{invalid_chars}."
            )
            sys.exit(1)

    print(f'Input validation passed: {len(df)} peptide–MHC pairs loaded.')

    # -- DataLoader --------------------------------------------------------
    dataset = PepMHCDataset(df['peptide'].tolist(), df['mhc_seq'].tolist())
    loader = DataLoader(dataset, batch_size=128, shuffle=False, num_workers=4)

    return df, loader


# ---------------------------------------------------------------------------
# Inference helpers
# ---------------------------------------------------------------------------

@torch.no_grad()
def run_presentation_inference(
    net: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> np.ndarray:
    """
    Run inference with a CALFP_PS model and return presentation probabilities.

    Returns:
        scores: (N,) float32 array, values in [0, 1]
    """
    net.eval()
    all_probs = []
    for pep_ids, mhc_ids in loader:
        pep_ids = pep_ids.to(device)
        mhc_ids = mhc_ids.to(device)
        logits = net(pep_ids, mhc_ids)                # (B, 2)
        probs  = F.softmax(logits, dim=1)[:, 1]       # (B,) positive class
        all_probs.append(probs.cpu())
    return torch.cat(all_probs).numpy().astype(np.float32)


@torch.no_grad()
def run_affinity_inference(
    net: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> np.ndarray:
    """
    Run inference with a CALFP_BA model and return rescaled affinity scores.

    Returns:
        scores: (N,) float32 array
    """
    net.eval()
    all_scores = []
    for pep_ids, mhc_ids in loader:
        pep_ids  = pep_ids.to(device)
        mhc_ids  = mhc_ids.to(device)
        scores   = net(pep_ids, mhc_ids)   # (B,)
        all_scores.append(scores.cpu())
    return torch.cat(all_scores).numpy().astype(np.float32)