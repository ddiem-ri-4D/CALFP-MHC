# datasets.py
# =============================================================================
# PyTorch Dataset classes for peptide–MHC binding tasks.
#
# Provides:
# - Index encoding of amino acid sequences
# - Dataset wrapper for MHC-II binding (with fingerprints + sequence indices)
# - Dataset wrapper for binding/core data
# =============================================================================

import numpy as np
import torch
from torch.utils.data import Dataset
from calfp.data_utils import to_fingerprint
from calfp.aminoacids import AAINDEX  

__all__ = ['EOMHCIIDataset']


# === Sequence indexing utility ===
def to_index(seq: str, max_len: int = 34) -> np.ndarray:
    """
    Convert an amino acid sequence into integer indices using AAINDEX.

    Parameters
    ----------
    seq : str
        Amino acid sequence (single-letter codes).
    max_len : int, default=34
        Maximum sequence length. Longer sequences are truncated;
        shorter sequences are zero-padded.

    Returns
    -------
    np.ndarray of shape (max_len,)
        Indexed sequence representation.
    """
    idx = np.zeros(max_len, dtype=np.int64)
    for i in range(min(len(seq), max_len)):
        idx[i] = AAINDEX.get(seq[i], 0)
    return idx


# === Dataset for MHC-II binding ===
class EOMHCIIDataset(Dataset):
    """
    Dataset for MHC-II binding using fingerprint + sequence index encoding.

    Each item returns:
        peptide_x : torch.LongTensor [max_len_pep]
        mhc_x     : torch.LongTensor [max_len_mhc]
        peptide_fp: torch.FloatTensor [fingerprint_dim]
        mhc_fp    : torch.FloatTensor [fingerprint_dim]
        label     : torch.FloatTensor (binding label)
    """

    def __init__(self, data_list, max_len_pep: int = 20, max_len_mhc: int = 34, **kwargs):
        self.data = []
        for row in data_list:
            if len(row) == 5:
                pep_seq, mhc_seq, peptide_fp, mhc_fp, label = row
                peptide_x = to_index(pep_seq, max_len_pep)
                mhc_x = to_index(mhc_seq, max_len_mhc)
                self.data.append((peptide_x, mhc_x, peptide_fp, mhc_fp, float(label)))

    def __getitem__(self, idx):
        peptide_x, mhc_x, peptide_fp, mhc_fp, label = self.data[idx]
        return (
            torch.tensor(peptide_x, dtype=torch.long),
            torch.tensor(mhc_x, dtype=torch.long),
            torch.tensor(peptide_fp, dtype=torch.float32),
            torch.tensor(mhc_fp, dtype=torch.float32),
            torch.tensor(label, dtype=torch.float32),
        )

    def __len__(self):
        return len(self.data)


# === Dataset for binding/core data ===
class BindingDataset(Dataset):
    """
    Dataset for peptide–MHC binding/core data.

    Each item returns:
        (peptide_x, mhc_x, peptide_fp, mhc_fp), None

    Notes
    -----
    - Samples with peptide length < 9 are automatically filtered out.
    - Label is not included (set as None).
    """

    def __init__(self, data_list, max_len_pep: int = 20, max_len_mhc: int = 34):
        self.data = []

        for row in data_list:
            if len(row) == 6:
                (_, mhc_name, core), pep_fp, mhc_fp, peptide_seq, mhc_seq = row

                if len(peptide_seq) < 9:
                    continue

                peptide_x = to_index(peptide_seq, max_len_pep)
                mhc_x = to_index(mhc_seq, max_len_mhc)

                self.data.append((peptide_x, mhc_x, pep_fp, mhc_fp))

        if len(self.data) == 0:
            print("[WARNING] BindingDataset: All samples were filtered out (e.g., peptide length < 9)")

    def __getitem__(self, idx):
        peptide_x, mhc_x, peptide_fp, mhc_fp = self.data[idx]
        return (
            torch.tensor(peptide_x, dtype=torch.long),
            torch.tensor(mhc_x, dtype=torch.long),
            torch.tensor(peptide_fp, dtype=torch.float32),
            torch.tensor(mhc_fp, dtype=torch.float32),
        ), None

    def __len__(self):
        return len(self.data)
