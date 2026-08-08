"""
train_data_utils.py
--------------------
Labeled-data utilities for training CALFP_PS (presentation) and CALFP_BA
(binding affinity), on top of the encoding helpers already in data_utils.py.

Expected training file columns (CSV/TSV/Parquet):
    peptide  — amino acid string
    allele   — HLA allele name (must exist in HLA_library.csv)
    label    — for presentation training: 0/1 (non-binder/binder)
             — for affinity training: continuous target (rescaled IC50 /
               %Rank, however you prepared it upstream — this script does
               not rescale for you)
"""

import pandas as pd
import torch
from torch.utils.data import Dataset

from calfp.data.data_utils import encode_sequence, PEP_MAX_LEN, MHC_PSEUDO_LEN


class LabeledPepMHCDataset(Dataset):
    """Same encoding as PepMHCDataset, plus a label column for training."""

    def __init__(self, df: pd.DataFrame, hla_lib: dict, label_dtype=torch.float32):
        missing = {'peptide', 'allele', 'label'} - set(df.columns)
        if missing:
            raise ValueError(f'Training file missing columns: {missing}')

        unknown = set(df['allele']) - set(hla_lib)
        if unknown:
            raise ValueError(f'Unrecognised allele(s) in training file: {unknown}')

        self.pep = torch.stack(
            [encode_sequence(p, PEP_MAX_LEN) for p in df['peptide']])
        self.mhc = torch.stack(
            [encode_sequence(hla_lib[a], MHC_PSEUDO_LEN) for a in df['allele']])
        self.label = torch.tensor(df['label'].values, dtype=label_dtype)

    def __len__(self):
        return len(self.pep)

    def __getitem__(self, idx):
        return self.pep[idx], self.mhc[idx], self.label[idx]


def load_hla_library(path: str) -> dict:
    hla_df = pd.read_csv(path)
    return dict(zip(hla_df['Allele Name'], hla_df['MHC pseudo-seq']))


def read_labeled_file(path: str) -> pd.DataFrame:
    if path.endswith('.parquet'):
        df = pd.read_csv(path) if path.endswith('.csv') else pd.read_parquet(path)
    elif path.endswith(('.tsv', '.txt')):
        df = pd.read_csv(path, sep='\t')
    else:
        df = pd.read_csv(path)
    df['peptide'] = df['peptide'].astype(str).str.strip()
    df['allele']  = df['allele'].astype(str).str.strip()
    return df


def pearson_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    1 - Pearson correlation coefficient, as a differentiable loss term.
    Combined with MSE per Methods: "MSE + Pearson correlation loss → IC50 and %Rank".
    """
    pred_c   = pred - pred.mean()
    target_c = target - target.mean()
    num = (pred_c * target_c).sum()
    den = torch.sqrt((pred_c ** 2).sum() * (target_c ** 2).sum()) + 1e-8
    pcc = num / den
    return 1.0 - pcc
