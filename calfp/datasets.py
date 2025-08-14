# datasets.py

import numpy as np
import torch
from torch.utils.data import Dataset
from calfp.data_utils import to_fingerprint

__all__ = ['EOMHCIIDataset']


from calfp.aminoacids import AAINDEX  

def to_index(seq, max_len=34):
    idx = np.zeros(max_len, dtype=np.int64)
    for i in range(min(len(seq), max_len)):
        idx[i] = AAINDEX.get(seq[i], 0)
    return idx

class EOMHCIIDataset(Dataset):
    """
    Dataset for MHC-II binding using fingerprint + sequence index encoding.
    Returns: peptide_x, mhc_x, peptide_fp, mhc_fp, label
    """

    def __init__(self, data_list, max_len_pep=20, max_len_mhc=34, **kwargs):
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

class BindingDataset(Dataset):
    def __init__(self, data_list, max_len_pep=20, max_len_mhc=34):
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




