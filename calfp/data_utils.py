# data_utils.py
# =============================================================================
# Data utility functions for peptide–MHC binding analysis.
#
# Provides functions to:
# - Load MHC name–to–sequence mappings
# - Load peptide–MHC binding datasets with fingerprints
# - Prepare binding/core data for structural studies
# - Prepare peptide sets for sequence logo visualization
# =============================================================================

__all__ = [
    'ACIDS', 
    'get_mhc_name_seq', 'get_data', 'get_binding_data',
    'get_seq2logo_data', 'to_fingerprint', 'ap_encoder'
]

from calfp.aminoacids import ap_encoder, to_fingerprint

# Amino acid alphabet (including 0 for padding)
ACIDS = '0-ACDEFGHIKLMNPQRSTVWY'


# === Load MHC name-to-sequence mapping ===
def get_mhc_name_seq(mhc_name_seq_file: str) -> dict:
    """
    Load mapping from MHC names to their amino acid sequences.

    Parameters
    ----------
    mhc_name_seq_file : str
        Path to a file containing MHC name and sequence per line.
        Format: "<mhc_name> <mhc_seq>"

    Returns
    -------
    dict
        Dictionary {mhc_name: mhc_sequence}
    """
    mhc_name_seq = {}
    with open(mhc_name_seq_file) as fp:
        for line in fp:
            mhc_name, mhc_seq = line.strip().split()
            mhc_name_seq[mhc_name] = mhc_seq
    return mhc_name_seq


# === Main data loading functions ===
def get_data(data_path: str, mhc_name_seq: dict = None) -> list:
    """
    Load peptide–MHC binding data with fingerprint conversion.

    Parameters
    ----------
    data_path : str
        Path to input file (tab-delimited). Each line should contain:
        <peptide_seq> <label> <mhc_name>
    mhc_name_seq : dict, optional
        Dictionary mapping MHC names to sequences (from `get_mhc_name_seq`).

    Returns
    -------
    list
        Each entry: [peptide_seq, mhc_seq, peptide_fp, mhc_fp, label (float)]
    """
    data = []
    skipped = 0
    with open(data_path) as fp:
        for line in fp:
            parts = line.strip().split('\t')
            if len(parts) != 3:
                print(f"[SKIP FORMAT] {line.strip()}")
                skipped += 1
                continue

            peptide_seq, label, mhc_name = parts
            try:
                peptide_fp = to_fingerprint(peptide_seq)
                mhc_fp = to_fingerprint(mhc_name_seq[mhc_name])
            except Exception as e:
                print(f"[SKIP INVALID] {peptide_seq} | {mhc_name}: {e}")
                skipped += 1
                continue

            data.append([
                peptide_seq,
                mhc_name_seq[mhc_name],
                peptide_fp,
                mhc_fp,
                float(label)
            ])
    print(f"[INFO] Loaded {len(data)} valid rows, skipped {skipped} rows")
    return data


def get_binding_data(data_file: str, mhc_name_seq: dict, core_len: int = 9) -> list:
    """
    Load binding/core data (e.g., from structural complexes).

    Parameters
    ----------
    data_file : str
        Path to data file containing: <pdb> <mhc_name> <mhc_seq> <peptide_seq> <core>
    mhc_name_seq : dict
        Dictionary mapping MHC names to sequences.
    core_len : int, default=9
        Expected core peptide length.

    Returns
    -------
    list
        Each entry: ((pdb, mhc_name, core), peptide_seq, mhc_seq, 0.0)
    """
    data_list = []
    with open(data_file) as fp:
        for line in fp:
            pdb, mhc_name, mhc_seq, peptide_seq, core = line.split()
            assert len(core) == core_len
            data_list.append(((pdb, mhc_name, core), peptide_seq, mhc_name_seq[mhc_name], 0.0))
    return data_list


def get_seq2logo_data(data_file: str, mhc_name: str, mhc_seq: str) -> list:
    """
    Load a list of peptides for sequence logo visualization.

    Parameters
    ----------
    data_file : str
        Path to file containing peptide sequences (one per line).
    mhc_name : str
        MHC name (for tagging outputs).
    mhc_seq : str
        MHC amino acid sequence.

    Returns
    -------
    list
        Each entry: (mhc_name, peptide_fp, mhc_fp, score=0.0)
    """
    try:
        mhc_fp = to_fingerprint(mhc_seq)
    except ValueError:
        return []

    data_list = []
    with open(data_file) as fp:
        for line in fp:
            pep = line.strip()
            if not pep:
                continue
            try:
                pep_fp = to_fingerprint(pep)
            except ValueError:
                continue
            data_list.append((mhc_name, pep_fp, mhc_fp, 0.0))
    return data_list
