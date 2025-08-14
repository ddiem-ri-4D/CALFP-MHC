# data_utils.py

__all__ = [ 'ACIDS', 
    'get_mhc_name_seq', 'get_data', 'get_binding_data',
    'get_seq2logo_data', 'to_fingerprint', 'ap_encoder'
]

from calfp.aminoacids import ap_encoder, to_fingerprint
ACIDS = '0-ACDEFGHIKLMNPQRSTVWY'

# === Load MHC name-to-sequence mapping ===
def get_mhc_name_seq(mhc_name_seq_file):
    """
    Reads a file containing MHC name and its amino acid sequence per line.
    Format: <mhc_name> <mhc_seq>
    """
    mhc_name_seq = {}
    with open(mhc_name_seq_file) as fp:
        for line in fp:
            mhc_name, mhc_seq = line.strip().split()
            mhc_name_seq[mhc_name] = mhc_seq
    return mhc_name_seq

# === Main data loading functions ===

def get_data(data_path, mhc_name_seq=None):
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

            data.append([peptide_seq, mhc_name_seq[mhc_name], peptide_fp, mhc_fp, float(label)])
    print(f"[INFO] Loaded {len(data)} valid rows, skipped {skipped} rows")
    return data


def get_binding_data(data_file, mhc_name_seq, core_len=9):
    data_list = []
    with open(data_file) as fp:
        for line in fp:
            pdb, mhc_name, mhc_seq, peptide_seq, core = line.split()
            assert len(core) == core_len
            data_list.append(((pdb, mhc_name, core), peptide_seq, mhc_name_seq[mhc_name], 0.0))
    return data_list

def get_seq2logo_data(data_file, mhc_name, mhc_seq):
    """
    Load a list of peptides for a given MHC to be visualized as a logo.
    Returns list of (mhc_name, peptide_fp, mhc_fp, score=0.0)
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

