# aminoacids.py

__all__ = ['AALETTER', 'AAINDEX', 'INVALID_ACIDS', 'to_fingerprint', 'ap_encoder']
ACIDS = '0-ACDEFGHIKLMNPQRSTVWY'

from amorprot import AmorProt

AALETTER = [
    'A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I',
    'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V'
]
AAINDEX = {aa: i + 1 for i, aa in enumerate(AALETTER)}
INVALID_ACIDS = {'U', 'O', 'B', 'Z', 'J', 'X', '*'}

# Initialize AmorProt encoder
ap_encoder = AmorProt(maccs=True, ecfp4=True, ecfp6=True, rdkit=True)

def to_fingerprint(seq):
    """
    Convert amino acid sequence to a fingerprint vector using AmorProt encoder.
    Any invalid amino acids are removed before encoding.
    """
    clean_seq = ''.join(aa for aa in seq if aa not in INVALID_ACIDS)
    
    if len(clean_seq) == 0:
        raise ValueError(f"Sequence becomes empty after filtering invalid AAs: {seq}")
    
    return ap_encoder.fingerprint(clean_seq).tolist()


