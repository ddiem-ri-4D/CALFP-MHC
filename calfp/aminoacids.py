# aminoacids.py
# =============================================================================
# Amino acid utilities and molecular fingerprint encoder
#
# This module defines amino acid alphabets, indexing, and provides a helper
# function to convert sequences into molecular fingerprints using AmorProt.
# =============================================================================

__all__ = ['AALETTER', 'AAINDEX', 'INVALID_ACIDS', 'to_fingerprint', 'ap_encoder']

# Amino acid alphabet (single-letter codes)
AALETTER = [
    'A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I',
    'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V'
]

# Mapping: amino acid → index (1-based)
AAINDEX = {aa: i + 1 for i, aa in enumerate(AALETTER)}

# Non-standard or invalid amino acids to exclude
INVALID_ACIDS = {'U', 'O', 'B', 'Z', 'J', 'X', '*'}

# Amino acid alphabet including 0/padding
ACIDS = '0-ACDEFGHIKLMNPQRSTVWY'

# Import AmorProt encoder (fingerprint generator)
from amorprot import AmorProt

# Initialize AmorProt encoder with multiple fingerprint schemes
ap_encoder = AmorProt(
    maccs=True,
    ecfp4=True,
    ecfp6=True,
    rdkit=True
)

def to_fingerprint(seq: str):
    """
    Convert an amino acid sequence into a fingerprint vector using AmorProt.

    Parameters
    ----------
    seq : str
        Amino acid sequence (string of single-letter codes).
        Invalid amino acids are removed automatically.

    Returns
    -------
    list[float]
        Fingerprint vector representing the input sequence.

    Raises
    ------
    ValueError
        If the sequence becomes empty after removing invalid amino acids.

    Example
    -------
    >>> to_fingerprint("ACDEFG")
    [0.0, 1.0, 0.0, 0.5, ...]
    """
    # Remove invalid amino acids
    clean_seq = ''.join(aa for aa in seq if aa not in INVALID_ACIDS)
    
    if len(clean_seq) == 0:
        raise ValueError(
            f"Sequence becomes empty after filtering invalid AAs: {seq}"
        )
    
    # Convert sequence to fingerprint vector
    return ap_encoder.fingerprint(clean_seq).tolist()
