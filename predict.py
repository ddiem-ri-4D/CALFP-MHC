"""
predict.py
----------
CALFP-MHC inference script.

Predicts peptide–MHC presentation probability (PB) and optionally binding
affinity (BA) for HLA class I and class II alleles using a 5-fold ensemble
of pre-trained models.

Usage
-----
    python predict.py --input peptides.csv --output results.csv

    python predict.py --input peptides.csv --output results.csv \\
                      --gpu True --BA True

Input CSV columns (required):
    peptide  — amino acid sequence, length 7–25
    allele   — HLA allele name (must be in HLA_library.csv)

Output CSV columns:
    peptide, allele, presentation_score [, affinity_score]

    presentation_score  — probability of MHC surface display (> 0.5 = presented)
    affinity_score      — rescaled IC50 score (> 0.5 ≈ IC50 < 500 nM),
                          reported when --BA True
"""

import argparse
import os
import sys
import torch
import pandas as pd
from tqdm import tqdm

from presentation_model import CALFP_PS
from affinity_model import CALFP_BA
from data_utils import load_input, run_presentation_inference, run_affinity_inference


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    desc = (
        "CALFP-MHC: predict peptide–MHC presentation probability and binding "
        "affinity for HLA class I and class II.\n\n"
        "Input CSV must have columns 'peptide' (length 7–25) and 'allele' "
        "(name must appear in HLA_library.csv)."
    )
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('--input',  required=True,
                        help='Path to input CSV (or TSV / parquet) file.')
    parser.add_argument('--output', required=True,
                        help='Path for output CSV file.')
    parser.add_argument('--gpu',    default='False',
                        help='Use GPU for inference: True or False (default False).')
    parser.add_argument('--BA',     default='True',
                        help='Also predict binding affinity: True or False (default True).')
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    # Resolve paths relative to this script's location
    script_dir  = os.path.dirname(os.path.abspath(__file__))
    hla_lib     = os.path.join(script_dir, 'HLA_library.csv')
    params_dir  = os.path.join(script_dir, 'params')
    log_path    = os.path.join(os.getcwd(), 'error.log')
    out_path    = os.path.join(os.getcwd(), args.output)

    if not os.path.isfile(args.input):
        print(f"[ERROR] Input file not found: {args.input}", file=sys.stderr)
        sys.exit(1)

    device = torch.device('cuda' if args.gpu.strip().lower() == 'true' else 'cpu')
    predict_ba = args.BA.strip().lower() == 'true'

    # -- Load and validate input -------------------------------------------
    df, loader = load_input(args.input, hla_lib, log_path)

    # -- Presentation score (5-fold ensemble) ------------------------------
    ps_folds = []
    print('Running 5-fold presentation score ensemble ...')
    for fold in tqdm(range(5), desc='PS folds'):
        params_path = os.path.join(params_dir, f'el_fold{fold}.params')
        net = CALFP_PS().to(device)
        net.load_state_dict(
            torch.load(params_path, map_location=device, weights_only=True)
        )
        scores = run_presentation_inference(net, loader, device)
        ps_folds.append(scores)
    df['presentation_score'] = sum(ps_folds) / len(ps_folds)

    # -- Binding affinity (5-fold ensemble, optional) ----------------------
    if predict_ba:
        ba_folds = []
        print('Running 5-fold binding affinity ensemble ...')
        for fold in tqdm(range(5), desc='BA folds'):
            params_path = os.path.join(params_dir, f'ba_fold{fold}.params')
            net = CALFP_BA().to(device)
            net.load_state_dict(
                torch.load(params_path, map_location=device, weights_only=True)
            )
            scores = run_affinity_inference(net, loader, device)
            ba_folds.append(scores)
        df['affinity_score'] = sum(ba_folds) / len(ba_folds)

    # -- Write output -------------------------------------------------------
    output_cols = ['peptide', 'allele', 'presentation_score']
    if predict_ba:
        output_cols.append('affinity_score')
    df[output_cols].to_csv(out_path, index=False)
    print(f'Done. Results written to: {out_path}')


if __name__ == '__main__':
    main()