"""
predict.py
----------
CALFP-MHC  —  Inference entry point

Runs a 5-fold ensemble of pre-trained CALFP_PS (presentation score) and
optionally CALFP_BA (binding affinity) models to score peptide–MHC pairs.

Usage
-----
Basic (presentation score only):
    python predict.py --input peptides.csv --output results.csv

With binding affinity:
    python predict.py --input peptides.csv --output results.csv --BA True

GPU inference:
    python predict.py --input peptides.csv --output results.csv --gpu True

Input format
------------
CSV file with columns:
    peptide  — amino acid sequence, length 7–25, standard residues only
    allele   — HLA allele name (must be present in HLA_library.csv)

TSV and Parquet files are also accepted (auto-detected by extension).

Output format
-------------
CSV with original columns plus:
    presentation_score  — probability of MHC surface display (> 0.5 = presented)
    affinity_score      — rescaled IC50 score (> 0.5 ≈ IC50 < 500 nM)
                          [only when --BA True]
"""

import argparse
import os
import sys
import torch
import numpy as np
from tqdm import tqdm

from presentation_model import CALFP_PS
from affinity_model import CALFP_BA
from data_utils import load_input, run_presentation_inference, run_affinity_inference


# ── CLI ───────────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog='predict.py',
        description=(
            'CALFP-MHC: predict peptide–MHC presentation probability '
            'and binding affinity for HLA class I and class II.'
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        '--input', required=True, metavar='FILE',
        help='Input CSV / TSV / Parquet file (columns: peptide, allele).')
    p.add_argument(
        '--output', required=True, metavar='FILE',
        help='Output CSV file path.')
    p.add_argument(
        '--gpu', default='False', metavar='BOOL',
        help='Use GPU for inference: True or False  (default: False).')
    p.add_argument(
        '--BA', default='True', metavar='BOOL',
        help='Also predict binding affinity: True or False  (default: True).')
    return p


# ── Helpers ───────────────────────────────────────────────────────────────────

def _str_to_bool(s: str) -> bool:
    return s.strip().lower() in ('true', '1', 'yes')


def _load_model(model_cls, params_path: str,
                device: torch.device) -> torch.nn.Module:
    net = model_cls()
    net.load_state_dict(
        torch.load(params_path, map_location=device, weights_only=True))
    net.to(device)
    net.eval()
    return net


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args   = build_parser().parse_args()
    device = torch.device('cuda' if _str_to_bool(args.gpu) else 'cpu')
    run_ba = _str_to_bool(args.BA)

    script_dir   = os.path.dirname(os.path.abspath(__file__))
    hla_lib_path = os.path.join(script_dir, 'HLA_library.csv')
    params_dir   = os.path.join(script_dir, 'params')
    log_path     = os.path.join(os.getcwd(), 'error.log')

    if not os.path.isfile(args.input):
        print(f'[ERROR] Input file not found: {args.input}', file=sys.stderr)
        sys.exit(1)

    # ── Load and validate input ────────────────────────────────────────────
    df, loader = load_input(args.input, hla_lib_path, log_path)

    # ── Presentation score  (5-fold ensemble) ─────────────────────────────
    print(f'\nPredicting presentation scores  [device: {device}]')
    ps_scores = np.zeros(len(df), dtype=np.float32)
    for fold in tqdm(range(5), desc='PS folds', unit='fold'):
        params_path = os.path.join(params_dir, f'el_fold{fold}.params')
        net = _load_model(CALFP_PS, params_path, device)
        ps_scores += run_presentation_inference(net, loader, device)
    df['presentation_score'] = ps_scores / 5

    # ── Binding affinity  (5-fold ensemble, optional) ─────────────────────
    if run_ba:
        print(f'\nPredicting binding affinity scores  [device: {device}]')
        ba_scores = np.zeros(len(df), dtype=np.float32)
        for fold in tqdm(range(5), desc='BA folds', unit='fold'):
            params_path = os.path.join(params_dir, f'ba_fold{fold}.params')
            net = _load_model(CALFP_BA, params_path, device)
            ba_scores += run_affinity_inference(net, loader, device)
        df['affinity_score'] = ba_scores / 5

    # ── Write output ───────────────────────────────────────────────────────
    out_cols = ['peptide', 'allele', 'presentation_score']
    if run_ba:
        out_cols.append('affinity_score')

    out_path = args.output
    df[out_cols].to_csv(out_path, index=False)
    print(f'\nResults written to: {out_path}')
    print(f'Total pairs scored: {len(df)}')


if __name__ == '__main__':
    main()
