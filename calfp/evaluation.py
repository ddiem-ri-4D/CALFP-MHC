# metrics_utils.py
# =============================================================================
# Utility functions for evaluation metrics in peptide–MHC binding prediction.
#
# Provides:
# - ROC AUC, Pearson correlation (PCC), Spearman correlation (SRCC)
# - Group-wise metrics per MHC allele
# - Outputting results to CSV (per allele + overall)
# =============================================================================

import math
import csv
import numpy as np
from collections import namedtuple
from pathlib import Path
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score
from logzero import logger

__all__ = [
    'CUTOFF', 'get_auc', 'get_pcc', 'get_srcc',
    'get_group_metrics', 'output_res'
]

# Binding cutoff (1 - log(IC50, 50000))
CUTOFF = 1.0 - math.log(500, 50000)

# Named tuple container for metrics
Metrics = namedtuple('Metrics', ['auc', 'pcc', 'srcc'])


# === Metric functions ===
def get_auc(targets, scores):
    """
    Compute ROC-AUC.

    Parameters
    ----------
    targets : array-like
        Ground-truth binding values.
    scores : array-like
        Predicted scores.

    Returns
    -------
    float
        ROC-AUC score (binary classification with CUTOFF threshold).
    """
    return roc_auc_score(targets >= CUTOFF, scores)


def get_pcc(targets, scores):
    """
    Compute Pearson correlation coefficient (PCC).

    Returns
    -------
    float
        Pearson correlation coefficient.
    """
    return np.corrcoef(targets, scores)[0, 1]


def get_srcc(targets, scores):
    """
    Compute Spearman rank correlation coefficient (SRCC).

    Returns
    -------
    float
        Spearman correlation coefficient.
    """
    return spearmanr(targets, scores)[0]


def get_group_metrics(mhc_names, targets, scores, reduce=True):
    """
    Compute metrics per MHC allele (grouped).

    Parameters
    ----------
    mhc_names : array-like
        MHC allele names.
    targets : array-like
        Ground-truth binding values.
    scores : array-like
        Predicted scores.
    reduce : bool, default=True
        If True, return mean metrics across alleles.
        If False, return per-allele results.

    Returns
    -------
    tuple
        - If reduce=True: (mean_auc, mean_pcc, mean_srcc)
        - If reduce=False: (mhc_groups, auc_list, pcc_list, srcc_list)
    """
    mhc_names, targets, scores = np.asarray(mhc_names), np.asarray(targets), np.asarray(scores)
    mhc_groups, metrics = [], Metrics([], [], [])

    for mhc_name_ in sorted(set(mhc_names)):
        t_ = targets[mhc_names == mhc_name_]
        s_ = scores[mhc_names == mhc_name_]

        # Need both positive and negative examples for AUC
        if len(np.unique(t_ >= CUTOFF)) == 2:
            mhc_groups.append(mhc_name_)
            metrics.auc.append(get_auc(t_, s_))
            metrics.pcc.append(get_pcc(t_, s_))
            metrics.srcc.append(get_srcc(t_, s_))

    return (np.mean(x) for x in metrics) if reduce else (mhc_groups,) + metrics


def output_res(mhc_names, targets, scores, output_path: Path):
    """
    Save evaluation results to .npy and .csv.

    Parameters
    ----------
    mhc_names : array-like
        MHC allele names.
    targets : array-like
        Ground-truth binding values.
    scores : array-like
        Predicted scores.
    output_path : pathlib.Path
        Output file path (scores will be saved as .npy, results as .csv).
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Save raw prediction scores
    np.save(output_path, scores)

    eval_out_path = output_path.with_suffix('.csv')
    mhc_names = np.asarray(mhc_names)
    targets = np.asarray(targets)
    scores = np.asarray(scores)
    metrics = []

    with open(eval_out_path, 'w') as fp:
        writer = csv.writer(fp)
        writer.writerow(['allele', 'total', 'positive', 'AUC', 'PCC', 'SRCC'])

        mhc_groups, auc, pcc, srcc = get_group_metrics(mhc_names, targets, scores, reduce=False)

        for mhc_name_, auc_, pcc_, srcc_ in zip(mhc_groups, auc, pcc, srcc):
            t_ = targets[mhc_names == mhc_name_]
            writer.writerow([mhc_name_, len(t_), len(t_[t_ >= CUTOFF]), auc_, pcc_, srcc_])
            metrics.append((auc_, pcc_, srcc_))

        # Overall mean metrics
        if len(metrics) == 0:
            metrics = [float('nan')] * 3
        else:
            metrics = np.mean(metrics, axis=0)
            if isinstance(metrics, (float, int)):
                metrics = [metrics]
            elif isinstance(metrics, np.ndarray):
                metrics = metrics.tolist()

        writer.writerow([''] * 3 + metrics)
        logger.info(f'AUC: {metrics[0]:.3f} PCC: {metrics[1]:.3f} SRCC: {metrics[2]:.3f}')
