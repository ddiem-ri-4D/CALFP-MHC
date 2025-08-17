"""
calfp_fine_tuning.py

Command-line interface (CLI) for fine-tuning CALFP models.

This script supports multiple fine-tuning and evaluation modes for CALFP, including:
- `train`: Standard training with optional validation split.
- `eval`: Model evaluation on test data without training.
- `5cv`: 5-fold cross-validation training and evaluation.
- `lomo`: Leave-one-molecule-out cross-validation.
- `binding`: Binding core prediction from structural binding data.
- `seq2logo`: Sequence-to-logo generation for motif visualization.

It loads data, initializes models, and orchestrates the fine-tuning process,
including training, validation, testing, and saving results.

Example usage:
    python calfp_fine_tuning.py --mode train -d config/data.yaml -m config/calfp_fine_tuning.yaml
"""

import os
import click
import numpy as np
from pathlib import Path
from functools import partial
from ruamel.yaml import YAML
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from logzero import logger
import torch
import torch.nn.functional as F

from calfp.data_utils import *
from calfp.datasets import EOMHCIIDataset, BindingDataset
from calfp.models_fine_tuning import ModelFineTuning
from calfp.models_binding import ModelBinding
from calfp.CALFP import LinCALFP, BinCALFP
from calfp.evaluation import output_res, CUTOFF


def train(model, data_cnf, model_cnf, train_data, test_data=None,
          data_group_name=None, cv_id=None, cv_=None,
          valid_data=None, random_state=2023):
    """
    Train a fine-tuning model.

    Args:
        model (ModelFineTuning): Model wrapper instance to be trained.
        data_cnf (dict): Dataset configuration dictionary.
        model_cnf (dict): Model and training configuration dictionary.
        train_data (list): Training dataset samples.
        test_data (list, optional): Test dataset samples for evaluation.
        data_group_name (list, optional): Group names of the dataset samples (for logging).
        cv_id (list, optional): Cross-validation IDs for the dataset.
        cv_ (int, optional): Current cross-validation fold.
        valid_data (list, optional): Validation dataset samples. If None, split from training data.
        random_state (int): Random seed for train/validation split.
    """
    logger.info(f'Start training model {model.model_path}')
    if valid_data is None:
        train_data, valid_data = train_test_split(
            train_data, test_size=data_cnf.get('valid', 1000), random_state=random_state
        )

    train_loader = DataLoader(
        EOMHCIIDataset(train_data, **model_cnf['padding']),
        batch_size=model_cnf['train']['batch_size'], shuffle=True
    )
    valid_loader = DataLoader(
        EOMHCIIDataset(valid_data, **model_cnf['padding']),
        batch_size=model_cnf['valid']['batch_size']
    )

    if test_data is not None:
        test_loader = DataLoader(
            EOMHCIIDataset(test_data, **model_cnf['padding']),
            batch_size=model_cnf['test']['batch_size']
        )
        model.train(train_loader, valid_loader, test_loader,
                    data_group_name, cv_id, cv_, **model_cnf['train'])
    else:
        model.train(train_loader, valid_loader, **model_cnf['train'])

    logger.info(f'Finish training model {model.model_path}')


def test(model, model_cnf, test_data):
    """
    Run inference on a test dataset.

    Args:
        model (ModelFineTuning or ModelBinding): Trained model wrapper.
        model_cnf (dict): Model configuration dictionary.
        test_data (list): Test dataset samples.

    Returns:
        np.ndarray: Model predictions for the test dataset.
    """
    data_loader = DataLoader(
        EOMHCIIDataset(test_data, **model_cnf['padding']),
        batch_size=model_cnf['test']['batch_size']
    )
    return model.predict(data_loader)


def get_binding_core(data_list, model_cnf, model_path, start_id, num_models):
    """
    Predict binding core positions using an ensemble of binding models.

    Args:
        data_list (list): List of binding dataset samples.
        model_cnf (dict): Model configuration dictionary.
        model_path (Path): Path to model checkpoint.
        start_id (int): Starting index for ensemble models.
        num_models (int): Number of ensemble models to aggregate.

    Returns:
        tuple:
            np.ndarray: Predicted binding core positions (argmax over sequence).
            np.ndarray: Raw prediction scores averaged over ensemble.
    """
    scores_list = []
    for model_id in range(start_id, start_id + num_models):
        model = ModelBinding(
            BinCALFP,
            model_path=model_path.with_name(f'{model_path.stem}-{model_id}{model_path.suffix}'),
            pooling=False,
            **model_cnf['model']
        )
        scores = test(model, model_cnf, data_list)
        if scores.shape[0] == 0:
            logger.warning(f"Model {model_id}: empty prediction — skipping.")
            continue
        scores_list.append(scores)

    if len(scores_list) == 0:
        logger.error("No model produced predictions. Cannot compute binding core.")
        return [], np.array([])

    scores = np.mean(scores_list, axis=0)
    return scores.argmax(-1), scores


def fix_padding_keys(d):
    """
    Adapt padding configuration keys for `BindingDataset`.

    Args:
        d (dict): Original padding configuration dictionary.

    Returns:
        dict: Updated configuration with renamed keys.
    """
    d = dict(d)  # avoid modifying in-place
    if 'peptide_len' in d:
        d['max_len_pep'] = d.pop('peptide_len')
    if 'mhc_len' in d:
        d['max_len_mhc'] = d.pop('mhc_len')
    d.pop('peptide_pad', None)
    return d


@click.command()
@click.option('-d', '--data-cnf', type=click.Path(exists=True), default="config/data.yaml",
              help="Path to dataset configuration file (YAML).")
@click.option('-m', '--model-cnf', type=click.Path(exists=True), default="config/calfp_fine_tuning.yaml",
              help="Path to model configuration file (YAML).")
@click.option('--mode', type=click.Choice(('train', 'eval', '5cv', 'lomo', 'binding', 'seq2logo')),
              default='train', help="Execution mode.")
@click.option('-s', '--start-id', default=0, help="Starting model index (for ensemble training).")
@click.option('-n', '--num_models', default=20, help="Number of models to train or evaluate.")
@click.option('-c', '--continue', 'continue_train', is_flag=True,
              help="Continue training if checkpoint exists.")
@click.option('-a', '--allele', default='DRB1_0101',
              help="Allele name for seq2logo mode.")
@click.option('--save-csv', type=click.Path(), default=None,
              help="Path to save binding predictions as CSV (binding mode).")
def main(data_cnf, model_cnf, mode, continue_train, start_id, num_models, allele, save_csv):
    """
    Entry point for CALFP fine-tuning and evaluation.

    Supports different modes for model training and analysis:
    - train: Standard fine-tuning with optional validation split.
    - eval: Evaluate trained models on test data.
    - 5cv: 5-fold cross-validation.
    - lomo: Leave-one-molecule-out evaluation.
    - binding: Binding core prediction from structural binding datasets.
    - seq2logo: Generate sequence logo motifs from high-scoring peptides.

    Args:
        data_cnf (str): Path to dataset configuration YAML file.
        model_cnf (str): Path to model configuration YAML file.
        mode (str): Execution mode.
        continue_train (bool): If True, skip training if checkpoints exist.
        start_id (int): Starting model index (for ensembles).
        num_models (int): Number of ensemble models.
        allele (str): Allele name (for seq2logo mode).
        save_csv (str): Path to save binding predictions as CSV.
    """
    yaml = YAML(typ='safe')
    data_cnf, model_cnf = yaml.load(Path(data_cnf)), yaml.load(Path(model_cnf))
    model_name = model_cnf['name']
    model_path = Path(model_cnf['path']) / f'{model_name}.pt'
    res_path = Path(data_cnf['results']) / f'{model_name}'
    model_cnf.setdefault('ensemble', 20)

    mhc_name_seq = get_mhc_name_seq(data_cnf['mhc_seq'])
    get_data_fn = partial(get_data, mhc_name_seq=mhc_name_seq)

    # --- execution modes ---
    if mode in ('train', 'eval'):
        train_data = get_data_fn(data_cnf['train']) if mode == 'train' else None
        valid_data = get_data_fn(data_cnf['valid']) if train_data is not None and 'valid' in data_cnf else None
        test_data = get_data_fn(data_cnf['test'])
        test_group_name, test_truth = [x[0] for x in test_data], [x[-1] for x in test_data]

        scores_list = []
        for model_id in range(start_id, start_id + num_models):
            model = ModelFineTuning(LinCALFP, model_path=model_path.with_name(f'{model_path.stem}-{model_id}{model_path.suffix}'),
                                    **model_cnf['model'])
            if train_data is not None and (not continue_train or not model.model_path.exists()):
                train(model, data_cnf, model_cnf, train_data=train_data, valid_data=valid_data,
                      test_data=test_data, data_group_name=test_group_name)

            scores_list.append(test(model, model_cnf, test_data=test_data))
            output_res(test_group_name, test_truth, np.mean(scores_list, axis=0),
                       res_path.with_name(f'{res_path.stem}-train-eval'))

    elif mode == '5cv':
        data = np.asarray(get_data_fn(data_cnf['train']), dtype=object)
        data_group_name, data_truth = [x[0] for x in data], [x[-1] for x in data]
        with open(data_cnf['cv_id']) as fp:
            cv_id = np.asarray([int(line) for line in fp])
        assert len(data) == len(cv_id)

        scores_list = []
        for model_id in range(start_id, start_id + num_models):
            scores_ = np.empty(len(data), dtype=np.float32)
            for cv_ in range(5):
                train_data, test_data = data[cv_id != cv_], data[cv_id == cv_]
                model = ModelFineTuning(LinCALFP,
                                        # model_path=model_path.with_stem(f'{model_path.stem}-{model_id}-CV{cv_}'),
                                        model_path=model_path.with_name(f'{model_path.stem}-{model_id}-CV{cv_}{model_path.suffix}'),
                                        **model_cnf['model'])
                if not continue_train or not model.model_path.exists():
                    train(model, data_cnf, model_cnf, train_data=train_data, test_data=test_data,
                          data_group_name=data_group_name, cv_id=cv_id, cv_=cv_)

                scores_[cv_id == cv_] = test(model, model_cnf, test_data=test_data)

            scores_list.append(scores_)
            output_res(data_group_name, data_truth, np.mean(scores_list, axis=0),
                       res_path.with_name(f'{res_path.stem}-5CV'))

    elif mode == 'lomo':
        data = np.asarray(get_data_fn(data_cnf['train']), dtype=object)
        with open(data_cnf['cv_id']) as fp:
            cv_id = np.asarray([int(line) for line in fp])

        scores_list = []
        for model_id in range(start_id, start_id + num_models):
            group_names, group_names_, truth_, scores_ = [x[0] for x in data], [], [], []
            for name_ in sorted(set(group_names)):
                train_data, train_cv_id = data[group_names != name_], cv_id[group_names != name_]
                test_data, test_cv_id = data[group_names == name_], cv_id[group_names == name_]
                if len(test_data) > 30 and sum(x[-1] >= CUTOFF for x in test_data) >= 3:
                    for cv_ in range(5):
                        model = ModelFineTuning(LinCALFP,
                                                model_path=model_path.with_name(f'{model_path.stem}-{name_}-{model_id}-CV{cv_}{model_path.suffix}'),
                                                **model_cnf['model'])
                        if not continue_train or not model.model_path.exists():
                            train(model, data_cnf, model_cnf, train_data[train_cv_id != cv_])

                        test_data_ = test_data[test_cv_id == cv_]
                        group_names_ += [x[0] for x in test_data_]
                        truth_ += [x[-1] for x in test_data_]
                        scores_ += test(model, model_cnf, test_data_).tolist()

            scores_list.append(scores_)
            output_res(group_names_, truth_, np.mean(scores_list, axis=0), res_path.with_name(f'{res_path.stem}-LOMO'))

    elif mode == 'binding':
        model_cnf['padding'] = model_cnf['binding']
        raw_data = get_binding_data(data_cnf['binding'], mhc_name_seq)

        logger.debug(f"[DEBUG] Loaded {len(raw_data)} raw binding samples")
        for i, row in enumerate(raw_data[:3]):
            print(f"[DEBUG] Sample {i}: {row}")

        dataset_input = []
        meta_info = []

        for (pdb, mhc_name, core), peptide_seq, mhc_seq, _ in raw_data:
            if not isinstance(peptide_seq, str) or not isinstance(mhc_seq, str):
                continue
            if len(peptide_seq) < 9:
                continue
            try:
                pep_fp = to_fingerprint(peptide_seq)
                mhc_fp = to_fingerprint(mhc_seq)
            except Exception as e:
                logger.warning(f"Skipping {pdb} due to fingerprint error: {e}")
                continue

            meta_info.append((pdb, mhc_name, core, peptide_seq))
            dataset_input.append((peptide_seq, mhc_seq, pep_fp, mhc_fp, 0.0))  # dummy label

        if len(dataset_input) == 0:
            logger.warning("No valid binding samples after preprocessing.")
            return

        core_pos, scores = get_binding_core(dataset_input, model_cnf, model_path, start_id, num_models)

        # Đánh giá
        correct = 0
        for (pdb, mhc_name, core, peptide_seq), core_pos_, scores_ in zip(meta_info, core_pos, scores):
            pred_core = peptide_seq[core_pos_: core_pos_ + len(core)]
            print(pdb, mhc_name, peptide_seq, core, pred_core, core == pred_core)
            if core != pred_core:
                for i, s in enumerate(scores_[:len(peptide_seq) - len(core) + 1]):
                    print(peptide_seq[i:i + len(core)], s)
            correct += core == pred_core

        logger.info(f'The number of correct prediction is {correct}.')

        if save_csv:
            import pandas as pd
            import torch.nn.functional as F
            import torch

            rows = []
            for (pdb, mhc_name, core, peptide_seq), core_pos_, scores_ in zip(meta_info, core_pos, scores):
                core_len = len(core)
                valid_len = len(peptide_seq) - core_len + 1
                valid_scores = scores_[:valid_len]
                # ---- Check NaN or inf here ----
                if any(np.isnan(valid_scores)) or any(np.isinf(valid_scores)):
                    print(f"[NaN case] PDB: {pdb} | peptide: {peptide_seq} | valid_scores: {valid_scores}")
                
                valid_scores = np.nan_to_num(valid_scores, nan=-1e6, posinf=1e6, neginf=-1e6)

                try:
                    probs = F.softmax(torch.tensor(valid_scores, dtype=torch.float32), dim=0).numpy()
                    max_prob = float(np.max(probs))
                    prob_vector = ";".join(f"{p:.4f}" for p in probs)
                except Exception as e:
                    logger.warning(f"Softmax failed for {pdb}: {e}")
                    probs = [float('nan')] * valid_len
                    max_prob = float('nan')
                    prob_vector = ";".join(["nan"] * valid_len)

                pred_core = peptide_seq[core_pos_: core_pos_ + core_len]

                row = {
                    "PDB": pdb,
                    "Allele": mhc_name,
                    "Peptide": peptide_seq,
                    "TrueCore": core,
                    "PredCore": pred_core,
                    "Correct": core == pred_core,
                    "CorePos": core_pos_,
                    "MaxProb": max_prob,
                    "ProbVector": prob_vector
                }
                rows.append(row)

            df = pd.DataFrame(rows)
            df.to_csv(save_csv, index=False)
            logger.info(f"Saved prediction results to {save_csv}")


    elif mode == 'seq2logo':
        assert allele in mhc_name_seq
        data_list = get_seq2logo_data(data_cnf['seq2logo'], allele, mhc_name_seq[allele])

        scores_list = []
        for model_id in range(start_id, start_id + num_models):
            model = ModelBinding(BinCALFP,
                                model_path=model_path.with_name(f'{model_path.stem}-{model_id}{model_path.suffix}'),
                                 pooling=False, **model_cnf['model'])
            scores_list.append(test(model, model_cnf, data_list))

        scores = np.mean(scores_list, axis=0)
        s_, p_ = scores.max(axis=1), scores.argmax(axis=1)
        res_path = Path(data_cnf['results_logos'])
        res_path.mkdir(parents=True, exist_ok=True)
        with open(res_path.joinpath(f'{res_path.stem}-{allele}-{start_id}_{start_id + num_models}.txt'), 'w') as fp:
            for k in (-s_).argsort()[:int(0.01 * len(s_))]:
                print(data_list[k][1][p_[k]: p_[k] + 9], file=fp)

if __name__ == '__main__':
    main()