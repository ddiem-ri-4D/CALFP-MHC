# models_fine_tuning.py
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from torch.utils.data import DataLoader
from tqdm import tqdm
from logzero import logger
from typing import Optional, Mapping, Tuple
from calfp.CALFP import LinearPredictor

from calfp.evaluation import get_auc, get_pcc, get_group_metrics
__all__ = ['ModelFineTuning']

import logzero
logzero.loglevel(logzero.logging.DEBUG)

class ModelFineTuning(object):
    """

    """

    def __init__(self, network, model_path, **kwargs):
        self.model = self.network = network(**kwargs).cuda()

        self.loss_fn, self.model_path = nn.MSELoss(), Path(model_path)
        self.model_path.parent.mkdir(parents=True, exist_ok=True)
        self.optimizer = None
        self.training_state = {}

        pretrain_model = model_path.parent.parent.joinpath('pre_models').joinpath(self.model_path.stem+'-epoch-best.pt')
        if pretrain_model.exists():
            self.model.network.load_state_dict(torch.load(pretrain_model))
            logger.info(f'Loading pretrain model from {pretrain_model}')
        else:
            logger.info(f'No pretrain model found in {pretrain_model}')


    def get_scores(self, inputs, **kwargs):
        output = self.model(inputs, **kwargs)
        return output

    def loss_and_backward(self, scores, targets):
        loss = self.loss_fn(scores, targets.cuda())
        loss.backward()
        return loss

    def train_step(self, inputs: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], targets: torch.Tensor,
                   **kwargs):
        self.optimizer.zero_grad()
        self.model.train()

        loss = self.loss_and_backward(self.get_scores(inputs, **kwargs), targets)
        self.optimizer.step(closure=None)
        return loss.item()

    @torch.no_grad()
    def predict_step(self, inputs: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], **kwargs):
        self.model.eval()
        return self.get_scores(inputs, **kwargs).cpu()

    def get_optimizer(self, optimizer_cls='Adadelta', weight_decay=1e-3, **kwargs):
        if isinstance(optimizer_cls, str):
            optimizer_cls = getattr(torch.optim, optimizer_cls)
        self.optimizer = optimizer_cls(self.model.parameters(), weight_decay=weight_decay, **kwargs)

    def train(self, train_loader: DataLoader, valid_loader: DataLoader, test_loader: DataLoader = None,
              data_group_name=None, cv_id=None, cv_=None, opt_params: Optional[Mapping] = (), num_epochs=20, **kwargs):
        self.get_optimizer(**dict(opt_params))
        self.training_state['best'] = 0.0
        self.training_state['best_epoch'] = 0
        for epoch_idx in range(num_epochs):
            train_loss = 0.0

            for peptide_x, mhc_x, peptide_fp, mhc_fp, targets in train_loader:
                inputs = (peptide_x, mhc_x, peptide_fp, mhc_fp)
                train_loss += self.train_step(inputs, targets, **kwargs) * len(targets)
            auc_valid, pcc_valid = self.valid(valid_loader, epoch_idx)

            if test_loader is None:
                logger.info(f'Epoch: {epoch_idx} '
                            f'-- Loss: {train_loss:.5f} '
                            f'-- Valid: AUC: {auc_valid:.5f} PCC: {pcc_valid:.3f} ')
                continue
            if cv_id is not None:
                mhc_names = np.asarray(data_group_name)[cv_id == cv_]
            else:
                mhc_names = np.asarray(data_group_name)
            auc_all, auc_group, pcc_group, srcc_group = self.test(test_loader, mhc_names)

            logger.info(f'Epoch: {epoch_idx} '
                        f'-- Loss: {train_loss:.5f} '
                        f'-- Valid: AUC: {auc_valid:.5f} PCC: {pcc_valid:.3f} '
                        f'-- Test: All AUC: {auc_all:.5f} - Group AUC: {auc_group:.5f} - PCC: '
                        f'{pcc_group:.3f} SRCC: {srcc_group:.3f}')

            if epoch_idx % 5 == 0:
                save_file = self.model_path.with_name(f'{self.model_path.stem}-epoch-{epoch_idx}{self.model_path.suffix}')
                torch.save(self.model.state_dict(), save_file)
        logger.info(f'Best Epoch: {self.training_state["best_epoch"]} ')

    def valid(self, valid_loader, epoch_idx, **kwargs):
        scores, targets = [], []
        for peptide_x, mhc_x, peptide_fp, mhc_fp, t in valid_loader:
            inputs = (peptide_x, mhc_x, peptide_fp, mhc_fp)
            scores.append(self.predict_step(inputs, **kwargs))
            targets.append(t)
        scores = np.hstack(scores)
        targets = torch.cat(targets).cpu().numpy()

        auc_valid, pcc_valid = get_auc(targets, scores), get_pcc(targets, scores)
        if pcc_valid > self.training_state['best']:
            self.save_model()
            self.training_state['best'] = pcc_valid
            self.training_state['best_epoch'] = epoch_idx
        return auc_valid, pcc_valid

    def predict(self, data_loader: DataLoader, valid=False, **kwargs):
        if not valid:
            self.load_model()
            
        return np.hstack([
            self.predict_step((peptide_x, mhc_x, peptide_fp, mhc_fp), **kwargs)
            for peptide_x, mhc_x, peptide_fp, mhc_fp, _ in data_loader
        ])

    def save_model(self):
        torch.save(self.model.state_dict(), self.model_path)

    def load_model(self):
        self.model.load_state_dict(torch.load(self.model_path))
        logger.info(f'Loading model from {self.model_path}')

    def test(self, test_loader, mhc_names, **kwargs):
        scores, targets = [], []
        for peptide_x, mhc_x, peptide_fp, mhc_fp, t in test_loader:
            inputs = (peptide_x, mhc_x, peptide_fp, mhc_fp)
            scores.append(self.predict_step(inputs, **kwargs))
            targets.append(t)
        scores = np.hstack(scores)
        targets = torch.cat(targets).cpu().numpy()

        # 🔍 Add this debug line here:
        logger.debug(f"mhc_names shape: {mhc_names.shape}, targets shape: {targets.shape}, scores shape: {scores.shape}")

        auc_all, pcc_all = get_auc(targets, scores), get_pcc(targets, scores)

        mhc_groups, auc, pcc, srcc = get_group_metrics(mhc_names, targets, scores, reduce=False)

        if len(auc) == 0:
            logger.warning("No valid MHC groups found for group metrics. Returning NaNs.")
            return auc_all, float('nan'), float('nan'), float('nan')

        metrics = np.mean(list(zip(auc, pcc, srcc)), axis=0)
        return auc_all, metrics[0], metrics[1], metrics[2]

