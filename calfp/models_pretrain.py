# models_pretrain.py
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from torch.utils.data import DataLoader
from logzero import logger
from typing import Optional, Mapping, Tuple

from calfp.evaluation import get_auc, get_pcc, get_group_metrics
from calfp.losses import SupConLoss
from calfp.evaluation import CUTOFF

__all__ = ['ModelPretrain']


class ModelPretrain(object):
    """
    A wrapper for training pretrain models with supervised contrastive loss (SupConLoss).
    Automatically saves the best model based on validation loss.
    """

    def __init__(self, network, model_path: str, **kwargs):
        """
        Args:
            network: model constructor (class).
            model_path: path to save the model.
            **kwargs: hyperparameters passed to the network.
        """
        self.model = self.network = network(**kwargs).cuda()
        self.model_path = Path(model_path)
        self.criterion = SupConLoss(temperature=0.01)
        self.model_path.parent.mkdir(parents=True, exist_ok=True)
        self.optimizer = None
        self.training_state = {}
        self.best_model_path = None

    def get_scores(self, inputs, **kwargs):
        """
        Forward pass of the model.
        
        Args:
            inputs: tuple (peptide_x, mhc_x, peptide_fp, mhc_fp).
        """
        peptide_x, mhc_x, peptide_fp, mhc_fp = inputs
        return self.model(
            peptide_x.cuda(),
            mhc_x.cuda(),
            peptide_fp.cuda(),
            mhc_fp.cuda(),
            **kwargs
        )

    def loss_and_backward(self, features: torch.Tensor, targets: torch.Tensor):
        """
        Compute contrastive loss and perform backward.
        """
        features = features.unsqueeze(1)
        loss = self.criterion(features, targets.cuda())
        loss.backward()
        return loss

    def train_step(self, inputs: Tuple[torch.Tensor, torch.Tensor], targets: torch.Tensor, **kwargs):
        """
        Perform one training step: forward, loss, backward, optimizer step.
        """
        self.optimizer.zero_grad()
        self.model.train()
        loss = self.loss_and_backward(self.get_scores(inputs, **kwargs), targets)
        self.optimizer.step()
        return loss.item()

    @torch.no_grad()
    def predict_step(self, inputs: Tuple[torch.Tensor, torch.Tensor], **kwargs):
        """
        Predict features for a batch.
        """
        self.model.eval()
        return self.get_scores(inputs, **kwargs)

    def get_optimizer(self, optimizer_cls='Adadelta', weight_decay=1e-3, **kwargs):
        """
        Initialize optimizer.

        Args:
            optimizer_cls: string or torch.optim class.
            weight_decay: weight decay regularization.
            **kwargs: optimizer hyperparameters (e.g., lr).
        """
        if isinstance(optimizer_cls, str):
            optimizer_cls = getattr(torch.optim, optimizer_cls)
        self.optimizer = optimizer_cls(self.model.parameters(), weight_decay=weight_decay, **kwargs)

    def train(self,
              train_loader: DataLoader,
              valid_loader: DataLoader,
              opt_params: Optional[Mapping] = None,
              num_epochs=20,
              verbose=True,
              **kwargs):
        """
        Train the model on train_loader and validate on valid_loader.
        Save the best model based on validation loss.

        Args:
            train_loader: DataLoader for training data.
            valid_loader: DataLoader for validation data.
            opt_params: dict of optimizer hyperparameters.
            num_epochs: number of training epochs.
            verbose: whether to print logs.
        """
        if opt_params is None:
            opt_params = {}
        self.get_optimizer(**opt_params)
        self.training_state['best'] = float('inf')
        self.training_state['best_epoch'] = 0

        for epoch_idx in range(num_epochs):
            train_loss = 0.0
            for peptide_x, mhc_x, peptide_fp, mhc_fp, targets in train_loader:
                inputs = (peptide_x, mhc_x, peptide_fp, mhc_fp)
                train_loss += self.train_step(inputs, targets, **kwargs) * len(targets)
            train_loss /= len(train_loader.dataset)

            valid_loss = self.valid(valid_loader, verbose, epoch_idx, train_loss, **kwargs)
            if verbose:
                logger.info(f'Epoch: {epoch_idx} -- Loss: {train_loss:.5f} -- Valid: {valid_loss:.5f}')

        logger.info(f'Best Epoch: {self.training_state["best_epoch"]}')

    def get_loss(self, features, targets):
        """
        Compute SupConLoss for (features, targets).
        """
        features = features.unsqueeze(1)
        return self.criterion(features, targets.cuda())

    def valid(self, valid_loader, verbose, epoch_idx, train_loss, **kwargs):
        """
        Validation loop. Update best model if current validation loss is lower.
        """
        scores, targets = self.predict(valid_loader, valid=True, **kwargs)
        valid_loss = self.get_loss(scores, targets).item()

        if valid_loss < self.training_state['best']:
            self.save_model()
            self.training_state['best'] = valid_loss
            self.training_state['best_epoch'] = epoch_idx
        return valid_loss

    def predict(self, data_loader: DataLoader, valid=False, **kwargs):
        """
        Predict features for the entire dataset.

        Args:
            data_loader: DataLoader for input data.
            valid: if True, use the current model; if False, load the best model.
        """
        if not valid:
            self.load_model()
        features = []
        sup_targets = []
        for peptide_x, mhc_x, peptide_fp, mhc_fp, targets in data_loader:
            inputs = (peptide_x, mhc_x, peptide_fp, mhc_fp)
            features.append(self.predict_step(inputs, **kwargs).cpu())
            sup_targets.append(targets.cpu())
        features = torch.cat(features, dim=0)
        sup_targets = torch.cat(sup_targets, dim=0)
        return features, sup_targets

    def save_model(self):
        """
        Save the best model to file.
        """
        save_file = self.model_path.with_name(f'{self.model_path.stem}-best{self.model_path.suffix}')
        torch.save(self.model.state_dict(), save_file)
        self.best_model_path = save_file

    def load_model(self):
        """
        Load the best model (if available), otherwise load from model_path.
        """
        load_file = self.best_model_path if self.best_model_path else self.model_path
        self.model.load_state_dict(torch.load(load_file))
