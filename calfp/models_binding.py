# models_binding.py
from pathlib import Path
from typing import Tuple
import numpy as np
import torch
import torch.nn as nn
from logzero import logger
from torch.utils.data import DataLoader

__all__ = ['ModelBinding']


class ModelBinding(object):
    """
    Wrapper for peptide–MHC binding prediction models.

    This class provides:
      - Model initialization and checkpoint loading
      - Inference functions (`predict_step`, `predict`)
      - Utility to handle both peptide/MHC sequences and fingerprint inputs

    Notes
    -----
    - The network passed must implement a method `forward_binding(peptide, mhc, pep_fp, mhc_fp)`.
    - Designed to be compatible with fine-tuned checkpoints (stored under `fine_tuning/`).
    - Prediction outputs are concatenated into a single NumPy array.
    """

    def __init__(self, network, model_path, **kwargs):
        """
        Parameters
        ----------
        network : nn.Module
            The neural network architecture (class) to be instantiated.
        model_path : str or Path
            Path to the trained model checkpoint.
        kwargs : dict
            Extra arguments passed to the network constructor.
        """
        self.model = self.network = network(**kwargs).cuda()

        self.loss_fn, self.model_path = nn.MSELoss(), Path(model_path)
        self.model_path.parent.mkdir(parents=True, exist_ok=True)
        self.optimizer = None
        self.training_state = {}

        # Automatically load fine-tuned checkpoint if available
        trained_model = model_path.parent.parent.joinpath('fine_tuning').joinpath(self.model_path.stem + '.pt')
        self.model.load_state_dict(torch.load(trained_model))
        logger.info(f'Loading fine-tuning model from {trained_model}')

    def get_scores(self, inputs, **kwargs):
        """
        Run the forward pass of the model for binding prediction.

        Parameters
        ----------
        inputs : tuple
            Either (peptide_seq, mhc_seq, pep_fp, mhc_fp, label) or 
                   (peptide_seq, mhc_seq, pep_fp, mhc_fp).
        kwargs : dict
            Extra arguments for model forward pass.

        Returns
        -------
        torch.Tensor
            Predicted binding scores.
        """
        if len(inputs) == 5:
            peptide_seq, mhc_seq, pep_fp, mhc_fp, label = inputs
        elif len(inputs) == 4:
            peptide_seq, mhc_seq, pep_fp, mhc_fp = inputs
        else:
            raise ValueError(f"Unexpected number of inputs: {len(inputs)}")

        return self.model.forward_binding(
            peptide_seq.cuda(),
            mhc_seq.cuda(),
            pep_fp.cuda(),
            mhc_fp.cuda()
        )

    @torch.no_grad()
    def predict_step(self, inputs: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], **kwargs):
        """
        Single inference step without gradient computation.

        Parameters
        ----------
        inputs : tuple of torch.Tensor
            Batch of (peptide_seq, mhc_seq, pep_fp, mhc_fp).
        
        Returns
        -------
        torch.Tensor
            Predictions moved to CPU.
        """
        self.model.eval()
        return self.get_scores(inputs, **kwargs).cpu()

    def predict(self, data_loader: DataLoader, valid=False, **kwargs):
        """
        Run predictions for a dataset.

        Parameters
        ----------
        data_loader : DataLoader
            PyTorch DataLoader yielding model inputs.
        valid : bool, default=False
            If False, reload the model before predicting.
        kwargs : dict
            Extra args forwarded to `predict_step`.

        Returns
        -------
        np.ndarray
            Concatenated predictions across batches.
        """
        if not valid:
            self.load_model()
        outputs = [self.predict_step(data_x, **kwargs) for data_x in data_loader]
        if not outputs:
            logger.warning("No data to predict on — returning empty array.")
            return np.empty((0, self.model.output_dim if hasattr(self.model, 'output_dim') else 0))
        return np.concatenate(outputs, axis=0)

    def load_model(self):
        """Reload model weights from self.model_path."""
        self.model.load_state_dict(torch.load(self.model_path))
