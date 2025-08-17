from __future__ import print_function

import torch
import torch.nn as nn

class SupConLoss(nn.Module):
    """
    Supervised Contrastive Loss (SupConLoss).

    Reference:
        - Khosla et al., Supervised Contrastive Learning,
          NeurIPS 2020. (https://arxiv.org/pdf/2004.11362.pdf)
        - For unsupervised variant (SimCLR): Chen et al., 2020
          (https://arxiv.org/pdf/2002.05709.pdf)

    This implementation supports both supervised and unsupervised
    contrastive learning objectives:
      - If `labels` is provided, samples of the same class are pulled closer.
      - If `labels` is None, falls back to SimCLR loss where only
        different augmented views of the same instance are positives.

    Args:
        temperature (float): Scaling factor for similarities. Default: 0.07
        contrast_mode (str): 'all' (use all views as anchors) or 'one'
            (use only one view as anchor). Default: 'all'
        base_temperature (float): Normalization constant. Default: 0.07

    Input:
        features (torch.Tensor): Hidden representations of shape
            [batch_size, n_views, feature_dim].
        labels (torch.Tensor, optional): Class labels [batch_size].
        mask (torch.Tensor, optional): Contrastive mask [batch_size, batch_size].
            mask_{i,j} = 1 if sample j is a positive for sample i.

    Returns:
        torch.Tensor: A scalar loss value.
    """

    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None):
        """
        Compute supervised (or unsupervised) contrastive loss.

        Workflow:
            1. Normalize features into shape [N, n_views, d].
            2. Build a mask of positives (from labels or user-provided mask).
            3. Compute pairwise similarities and apply temperature scaling.
            4. Compute log-probabilities over the contrastive set.
            5. Average over positives to obtain the final loss.
        """
        device = features.device

        # features should be [batch_size, n_views, ...]
        if len(features.shape) < 3:
            raise ValueError("`features` must have at least 3 dims: "
                             "[batch_size, n_views, feature_dim]")
        if len(features.shape) > 3:
            # Flatten extra dimensions
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]

        # Build contrastive mask
        if labels is not None and mask is not None:
            raise ValueError("Cannot define both `labels` and `mask`")
        elif labels is None and mask is None:
            # Unsupervised: only self-positives
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            # Supervised: positives are samples with the same label
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError("Num of labels does not match num of features")
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            # User-provided mask
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        # Concatenate features from all views: [batch_size * n_views, feature_dim]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)

        if self.contrast_mode == 'one':
            # Use only the first view as anchors
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            # Use all views as anchors
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError(f"Unknown mode: {self.contrast_mode}")

        # Cosine similarity scaled by temperature
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature
        )

        # Numerical stability: subtract max per row
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # Repeat mask for all anchors/views
        mask = mask.repeat(anchor_count, contrast_count)

        # Mask-out self-contrast (i == j)
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # Compute log-probabilities
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-6)

        # Mean log-likelihood over positives
        mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1) + 1e-6)

        # Loss = scaled negative log-likelihood
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss
