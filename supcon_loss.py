"""
supcon_loss.py
--------------
Supervised Contrastive Loss (Khosla et al. 2020), matching the manuscript's
equation exactly:

    L_sup = sum_i [ -1/|P(i)| * sum_{p in P(i)} log( exp(z_i . z_p / tau) /
                                                       sum_{a in A(i)} exp(z_i . z_a / tau) ) ]

where P(i) = other samples in the batch sharing i's binder/non-binder label,
A(i) = all other samples in the batch, tau = temperature (0.07, per Methods).

z is assumed already L2-normalized (unit hypersphere) — the projection
head in presentation_model.py / affinity_model.py should normalize its
output before passing it here.
"""

import torch
import torch.nn as nn


class SupConLoss(nn.Module):
    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, z: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z:      (B, D)  L2-normalized embeddings
            labels: (B,)    binder/non-binder labels {0,1}
        Returns:
            scalar loss
        """
        device = z.device
        B = z.shape[0]
        labels = labels.view(-1, 1)

        # Cosine similarity matrix / temperature
        sim = torch.matmul(z, z.T) / self.temperature          # (B,B)
        sim_max, _ = sim.max(dim=1, keepdim=True)
        sim = sim - sim_max.detach()                            # numerical stability

        # Mask out self-comparisons
        self_mask = torch.eye(B, dtype=torch.bool, device=device)
        # P(i): same label, not self
        pos_mask = (labels == labels.T) & ~self_mask
        # A(i): everything except self
        logits_mask = ~self_mask

        exp_sim = torch.exp(sim) * logits_mask
        log_prob = sim - torch.log(exp_sim.sum(dim=1, keepdim=True) + 1e-12)

        pos_counts = pos_mask.sum(dim=1)
        # Avoid div-by-zero for samples with no positives in this batch
        valid = pos_counts > 0
        mean_log_prob_pos = torch.zeros(B, device=device)
        mean_log_prob_pos[valid] = (
            (pos_mask[valid] * log_prob[valid]).sum(dim=1) / pos_counts[valid]
        )

        loss = -mean_log_prob_pos[valid].mean()
        return loss
