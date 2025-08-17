import torch

__all__ = ['truncated_normal_']


@torch.no_grad()
def truncated_normal_(tensor, mean=0.0, std=1.0):
    """
    Fill the input tensor with values drawn from a truncated normal distribution.

    - Truncation range: (-2, 2) standard deviations
    - Values are sampled from a normal distribution and re-scaled to (mean, std).
    - This is often used for initializing neural network weights
      to avoid extreme outliers that standard normal_() can produce.

    Args:
        tensor (torch.Tensor): The tensor to be filled in-place.
        mean (float): Mean of the truncated normal distribution.
        std (float): Standard deviation of the truncated normal distribution.

    Returns:
        torch.Tensor: The tensor filled with truncated normal values (in-place).
    """
    size = tensor.shape
    # Sample extra values (last dim = 4) to ensure enough within (-2, 2)
    tmp = tensor.new_empty(size + (4,)).normal_()
    # Mask valid values within the range (-2, 2)
    valid = (tmp < 2) & (tmp > -2)
    # Choose one valid sample per position
    ind = valid.max(-1, keepdim=True)[1]
    # Copy selected values into the original tensor
    tensor.data.copy_(tmp.gather(-1, ind).squeeze(-1))
    # Scale to (mean, std)
    tensor.data.mul_(std).add_(mean)
    return tensor
