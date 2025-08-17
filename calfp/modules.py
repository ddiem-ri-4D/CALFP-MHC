# modules.py
import torch
import torch.nn as nn
import torch.nn.functional as F

from calfp.init import truncated_normal_

__all__ = ['IConv', 'LinearAndOut']


class IConv(nn.Module):
    """
    Interaction-aware 1D convolution module for peptide–MHC representation.

    This module dynamically generates convolution kernels conditioned on the MHC sequence 
    and applies them to peptide embeddings. It implements a bilinear interaction where the 
    peptide features are convolved with MHC-dependent filters.

    Parameters
    ----------
    out_channels : int
        Number of output channels of the convolution.
    kernel_size : int
        Size of the convolution kernel.
    mhc_len : int, default=34
        Length of the MHC sequence (used in kernel generation).
    stride : int, default=1
        Stride of the convolution.
    **kwargs : dict
        Additional unused keyword arguments for compatibility.

    Attributes
    ----------
    weight : torch.nn.Parameter
        Learnable kernel weights of shape (out_channels, kernel_size, mhc_len).
    bias : torch.nn.Parameter
        Learnable bias of shape (out_channels,).
    stride : int
        Convolution stride.
    kernel_size : int
        Kernel size.

    Methods
    -------
    forward(peptide_x, mhc_x, **kwargs)
        Forward pass: generates interaction-specific convolution kernels from `mhc_x`
        and applies them to `peptide_x`.
    reset_parameters()
        Initializes weights with truncated normal distribution and bias with zeros.
    """

    def __init__(self, out_channels, kernel_size, mhc_len=34, stride=1, **kwargs):
        super(IConv, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(out_channels, kernel_size, mhc_len))
        self.bias = nn.Parameter(torch.Tensor(out_channels))
        self.stride, self.kernel_size = stride, kernel_size
        self.reset_parameters()

    def forward(self, peptide_x, mhc_x, **kwargs):
        """
        Forward computation of interaction-aware convolution.

        Parameters
        ----------
        peptide_x : torch.Tensor
            Input peptide embeddings of shape (batch_size, seq_len, hidden_dim).
        mhc_x : torch.Tensor
            Input MHC embeddings of shape (batch_size, mhc_len, hidden_dim).
        **kwargs : dict
            Additional arguments (unused).

        Returns
        -------
        torch.Tensor
            Output tensor of shape (batch_size, out_channels, new_seq_len),
            where `new_seq_len` depends on convolution stride and kernel size.
        """
        bs = peptide_x.shape[0]
        # Generate interaction kernels from MHC features
        kernel = F.relu(torch.einsum('nld,okl->nodk', mhc_x, self.weight))
        # Perform grouped 1D convolution for each sample
        outputs = F.conv1d(
            peptide_x.transpose(1, 2).reshape(1, -1, peptide_x.shape[1]),
            kernel.contiguous().view(-1, *kernel.shape[-2:]),
            stride=self.stride,
            groups=bs
        )
        return outputs.view(bs, -1, outputs.shape[-1]) + self.bias[:, None]

    def reset_parameters(self):
        """Initialize weights with truncated normal distribution and biases with zeros."""
        truncated_normal_(self.weight, std=0.02)
        nn.init.zeros_(self.bias)


class LinearAndOut(nn.Module):
    """
    Multi-layer perceptron (MLP) with sigmoid output for binary prediction.

    This module applies a stack of fully connected layers with ReLU activations,
    followed by a final linear layer and sigmoid activation to produce a probability score.

    Parameters
    ----------
    linear_size : list of int
        List specifying the dimensions of the linear layers. For example,
        [128, 64, 32] will create two hidden layers: 128→64 and 64→32,
        followed by an output layer 32→1.

    Attributes
    ----------
    linear : torch.nn.ModuleList
        List of linear layers for hidden feature transformation.
    output : torch.nn.Linear
        Final linear layer that maps to a single logit.
    
    Methods
    -------
    forward(inputs)
        Applies the MLP and outputs probabilities in [0, 1].
    reset_parameters()
        Initializes weights with truncated normal distribution and resets biases.
    """

    def __init__(self, linear_size):
        super(LinearAndOut, self).__init__()
        self.linear = nn.ModuleList(
            nn.Linear(in_s, out_s) for in_s, out_s in zip(linear_size[:-1], linear_size[1:])
        )
        self.output = nn.Linear(linear_size[-1], 1)
        self.reset_parameters()

    def forward(self, inputs):
        """
        Forward computation of the MLP classifier.

        Parameters
        ----------
        inputs : torch.Tensor
            Input tensor of shape (batch_size, input_dim).

        Returns
        -------
        torch.Tensor
            Probability scores of shape (batch_size,), with values in [0, 1].
        """
        linear_out = inputs
        for linear in self.linear:
            linear_out = F.relu(linear(linear_out))
        return torch.sigmoid(torch.squeeze(self.output(linear_out), -1))

    def reset_parameters(self):
        """Reinitialize all weights using truncated normal initialization."""
        for linear in self.linear:
            linear.reset_parameters()
            truncated_normal_(linear.weight, std=0.1)
        self.output.reset_parameters()
        truncated_normal_(self.output.weight, std=0.1)
