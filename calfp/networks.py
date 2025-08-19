#networks.py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from calfp.data_utils import ACIDS
from calfp.modules import *
from calfp.init import truncated_normal_

__all__ = ['DeepMHCII', 'ResMHCII']


class Network(nn.Module):
    """
    Base network that provides peptide and MHC embeddings.

    Args:
        emb_size (int): Dimension of embedding vectors.
        vocab_size (int, optional): Vocabulary size for amino acids (default: len(ACIDS)).
        padding_idx (int, optional): Index used for padding tokens (default: 0).
        peptide_pad (int, optional): Padding length at peptide sequence ends (default: 3).
        mhc_len (int, optional): Length of MHC sequence (default: 34).
    """
    def __init__(self, *, emb_size, vocab_size=len(ACIDS), padding_idx=0, peptide_pad=3, mhc_len=34, **kwargs):
        super(Network, self).__init__()
        self.peptide_emb = nn.Embedding(vocab_size, emb_size)
        self.mhc_emb = nn.Embedding(vocab_size, emb_size)
        self.peptide_pad, self.padding_idx, self.mhc_len = peptide_pad, padding_idx, mhc_len

    def forward(self, peptide_x, mhc_x, *args, **kwargs):
        """
        Embed peptide and MHC sequences and compute peptide masks.

        Args:
            peptide_x (Tensor): Peptide token indices (B × L).
            mhc_x (Tensor): MHC token indices (B × L).

        Returns:
            peptide_emb (Tensor): Embedded peptide tensor.
            mhc_emb (Tensor): Embedded MHC tensor.
            masks (BoolTensor): Mask for valid peptide residues (ignores padding).
        """
        masks = peptide_x[:, self.peptide_pad: peptide_x.shape[1] - self.peptide_pad] != self.padding_idx
        return self.peptide_emb(peptide_x), self.mhc_emb(mhc_x), masks

    def reset_parameters(self):
        """Reset embedding weights uniformly."""
        nn.init.uniform_(self.peptide_emb.weight, -0.1, 0.1)
        nn.init.uniform_(self.mhc_emb.weight, -0.1, 0.1)


class DeepMHCII(Network):
    """
    Deep convolutional neural network for peptide–MHCII binding prediction.

    Combines:
      - Peptide/MHC embeddings
      - One-hot encodings processed via convolution
      - Multiple convolutional + linear layers with dropout
      - Optional pooling for sequence-level output

    Args:
        conv_num (list[int]): Number of convolutional filters per layer.
        conv_size (list[int]): Kernel sizes for convolutional filters.
        conv_off (list[int]): Offset values for peptide subsequence cropping.
        linear_size (list[int]): Sizes of intermediate linear layers.
        dropout (float, optional): Dropout rate (default: 0.5).
        pooling (bool, optional): Whether to apply global max pooling (default: True).
    """
    def __init__(self, *, conv_num, conv_size, conv_off, linear_size, dropout=0.5, pooling=True, **kwargs):
        super(DeepMHCII, self).__init__(**kwargs)
        self.conv_off = conv_off
        self.pooling = pooling
        self.dropout = nn.Dropout(dropout)

        # Convolutions for one-hot and embedding inputs
        self.conv_onehot = nn.ModuleList(IConv(cn, cs, self.mhc_len) for cn, cs in zip(conv_num, conv_size))
        self.conv = nn.ModuleList(IConv(cn, cs, self.mhc_len) for cn, cs in zip(conv_num, conv_size))
        self.conv_onehot_bn = nn.ModuleList(nn.BatchNorm1d(cn) for cn in conv_num)
        self.conv_bn = nn.ModuleList(nn.BatchNorm1d(cn) for cn in conv_num)

        # Linear transformation layers
        linear_size = [sum(conv_num) * 2] + linear_size
        self.linear = nn.ModuleList(nn.Conv1d(in_s, out_s, 1) for in_s, out_s in zip(linear_size[:-1], linear_size[1:]))
        self.linear_bn = nn.ModuleList(nn.BatchNorm1d(out_s) for out_s in linear_size[1:])
        self.output = nn.Conv1d(linear_size[-1], 1, 1)

        # Pre-processing convolutions for one-hot encoding
        self.conv_mhc = nn.Conv1d(21, 64, kernel_size=3, padding=1)
        self.conv_pep = nn.Conv1d(21, 64, kernel_size=3, padding=1)
        self.relu = nn.ReLU()

        self.reset_parameters()

    def forward(self, peptide_x, mhc_x, peptide_one_hot, mhc_one_hot, pooling=None, **kwargs):
        """
        Forward pass of DeepMHCII.

        Args:
            peptide_x (Tensor): Peptide indices (B × L).
            mhc_x (Tensor): MHC indices (B × L).
            peptide_one_hot (Tensor): One-hot encoded peptide (B × L × 21).
            mhc_one_hot (Tensor): One-hot encoded MHC (B × L × 21).
            pooling (bool, optional): Override default pooling behavior.

        Returns:
            Tensor: Binding affinity predictions (B,).
        """
        peptide_x, mhc_x, masks = super().forward(peptide_x, mhc_x)

        # One-hot convolutions
        peptide_one_hot = self.conv_pep(peptide_one_hot.transpose(1, 2)).transpose(1, 2)
        mhc_one_hot = self.conv_mhc(mhc_one_hot.transpose(1, 2)).transpose(1, 2)

        conv_one_hot_out = torch.cat([
            bn(F.relu(conv(peptide_one_hot[:, off: peptide_one_hot.shape[1] - off], mhc_one_hot)))
            for conv, bn, off in zip(self.conv_onehot, self.conv_onehot_bn, self.conv_off)
        ], dim=1)
        conv_one_hot_out = self.dropout(conv_one_hot_out)

        # Embedded convolutions
        conv_embeded_out = torch.cat([
            bn(F.relu(conv(peptide_x[:, off: peptide_x.shape[1] - off], mhc_x)))
            for conv, bn, off in zip(self.conv, self.conv_bn, self.conv_off)
        ], dim=1)
        conv_embeded_out = self.dropout(conv_embeded_out)

        conv_out = torch.cat([conv_one_hot_out, conv_embeded_out], dim=1)

        # Linear + batch norm
        for linear, bn in zip(self.linear, self.linear_bn):
            conv_out = bn(F.relu(linear(conv_out)))
        conv_out = self.dropout(conv_out)

        # Mask + output
        masks = masks[:, None, -conv_out.shape[2]:]
        if pooling or self.pooling:
            pool_out, _ = conv_out.masked_fill(~masks, -np.inf).max(dim=2, keepdim=True)
            return torch.sigmoid(self.output(pool_out).flatten())
        else:
            return torch.sigmoid(self.output(conv_out)).masked_fill(~masks, -np.inf).squeeze(1)

    def reset_parameters(self):
        """Reset all parameters with proper initializations."""
        super().reset_parameters()
        self.conv_pep.reset_parameters()
        self.conv_mhc.reset_parameters()

        for conv, bn in zip(self.conv, self.conv_bn):
            conv.reset_parameters()
            bn.reset_parameters()
            nn.init.normal_(bn.weight.data, mean=1.0, std=0.002)

        for linear, bn in zip(self.linear, self.linear_bn):
            truncated_normal_(linear.weight, std=0.02)
            nn.init.zeros_(linear.bias)
            bn.reset_parameters()
            nn.init.normal_(bn.weight.data, mean=1.0, std=0.002)

        truncated_normal_(self.output.weight, std=0.1)
        nn.init.zeros_(self.output.bias)


class ResMHCII(Network):
    """
    Residual convolutional architecture for peptide–MHCII binding prediction.

    Uses:
      - Embedding and one-hot convolution branches
      - ResNet-style bottleneck blocks
      - Global average pooling + fully connected layers

    Args:
        conv_num (list[int]): Number of convolutional filters per layer.
        conv_size (list[int]): Kernel sizes for convolutional filters.
        conv_off (list[int]): Offset values for peptide subsequence cropping.
        linear_size (list[int]): (Unused here, kept for API consistency).
        dropout (float, optional): Dropout rate (default: 0.5).
        pooling (bool, optional): Global pooling flag (default: True).
    """
    def __init__(self, *, conv_num, conv_size, conv_off, linear_size, dropout=0.5, pooling=True, **kwargs):
        super(ResMHCII, self).__init__(**kwargs)
        self.conv_off = conv_off
        self.pooling = pooling
        self.dropout = nn.Dropout(dropout)

        # Convolutional modules
        self.conv_onehot = nn.ModuleList(IConv(cn, cs, self.mhc_len) for cn, cs in zip(conv_num, conv_size))
        self.conv = nn.ModuleList(IConv(cn, cs, self.mhc_len) for cn, cs in zip(conv_num, conv_size))
        self.conv_onehot_bn = nn.ModuleList(nn.BatchNorm1d(cn) for cn in conv_num)
        self.conv_bn = nn.ModuleList(nn.BatchNorm1d(cn) for cn in conv_num)

        # One-hot preprocessing convolutions
        self.conv_mhc = nn.Conv1d(21, 64, kernel_size=3, padding=1)
        self.conv_pep = nn.Conv1d(21, 64, kernel_size=3, padding=1)
        self.relu = nn.ReLU()

        # ResNet-like blocks
        block = BottleNeck
        num_block = [3, 4, 6, 3]
        self.in_channels = 128
        self.conv2_x = self._make_layer(block, 128, num_block[0], 1)
        self.conv3_x = self._make_layer(block, 128, num_block[1], 2)
        self.conv4_x = self._make_layer(block, 256, num_block[2], 2)

        # Fully connected layers
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(256 * block.expansion, 32)
        self.fc2 = nn.Linear(32, 1)

        self.reset_parameters()

    def forward(self, peptide_x, mhc_x, peptide_one_hot, mhc_one_hot, pooling=None, **kwargs):
        """
        Forward pass of ResMHCII.

        Returns:
            Tensor: Binding affinity predictions (B,).
        """
        peptide_x, mhc_x, masks = super().forward(peptide_x, mhc_x)

        peptide_one_hot = self.conv_pep(peptide_one_hot.transpose(1, 2)).transpose(1, 2)
        mhc_one_hot = self.conv_mhc(mhc_one_hot.transpose(1, 2)).transpose(1, 2)

        conv_one_hot_out = torch.cat([
            bn(F.relu(conv(peptide_one_hot[:, off: peptide_one_hot.shape[1] - off], mhc_one_hot)))
            for conv, bn, off in zip(self.conv_onehot, self.conv_onehot_bn, self.conv_off)
        ], dim=1)
        conv_one_hot_out = self.dropout(conv_one_hot_out)

        conv_embeded_out = torch.cat([
            bn(F.relu(conv(peptide_x[:, off: peptide_x.shape[1] - off], mhc_x)))
            for conv, bn, off in zip(self.conv, self.conv_bn, self.conv_off)
        ], dim=1)
        conv_embeded_out = self.dropout(conv_embeded_out)

        x = torch.cat([conv_one_hot_out, conv_embeded_out], dim=1)
        x = self.conv2_x(x)
        x = self.conv3_x(x)
        x = self.conv4_x(x)

        x = self.avg_pool(x).view(x.size(0), -1)
        x = self.fc(x)
        x = self.fc2(x)
        return torch.sigmoid(x.flatten())

    def reset_parameters(self):
        """Reset all parameters with appropriate initialization."""
        super().reset_parameters()
        self.conv_pep.reset_parameters()
        self.conv_mhc.reset_parameters()

        for conv, bn in zip(self.conv, self.conv_bn):
            conv.reset_parameters()
            bn.reset_parameters()
            nn.init.normal_(bn.weight.data, mean=1.0, std=0.002)

    def _make_layer(self, block, out_channels, num_blocks, stride):
        """Construct a ResNet-like layer with repeated blocks."""
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(block(self.in_channels, out_channels, s))
            self.in_channels = out_channels * block.expansion
        return nn.Sequential(*layers)


class BottleNeck(nn.Module):
    """
    Residual bottleneck block for ResMHCII.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        stride (int, optional): Convolution stride (default: 1).
    """
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.residual_function = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(out_channels, out_channels * self.expansion, kernel_size=1, bias=False),
            nn.BatchNorm1d(out_channels * self.expansion),
        )

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels * self.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels * self.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels * self.expansion)
            )

    def forward(self, x):
        """Forward pass with residual connection."""
        return F.relu(self.residual_function(x) + self.shortcut(x))
