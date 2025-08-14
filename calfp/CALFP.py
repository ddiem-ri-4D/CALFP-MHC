#CALFP.py

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from calfp.data_utils import ACIDS
from calfp.modules import *

__all__ = ['CALFP', 'LinearPredictor', 'SupCALFP', 'LinCALFP', 'BinCALFP']

def pad_to_length(x, length):
    if x.dim() == 2:
        x = x.unsqueeze(-1)
    if x.size(1) > length:
        return x[:, :length, :]
    elif x.size(1) < length:
        return F.pad(x, (0, 0, 0, length - x.size(1)))
    return x

class EmbeddingLayer(nn.Module):
    def __init__(self, *, emb_size, vocab_size=len(ACIDS), padding_idx=0, peptide_pad=3, mhc_len=34, **kwargs):
        super(EmbeddingLayer, self).__init__()
        self.peptide_emb = nn.Embedding(vocab_size, emb_size)
        self.mhc_emb = nn.Embedding(vocab_size, emb_size)
        self.peptide_pad, self.padding_idx, self.mhc_len = peptide_pad, padding_idx, mhc_len

    def forward(self, peptide_x, mhc_x, *args, **kwargs):
        masks = peptide_x[:, self.peptide_pad: peptide_x.shape[1] - self.peptide_pad] != self.padding_idx
        return self.peptide_emb(peptide_x.long()), self.mhc_emb(mhc_x.long()), masks

    def reset_parameters(self):
        nn.init.uniform_(self.peptide_emb.weight, -0.1, 0.1)
        nn.init.uniform_(self.mhc_emb.weight, -0.1, 0.1)


class MHSA(nn.Module):
    def __init__(self, n_dims, width=14, heads=4):
        super(MHSA, self).__init__()
        self.heads = heads

        self.query = nn.Conv1d(n_dims, n_dims, kernel_size=1)
        self.key = nn.Conv1d(n_dims, n_dims, kernel_size=1)
        self.value = nn.Conv1d(n_dims, n_dims, kernel_size=1)

        self.rel_w = nn.Parameter(torch.randn([1, heads, n_dims // heads, width]), requires_grad=True)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        n_batch, C, width = x.size()
        q = self.query(x).view(n_batch, self.heads, C // self.heads, -1)
        k = self.key(x).view(n_batch, self.heads, C // self.heads, -1)
        v = self.value(x).view(n_batch, self.heads, C // self.heads, -1)

        content_content = torch.matmul(q.permute(0, 1, 3, 2), k)

        rel_w = self.rel_w  # (1, heads, C//heads, width)
        if rel_w.shape[-1] != q.shape[-1]:
            rel_w = rel_w.squeeze(0)  # (heads, C//heads, width)
            rel_w = F.interpolate(rel_w, size=q.shape[-1], mode='linear', align_corners=False)  # (heads, C//heads, new_width)
            rel_w = rel_w.unsqueeze(0)  # (1, heads, C//heads, new_width)

        content_position = torch.matmul(rel_w.permute(0, 1, 3, 2), q)
        energy = content_content + content_position
        attention = self.softmax(energy)

        out = torch.matmul(v, attention.permute(0, 1, 3, 2))
        out = out.view(n_batch, C, width)
        return out

class Bottleneck(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, heads=4, mhsa=False, resolution=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv1d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(planes)
        if not mhsa:
            self.conv2 = nn.Conv1d(planes, planes, kernel_size=3, padding=1, stride=stride, bias=False)
        else:
            self.conv2 = nn.Sequential(
                MHSA(planes, width=int(resolution), heads=heads),
                nn.AvgPool1d(2) if stride == 2 else nn.Identity()
            )
        self.bn2 = nn.BatchNorm1d(planes)
        self.conv3 = nn.Conv1d(planes, self.expansion * planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm1d(self.expansion * planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_planes, self.expansion * planes, kernel_size=1, stride=stride),
                nn.BatchNorm1d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class CALFP(EmbeddingLayer):
    def __init__(self, *, conv_num, conv_size, conv_off, heads=4, dropout=0.5, pooling=True, **kwargs):
        super(CALFP, self).__init__(**kwargs)
        self.conv_fp = nn.ModuleList(IConv(cn, cs, self.mhc_len) for cn, cs in zip(conv_num, conv_size))
        self.conv_emb = nn.ModuleList(IConv(cn, cs, self.mhc_len) for cn, cs in zip(conv_num, conv_size))
        self.bn_fp = nn.ModuleList(nn.BatchNorm1d(cn) for cn in conv_num)
        self.bn_emb = nn.ModuleList(nn.BatchNorm1d(cn) for cn in conv_num)
        self.conv_off = conv_off
        self.dropout = nn.Dropout(dropout)
        self.pooling = pooling

        self.conv_mhc = nn.Conv1d(1, 64, kernel_size=3, padding=1)
        self.conv_pep = nn.Conv1d(1, 64, kernel_size=3, padding=1)

        self.in_planes = 128
        self.resolution = 16
        block = Bottleneck
        num_blocks = [2, 3, 2]
        self.layer1 = self._make_layer(block, 128, num_blocks[0], stride=1, heads=heads, mhsa=True)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.avgpool = nn.AdaptiveAvgPool1d(1)

    def _make_layer(self, block, planes, num_blocks, stride=1, heads=4, mhsa=False):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, heads, mhsa, self.resolution))
            if stride == 2:
                self.resolution /= 2
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, peptide_x, mhc_x, peptide_fp, mhc_fp, pooling=None, **kwargs):
        peptide_x, mhc_x, masks = super().forward(peptide_x, mhc_x)

        max_fp_len = self.mhc_len  # thường là 34

        peptide_fp = pad_to_length(peptide_fp, max_fp_len)
        mhc_fp = pad_to_length(mhc_fp, max_fp_len)

        pep_fp = self.conv_pep(peptide_fp.permute(0, 2, 1))  # [B, D, L]
        mhc_fp = self.conv_mhc(mhc_fp.permute(0, 2, 1))      # [B, D, L]

        pep_fp = pep_fp.transpose(1, 2)
        mhc_fp = mhc_fp.transpose(1, 2)

        fp_out = torch.cat([
            bn(F.relu(conv(pep_fp[:, off: max(pep_fp.shape[1] - off, off + 1)], mhc_fp)))
            for conv, bn, off in zip(self.conv_fp, self.bn_fp, self.conv_off)
        ], dim=1)
        fp_out = self.dropout(fp_out)

        emb_out = torch.cat([
            bn(F.relu(conv(peptide_x[:, off: max(peptide_x.shape[1] - off, off + 1)], mhc_x)))
            for conv, bn, off in zip(self.conv_emb, self.bn_emb, self.conv_off)
        ], dim=1)

        emb_out = self.dropout(emb_out)

        min_len = min(fp_out.shape[2], emb_out.shape[2])
        fp_out = fp_out[:, :, :min_len]
        emb_out = emb_out[:, :, :min_len]

        conv_out = torch.cat([fp_out, emb_out], dim=1)
        masks = masks[:, -conv_out.shape[2]:]

        x = self.layer1(conv_out)
        if pooling or self.pooling:
            x = self.layer2(x)
            x = self.layer3(x)
            x = self.avgpool(x)
            x = torch.flatten(x, 1)
        else:
            x = torch.sigmoid(torch.mean(x, dim=1)).masked_fill(~masks, -np.inf)
        return x

class SupCALFP(nn.Module):
    def __init__(self, *, conv_num, conv_size, conv_off, heads=4, dropout=0.5, pooling=True, **kwargs):
        super().__init__()
        self.encoder = CALFP(conv_num=conv_num, conv_size=conv_size, conv_off=conv_off,
                              heads=heads, dropout=dropout, pooling=pooling, **kwargs)
        self.head = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(256, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 16)
        )

    def forward(self, peptide_x, mhc_x, peptide_fp, mhc_fp, pooling=None, **kwargs):
        feature = self.encoder(peptide_x, mhc_x, peptide_fp, mhc_fp, pooling=pooling, **kwargs)
        return F.normalize(self.head(feature), dim=1)


class LinearPredictor(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.fc = nn.Linear(input_size, 1)

    def forward(self, x):
        return torch.sigmoid(self.fc(x).flatten())


class LinCALFP(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.network = SupCALFP(**kwargs).cuda()
        self.classifier = LinearPredictor(256).cuda()

    def forward(self, inputs, **kwargs):
        features = self.network.encoder(*(x.cuda() for x in inputs), **kwargs)
        return self.classifier(features)

class BinCALFP(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.network = SupCALFP(**kwargs).cuda()
        self.classifier = LinearPredictor(256).cuda()

    def forward(self, inputs, **kwargs):
        return self.network.encoder(*(x.cuda() for x in inputs), **kwargs)

    def forward_binding(self, peptide_x, mhc_x, peptide_fp, mhc_fp):
        # Không dùng pooling => trả về score theo từng vị trí
        return self.network.encoder(peptide_x, mhc_x, peptide_fp, mhc_fp, pooling=False)
