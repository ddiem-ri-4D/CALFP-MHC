"""
train_affinity.py
------------------
Two-stage training for CALFP_BA, matching the manuscript's Methods:

Stage 1 — Supervised contrastive pre-training (full network trainable):
    Same as train_presentation.py — SupCon needs a binary label, so this
    script expects a 'label' column (0/1 binder) for Stage 1 even though
    Stage 2 trains on a continuous target ('affinity' column: rescaled
    IC50 or %Rank, however you prepared it upstream).
Stage 2 — Fine-tuning (encoder frozen):
    only feature_selection (regression head) is trained,
    loss = MSE + Pearson correlation loss (Methods: "BA Head: MSE +
    Pearson loss → IC50 & %Rank").

Usage:
    python train_affinity.py \\
        --train_csv data/ba_train_fold0.csv \\
        --val_csv   data/ba_val_fold0.csv \\
        --hla_lib   HLA_library.csv \\
        --fold 0 \\
        --output_dir params_new/ \\
        --epochs_pretrain 30 --epochs_finetune 100 \\
        --batch_size 256 --lr_pretrain 1e-4 --lr_finetune 1e-4 \\
        --pearson_weight 1.0 --patience 10 --device cuda

Input CSV columns required: peptide, allele, label (0/1, for Stage-1
SupCon only), affinity (continuous target for Stage-2 regression).
Output: params_new/ba_fold{fold}.params
"""

import argparse
import copy
import os
import time

import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from affinity_model import CALFP_BA
from presentation_model import ContrastiveProjectionHead
from supcon_loss import SupConLoss
from train_data_utils import (
    load_hla_library, read_labeled_file, pearson_loss,
)
from data_utils import encode_sequence, PEP_MAX_LEN, MHC_PSEUDO_LEN


class AffinityDataset(Dataset):
    """Peptide/MHC + binary label (Stage 1) + continuous affinity (Stage 2)."""

    def __init__(self, df: pd.DataFrame, hla_lib: dict):
        missing = {'peptide', 'allele', 'label', 'affinity'} - set(df.columns)
        if missing:
            raise ValueError(f'Training file missing columns: {missing}')
        unknown = set(df['allele']) - set(hla_lib)
        if unknown:
            raise ValueError(f'Unrecognised allele(s): {unknown}')

        self.pep = torch.stack(
            [encode_sequence(p, PEP_MAX_LEN) for p in df['peptide']])
        self.mhc = torch.stack(
            [encode_sequence(hla_lib[a], MHC_PSEUDO_LEN) for a in df['allele']])
        self.label = torch.tensor(df['label'].values, dtype=torch.float32)
        self.affinity = torch.tensor(df['affinity'].values, dtype=torch.float32)

    def __len__(self):
        return len(self.pep)

    def __getitem__(self, idx):
        return self.pep[idx], self.mhc[idx], self.label[idx], self.affinity[idx]


def build_parser():
    p = argparse.ArgumentParser(description='Train CALFP_BA (binding affinity model)')
    p.add_argument('--train_csv', required=True)
    p.add_argument('--val_csv', required=True)
    p.add_argument('--hla_lib', default='HLA_library.csv')
    p.add_argument('--fold', type=int, default=0)
    p.add_argument('--output_dir', default='params_new')
    p.add_argument('--epochs_pretrain', type=int, default=30)
    p.add_argument('--epochs_finetune', type=int, default=100)
    p.add_argument('--batch_size', type=int, default=256)
    p.add_argument('--lr_pretrain', type=float, default=1e-4)
    p.add_argument('--lr_finetune', type=float, default=1e-4)
    p.add_argument('--weight_decay', type=float, default=1e-4)
    p.add_argument('--temperature', type=float, default=0.07)
    p.add_argument('--pearson_weight', type=float, default=1.0,
                    help='Weight on (1 - Pearson r) term added to MSE.')
    p.add_argument('--patience', type=int, default=10)
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--num_workers', type=int, default=4)
    p.add_argument('--seed', type=int, default=0)
    return p


def make_loaders(args):
    hla_lib = load_hla_library(args.hla_lib)
    train_df = read_labeled_file(args.train_csv)
    val_df = read_labeled_file(args.val_csv)
    train_ds = AffinityDataset(train_df, hla_lib)
    val_ds = AffinityDataset(val_df, hla_lib)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                               num_workers=args.num_workers, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers)
    return train_loader, val_loader


def stage1_pretrain(net, proj_head, train_loader, args, device):
    print(f'\n[Stage 1] SupCon pretraining — {args.epochs_pretrain} epochs, '
          f'tau={args.temperature}, lr={args.lr_pretrain}')
    params = list(net.parameters()) + list(proj_head.parameters())
    optim = torch.optim.Adam(params, lr=args.lr_pretrain, weight_decay=args.weight_decay)
    criterion = SupConLoss(temperature=args.temperature)

    net.train()
    proj_head.train()
    for epoch in range(args.epochs_pretrain):
        t0 = time.time()
        total_loss, n_batches = 0.0, 0
        for pep, mhc, label, _affinity in train_loader:
            pep, mhc, label = pep.to(device), mhc.to(device), label.to(device)
            if label.unique().numel() < 2:
                continue
            optim.zero_grad()
            g = net.encode(pep, mhc)
            z = proj_head(g)
            loss = criterion(z, label)
            loss.backward()
            optim.step()
            total_loss += loss.item()
            n_batches += 1
        avg = total_loss / max(n_batches, 1)
        print(f'  epoch {epoch+1:3d}/{args.epochs_pretrain}  '
              f'supcon_loss={avg:.4f}  ({time.time()-t0:.1f}s)')
    return net


@torch.no_grad()
def evaluate_mse_pearson(net, loader, device, pearson_weight):
    net.eval()
    total, n = 0.0, 0
    for pep, mhc, _label, affinity in loader:
        pep, mhc, affinity = pep.to(device), mhc.to(device), affinity.to(device)
        pred = net(pep, mhc)
        mse = torch.nn.functional.mse_loss(pred, affinity)
        pr = pearson_loss(pred, affinity)
        loss = mse + pearson_weight * pr
        total += loss.item()
        n += 1
    return total / max(n, 1)


def stage2_finetune(net, train_loader, val_loader, args, device):
    print(f'\n[Stage 2] Fine-tuning regression head (encoder frozen) — '
          f'up to {args.epochs_finetune} epochs, patience={args.patience}, '
          f'lr={args.lr_finetune}, loss=MSE + {args.pearson_weight}*(1-Pearson r)')

    for p in net.parameters():
        p.requires_grad = False
    for p in net.feature_selection.parameters():
        p.requires_grad = True

    optim = torch.optim.Adam(
        net.feature_selection.parameters(),
        lr=args.lr_finetune, weight_decay=args.weight_decay,
    )

    best_val = float('inf')
    best_state = copy.deepcopy(net.state_dict())
    epochs_no_improve = 0

    for epoch in range(args.epochs_finetune):
        t0 = time.time()
        net.train()
        net.fp_encoder.eval()
        net.conv.eval()
        net.selfattention.eval()

        total_loss, n_batches = 0.0, 0
        for pep, mhc, _label, affinity in train_loader:
            pep, mhc, affinity = pep.to(device), mhc.to(device), affinity.to(device)
            optim.zero_grad()
            pred = net(pep, mhc)
            mse = torch.nn.functional.mse_loss(pred, affinity)
            pr = pearson_loss(pred, affinity)
            loss = mse + args.pearson_weight * pr
            loss.backward()
            optim.step()
            total_loss += loss.item()
            n_batches += 1
        train_loss = total_loss / max(n_batches, 1)

        val_loss = evaluate_mse_pearson(net, val_loader, device, args.pearson_weight)
        print(f'  epoch {epoch+1:3d}/{args.epochs_finetune}  '
              f'train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  '
              f'({time.time()-t0:.1f}s)')

        if val_loss < best_val - 1e-5:
            best_val = val_loss
            best_state = copy.deepcopy(net.state_dict())
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= args.patience:
                print(f'  Early stopping at epoch {epoch+1} '
                      f'(no improvement for {args.patience} epochs).')
                break

    net.load_state_dict(best_state)
    return net, best_val


def main():
    args = build_parser().parse_args()
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    os.makedirs(args.output_dir, exist_ok=True)

    train_loader, val_loader = make_loaders(args)

    net = CALFP_BA().to(device)
    proj_head = ContrastiveProjectionHead(in_dim=net.model_dim).to(device)

    net = stage1_pretrain(net, proj_head, train_loader, args, device)
    net, best_val = stage2_finetune(net, train_loader, val_loader, args, device)

    out_path = os.path.join(args.output_dir, f'ba_fold{args.fold}.params')
    torch.save(net.state_dict(), out_path)
    print(f'\nSaved: {out_path}  (best val loss = {best_val:.4f})')


if __name__ == '__main__':
    main()
