"""
train_presentation.py
----------------------
Two-stage training for CALFP_PS, matching the manuscript's Methods:

Stage 1 — Supervised contrastive pre-training (full network trainable):
    pooled encoder output -> ContrastiveProjectionHead -> SupConLoss (tau=0.07)
Stage 2 — Fine-tuning (encoder frozen, per Fig.1d "Frozen encoder"):
    only feature_selection (classification head) is trained, BCE loss.

Usage:
    python train_presentation.py \\
        --train_csv data/el_train_fold0.csv \\
        --val_csv   data/el_val_fold0.csv \\
        --hla_lib   HLA_library.csv \\
        --fold 0 \\
        --output_dir params_new/ \\
        --epochs_pretrain 30 --epochs_finetune 100 \\
        --batch_size 256 --lr_pretrain 1e-4 --lr_finetune 1e-4 \\
        --patience 10 --device cuda

Input CSV columns required: peptide, allele, label (0/1 binder).
Output: params_new/el_fold{fold}.params  (state_dict, loadable by predict.py)
"""

import os as _os
import sys as _sys
_sys.path.insert(0, _os.path.abspath(_os.path.join(_os.path.dirname(__file__), '..')))


import argparse
import copy
import os
import sys
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from calfp.models.presentation_model import CALFP_PS, ContrastiveProjectionHead
from calfp.losses.supcon_loss import SupConLoss
from calfp.data.train_data_utils import LabeledPepMHCDataset, load_hla_library, read_labeled_file


def build_parser():
    p = argparse.ArgumentParser(description='Train CALFP_PS (presentation score model)')
    p.add_argument('--train_csv', required=True)
    p.add_argument('--val_csv', required=True)
    p.add_argument('--hla_lib', default='resources/HLA_library.csv')
    p.add_argument('--fold', type=int, default=0)
    p.add_argument('--output_dir', default='params_new')
    p.add_argument('--epochs_pretrain', type=int, default=30)
    p.add_argument('--epochs_finetune', type=int, default=100)
    p.add_argument('--batch_size', type=int, default=256)
    p.add_argument('--lr_pretrain', type=float, default=1e-4)
    p.add_argument('--lr_finetune', type=float, default=1e-4)
    p.add_argument('--weight_decay', type=float, default=1e-4)
    p.add_argument('--temperature', type=float, default=0.07)
    p.add_argument('--patience', type=int, default=10,
                    help='Early-stopping patience on val loss (Stage 2 only).')
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--num_workers', type=int, default=4)
    p.add_argument('--seed', type=int, default=0)
    return p


def make_loaders(args):
    hla_lib = load_hla_library(args.hla_lib)
    train_df = read_labeled_file(args.train_csv)
    val_df = read_labeled_file(args.val_csv)
    train_ds = LabeledPepMHCDataset(train_df, hla_lib)
    val_ds = LabeledPepMHCDataset(val_df, hla_lib)
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
        for pep, mhc, label in train_loader:
            pep, mhc, label = pep.to(device), mhc.to(device), label.to(device)
            # Skip batches with fewer than 2 classes present (SupCon needs positives)
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
def evaluate_bce(net, loader, device, criterion):
    net.eval()
    total, n = 0.0, 0
    for pep, mhc, label in loader:
        pep, mhc, label = pep.to(device), mhc.to(device), label.to(device).long()
        logits = net(pep, mhc)
        loss = criterion(logits, label)
        total += loss.item()
        n += 1
    return total / max(n, 1)


def stage2_finetune(net, train_loader, val_loader, args, device):
    print(f'\n[Stage 2] Fine-tuning classification head (encoder frozen) — '
          f'up to {args.epochs_finetune} epochs, patience={args.patience}, '
          f'lr={args.lr_finetune}')

    # Freeze everything except the final classification head, per Fig.1d
    # ("Frozen encoder + 2 independent linear heads").
    for p in net.parameters():
        p.requires_grad = False
    for p in net.head.parameters():
        p.requires_grad = True

    optim = torch.optim.Adam(
        net.head.parameters(),
        lr=args.lr_finetune, weight_decay=args.weight_decay,
    )
    criterion = nn.CrossEntropyLoss()

    best_val = float('inf')
    best_state = copy.deepcopy(net.state_dict())
    epochs_no_improve = 0

    for epoch in range(args.epochs_finetune):
        t0 = time.time()
        net.train()
        # Keep frozen submodules in eval mode (BatchNorm/Dropout stability)
        net.encoder.eval()

        total_loss, n_batches = 0.0, 0
        for pep, mhc, label in train_loader:
            pep, mhc, label = pep.to(device), mhc.to(device), label.to(device).long()
            optim.zero_grad()
            logits = net(pep, mhc)
            loss = criterion(logits, label)
            loss.backward()
            optim.step()
            total_loss += loss.item()
            n_batches += 1
        train_loss = total_loss / max(n_batches, 1)

        val_loss = evaluate_bce(net, val_loader, device, criterion)
        print(f'  epoch {epoch+1:3d}/{args.epochs_finetune}  '
              f'train_bce={train_loss:.4f}  val_bce={val_loss:.4f}  '
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

    net = CALFP_PS().to(device)
    proj_head = ContrastiveProjectionHead(in_dim=net.model_dim).to(device)

    net = stage1_pretrain(net, proj_head, train_loader, args, device)
    net, best_val = stage2_finetune(net, train_loader, val_loader, args, device)

    out_path = os.path.join(args.output_dir, f'el_fold{args.fold}.params')
    torch.save(net.state_dict(), out_path)
    print(f'\nSaved: {out_path}  (best val BCE = {best_val:.4f})')


if __name__ == '__main__':
    main()
