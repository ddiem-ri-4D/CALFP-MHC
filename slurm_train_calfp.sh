#!/bin/bash
#SBATCH --job-name=calfp_train
#SBATCH --output=logs/calfp_train_%A_%a.out
#SBATCH --error=logs/calfp_train_%A_%a.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a4000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=48:00:00
#SBATCH --array=0-9

# ─────────────────────────────────────────────────────────────────────────
# Trains CALFP_PS (presentation) and CALFP_BA (affinity), 5 folds each,
# as one SLURM array job (10 tasks total: 5 EL folds + 5 BA folds).
#
#   array index 0-4 -> EL fold 0-4  (train_presentation.py)
#   array index 5-9 -> BA fold 0-4  (train_affinity.py)
#
# Edit TRAIN_DIR / DATA_ROOT below to point at your actual fold CSVs
# (data/el_train_fold{N}.csv, data/el_val_fold{N}.csv, etc. — you need
# to have already split your labeled data into 5 folds per head).
#
# Usage:
#   mkdir -p logs
#   sbatch slurm_train_calfp.sh
# ─────────────────────────────────────────────────────────────────────────

set -euo pipefail

# --- environment ------------------------------------------------------
eval "$(micromamba shell hook --shell bash)"
micromamba activate calfp   # adjust to your actual env name

# --- paths --------------------------------------------------------------
PROJECT_DIR="$(pwd)"
DATA_ROOT="${PROJECT_DIR}/data"          # expects el_train_fold{N}.csv etc.
HLA_LIB="${PROJECT_DIR}/HLA_library.csv"
OUTPUT_DIR="${PROJECT_DIR}/params_new"
mkdir -p "${OUTPUT_DIR}" logs

FOLD=$(( SLURM_ARRAY_TASK_ID % 5 ))

if [ "${SLURM_ARRAY_TASK_ID}" -lt 5 ]; then
    echo "=== Training EL (presentation) fold ${FOLD} ==="
    python train_presentation.py \
        --train_csv "${DATA_ROOT}/el_train_fold${FOLD}.csv" \
        --val_csv   "${DATA_ROOT}/el_val_fold${FOLD}.csv" \
        --hla_lib   "${HLA_LIB}" \
        --fold ${FOLD} \
        --output_dir "${OUTPUT_DIR}" \
        --epochs_pretrain 30 \
        --epochs_finetune 100 \
        --batch_size 256 \
        --lr_pretrain 1e-4 \
        --lr_finetune 1e-4 \
        --weight_decay 1e-4 \
        --temperature 0.07 \
        --patience 10 \
        --device cuda
else
    echo "=== Training BA (affinity) fold ${FOLD} ==="
    python train_affinity.py \
        --train_csv "${DATA_ROOT}/ba_train_fold${FOLD}.csv" \
        --val_csv   "${DATA_ROOT}/ba_val_fold${FOLD}.csv" \
        --hla_lib   "${HLA_LIB}" \
        --fold ${FOLD} \
        --output_dir "${OUTPUT_DIR}" \
        --epochs_pretrain 30 \
        --epochs_finetune 100 \
        --batch_size 256 \
        --lr_pretrain 1e-4 \
        --lr_finetune 1e-4 \
        --weight_decay 1e-4 \
        --temperature 0.07 \
        --pearson_weight 1.0 \
        --patience 10 \
        --device cuda
fi

echo "Done: array task ${SLURM_ARRAY_TASK_ID} (fold ${FOLD})"
