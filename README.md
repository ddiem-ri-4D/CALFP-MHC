# CALFP-MHC: Interpretable Pan-Allelic Prediction of Peptide-MHC Binding and Presentation Using Chemically Grounded Fingerprints and Contrastive Learning 

A deep learning framework using **repurposed amino acid molecular fingerprints** and **supervised contrastive learning** to jointly predict peptide–MHC **binding affinity (BA)** and **presentation probability (PB)** for both **HLA class I and class II** in a single unified model.

![pipeline](https://github.com/ddiem-ri-4D/CALFP-MHC/blob/main/figs/CALFP_v2.png)

---

## Build

Users can configure the environment themselves, or use the Conda YAML file provided by us:

```bash
conda env create -f calfp.yaml
conda activate CALFP
```

## Usage

`scripts/predict.py` is used for making predictions of binding affinity (BA) and presentation score (PB) for HLA class I and class II.

### Command

Run from the repository root:

```bash
python scripts/predict.py --input input/test.parquet --output prediction/test_predictions.parquet --gpu False --BA True
```

Training (2-stage: SupCon pretrain → fine-tune), also run from the repository root:

```bash
python scripts/train_presentation.py --train_csv data/el_train_fold0.csv --val_csv data/el_val_fold0.csv --fold 0
python scripts/train_affinity.py     --train_csv data/ba_train_fold0.csv --val_csv data/ba_val_fold0.csv --fold 0
```

On the Artemis SLURM cluster, submit the array job (5 EL folds + 5 BA folds) from the repository root:

```bash
mkdir -p logs
sbatch slurm/slurm_train_calfp.sh
```

## Project structure

```
CALFP-MHC/
├── calfp/                     # importable package — no need to run these directly
│   ├── models/                # CALFP_PS, CALFP_BA, ContrastiveProjectionHead
│   ├── encoding/               # FingerprintResidueEncoder (MACCS+ECFP4+ECFP6+RDKit, positional encoding)
│   ├── data/                   # dataset / CSV loading utilities
│   └── losses/                 # SupConLoss
├── scripts/                    # entry points — run these
│   ├── predict.py
│   ├── train_presentation.py
│   └── train_affinity.py
├── slurm/                      # SLURM batch scripts for Artemis
├── resources/
│   └── HLA_library.csv         # allele -> pseudo-sequence lookup
├── params/                     # trained model checkpoints (el_fold*.params, ba_fold*.params)
├── data/, input/, prediction/, figs/
├── calfp.yaml, README.md, LICENSE
```

## Input Format

The input CSV file requires the following columns:

| peptide | Allele Name |
|---|---|
| GILGFVFTL | HLA-A*02:01 |
| SAVRLRSSVPGVR | HLA-B*08:01 |
| PKYVKQNTLKLAT | HLA-DRB1*03:01 |

- **peptide** — amino acid sequence (HLA-I: 8–11 aa; HLA-II: 13–18 aa)
- **Allele Name** — HLA allele in standard format (e.g. `HLA-A*02:01`, `HLA-DRB1*03:01`)
---
## Output Format

| peptide | Allele Name | PB_score | BA_score |
|---|---|---|---|
| GILGFVFTL | HLA-A*02:01 | 0.91 | 0.87 |
| SAVRLRSSVPGVR | HLA-B*08:01 | 0.74 | 0.65 |
| PKYVKQNTLKLAT | HLA-DRB1*03:01 | 0.83 | 0.71 |

- **PB_score** — presentation probability (> 0.5 = likely presented on cell surface)
- **BA_score** — binding affinity score (> 0.5 ≈ IC50 < 500 nM, reported when `--BA True`)

---

## Supported HLA Alleles

CALFP-MHC supports pan-allelic prediction for:
- **HLA class I**: HLA-A, HLA-B, HLA-C (48 alleles benchmarked)
- **HLA class II**: HLA-DR, HLA-DP, HLA-DQ (53 alleles benchmarked)

For a full list of supported alleles, see `data/supported_alleles.txt`.

---

## Citation

If you use this code or data in your research, please cite:

