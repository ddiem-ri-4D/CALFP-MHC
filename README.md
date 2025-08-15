# CALFP-MHC: Repurposing amino acid molecular fingerprints for supervised contrastive learning in MHC class I and II peptide binding prediction

This repository contains code and data for training and evaluating **CALFP-MHC**, a deep learning framework that leverages **repurposed amino acid molecular fingerprints** for supervised contrastive learning in predicting MHC class I and II peptide binding.

![pipeline](https://github.com/ddiem-ri-4D/CALFP-MHC/blob/main/figs/calfp_model.png)

---

## Requirements

- Python >= 3.6.8  
- Keras == 2.6.0  
- TensorFlow == 2.6.0  
- Other dependencies are listed in `environment.yml`

---

## Installation

```bash
# Clone repository
git clone https://github.com/ddiem-ri-4D/CALFP-MHC
cd CALFP-MHC/

# Create and activate conda environment
conda env create -f environment.yml
conda activate CALFP

## Data Format

The model requires **Parquet** format (`.parquet`) with at least the following columns:

| Peptide       | MHC       | Label |
|---------------|-----------|-------|
| AASSYGQNFV    | QIKVRVDMV | 1     |
| AIRAGGDEQ     | HSKKKCDEL | 1     |
| AISETDKLG     | LPPIVAKEI | 1     |
| SARDRVRTDTQY  | FVSKLYYFE | 0     |
| SARDRVRTDTQY  | KLSHQPVLL | 0     |

Columns:
- Peptide: amino acid sequence of the peptide  
- MHC: amino acid sequence of the MHC allele  
- Label: binding indicator (`1` = binding, `0` = non-binding) — only required for training  

Example files:
- `train.parquet` — training set  
- `test.parquet` — independent test set  

---
## How to Run CALFP-MHC

### 1. Train and evaluate on an independent test set
```bash
python -u conbotnet_pretrain.py -d config/data.yaml --mode train
python -u conbotnet_fine_tuning.py -d config/data.yaml --mode train
```

### 2. 5-fold cross-validation
```bash 
python -u conbotnet_pretrain.py -d config/data.yaml --mode 5cv
python -u conbotnet_fine_tuning.py -d config/data.yaml --mode 5cv
```

### 3. Leave-One-Molecule-Out cross-validation
```bash
python -u conbotnet_pretrain.py -d config/data.yaml --mode lomo
python -u conbotnet_fine_tuning.py -d config/data.yaml --mode lomo
```

## Citation
If you use this code or data in your research, please cite: