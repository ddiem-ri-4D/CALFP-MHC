# CALFP-MHC: Repurposing amino acid molecular fingerprints for supervised contrastive learning in MHC class I and II peptide binding prediction

This repository contains code and data for training and evaluating **CALFP-MHC**, a deep learning framework that leverages **repurposed amino acid molecular fingerprints** for supervised contrastive learning in predicting MHC class I and II peptide binding.

![pipeline](https://github.com/ddiem-ri-4D/CALFP-MHC/blob/main/figs/CALFP.png)

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
```

## Create and activate conda environment
```bash
conda env create -f environment.yml
conda activate CALFP
```

## Data format

The model requires **Parquet** format (`.parquet`) with at least the following columns:

| Peptide          | Allele    | MHC sequence (pseudosequence)          | Label |
|---------------   |-----------|----------------------------------------|-------|
| SAVRLRSSVPGVR    | DRB1_0401 | QEFFIASGAAVDAIMEVHFDYYDLQKATYHVGFT     | 1     |
| NPVVHFFKNIVTPRTP | DRB5_0101 | QEFFIASGAAVDAIMQDYFHDYDFDRATYHVGFT     | 0     |
| ENPVVHFFKNIVTP   | DRB1_1501 | QEFFIASGAAVDAIMWPRFDYFDIQAATYHVVFT     | 1     |
| SAVRLRSSVPGVR    | DRB1_0402 | QEFFIASGAAVDAIMEVHFDYYDIDEATYHVVFT     | 0     |
| MPLAQMLLPTAMRMKM | DRB1_0101 | QEFFIASGAAVDAIMWLFLECYDLQRATYHVGFT     | 1     |

**Columns**:  
- **Peptide** — amino acid sequence of the peptide  
- **Allele** — HLA class II allele identifier (e.g., `DRB1_0401`, `DRB5_0101`)  
- **MHC sequence** — pseudosequence of the MHC allele  
- **Label** — binding indicator (`1` = binding, `0` = non-binding), required only for training   

---
## How to Run CALFP-MHC

### 1. Train and evaluate on an independent test set
```bash
python -u calfp_pretrain.py -d config/data.yaml --mode train
python -u calfp_fine_tuning.py -d config/data.yaml --mode train
```

### 2. 5-fold cross-validation
```bash 
python -u calfp_pretrain.py -d config/data.yaml --mode 5cv
python -u calfp_fine_tuning.py -d config/data.yaml --mode 5cv
```

### 3. Leave-One-Molecule-Out cross-validation
```bash
python -u calfp_pretrain.py -d config/data.yaml --mode lomo
python -u calfp_fine_tuning.py -d config/data.yaml --mode lomo
```

## Citation
If you use this code or data in your research, please cite:


