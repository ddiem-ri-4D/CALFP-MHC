# CALFP-MHC: Molecular Fingerprint Representations and Supervised Contrastive Learning for Pan-Allelic Peptide–MHC Binding and Presentation Prediction 

A deep learning framework using **repurposed amino acid molecular fingerprints** and **supervised contrastive learning** to jointly predict peptide–MHC **binding affinity (BA)** and **presentation probability (PB)** for both **HLA class I and class II** in a single unified model.

![pipeline](https://github.com/ddiem-ri-4D/CALFP-MHC/blob/main/figs/CALFP_v2.png)

---

## Requirements

- python >= 3.6.8  
- torch==2.6.0
- torchvision==0.21.0
- numpy==2.0.2
- pyyaml==6.0.2
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

| peptide          | Allele Name |
|---------------   |-------------|
| SAVRLRSSVPGVR    | HLA-B*08:01 | 
| NPVVHFFKNIVTPRTP | HLA-A*02:01 | 
| ENPVVHFFKNIVTP   | HLA-B*18:01 |
| SAVRLRSSVPGVR    | HLA-A*01:01 | 
| MPLAQMLLPTAMRMKM | HLA-A*02:01 | 

**Columns**:  
- **peptide** — amino acid sequence of the peptide  
- **Allele** — HLA class II allele identifier (e.g., `HLA-B*08:01`, `HLA-A*01:01`)
---
## Test

To test your installation, make sure you are in the CALFP-MHC directory and run:

```bash
python CALFP_MHC.py --input test.parquet --output test_out.parquet
```

## Usage

`CALFP_MHC.py` is used for making predictions of binding affinity (BA) and presentation score (PB) for HLA class I and class II.

## Supported HLA Alleles

CALFP-MHC supports pan-allelic prediction for:
- **HLA class I**: HLA-A, HLA-B, HLA-C (48 alleles benchmarked)
- **HLA class II**: HLA-DR, HLA-DP, HLA-DQ (53 alleles benchmarked)

For a full list of supported alleles, see `data/supported_alleles.txt`.

---

## Citation

If you use this code or data in your research, please cite:

