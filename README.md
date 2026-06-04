# CALFP-MHC: Molecular Fingerprint Representations and Supervised Contrastive Learning for Pan-Allelic Peptide–MHC Binding and Presentation Prediction 

A deep learning framework using **repurposed amino acid molecular fingerprints** and **supervised contrastive learning** to jointly predict peptide–MHC **binding affinity (BA)** and **presentation probability (PB)** for both **HLA class I and class II** in a single unified model.

![pipeline](https://github.com/ddiem-ri-4D/CALFP-MHC/blob/main/figs/CALFP_v2.png)

---

## Build

Users can configure the environment themselves, or use the Conda YAML file provided by us:

```bash
conda env create -f calfp.yaml
conda activate CALFP
```

---
## Test

To test your installation, make sure you are in the CALFP-MHC directory and run:

```bash
python CALFP_MHC.py --input test.csv --output test_out.csv
```

---
## Usage

`CALFP_MHC.py` is used for making predictions of binding affinity (BA) and presentation score (PB) for HLA class I and class II.

### Command

```bash
python CALFP_MHC.py --input test.csv --output test_out.csv --gpu False --BA True
```

## Input Format

The input CSV file requires the following columns:

| peptide | allele |
|---|---|
| GILGFVFTL | HLA-A*02:01 |
| SAVRLRSSVPGVR | HLA-B*08:01 |
| PKYVKQNTLKLAT | HLA-DRB1*03:01 |

- **peptide** — amino acid sequence (HLA-I: 8–11 aa; HLA-II: 13–18 aa)
- **allele** — HLA allele in standard format (e.g. `HLA-A*02:01`, `HLA-DRB1*03:01`)
---
## Output Format

| peptide | allele | PB_score | BA_score |
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

