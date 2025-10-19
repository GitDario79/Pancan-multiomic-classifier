![CI](https://github.com/GitDario79/pancan-multiomic-classifier/actions/workflows/ci.yml/badge.svg)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)
![Python](https://img.shields.io/badge/Python-3.11+-blue)


# Multi‑Omic Cancer Type Classification (TCGA PanCancer)

A reproducible pipeline and notebook for classifying tumour types from **multi‑omics** data (RNA‑seq, DNA methylation, miRNA‑seq) using the TCGA Pan‑Cancer Atlas dataset.

## Highlights
- **Goal:** Predict `CANCER_TYPE` across 18 cancers from ~549k features per patient.
- **Modality fusion:** RNA‑seq, DNA methylation (beta values), and miRNA‑seq.
- **Dimensionality reduction:** PCA → **300 components** after StandardScaler.
- **Best model:** L1‑regularised Logistic Regression (`saga`) on PCA features.
- **Test performance:** **97.45% accuracy**, **0.97 weighted F1** (stratified 80/20 split).
- **Key insight:** DNA **methylation** signals were consistently the strongest contributors to discriminating cancer types.
- **Challenge:** Extreme class imbalance—rare classes (support < 10) remain hard to model.

> The accompanying notebook (`notebooks/Pancan_project.ipynb`) contains exploratory analysis, model training, and comparisons (LogReg, RandomForest, XGBoost). A report is available in `reports/`.

## Repository Structure
```
pancan-multiomic-classifier/
├─ src/
│  ├─ train.py            # Train/evaluate pipeline from CSV/Parquet paths
│  └─ utils.py            # I/O helpers and metrics
├─ notebooks/
│  └─ Pancan_project.ipynb
├─ reports/
│  └─ Pancan_Project_Report.docx
├─ data/
│  ├─ raw/                # Place downloaded TCGA files here
│  └─ processed/          # Intermediate files (PCA, splits)
├─ models/                # Trained models and artefacts
├─ figures/               # Exported plots
├─ requirements.txt
├─ environment.yml
├─ CITATION.cff
├─ LICENSE
├─ .gitignore
└─ README.md
```

## Data
This project uses multi‑omics data from the **TCGA Pan‑Cancer Atlas** (commonly distributed on Kaggle/TCGA mirrors). You will need three aligned tables per sample:
- RNA‑seq (e.g., TPM)
- DNA methylation (beta values)
- miRNA‑seq (RPM)

> **Note:** Large datasets are not included. See `src/utils.py` for expected input formats and column conventions. Minimal sample schema is documented inline.

## Reproduce
### 1) Create environment
```
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

Or with conda/mamba:
```
mamba env create -f environment.yml
mamba activate pancan-ml
```

### 2) Prepare data
Place modality files under `data/raw/` and ensure sample IDs align across modalities.

### 3) Train baseline
```
python src/train.py   --rna data/raw/rna.csv   --meth data/raw/methylation.csv   --mirna data/raw/mirna.csv   --label_column CANCER_TYPE   --output_dir models   --n_components 300
```


This will:
- scale → PCA (n=300)
- fit **LogisticRegression** (`saga`, `penalty='l1'`)
- print accuracy/F1, save `model.joblib`, `label_encoder.joblib`, and `metrics.json`
- write a confusion matrix to `figures/confusion_matrix.png`

### 4) Evaluate on your split
If you provide a `--test_size` (default 0.2) and `--random_state`, the script will generate a stratified split and report **accuracy** and **weighted F1**.

## Results Summary
- **LogReg (opt):** 97.45% accuracy, 0.97 weighted F1. Rare classes improved vs trees/boosting.
- **RandomForest / XGBoost:** ~95% accuracy; struggled with rare classes (sometimes F1=0.0).
- **PCA impact:** Reducing ~549k features to 300 components delivered >100× speedups with minimal accuracy loss.
- **Biology:** DNA methylation carried strong discriminative signal between cancers.

> See `reports/Pancan_Project_Report.docx` for the full narrative and `notebooks/` for the analysis steps.

## Next Steps (Roadmap)
- **Variance‑targeted PCA** (`n_components=0.95`) to probe rare‑class signals.
- **Imbalance methods:** SMOTE/ADASYN or class‑balanced ensembles.
- **Modality‑specific models:** Methylation‑only baselines; early/late fusion strategies.
- **Interpretability:** Map high‑loading features → gene sets/pathways.

## License
Released under the MIT License (see `LICENSE`).

## Citation
If you use this work, please cite via `CITATION.cff` or:
```
Meacci, D. (2025). Multi‑Omic Cancer Type Classification (TCGA PanCancer). GitHub repository.
```

(Sync test)

