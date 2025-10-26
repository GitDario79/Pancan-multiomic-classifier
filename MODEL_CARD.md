# Model Card — Pancan Multi‑Omic Cancer Type Classifier

**Task**: Multiclass classification of cancer type from multi‑omic features (e.g., RNA‑seq + additional omics).  
**Intended use**: Educational and research showcase; **not for clinical use**.  
**Inputs**: Tabular features derived from multi‑omics (toy: synthetic data generator).  
**Outputs**: Predicted class and class probabilities.

## Data
- **Toy**: Generated with `scripts/generate_synthetic_multiomic.py` (balanced/unbalanced options).
- **Real**: TCGA-based processed features (not bundled). Document how to obtain in your main README.

## Metrics
- Macro F1, balanced accuracy, ROC-AUC (one-vs-rest), confusion matrix.
- On toy data, expect modest but stable metrics; real metrics depend on your feature set and preprocessing.

## Training & Evaluation
- One command toy run: `make run-toy` (creates data → trains model → saves figures/metrics).
- See `tests/` for sanity checks (non‑NaN metrics, expected shapes).

## Ethical Considerations & Limitations
- Clinical predictions are **unsafe** without rigorous validation, bias assessment, and regulatory review.
- Class imbalance and domain shift can degrade performance.
- Interpretability: include SHAP/feature importances if appropriate.

## Versioning
- Release tagged as `v0.1-toy` contains toy-trained artifacts for demonstration.
