# 🚀 Quickstart (Toy Data)
Reproduce the pipeline **without downloading TCGA** using a tiny synthetic dataset.

```bash
# create and activate env (example)
python -m venv .venv && . .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements-dev.txt

# run end-to-end on toy data
make run-toy
```

## What this adds
- `data/synthetic/` generator for a **toy multi-omic dataset**
- `Makefile` with reproducible targets (`env`, `data`, `train`, `eval`)
- Minimal `tests/` and GitHub Actions CI
- `MODEL_CARD.md` documenting task, metrics, and limitations
- README embeds for **confusion matrix** and **ROC/AUC** (ensure your training code saves the figures into `figures/`)

> After verifying toy run, switch your config back to the real dataset for proper results.
