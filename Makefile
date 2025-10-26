.PHONY: env data train eval run-toy lint test ci

env:
	python -m pip install --upgrade pip
	pip install -r requirements-dev.txt

data:
	python scripts/generate_synthetic_multiomic.py --out data/synthetic --n 600 --omics rna,proteomics,cnv --classes 6 --imbalance 0.2

train:
	python scripts/train_classifier.py --data_dir data/synthetic --out_dir artifacts/toy

eval:
	python scripts/evaluate_classifier.py --artifact_dir artifacts/toy --fig_dir figures

run-toy: env data train eval

lint:
	ruff check .
	black --check .
	isort --check-only .

test:
	pytest -q

ci: lint test
