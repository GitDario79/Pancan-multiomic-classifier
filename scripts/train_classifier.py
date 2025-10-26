import argparse, os, joblib, numpy as np, pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, default="data/synthetic")
    ap.add_argument("--out_dir", type=str, default="artifacts/toy")
    args = ap.parse_args()

    X = pd.read_csv(os.path.join(args.data_dir, "X.csv"))
    y = pd.read_csv(os.path.join(args.data_dir, "y.csv"))["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    pipe = Pipeline([
        ("scaler", StandardScaler(with_mean=True, with_std=True)),
        ("pca", PCA(n_components=0.95, svd_solver="full")),
        ("clf", LogisticRegression(max_iter=200, n_jobs=None, multi_class="auto"))
    ])

    pipe.fit(X_train, y_train)

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    joblib.dump(pipe, os.path.join(args.out_dir, "model.joblib"))
    X_test.to_csv(os.path.join(args.out_dir, "X_test.csv"), index=False)
    y_test.to_csv(os.path.join(args.out_dir, "y_test.csv"), index=False)
    print("Saved model and test split.")

if __name__ == "__main__":
    main()
