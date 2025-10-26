import argparse, os, json
import numpy as np
import pandas as pd
from pathlib import Path

def make_block(n, p, scale=1.0, seed=None):
    rng = np.random.default_rng(seed)
    return rng.normal(0, scale, size=(n, p))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=str, default="data/synthetic")
    ap.add_argument("--n", type=int, default=600)
    ap.add_argument("--omics", type=str, default="rna,proteomics,cnv")
    ap.add_argument("--classes", type=int, default=6)
    ap.add_argument("--imbalance", type=float, default=0.0, help="0 balanced, up to ~0.8 skew")
    args = ap.parse_args()

    Path(args.out).mkdir(parents=True, exist_ok=True)

    omics = [o.strip() for o in args.omics.split(",") if o.strip()]
    n = args.n
    k = args.classes
    rng = np.random.default_rng(42)

    # class distribution
    if args.imbalance > 0:
        weights = np.array([ (1 - args.imbalance) ** i for i in range(k) ], dtype=float)
        weights = weights / weights.sum()
    else:
        weights = np.ones(k) / k

    y = rng.choice(np.arange(k), size=n, p=weights)
    df_list = []
    meta = {"omics": {}, "classes": int(k), "n_samples": int(n), "weights": weights.tolist()}

    for om in omics:
        p = {"rna": 300, "proteomics": 80, "cnv": 60}.get(om, 50)  # simple dims
        X = make_block(n, p, scale=1.0, seed=hash(om) % (2**32))
        # class-specific shift to make it learnable
        centers = rng.normal(0, 2.0, size=(k, p))
        for i in range(n):
            X[i] += centers[y[i]]
        cols = [f"{om}_{i:03d}" for i in range(p)]
        df_list.append(pd.DataFrame(X, columns=cols))

        meta["omics"][om] = {"features": p}

    X_df = pd.concat(df_list, axis=1)
    y_df = pd.DataFrame({"target": y})

    X_df.to_csv(os.path.join(args.out, "X.csv"), index=False)
    y_df.to_csv(os.path.join(args.out, "y.csv"), index=False)
    with open(os.path.join(args.out, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Wrote synthetic dataset to {args.out} with shape {X_df.shape}")

if __name__ == "__main__":
    main()
