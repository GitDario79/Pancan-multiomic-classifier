import argparse, os, joblib, numpy as np, pandas as pd
from pathlib import Path
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifact_dir", type=str, default="artifacts/toy")
    ap.add_argument("--fig_dir", type=str, default="figures")
    args = ap.parse_args()

    model = joblib.load(os.path.join(args.artifact_dir, "model.joblib"))
    X_test = pd.read_csv(os.path.join(args.artifact_dir, "X_test.csv"))
    y_test = pd.read_csv(os.path.join(args.artifact_dir, "y_test.csv"))["target"]

    y_pred = model.predict(X_test)
    y_proba = None
    try:
        y_proba = model.predict_proba(X_test)
    except Exception:
        pass

    print(classification_report(y_test, y_pred, digits=3))

    cm = confusion_matrix(y_test, y_pred)
    Path(args.fig_dir).mkdir(parents=True, exist_ok=True)

    # Confusion Matrix Plot
    plt.figure()
    import itertools
    classes = np.unique(y_test)
    plt.imshow(cm, interpolation="nearest")
    plt.title("Confusion Matrix")
    plt.xticks(range(len(classes)), classes)
    plt.yticks(range(len(classes)), classes)
    thresh = cm.max() / 2.0
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], "d"),
                 horizontalalignment="center")
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.tight_layout()
    plt.savefig(os.path.join(args.fig_dir, "confusion_matrix.png"), dpi=150)

    # ROC-AUC (one-vs-rest) if we have probabilities
    if y_proba is not None:
        y_true_bin = label_binarize(y_test, classes=classes)
        try:
            auc = roc_auc_score(y_true_bin, y_proba, average="macro", multi_class="ovr")
            with open(os.path.join(args.fig_dir, "metrics.txt"), "w") as f:
                f.write(f"macro_ovr_roc_auc: {auc:.4f}\n")
        except Exception as e:
            with open(os.path.join(args.fig_dir, "metrics.txt"), "w") as f:
                f.write(f"roc_auc_error: {e}\n")

if __name__ == "__main__":
    main()
