import argparse, json, os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import joblib

from utils import load_modalities, encode_labels

def plot_conf_mat(cm, classes, out_path):
    fig = plt.figure()
    plt.imshow(cm, interpolation='nearest')
    plt.title('Confusion Matrix')
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=90)
    plt.yticks(tick_marks, classes)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    fig.savefig(out_path, bbox_inches='tight', dpi=150)
    plt.close(fig)

def main(args):
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.figures_dir, exist_ok=True)

    X, y = load_modalities(args.rna, args.meth, args.mirna, args.label_column)
    y_enc, le = encode_labels(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_enc, test_size=args.test_size, random_state=args.random_state, stratify=y_enc
    )

    scaler = StandardScaler(with_mean=True, with_std=True)
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    n_components = args.n_components if args.n_components > 1 else args.n_components
    pca = PCA(n_components=n_components, random_state=args.random_state)
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_test_pca = pca.transform(X_test_scaled)

    clf = LogisticRegression(
        solver='saga',
        penalty='l1',
        C=1.0,
        max_iter=2000,
        n_jobs=None # saga ignores n_jobs; added for clarity
    )
    clf.fit(X_train_pca, y_train)

    y_pred = clf.predict(X_test_pca)
    acc = accuracy_score(y_test, y_pred)
    f1w = f1_score(y_test, y_pred, average='weighted')

    # Save metrics
    metrics = {
        "accuracy": acc,
        "f1_weighted": f1w,
        "n_components": args.n_components,
        "test_size": args.test_size,
        "random_state": args.random_state,
        "classes": le.classes_.tolist()
    }
    with open(os.path.join(args.output_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    # Save model artifacts
    joblib.dump(clf, os.path.join(args.output_dir, "model.joblib"))
    joblib.dump(pca, os.path.join(args.output_dir, "pca.joblib"))
    joblib.dump(scaler, os.path.join(args.output_dir, "scaler.joblib"))
    joblib.dump(le, os.path.join(args.output_dir, "label_encoder.joblib"))

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred, labels=np.arange(len(le.classes_)))
    plot_conf_mat(cm, le.classes_, os.path.join(args.figures_dir, "confusion_matrix.png"))

    # Classification report to file
    report = classification_report(y_test, y_pred, target_names=le.classes_)
    with open(os.path.join(args.output_dir, "classification_report.txt"), "w") as f:
        f.write(report)

    print(json.dumps(metrics, indent=2))
    print("\nClassification Report:\n")
    print(report)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train PCA+LogReg cancer-type classifier.")
    parser.add_argument('--rna', required=True, help='Path to RNA-seq features CSV (rows=samples, cols=genes).')
    parser.add_argument('--meth', required=True, help='Path to DNA methylation features CSV (rows=samples, cols=probes).')
    parser.add_argument('--mirna', required=True, help='Path to miRNA-seq features CSV (rows=samples, cols=miRNAs).')
    parser.add_argument('--label_column', default='CANCER_TYPE', help='Name of the label column (present in one file).')
    parser.add_argument('--output_dir', default='models', help='Directory to store model artifacts/metrics.')
    parser.add_argument('--figures_dir', default='figures', help='Directory to store plots.')
    parser.add_argument('--test_size', type=float, default=0.2)
    parser.add_argument('--random_state', type=int, default=42)
    parser.add_argument('--n_components', type=int, default=300, help='Number of PCA components (int or variance if <1).')
    args = parser.parse_args()
    main(args)
