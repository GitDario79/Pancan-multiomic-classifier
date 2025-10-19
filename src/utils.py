import pandas as pd
from typing import Tuple, Optional
from sklearn.preprocessing import LabelEncoder

def load_modalities(rna_path: str, meth_path: str, mirna_path: str, label_column: str) -> Tuple[pd.DataFrame, pd.Series]:
    """Load and align three modality tables by sample ID index.
    Each CSV is expected to have samples as rows and features as columns.
    The label column must be present in *exactly one* of the tables.
    """
    rna = pd.read_csv(rna_path, index_col=0)
    meth = pd.read_csv(meth_path, index_col=0)
    mirna = pd.read_csv(mirna_path, index_col=0)

    # Identify table with labels
    label_series = None
    for df in (rna, meth, mirna):
        if label_column in df.columns:
            label_series = df[label_column]
            df.drop(columns=[label_column], inplace=True)
            break
    if label_series is None:
        raise ValueError(f"Label column '{label_column}' not found in provided files.")

    # Intersect sample IDs
    common_ids = rna.index.intersection(meth.index).intersection(mirna.index).intersection(label_series.index)
    if len(common_ids) == 0:
        raise ValueError("No overlapping sample IDs across modalities and label.")

    rna = rna.loc[common_ids]
    meth = meth.loc[common_ids]
    mirna = mirna.loc[common_ids]
    y = label_series.loc[common_ids]

    # Concatenate features
    X = pd.concat([rna, meth, mirna], axis=1)
    # Fill any remaining NaNs with 0 as per project report
    X = X.fillna(0)
    return X, y

def encode_labels(y: pd.Series) -> Tuple[pd.Series, LabelEncoder]:
    le = LabelEncoder()
    y_enc = pd.Series(le.fit_transform(y.values), index=y.index, name=y.name)
    return y_enc, le
