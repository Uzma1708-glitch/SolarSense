"""
encoding.py
--------------------------------------------------------------------
Handles categorical variable detection and encoding.
Includes:
  - Automatic detection of categorical columns
  - One-Hot Encoding (OHE) for nominal columns (PLANT_ID, SOURCE_KEY)
  - Label Encoding for columns
  - Dynamic details (unique values count, before/after shapes, and newly created columns)
--------------------------------------------------------------------
"""

import pandas as pd
from sklearn.preprocessing import LabelEncoder


def detect_categorical_columns(df: pd.DataFrame) -> list:
    """Detect categorical columns based on dtype or low cardinality objects."""
    cols = []
    for col in df.columns:
        # Ignore temporal columns
        if col in ["DATE_TIME", "DATE", "HOUR", "DAY_NUM"]:
            continue
        # Check if dtype is object, category, or integer with low cardinality (like PLANT_ID)
        if df[col].dtype == "object" or isinstance(df[col].dtype, pd.CategoricalDtype):
            cols.append(col)
        elif df[col].dtype in ["int64", "int32"] and df[col].nunique() < 30:
            # Low cardinality identifiers (like PLANT_ID) are categorical in nature
            cols.append(col)
    return cols


def apply_encoding(df: pd.DataFrame, method: str = "none") -> tuple:
    """
    Apply either One-Hot Encoding or Label Encoding to detected categorical columns.
    Returns:
        - Encoded DataFrame
        - Dict with encoding details: before_shape, after_shape, encoded_cols, new_cols_created
    """
    df = df.copy()
    cat_cols = detect_categorical_columns(df)

    before_shape = df.shape
    new_cols_created = []

    if method == "none" or not cat_cols:
        return df, {
            "method": method,
            "cat_cols": cat_cols,
            "before_shape": before_shape,
            "after_shape": before_shape,
            "new_cols_created": [],
            "unique_counts": {col: df[col].nunique() for col in cat_cols}
        }

    unique_counts = {col: df[col].nunique() for col in cat_cols}

    if method == "onehot":
        # One-Hot encoding using pandas.get_dummies
        df_encoded = pd.get_dummies(df, columns=cat_cols, dtype=int)
        after_shape = df_encoded.shape
        new_cols_created = [col for col in df_encoded.columns if col not in df.columns]
        df = df_encoded
    elif method == "label":
        # Label Encoding using sklearn's LabelEncoder
        for col in cat_cols:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            new_cols_created.append(col)
        after_shape = df.shape

    return df, {
        "method": method,
        "cat_cols": cat_cols,
        "before_shape": before_shape,
        "after_shape": after_shape,
        "new_cols_created": new_cols_created,
        "unique_counts": unique_counts
    }


if __name__ == "__main__":
    from load_data import load_clean

    df = load_clean()
    print("Categorical columns detected:", detect_categorical_columns(df))
    df_enc, report = apply_encoding(df, method="onehot")
    print("Encoding report:", report)
    print("Encoded df head columns:", list(df_enc.columns[:10]))
