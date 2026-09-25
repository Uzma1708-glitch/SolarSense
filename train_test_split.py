"""
train_test_split.py
--------------------------------------------------------------------
Handles partitioning the processed dataset into training and testing sets.
Includes:
  - Random split (using sklearn train_test_split)
  - Chronological split (preserving date_time sequence, crucial for time-series)
  - Target variable separation (AC_POWER as target, avoiding target leakage)
  - Report of shapes, row counts, and percentages for X_train, X_test, y_train, y_test
--------------------------------------------------------------------
"""

import pandas as pd
from sklearn.model_selection import train_test_split as sklearn_split


def split_data(df: pd.DataFrame, test_size: float = 0.2, split_method: str = "chronological",
               target_col: str = "AC_POWER") -> dict:
    """
    Splits the data into training and testing sets.
    Separates X (input features) and y (target variable).
    Ensures target column is excluded from X to prevent target leakage.
    Also strips helper date columns from features to ensure mathematical input.
    """
    df = df.copy()

    # Define features to exclude to prevent target leakage and handle metadata
    # AC_POWER is target.
    # DATE_TIME, DATE, and any metadata/index column should be excluded or treated.
    # PLANT_ID is constant in this dataset, so it offers no variance and is excluded.
    exclude_cols = [target_col, "DATE_TIME", "DATE", "PLANT_ID"]

    # We also check for derived efficiency columns or other columns containing target info
    for col in ["EFFICIENCY", "IS_ANOMALY"]:
        if col in df.columns:
            exclude_cols.append(col)

    feature_cols = [c for c in df.columns if c not in exclude_cols]

    # X and y
    X = df[feature_cols]
    y = df[target_col]

    n_rows = len(df)
    test_rows = int(n_rows * test_size)
    train_rows = n_rows - test_rows

    if split_method == "chronological":
        # Sort by DATE_TIME if available to ensure correct temporal ordering
        if "DATE_TIME" in df.columns:
            df_sorted = df.sort_values("DATE_TIME")
            X = df_sorted[feature_cols]
            y = df_sorted[target_col]

        X_train = X.iloc[:train_rows]
        X_test = X.iloc[train_rows:]
        y_train = y.iloc[:train_rows]
        y_test = y.iloc[train_rows:]
    else:
        # Random split
        X_train, X_test, y_train, y_test = sklearn_split(
            X, y, test_size=test_size, random_state=42
        )

    return {
        "X_train_shape": X_train.shape,
        "X_test_shape": X_test.shape,
        "y_train_shape": y_train.shape,
        "y_test_shape": y_test.shape,
        "train_rows": len(X_train),
        "test_rows": len(X_test),
        "train_pct": round((len(X_train) / n_rows) * 100, 1),
        "test_pct": round((len(X_test) / n_rows) * 100, 1),
        "features": list(X.columns),
        "target": target_col,
        "split_method": split_method,
        "X_train_preview": X_train.head(5).to_dict("records"),
        "y_train_preview": y_train.head(5).tolist()
    }


if __name__ == "__main__":
    from load_data import load_clean

    df = load_clean()
    # Let's mock encoding SOURCE_KEY using dummies
    df_encoded = pd.get_dummies(df, columns=["SOURCE_KEY"], drop_first=True, dtype=int)
    report = split_data(df_encoded, test_size=0.2, split_method="chronological")
    print("Split features:", report["features"])
    print("Train rows:", report["train_rows"], f"({report['train_pct']}%)")
    print("Test rows:", report["test_rows"], f"({report['test_pct']}%)")
