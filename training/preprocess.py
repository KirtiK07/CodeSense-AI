import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

DATA_PATH = "data/jm1.csv"

# Candidate names for the target column, in priority order.
TARGET_CANDIDATES = ["defects", "defect", "defective", "class", "bug", "label"]

# Values that should map to 1 (defective) / 0 (non-defective).
TRUE_VALUES = {"true", "yes", "y", "1", "1.0", True, 1}
FALSE_VALUES = {"false", "no", "n", "0", "0.0", False, 0}


def find_target_column(df: pd.DataFrame) -> str:
    """Locate the defect-label column regardless of exact naming."""
    lower_cols = {c.lower().strip(): c for c in df.columns}
    for candidate in TARGET_CANDIDATES:
        if candidate in lower_cols:
            return lower_cols[candidate]
    # Fall back: last column is almost always the label in this dataset family.
    return df.columns[-1]


def normalize_target(series: pd.Series) -> pd.Series:
    """Map whatever encoding the target uses (bool/str/int) to clean 0/1 ints."""
    def _map(v):
        if pd.isna(v):
            return np.nan
        v_norm = str(v).strip().lower()
        if v_norm in TRUE_VALUES or v in TRUE_VALUES:
            return 1
        if v_norm in FALSE_VALUES or v in FALSE_VALUES:
            return 0
        # Already numeric 0/1
        try:
            return int(float(v))
        except (ValueError, TypeError):
            return np.nan
    return series.apply(_map)


def load_and_clean(path: str = DATA_PATH):
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]

    target_col = find_target_column(df)
    print(f"[preprocess] Using '{target_col}' as the target column.")

    df[target_col] = normalize_target(df[target_col])
    df = df.dropna(subset=[target_col])
    df[target_col] = df[target_col].astype(int)

    feature_cols = [c for c in df.columns if c != target_col]

    # JM1 has a handful of missing numeric values (e.g. in 'uniq_Op'/'uniq_Opnd').
    # Median imputation keeps outlier-heavy metrics (loc, effort) from being skewed
    # the way mean imputation would.
    for col in feature_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        if df[col].isna().any():
            df[col] = df[col].fillna(df[col].median())

    X = df[feature_cols]
    y = df[target_col]

    print(f"[preprocess] Loaded {len(df)} rows, {len(feature_cols)} features.")
    print(f"[preprocess] Class balance before SMOTE: "
          f"{y.value_counts(normalize=True).round(3).to_dict()}")

    return X, y, feature_cols


def get_train_test_split(test_size: float = 0.2, random_state: int = 42):
    """
    Returns X_train, X_test, y_train, y_test, feature_cols.

    SMOTE is applied ONLY to the training set — never to the test set —
    so evaluation numbers reflect real-world class imbalance, not synthetic data.
    """
    X, y, feature_cols = load_and_clean()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    smote = SMOTE(random_state=random_state)
    X_train_bal, y_train_bal = smote.fit_resample(X_train, y_train)

    print(f"[preprocess] Training rows before SMOTE: {len(X_train)}, "
          f"after SMOTE: {len(X_train_bal)}")

    return X_train_bal, X_test, y_train_bal, y_test, feature_cols


if __name__ == "__main__":
    # Quick manual check: run `python training/preprocess.py` from the project root
    # to confirm the dataset loads and balances correctly before training.
    X_train, X_test, y_train, y_test, feature_cols = get_train_test_split()
    print("Feature columns:", feature_cols)
    print("Train shape:", X_train.shape, "Test shape:", X_test.shape)