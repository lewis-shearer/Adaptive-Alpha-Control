"""German Credit data loading and preprocessing — second dataset, same
pattern as src/data.py, for a cross-dataset generalization check.

Sensitive attribute is gender (M/F), target is classification (1=good
credit, 0=bad credit). No filtering needed (unlike COMPAS) — this dataset
is already clean, 1000 rows, no missing values.
"""
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

RAW_TSV = "data/German.tsv"

NUMERIC_COLUMNS = [
    "duration", "creditamount", "installmentrate", "residencesince",
    "existingcredits", "peopleliable",
]
CATEGORICAL_COLUMNS = [
    "existingchecking", "credithistory", "purpose", "savings",
    "employmentsince", "otherdebts", "property", "otherinstallmentplans",
    "housing", "job", "telephone", "foreignworker",
]
TARGET_COLUMN = "classification"
SENSITIVE_COLUMN = "gender"


def load_raw(path: str = RAW_TSV) -> pd.DataFrame:
    return pd.read_csv(path, sep="\t")


def preprocess(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42):
    """Returns (X_train, X_test, y_train, y_test, z_train, z_test).

    y = target (classification, already 1/0). z = sensitive attribute,
    encoded as 0/1 where 1 = F, 0 = M (order fixed here so every run/seed
    encodes it identically).
    """
    y = df[TARGET_COLUMN].astype(int).to_numpy()
    z = (df[SENSITIVE_COLUMN] == "F").astype(int).to_numpy()

    X = df[NUMERIC_COLUMNS + CATEGORICAL_COLUMNS].copy()
    X = pd.get_dummies(X, columns=CATEGORICAL_COLUMNS)

    (
        X_train, X_test,
        y_train, y_test,
        z_train, z_test,
    ) = train_test_split(
        X, y, z,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    scaler = StandardScaler()
    X_train = X_train.copy()
    X_test = X_test.copy()
    X_train[NUMERIC_COLUMNS] = scaler.fit_transform(X_train[NUMERIC_COLUMNS])
    X_test[NUMERIC_COLUMNS] = scaler.transform(X_test[NUMERIC_COLUMNS])

    return (
        X_train.to_numpy(dtype="float32"),
        X_test.to_numpy(dtype="float32"),
        y_train.astype("float32"),
        y_test.astype("float32"),
        z_train.astype("float32"),
        z_test.astype("float32"),
    )


def load_german(path: str = RAW_TSV, test_size: float = 0.2, random_state: int = 42):
    df = load_raw(path)
    return preprocess(df, test_size=test_size, random_state=random_state)


if __name__ == "__main__":
    X_train, X_test, y_train, y_test, z_train, z_test = load_german()
    print(f"train: {X_train.shape}, test: {X_test.shape}")
    print(f"features: {X_train.shape[1]}")
    print(f"base rate (y_train mean): {y_train.mean():.4f}")
    print(f"sensitive attr rate (z_train mean, 1=F): {z_train.mean():.4f}")
