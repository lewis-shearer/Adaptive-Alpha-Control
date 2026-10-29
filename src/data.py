"""COMPAS data loading and preprocessing.

Filtering follows the original ProPublica analysis (Angwin et al., 2016,
"Machine Bias") — this is the standard filtering used by most COMPAS
fairness papers, not an arbitrary choice:

  - |days_b_screening_arrest| <= 30: screening must happen close to arrest,
    otherwise the charge used for screening may not match the actual arrest.
  - is_recid != -1: -1 means recidivism data was simply missing for that
    person, not that they didn't recidivate.
  - c_charge_degree != 'O': ordinary traffic offenses carry no jail time,
    so they're not meaningful recidivism signal.
  - score_text != 'N/A': rows where COMPAS itself didn't produce a score.

Sensitive attribute is restricted to race in {African-American, Caucasian}
(the two groups ProPublica's own comparison focuses on, and the convention
most follow-up fairness papers use) — other races are dropped. This is a
limitation worth stating explicitly in the paper, same as the binary-gender
limitation in Adult/German.
"""
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

RAW_CSV = "data/compas-scores-two-years.csv"

FEATURE_COLUMNS = [
    "sex",
    "age",
    "age_cat",
    "juv_fel_count",
    "juv_misd_count",
    "juv_other_count",
    "priors_count",
    "c_charge_degree",
]
TARGET_COLUMN = "two_year_recid"
SENSITIVE_COLUMN = "race"
SENSITIVE_GROUPS = ["African-American", "Caucasian"]


def load_raw(path: str = RAW_CSV) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df[
        (df["days_b_screening_arrest"] <= 30)
        & (df["days_b_screening_arrest"] >= -30)
        & (df["is_recid"] != -1)
        & (df["c_charge_degree"] != "O")
        & (df["score_text"] != "N/A")
    ]
    df = df[df[SENSITIVE_COLUMN].isin(SENSITIVE_GROUPS)]
    return df.reset_index(drop=True)


def preprocess(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42):
    """Returns (X_train, X_test, y_train, y_test, z_train, z_test).

    y = target label (two_year_recid). z = sensitive attribute, encoded as
    0/1 where 1 = African-American, 0 = Caucasian (order fixed here so every
    run/seed encodes it identically).
    """
    y = df[TARGET_COLUMN].astype(int).to_numpy()
    z = (df[SENSITIVE_COLUMN] == "African-American").astype(int).to_numpy()

    X = df[FEATURE_COLUMNS].copy()
    X = pd.get_dummies(X, columns=["sex", "age_cat", "c_charge_degree"])

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
    numeric_cols = ["age", "juv_fel_count", "juv_misd_count", "juv_other_count", "priors_count"]
    X_train = X_train.copy()
    X_test = X_test.copy()
    X_train[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
    X_test[numeric_cols] = scaler.transform(X_test[numeric_cols])

    return (
        X_train.to_numpy(dtype="float32"),
        X_test.to_numpy(dtype="float32"),
        y_train.astype("float32"),
        y_test.astype("float32"),
        z_train.astype("float32"),
        z_test.astype("float32"),
    )


def load_compas(path: str = RAW_CSV, test_size: float = 0.2, random_state: int = 42):
    df = load_raw(path)
    return preprocess(df, test_size=test_size, random_state=random_state)


if __name__ == "__main__":
    X_train, X_test, y_train, y_test, z_train, z_test = load_compas()
    print(f"train: {X_train.shape}, test: {X_test.shape}")
    print(f"features: {X_train.shape[1]}")
    print(f"base rate (y_train mean): {y_train.mean():.4f}")
    print(f"sensitive attr rate (z_train mean, 1=African-American): {z_train.mean():.4f}")
