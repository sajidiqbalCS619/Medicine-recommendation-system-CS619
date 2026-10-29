"""
trainer.py - data preparation, model training, tuning and evaluation.

Implements SRS FR-01 .. FR-16 (data collection/pre-processing, training of
several models, hyper-parameter tuning, cross-validation, metric comparison,
model persistence).  The same functions are used by

  * Notebooks/model_comparison.ipynb   (offline experiments), and
  * the Admin panel of the app          (UC-06 dataset upload, UC-07 retrain).

Key design decisions (documented in the report, chapter 3):

1. The dataset contains only 304 unique symptom patterns repeated ~16 times.
   Removing duplicates (FR-03) BEFORE splitting prevents identical rows from
   appearing in both train and test, which would inflate accuracy.
2. Chat users type 2-4 symptoms while training rows have 3-17.  To make the
   model robust to sparse input, the TRAINING part only is augmented with
   random symptom subsets ("symptom dropout").  The test part is never
   augmented, so reported scores stay honest.
3. Hyper-parameters are tuned with GridSearchCV (stratified k-fold).  The
   tuning objective is the mean of accuracy on full symptom sets and accuracy
   when only 3 symptoms are kept, because that is what the chatbot sees.
"""

from __future__ import annotations

import gzip
import json
import pickle
import time
import warnings
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.exceptions import ConvergenceWarning
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=ConvergenceWarning)

TARGET = "prognosis"
RANDOM_STATE = 42
MAX_UPLOAD_MB = 50
SPARSE_KS = (2, 3, 4)  # number of symptoms kept in the "sparse input" test


class DatasetError(ValueError):
    """Raised when an uploaded dataset is not usable (UC-06 alternative path)."""


# ---------------------------------------------------------------------------
# FR-01 .. FR-05  data loading, validation, cleaning
# ---------------------------------------------------------------------------
INVALID_FORMAT_MSG = (
    "Invalid file format. Please upload a CSV or JSON file with required columns."
)


def load_dataset(file_or_path, filename: Optional[str] = None) -> pd.DataFrame:
    """Read a CSV or JSON file (path or file-like object) into a DataFrame."""
    name = (filename or getattr(file_or_path, "name", None) or str(file_or_path)).lower()
    try:
        if name.endswith(".csv"):
            return pd.read_csv(file_or_path)
        if name.endswith(".json"):
            return pd.read_json(file_or_path)
    except Exception as exc:  # malformed content
        raise DatasetError(INVALID_FORMAT_MSG) from exc
    raise DatasetError(INVALID_FORMAT_MSG)


def validate_dataframe(df: pd.DataFrame) -> None:
    """Check that the frame has the required columns (UC-06 step 4)."""
    if df is None or df.empty:
        raise DatasetError(INVALID_FORMAT_MSG)
    cols = [str(c).strip() for c in df.columns]
    if TARGET not in cols:
        raise DatasetError(INVALID_FORMAT_MSG + f" Missing column: '{TARGET}'.")
    symptom_cols = [c for c in cols if c != TARGET and not c.lower().startswith("unnamed")]
    if len(symptom_cols) < 2:
        raise DatasetError(INVALID_FORMAT_MSG + " At least 2 symptom columns are required.")


def clean_dataset(df: pd.DataFrame, dedupe: bool = True) -> Tuple[pd.DataFrame, pd.Series, dict]:
    """FR-02..FR-04: missing values, duplicates, normalisation.

    Returns (X, y, report).  X holds 0/1 symptom columns, y the disease names.
    """
    validate_dataframe(df)
    report = {"rows_in": int(len(df)), "cols_in": int(df.shape[1])}
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]

    unnamed = [c for c in df.columns if c.lower().startswith("unnamed")]
    df = df.drop(columns=unnamed)
    report["unnamed_columns_dropped"] = len(unnamed)

    # target: strip whitespace, drop rows without a label
    df[TARGET] = df[TARGET].astype(str).str.strip()
    df = df[df[TARGET].str.lower().ne("nan") & df[TARGET].ne("")]

    symptom_cols = [c for c in df.columns if c != TARGET]
    sym = df[symptom_cols].apply(pd.to_numeric, errors="coerce")
    report["missing_values_filled"] = int(sym.isna().sum().sum())
    sym = sym.fillna(0)
    sym = (sym > 0).astype(int)  # binarise
    # drop symptom columns that never occur (carry no information)
    nonconst = [c for c in symptom_cols if sym[c].sum() > 0]
    report["empty_symptom_columns_dropped"] = len(symptom_cols) - len(nonconst)
    sym = sym[nonconst]

    clean = pd.concat([sym, df[TARGET]], axis=1)
    if dedupe:
        before = len(clean)
        clean = clean.drop_duplicates()
        report["duplicates_removed"] = int(before - len(clean))
    else:
        report["duplicates_removed"] = 0
    clean = clean.reset_index(drop=True)

    counts = clean[TARGET].value_counts()
    if counts.size < 2:
        raise DatasetError("The dataset must contain at least 2 different diseases.")
    if counts.min() < 2:
        bad = ", ".join(counts[counts < 2].index[:5])
        raise DatasetError(
            f"Every disease needs at least 2 distinct records after cleaning. Too few: {bad}."
        )

    X = clean.drop(columns=TARGET)
    y = clean[TARGET]
    report.update(
        rows_out=int(len(clean)),
        n_symptoms=int(X.shape[1]),
        n_diseases=int(counts.size),
        min_rows_per_disease=int(counts.min()),
        max_rows_per_disease=int(counts.max()),
        avg_symptoms_per_row=round(float(X.sum(axis=1).mean()), 2),
    )
    return X, y, report


def encode_labels(y: pd.Series) -> Tuple[np.ndarray, LabelEncoder]:
    """FR-05: label-encode the disease names."""
    le = LabelEncoder()
    return le.fit_transform(y), le


# ---------------------------------------------------------------------------
# Sparse-input helpers (augmentation + robustness evaluation)
# ---------------------------------------------------------------------------
def keep_k_symptoms(X: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
    """Return a copy of X where every row keeps at most k random active symptoms."""
    out = np.zeros_like(X)
    for i, row in enumerate(X):
        idx = np.flatnonzero(row)
        if len(idx) > k:
            idx = rng.choice(idx, k, replace=False)
        out[i, idx] = 1
    return out


def augment_symptom_subsets(
    X: np.ndarray, y: np.ndarray, n_per_row: int = 30, kmin: int = 2, seed: int = RANDOM_STATE
) -> Tuple[np.ndarray, np.ndarray]:
    """Add `n_per_row` random symptom subsets (size kmin..all) for every row."""
    rng = np.random.default_rng(seed)
    rows, labels = [X], [y]
    for _ in range(n_per_row):
        Xa = np.zeros_like(X)
        for i, row in enumerate(X):
            idx = np.flatnonzero(row)
            lo = min(kmin, len(idx))
            k = rng.integers(lo, len(idx) + 1) if len(idx) else 0
            keep = rng.choice(idx, k, replace=False) if k else []
            Xa[i, keep] = 1
        rows.append(Xa)
        labels.append(y)
    return np.vstack(rows), np.concatenate(labels)


class AugmentedClassifier(BaseEstimator, ClassifierMixin):
    """Wrapper that augments the training data inside fit().

    This lets GridSearchCV / cross-validation augment only the TRAIN folds, so
    no augmented copy of a validation row can leak into training.
    """

    def __init__(self, base=None, n_aug: int = 30, kmin: int = 2, seed: int = RANDOM_STATE):
        self.base = base
        self.n_aug = n_aug
        self.kmin = kmin
        self.seed = seed

    def fit(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y)
        if self.n_aug > 0:
            X, y = augment_symptom_subsets(X, y, self.n_aug, self.kmin, self.seed)
        self.model_ = clone(self.base).fit(X, y)
        self.classes_ = self.model_.classes_
        return self

    def predict(self, X):
        return self.model_.predict(np.asarray(X))

    def predict_proba(self, X):
        return self.model_.predict_proba(np.asarray(X))


def sparse_scorer(estimator, X, y, k: int = 3, seed: int = 0) -> float:
    """Tuning objective: mean(accuracy on full input, accuracy with k symptoms)."""
    Xv, yv = np.asarray(X), np.asarray(y)
    full = accuracy_score(yv, estimator.predict(Xv))
    sparse = accuracy_score(yv, estimator.predict(keep_k_symptoms(Xv, k, np.random.default_rng(seed))))
    return (full + sparse) / 2


# ---------------------------------------------------------------------------
# FR-06 .. FR-09  model zoo
# ---------------------------------------------------------------------------
def model_zoo() -> Dict[str, Tuple[BaseEstimator, dict]]:
    """name -> (base estimator, GridSearchCV parameter grid)."""
    return {
        "Random Forest": (
            RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1),
            {"base__n_estimators": [100, 200], "base__max_depth": [None, 20],
             "base__min_samples_leaf": [1, 2]},
        ),
        "Decision Tree": (
            DecisionTreeClassifier(random_state=RANDOM_STATE),
            {"base__max_depth": [None, 10, 20], "base__min_samples_leaf": [1, 2],
             "base__criterion": ["gini", "entropy"]},
        ),
        "SVM": (
            SVC(kernel="rbf", probability=False, random_state=RANDOM_STATE),
            {"base__C": [1, 10, 100], "base__gamma": ["scale", 0.05]},
        ),
        "Neural Network (MLP)": (
            MLPClassifier(hidden_layer_sizes=(128,), max_iter=300, random_state=RANDOM_STATE),
            {"base__hidden_layer_sizes": [(128,), (128, 64)], "base__alpha": [1e-4, 1e-2]},
        ),
    }


try:  # XGBoost is named in the SRS; it is optional so the app still runs without it
    from xgboost import XGBClassifier

    _zoo_base = model_zoo

    def model_zoo() -> Dict[str, Tuple[BaseEstimator, dict]]:  # noqa: F811
        zoo = _zoo_base()
        zoo["XGBoost"] = (
            XGBClassifier(n_estimators=150, max_depth=6, learning_rate=0.1, tree_method="hist",
                          random_state=RANDOM_STATE, n_jobs=-1),
            {"base__n_estimators": [100, 200], "base__max_depth": [4, 6], "base__learning_rate": [0.1, 0.3]},
        )
        return zoo
except ImportError:
    pass

MODEL_NAMES = list(model_zoo().keys())


# ---------------------------------------------------------------------------
# FR-10 .. FR-11  evaluation
# ---------------------------------------------------------------------------
def evaluate(model, X: np.ndarray, y: np.ndarray, seed: int = 123) -> dict:
    """Accuracy, precision, recall, F1 (weighted + macro) and sparse-input accuracy."""
    pred = model.predict(X)
    p_w, r_w, f_w, _ = precision_recall_fscore_support(y, pred, average="weighted", zero_division=0)
    _, _, f_m, _ = precision_recall_fscore_support(y, pred, average="macro", zero_division=0)
    out = {
        "accuracy": float(accuracy_score(y, pred)),
        "precision": float(p_w),
        "recall": float(r_w),
        "f1": float(f_w),
        "f1_macro": float(f_m),
    }
    for k in SPARSE_KS:
        accs = [
            accuracy_score(y, model.predict(keep_k_symptoms(X, k, np.random.default_rng(seed + r))))
            for r in range(10)
        ]
        out[f"acc_{k}_symptoms"] = float(np.mean(accs))
    return out


def cross_validate_model(base, X: np.ndarray, y: np.ndarray, n_aug: int = 30, n_splits: int = 5) -> dict:
    """k-fold CV (FR-15).  Augmentation happens only on the training folds."""
    n_splits = max(2, min(n_splits, int(np.bincount(y).min())))
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    full, k3 = [], []
    for tr, va in skf.split(X, y):
        clf = AugmentedClassifier(base=base, n_aug=n_aug).fit(X[tr], y[tr])
        full.append(accuracy_score(y[va], clf.predict(X[va])))
        k3.append(accuracy_score(y[va], clf.predict(keep_k_symptoms(X[va], 3, np.random.default_rng(0)))))
    return {
        "cv_folds": n_splits,
        "cv_accuracy_mean": float(np.mean(full)), "cv_accuracy_std": float(np.std(full)),
        "cv_acc_3_symptoms_mean": float(np.mean(k3)), "cv_acc_3_symptoms_std": float(np.std(k3)),
    }


# ---------------------------------------------------------------------------
# FR-13 .. FR-16  train, tune, compare, select
# ---------------------------------------------------------------------------
def train_and_compare(
    X: pd.DataFrame,
    y: pd.Series,
    models: Optional[List[str]] = None,
    tune: bool = True,
    augment: bool = True,
    n_aug: int = 30,
    test_size: float = 0.2,
    progress: Optional[Callable[[str], None]] = None,
) -> dict:
    """Train every selected model, evaluate on the held-out test split and
    pick the best.  Returns a dict with fitted models and JSON-safe metrics."""
    say = progress or (lambda msg: None)
    models = models or MODEL_NAMES
    zoo = model_zoo()
    symptom_columns = list(X.columns)
    y_enc, le = encode_labels(y)
    Xv = X.values.astype(int)

    # a stratified split needs at least one test row per class (matters for small uploads)
    n_classes = len(le.classes_)
    test_size = max(test_size, min((n_classes + 1) / len(y_enc), 0.5))
    X_tr, X_te, y_tr, y_te = train_test_split(
        Xv, y_enc, test_size=test_size, random_state=RANDOM_STATE, stratify=y_enc
    )
    aug_n = n_aug if augment else 0
    results = {}
    for name in models:
        base, grid = zoo[name]
        say(f"Training {name} ...")
        t0 = time.time()
        best_params, tune_score = {}, None
        if tune:
            n_splits = max(2, min(4, int(np.bincount(y_tr).min())))
            gs = GridSearchCV(
                AugmentedClassifier(base=base, n_aug=aug_n),
                grid,
                scoring=sparse_scorer,
                cv=StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE),
                n_jobs=1,
            )
            gs.fit(X_tr, y_tr)
            best_params = {k.replace("base__", ""): _py(v) for k, v in gs.best_params_.items()}
            tune_score = float(gs.best_score_)
            base = clone(base).set_params(**best_params)
        if isinstance(base, SVC):
            base = clone(base).set_params(probability=True)  # needed for confidence scores

        fitted = AugmentedClassifier(base=base, n_aug=aug_n).fit(X_tr, y_tr)
        train_time = time.time() - t0
        plain = fitted.model_  # plain scikit-learn model (picklable, no custom class)

        test_m = evaluate(plain, X_te, y_te)
        train_acc = float(accuracy_score(y_tr, plain.predict(X_tr)))
        cv_m = cross_validate_model(clone(base), Xv, y_enc, n_aug=aug_n)
        robust_cv = (cv_m["cv_accuracy_mean"] + cv_m["cv_acc_3_symptoms_mean"]) / 2
        results[name] = {
            "estimator": plain,
            "best_params": best_params,
            "tuning_score": tune_score,
            "train_accuracy": train_acc,
            "overfit_gap": train_acc - test_m["accuracy"],
            "train_seconds": round(train_time, 2),
            "robust_cv_score": robust_cv,
            **{f"test_{k}": v for k, v in test_m.items()},
            **cv_m,
        }
        say(f"  {name}: test acc {test_m['accuracy']:.3f}, CV robust {robust_cv:.3f}")

    best = select_best(results, models)
    return {
        "label_encoder": le,
        "symptom_columns": symptom_columns,
        "results": results,
        "best": best,
        "split": {"train_rows": int(len(y_tr)), "test_rows": int(len(y_te)),
                  "augmented_train_rows": int(len(y_tr) * (aug_n + 1)) if aug_n else int(len(y_tr))},
        "settings": {"tuned": tune, "augmented": augment, "n_aug": aug_n},
        "X_train": X_tr, "y_train": y_tr, "X_test": X_te, "y_test": y_te,
    }


SELECTION_TOLERANCE = 0.005  # differences below 0.5 percentage points are ties


def select_best(results: dict, order_hint: Optional[List[str]] = None) -> str:
    """FR-16: pick the best model.

    Primary criterion: cross-validated robust score (mean of accuracy on full
    input and accuracy with 3 symptoms).  Models within SELECTION_TOLERANCE of
    the top score are treated as tied; ties go to the higher test F1 and then
    to the earlier model in the list (Random Forest first: interpretable).
    """
    names = list(results)
    order = {n: i for i, n in enumerate(order_hint or names)}
    top = max(results[n]["robust_cv_score"] for n in names)
    tied = [n for n in names if results[n]["robust_cv_score"] >= top - SELECTION_TOLERANCE]
    return sorted(tied, key=lambda n: (-round(results[n]["test_f1"], 4), order.get(n, 99)))[0]


def _py(v):
    """Convert numpy scalars to plain Python for JSON."""
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return float(v)
    return v


def metrics_table(run: dict) -> pd.DataFrame:
    """FR-16: model comparison table."""
    rows = []
    for name, r in run["results"].items():
        rows.append({
            "Model": name,
            "Accuracy": r["test_accuracy"], "Precision": r["test_precision"],
            "Recall": r["test_recall"], "F1": r["test_f1"],
            "Acc@2 symptoms": r["test_acc_2_symptoms"], "Acc@3 symptoms": r["test_acc_3_symptoms"],
            "Acc@4 symptoms": r["test_acc_4_symptoms"],
            "CV acc (mean)": r["cv_accuracy_mean"], "CV acc (std)": r["cv_accuracy_std"],
            "CV acc@3 (mean)": r["cv_acc_3_symptoms_mean"],
            "Train acc": r["train_accuracy"], "Overfit gap": r["overfit_gap"],
            "Train sec": r["train_seconds"], "Best params": json.dumps(r["best_params"]),
        })
    df = pd.DataFrame(rows).set_index("Model")
    df["Selected"] = ["yes" if m == run["best"] else "" for m in df.index]
    return df


def json_safe_metrics(run: dict, name: str) -> dict:
    """Metrics of one model without the estimator object."""
    return {k: _py(v) for k, v in run["results"][name].items() if k != "estimator"}


# ---------------------------------------------------------------------------
# FR-12  persistence (.pkl bundles)
# ---------------------------------------------------------------------------
def save_bundle(path: str, model, label_encoder, symptom_columns: List[str], metadata: dict) -> None:
    """One .pkl file = model + label encoder + symptom columns + metadata.

    The pickle is gzip-compressed (a tuned Random Forest is ~50 MB raw, ~10 MB
    compressed) so that it fits GitHub's web-upload limit.  load_bundle() reads
    both compressed bundles and plain pickles.
    """
    payload = {"model": model, "label_encoder": label_encoder,
               "symptom_columns": list(symptom_columns), "metadata": metadata}
    with gzip.open(path, "wb", compresslevel=6) as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_bundle(path: str) -> dict:
    with open(path, "rb") as f:
        magic = f.read(2)
    opener = gzip.open if magic == b"\x1f\x8b" else open
    with opener(path, "rb") as f:
        return pickle.load(f)


def predict_proba_from_symptoms(bundle: dict, symptoms: List[str]) -> np.ndarray:
    cols = bundle["symptom_columns"]
    vec = np.zeros((1, len(cols)))
    for s in symptoms:
        if s in cols:
            vec[0, cols.index(s)] = 1
    model = bundle["model"]
    if hasattr(model, "feature_names_in_"):  # model was fitted on a DataFrame
        vec = pd.DataFrame(vec, columns=cols)
    return model.predict_proba(vec)[0]
