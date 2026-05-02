"""Train a LightGBM baseline with a time-based split."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

MATPLOTLIB_CONFIG_DIR = Path("/private/tmp/lob_project_matplotlib")
XDG_CACHE_DIR = Path("/private/tmp/lob_project_cache")
MATPLOTLIB_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
XDG_CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MATPLOTLIB_CONFIG_DIR))
os.environ.setdefault("XDG_CACHE_HOME", str(XDG_CACHE_DIR))

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_recall_fscore_support


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET_DIR = PROJECT_ROOT / "data" / "processed" / "model_dataset"
DEFAULT_MODEL_DIR = PROJECT_ROOT / "models"
DEFAULT_PREDICTIONS_DIR = PROJECT_ROOT / "data" / "processed" / "model_dataset"
DEFAULT_METRICS_DIR = PROJECT_ROOT / "reports"
TARGET_COL = "target"
CLASS_LABELS = [-1, 0, 1]
PROBABILITY_COLUMNS = {-1: "prob_down", 0: "prob_neutral", 1: "prob_up"}
NON_FEATURE_COLUMNS = {
    "target",
    "timestamp",
    "event_time",
    "exchange",
    "symbol",
    "session_id",
    "future_mid_price",
    "future_price_change",
    "future_return",
    "future_return_bps",
    "split",
    "y_true",
    "y_pred",
    "prob_down",
    "prob_neutral",
    "prob_up",
}
RAW_ABSOLUTE_PRICE_COLUMNS = {"mid_price", "microprice"}
EXPLICIT_FEATURE_COLUMNS = {
    "spread",
    "relative_spread",
    "weighted_bid_depth",
    "weighted_ask_depth",
    "weighted_imbalance",
    "microprice_minus_mid",
    "delta_mid_price",
    "delta_spread",
}


@dataclass(frozen=True)
class TimeSplits:
    """Chronological train/validation/test row slices."""

    train: slice
    validation: slice
    test: slice


@dataclass(frozen=True)
class TrainingSummary:
    """Summary of LightGBM training artifacts."""

    model_path: Path
    predictions_path: Path
    metrics_path: Path
    feature_importance_path: Path
    rows: int
    feature_count: int


@dataclass(frozen=True)
class FeatureSelectionReport:
    """Selected and excluded feature columns for no-raw-price training."""

    total_columns: int
    numeric_columns: list[str]
    selected_features: list[str]
    excluded_raw_price_features: list[str]
    excluded_non_feature_columns: list[str]
    excluded_other_numeric_columns: list[str]

    @property
    def excluded_features(self) -> list[str]:
        """Return every numeric column excluded from model features."""
        return [
            *self.excluded_raw_price_features,
            *self.excluded_non_feature_columns,
            *self.excluded_other_numeric_columns,
        ]


@contextmanager
def suppress_native_stderr():
    """Temporarily suppress native library stderr noise during Parquet I/O."""
    sys.stderr.flush()
    original_stderr_fd = os.dup(2)
    try:
        with open(os.devnull, "w", encoding="utf-8") as devnull:
            os.dup2(devnull.fileno(), 2)
            yield
    finally:
        os.dup2(original_stderr_fd, 2)
        os.close(original_stderr_fd)


def find_newest_model_dataset(dataset_dir: Path = DEFAULT_DATASET_DIR) -> Path:
    """Find the newest true model dataset Parquet file by modification time."""
    files = sorted(dataset_dir.glob("*_model_dataset_*.parquet"), key=lambda path: path.stat().st_mtime)
    files = [path for path in files if "_predictions" not in path.stem]
    if not files:
        raise FileNotFoundError(f"No model dataset Parquet files found in {dataset_dir}")
    return files[-1]


def artifact_stem(input_path: Path) -> str:
    """Return a stable artifact stem from a model dataset filename."""
    stem = input_path.stem
    if "_model_dataset_" in stem:
        symbol, date_part = stem.split("_model_dataset_", maxsplit=1)
        return f"{symbol}_lightgbm_no_raw_prices_balanced_{date_part}"
    return f"{stem}_lightgbm_no_raw_prices_balanced"


def read_model_dataset(input_path: Path) -> pd.DataFrame:
    """Load a model dataset."""
    with suppress_native_stderr():
        return pd.read_parquet(input_path)


def write_predictions(predictions: pd.DataFrame, output_path: Path) -> None:
    """Write out-of-sample predictions to Parquet."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_suffix(".tmp.parquet")
    with suppress_native_stderr():
        predictions.to_parquet(tmp_path, index=False)
    tmp_path.replace(output_path)


def write_metrics(metrics: dict[str, Any], output_path: Path) -> None:
    """Write metrics as JSON."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")


def write_feature_importance(feature_importance: pd.DataFrame, output_path: Path) -> None:
    """Write feature importance to CSV."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    feature_importance.to_csv(output_path, index=False)


def is_raw_absolute_price_feature(column: str) -> bool:
    """Return True for raw absolute price levels and absolute price features."""
    return column.startswith("bid_price_") or column.startswith("ask_price_") or column in RAW_ABSOLUTE_PRICE_COLUMNS


def is_allowed_microstructure_feature(column: str) -> bool:
    """Return True for the no-raw-price microstructure feature policy."""
    return (
        column.startswith("bid_size_")
        or column.startswith("ask_size_")
        or column.startswith("bid_depth_")
        or column.startswith("ask_depth_")
        or column.startswith("imbalance_")
        or column.startswith("delta_imbalance_")
        or column in EXPLICIT_FEATURE_COLUMNS
    )


def select_feature_columns(dataset: pd.DataFrame) -> FeatureSelectionReport:
    """Select numeric model features while excluding raw absolute price features."""
    numeric_columns = dataset.select_dtypes(include=[np.number]).columns.tolist()
    selected_features: list[str] = []
    excluded_raw_price_features: list[str] = []
    excluded_non_feature_columns: list[str] = []
    excluded_other_numeric_columns: list[str] = []

    for column in numeric_columns:
        if column in NON_FEATURE_COLUMNS:
            excluded_non_feature_columns.append(column)
        elif is_raw_absolute_price_feature(column):
            excluded_raw_price_features.append(column)
        elif is_allowed_microstructure_feature(column):
            selected_features.append(column)
        else:
            excluded_other_numeric_columns.append(column)

    return FeatureSelectionReport(
        total_columns=len(dataset.columns),
        numeric_columns=numeric_columns,
        selected_features=selected_features,
        excluded_raw_price_features=excluded_raw_price_features,
        excluded_non_feature_columns=excluded_non_feature_columns,
        excluded_other_numeric_columns=excluded_other_numeric_columns,
    )


def print_feature_selection_report(report: FeatureSelectionReport) -> None:
    """Print a clear no-raw-price feature selection report before training."""
    print("Feature selection report", flush=True)
    print("------------------------", flush=True)
    print(f"Total columns in dataset: {report.total_columns}", flush=True)
    print(f"Number of selected features: {len(report.selected_features)}", flush=True)
    print(f"Number of excluded raw price features: {len(report.excluded_raw_price_features)}", flush=True)
    print("Selected feature list:", flush=True)
    for column in report.selected_features:
        print(f"- {column}", flush=True)
    print("Excluded feature list:", flush=True)
    for column in report.excluded_features:
        print(f"- {column}", flush=True)


def prepare_dataset(dataset: pd.DataFrame, target_col: str = TARGET_COL) -> tuple[pd.DataFrame, FeatureSelectionReport]:
    """Sort data chronologically and select numeric model features."""
    if "timestamp" not in dataset.columns:
        raise ValueError("Missing required column: timestamp")
    if target_col not in dataset.columns:
        raise ValueError(f"Missing required column: {target_col}")

    prepared = dataset.copy()
    prepared["timestamp"] = pd.to_datetime(prepared["timestamp"], utc=True, errors="coerce")
    prepared[target_col] = pd.to_numeric(prepared[target_col], errors="coerce")
    prepared = prepared.dropna(subset=["timestamp", target_col])
    prepared = prepared.sort_values("timestamp").reset_index(drop=True)

    feature_report = select_feature_columns(prepared)
    if not feature_report.selected_features:
        raise ValueError("No numeric feature columns available for training")

    prepared = prepared.dropna(subset=[target_col, *feature_report.selected_features]).reset_index(drop=True)
    prepared[target_col] = prepared[target_col].astype(int)
    return prepared, feature_report


def make_time_splits(row_count: int, train_fraction: float = 0.70, validation_fraction: float = 0.15) -> TimeSplits:
    """Create chronological train/validation/test slices."""
    if row_count < 10:
        raise ValueError(f"Need at least 10 rows for a time-based split, found {row_count}")

    train_end = int(row_count * train_fraction)
    validation_end = int(row_count * (train_fraction + validation_fraction))

    if train_end <= 0 or validation_end <= train_end or validation_end >= row_count:
        raise ValueError(
            "Invalid split sizes. Need non-empty train, validation, and test sets "
            f"for {row_count} rows."
        )

    return TimeSplits(
        train=slice(0, train_end),
        validation=slice(train_end, validation_end),
        test=slice(validation_end, row_count),
    )


def class_balance(values: pd.Series) -> dict[str, int]:
    """Return sorted class counts with JSON-friendly keys."""
    counts = values.value_counts().sort_index()
    return {str(int(label)): int(count) for label, count in counts.items()}


def probability_summary(probabilities: np.ndarray, classes: np.ndarray) -> dict[str, dict[str, float]]:
    """Summarize predicted probabilities by class."""
    summary: dict[str, dict[str, float]] = {}
    for index, class_label in enumerate(classes):
        probability_col = PROBABILITY_COLUMNS.get(int(class_label), f"prob_{class_label}")
        class_probabilities = probabilities[:, index]
        summary[probability_col] = {
            "min": float(np.min(class_probabilities)),
            "mean": float(np.mean(class_probabilities)),
            "max": float(np.max(class_probabilities)),
        }
    return summary


def evaluate_predictions(
    y_true: pd.Series,
    y_pred: np.ndarray,
    probabilities: np.ndarray,
    classes: np.ndarray,
) -> dict[str, Any]:
    """Compute classification metrics for one split."""
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="macro",
        zero_division=0,
    )
    precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="weighted",
        zero_division=0,
    )

    return {
        "rows": int(len(y_true)),
        "class_balance": class_balance(y_true),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(precision_macro),
        "recall_macro": float(recall_macro),
        "f1_macro": float(f1_macro),
        "precision_weighted": float(precision_weighted),
        "recall_weighted": float(recall_weighted),
        "f1_weighted": float(f1_weighted),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=CLASS_LABELS).tolist(),
        "confusion_matrix_labels": CLASS_LABELS,
        "classification_report": classification_report(
            y_true,
            y_pred,
            labels=CLASS_LABELS,
            output_dict=True,
            zero_division=0,
        ),
        "probability_summary": probability_summary(probabilities, classes),
    }


def prediction_frame(
    dataset: pd.DataFrame,
    row_slice: slice,
    split_name: str,
    y_pred: np.ndarray,
    probabilities: np.ndarray,
    classes: np.ndarray,
    target_col: str = TARGET_COL,
) -> pd.DataFrame:
    """Build a prediction artifact for one out-of-sample split."""
    metadata_columns = [column for column in ["timestamp", "event_time", "exchange", "symbol"] if column in dataset.columns]
    output = dataset.iloc[row_slice][metadata_columns].copy()
    output["split"] = split_name
    output["y_true"] = dataset.iloc[row_slice][target_col].to_numpy()
    output["y_pred"] = y_pred

    class_to_index = {int(class_label): index for index, class_label in enumerate(classes)}
    for class_label, column in PROBABILITY_COLUMNS.items():
        if class_label in class_to_index:
            output[column] = probabilities[:, class_to_index[class_label]]
        else:
            output[column] = 0.0

    return output


def train_lightgbm_classifier(
    dataset: pd.DataFrame,
    feature_columns: list[str],
    splits: TimeSplits,
    target_col: str = TARGET_COL,
    random_state: int = 42,
) -> LGBMClassifier:
    """Fit LightGBM on the chronological training split only."""
    x_train = dataset.iloc[splits.train][feature_columns]
    y_train = dataset.iloc[splits.train][target_col]

    if y_train.nunique() < 2:
        raise ValueError("Training split must contain at least two target classes")

    model = LGBMClassifier(
        n_estimators=300,
        learning_rate=0.05,
        num_leaves=31,
        random_state=random_state,
        n_jobs=-1,
        verbosity=-1,
        deterministic=True,
        force_col_wise=True,
        class_weight="balanced",
    )
    model.fit(x_train, y_train)
    return model


def build_feature_importance(model: LGBMClassifier, feature_columns: list[str]) -> pd.DataFrame:
    """Build split and gain feature importance table."""
    booster = model.booster_
    importance = pd.DataFrame(
        {
            "feature": feature_columns,
            "importance_split": booster.feature_importance(importance_type="split"),
            "importance_gain": booster.feature_importance(importance_type="gain"),
        }
    )
    importance["importance_gain"] = importance["importance_gain"].astype(float)
    importance["importance_split"] = importance["importance_split"].astype(int)
    return importance.sort_values(
        ["importance_gain", "importance_split", "feature"],
        ascending=[False, False, True],
    ).reset_index(drop=True)


def train_lightgbm_file(
    input_path: Path,
    model_dir: Path = DEFAULT_MODEL_DIR,
    predictions_dir: Path = DEFAULT_PREDICTIONS_DIR,
    metrics_dir: Path = DEFAULT_METRICS_DIR,
    random_state: int = 42,
) -> TrainingSummary:
    """Train LightGBM and save model, predictions, metrics, and feature importance."""
    raw_dataset = read_model_dataset(input_path)
    dataset, feature_report = prepare_dataset(raw_dataset)
    print_feature_selection_report(feature_report)
    feature_columns = feature_report.selected_features
    splits = make_time_splits(len(dataset))
    model = train_lightgbm_classifier(dataset, feature_columns, splits, random_state=random_state)

    x_validation = dataset.iloc[splits.validation][feature_columns]
    y_validation = dataset.iloc[splits.validation][TARGET_COL]
    validation_pred = model.predict(x_validation)
    validation_probabilities = model.predict_proba(x_validation)

    x_test = dataset.iloc[splits.test][feature_columns]
    y_test = dataset.iloc[splits.test][TARGET_COL]
    test_pred = model.predict(x_test)
    test_probabilities = model.predict_proba(x_test)

    stem = artifact_stem(input_path)
    model_path = model_dir / f"{stem}.joblib"
    predictions_path = predictions_dir / f"{stem}_predictions.parquet"
    metrics_path = metrics_dir / f"{stem}_metrics.json"
    feature_importance_path = metrics_dir / f"{stem}_feature_importance.csv"

    predictions = pd.concat(
        [
            prediction_frame(
                dataset,
                splits.validation,
                "validation",
                validation_pred,
                validation_probabilities,
                model.classes_,
            ),
            prediction_frame(dataset, splits.test, "test", test_pred, test_probabilities, model.classes_),
        ],
        ignore_index=True,
    )
    feature_importance = build_feature_importance(model, feature_columns)

    metrics = {
        "input_path": str(input_path),
        "model_path": str(model_path),
        "predictions_path": str(predictions_path),
        "feature_importance_path": str(feature_importance_path),
        "rows_before_dropna": int(len(raw_dataset)),
        "rows_used": int(len(dataset)),
        "dropped_rows": int(len(raw_dataset) - len(dataset)),
        "feature_count": int(len(feature_columns)),
        "feature_columns": feature_columns,
        "feature_policy": "no_raw_absolute_prices",
        "excluded_raw_price_feature_count": int(len(feature_report.excluded_raw_price_features)),
        "excluded_raw_price_features": feature_report.excluded_raw_price_features,
        "excluded_non_feature_columns": feature_report.excluded_non_feature_columns,
        "excluded_other_numeric_columns": feature_report.excluded_other_numeric_columns,
        "class_weight": "balanced",
        "model_params": model.get_params(),
        "split": {
            "method": "chronological_time_based",
            "train_rows": int(splits.train.stop - splits.train.start),
            "validation_rows": int(splits.validation.stop - splits.validation.start),
            "test_rows": int(splits.test.stop - splits.test.start),
            "train_fraction": 0.70,
            "validation_fraction": 0.15,
            "test_fraction": 0.15,
        },
        "class_balance": {
            "all": class_balance(dataset[TARGET_COL]),
            "train": class_balance(dataset.iloc[splits.train][TARGET_COL]),
            "validation": class_balance(y_validation),
            "test": class_balance(y_test),
        },
        "validation": evaluate_predictions(y_validation, validation_pred, validation_probabilities, model.classes_),
        "test": evaluate_predictions(y_test, test_pred, test_probabilities, model.classes_),
        "top_20_feature_importance_gain": feature_importance.head(20).to_dict(orient="records"),
    }

    model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model": model,
            "feature_columns": feature_columns,
            "target_col": TARGET_COL,
            "classes": model.classes_.tolist(),
            "split_method": "chronological_time_based_70_15_15",
        },
        model_path,
    )
    write_predictions(predictions, predictions_path)
    write_metrics(metrics, metrics_path)
    write_feature_importance(feature_importance, feature_importance_path)

    return TrainingSummary(
        model_path=model_path,
        predictions_path=predictions_path,
        metrics_path=metrics_path,
        feature_importance_path=feature_importance_path,
        rows=len(dataset),
        feature_count=len(feature_columns),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--input-path", type=Path, default=None)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--predictions-dir", type=Path, default=DEFAULT_PREDICTIONS_DIR)
    parser.add_argument("--metrics-dir", type=Path, default=DEFAULT_METRICS_DIR)
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = args.input_path or find_newest_model_dataset(args.dataset_dir)
    summary = train_lightgbm_file(
        input_path=input_path,
        model_dir=args.model_dir,
        predictions_dir=args.predictions_dir,
        metrics_dir=args.metrics_dir,
        random_state=args.random_state,
    )

    print(f"Saved model: {summary.model_path}", flush=True)
    print(f"Saved predictions: {summary.predictions_path}", flush=True)
    print(f"Saved metrics: {summary.metrics_path}", flush=True)
    print(f"Saved feature importance: {summary.feature_importance_path}", flush=True)
    print(f"Rows used: {summary.rows}", flush=True)
    print(f"Features used: {summary.feature_count}", flush=True)


if __name__ == "__main__":
    main()
