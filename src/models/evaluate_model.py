"""Compare logistic regression and LightGBM model outputs."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import dataclass
import json
import os
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_REPORTS_DIR = PROJECT_ROOT / "reports"
DEFAULT_PREDICTIONS_DIR = PROJECT_ROOT / "data" / "processed" / "model_dataset"
DEFAULT_REPORT_PATH = DEFAULT_REPORTS_DIR / "model_comparison_report.txt"
DEFAULT_JSON_PATH = DEFAULT_REPORTS_DIR / "model_comparison_report.json"
CLASS_LABELS = [-1, 0, 1]
MODEL_KEYS = {
    "logistic": "logistic_regression",
    "lightgbm": "lightgbm",
}


@dataclass(frozen=True)
class ModelArtifacts:
    """Saved artifacts for one trained model."""

    name: str
    metrics_path: Path
    predictions_path: Path
    feature_importance_path: Path | None = None


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


def read_json(path: Path) -> dict[str, Any]:
    """Read a JSON file."""
    return json.loads(path.read_text(encoding="utf-8"))


def read_predictions(path: Path) -> pd.DataFrame:
    """Read saved prediction Parquet output."""
    with suppress_native_stderr():
        return pd.read_parquet(path)


def artifact_date(path: Path, model_key: str, suffix: str) -> str | None:
    """Extract date/token from a model artifact filename."""
    prefix = f"BTCUSDT_{model_key}_"
    name = path.name
    if not name.startswith(prefix) or not name.endswith(suffix):
        return None
    return name[len(prefix) : -len(suffix)]


def discover_artifact_pairs(
    reports_dir: Path = DEFAULT_REPORTS_DIR,
    predictions_dir: Path = DEFAULT_PREDICTIONS_DIR,
) -> tuple[ModelArtifacts, ModelArtifacts, str]:
    """Find the newest date with both logistic and LightGBM metrics and predictions."""
    discovered: dict[str, dict[str, dict[str, Path]]] = {
        model_name: {} for model_name in MODEL_KEYS
    }

    for model_name, model_key in MODEL_KEYS.items():
        for metrics_path in reports_dir.glob(f"BTCUSDT_{model_key}_*_metrics.json"):
            date_part = artifact_date(metrics_path, model_key, "_metrics.json")
            if date_part is not None:
                discovered[model_name].setdefault(date_part, {})["metrics"] = metrics_path

        for predictions_path in predictions_dir.glob(f"BTCUSDT_{model_key}_*_predictions.parquet"):
            date_part = artifact_date(predictions_path, model_key, "_predictions.parquet")
            if date_part is not None:
                discovered[model_name].setdefault(date_part, {})["predictions"] = predictions_path

    common_dates = sorted(
        date_part
        for date_part in set(discovered["logistic"]) & set(discovered["lightgbm"])
        if {"metrics", "predictions"}.issubset(discovered["logistic"][date_part])
        and {"metrics", "predictions"}.issubset(discovered["lightgbm"][date_part])
    )
    if not common_dates:
        raise FileNotFoundError(
            "Could not find matching logistic regression and LightGBM metrics/predictions "
            f"in {reports_dir} and {predictions_dir}. Train both models for the same dataset date first."
        )

    date_part = common_dates[-1]
    logistic = ModelArtifacts(
        name="Logistic Regression",
        metrics_path=discovered["logistic"][date_part]["metrics"],
        predictions_path=discovered["logistic"][date_part]["predictions"],
    )
    lightgbm_feature_importance = reports_dir / f"BTCUSDT_lightgbm_{date_part}_feature_importance.csv"
    lightgbm = ModelArtifacts(
        name="LightGBM",
        metrics_path=discovered["lightgbm"][date_part]["metrics"],
        predictions_path=discovered["lightgbm"][date_part]["predictions"],
        feature_importance_path=lightgbm_feature_importance if lightgbm_feature_importance.exists() else None,
    )
    return logistic, lightgbm, date_part


def explicit_artifacts(args: argparse.Namespace) -> tuple[ModelArtifacts, ModelArtifacts, str | None]:
    """Build artifacts from explicit CLI paths when all required paths are provided."""
    explicit_paths = [
        args.logistic_metrics_path,
        args.logistic_predictions_path,
        args.lightgbm_metrics_path,
        args.lightgbm_predictions_path,
    ]
    if not any(explicit_paths):
        raise ValueError("No explicit paths were provided")
    if not all(explicit_paths):
        raise ValueError(
            "When using explicit artifact paths, provide all of: "
            "--logistic-metrics-path, --logistic-predictions-path, "
            "--lightgbm-metrics-path, --lightgbm-predictions-path."
        )

    return (
        ModelArtifacts(
            name="Logistic Regression",
            metrics_path=args.logistic_metrics_path,
            predictions_path=args.logistic_predictions_path,
        ),
        ModelArtifacts(
            name="LightGBM",
            metrics_path=args.lightgbm_metrics_path,
            predictions_path=args.lightgbm_predictions_path,
            feature_importance_path=args.lightgbm_feature_importance_path,
        ),
        None,
    )


def metric_value(metrics: dict[str, Any], split: str, key: str) -> float | None:
    """Safely read a split metric value."""
    value = metrics.get(split, {}).get(key)
    if value is None:
        return None
    return float(value)


def summarize_metrics(metrics: dict[str, Any], split: str) -> dict[str, float | int | None]:
    """Extract key model metrics for one split."""
    split_metrics = metrics.get(split, {})
    return {
        "rows": split_metrics.get("rows"),
        "accuracy": metric_value(metrics, split, "accuracy"),
        "precision_macro": metric_value(metrics, split, "precision_macro"),
        "recall_macro": metric_value(metrics, split, "recall_macro"),
        "f1_macro": metric_value(metrics, split, "f1_macro"),
        "f1_weighted": metric_value(metrics, split, "f1_weighted"),
    }


def confusion_from_predictions(predictions: pd.DataFrame, split: str) -> list[list[int]]:
    """Compute confusion matrix for one prediction split."""
    split_predictions = predictions.loc[predictions["split"] == split]
    if split_predictions.empty:
        return []
    return confusion_matrix(
        split_predictions["y_true"],
        split_predictions["y_pred"],
        labels=CLASS_LABELS,
    ).tolist()


def prepare_predictions(predictions: pd.DataFrame, model_prefix: str) -> pd.DataFrame:
    """Prepare prediction rows for cross-model comparison."""
    required = {"timestamp", "split", "y_true", "y_pred"}
    missing = sorted(required - set(predictions.columns))
    if missing:
        raise ValueError(f"Prediction file is missing required columns: {missing}")

    output = predictions.copy()
    output["timestamp"] = pd.to_datetime(output["timestamp"], utc=True, errors="coerce")
    output = output.dropna(subset=["timestamp", "split", "y_true", "y_pred"])
    output = output.sort_values(["split", "timestamp"]).reset_index(drop=True)
    output = output.rename(
        columns={
            "y_pred": f"{model_prefix}_pred",
            "prob_down": f"{model_prefix}_prob_down",
            "prob_neutral": f"{model_prefix}_prob_neutral",
            "prob_up": f"{model_prefix}_prob_up",
        }
    )
    keep_columns = [
        "timestamp",
        "split",
        "y_true",
        f"{model_prefix}_pred",
        f"{model_prefix}_prob_down",
        f"{model_prefix}_prob_neutral",
        f"{model_prefix}_prob_up",
    ]
    return output[[column for column in keep_columns if column in output.columns]]


def compare_prediction_agreement(
    logistic_predictions: pd.DataFrame,
    lightgbm_predictions: pd.DataFrame,
) -> dict[str, Any]:
    """Compare logistic and LightGBM predictions on overlapping timestamps."""
    logistic = prepare_predictions(logistic_predictions, "logistic")
    lightgbm = prepare_predictions(lightgbm_predictions, "lightgbm")

    merged = logistic.merge(
        lightgbm,
        on=["timestamp", "split", "y_true"],
        how="inner",
    )
    if merged.empty:
        return {
            "overlap_rows": 0,
            "by_split": {},
        }

    merged["models_agree"] = merged["logistic_pred"] == merged["lightgbm_pred"]
    merged["logistic_correct"] = merged["logistic_pred"] == merged["y_true"]
    merged["lightgbm_correct"] = merged["lightgbm_pred"] == merged["y_true"]
    merged["both_correct"] = merged["logistic_correct"] & merged["lightgbm_correct"]
    merged["both_wrong"] = ~merged["logistic_correct"] & ~merged["lightgbm_correct"]
    merged["logistic_only_correct"] = merged["logistic_correct"] & ~merged["lightgbm_correct"]
    merged["lightgbm_only_correct"] = ~merged["logistic_correct"] & merged["lightgbm_correct"]

    by_split: dict[str, dict[str, float | int]] = {}
    for split_name, group in merged.groupby("split", sort=True):
        rows = len(group)
        by_split[str(split_name)] = {
            "rows": int(rows),
            "agreement_rate": float(group["models_agree"].mean()),
            "logistic_accuracy_on_overlap": float(group["logistic_correct"].mean()),
            "lightgbm_accuracy_on_overlap": float(group["lightgbm_correct"].mean()),
            "both_correct": int(group["both_correct"].sum()),
            "both_wrong": int(group["both_wrong"].sum()),
            "logistic_only_correct": int(group["logistic_only_correct"].sum()),
            "lightgbm_only_correct": int(group["lightgbm_only_correct"].sum()),
        }

    return {
        "overlap_rows": int(len(merged)),
        "by_split": by_split,
    }


def load_feature_importance(path: Path | None, top_n: int = 20) -> list[dict[str, Any]]:
    """Load top LightGBM feature importance rows if available."""
    if path is None or not path.exists():
        return []
    frame = pd.read_csv(path)
    return frame.head(top_n).to_dict(orient="records")


def build_comparison(
    logistic_artifacts: ModelArtifacts,
    lightgbm_artifacts: ModelArtifacts,
    artifact_date_value: str | None,
) -> dict[str, Any]:
    """Load artifacts and build a complete comparison summary."""
    logistic_metrics = read_json(logistic_artifacts.metrics_path)
    lightgbm_metrics = read_json(lightgbm_artifacts.metrics_path)
    logistic_predictions = read_predictions(logistic_artifacts.predictions_path)
    lightgbm_predictions = read_predictions(lightgbm_artifacts.predictions_path)

    comparison = {
        "artifact_date": artifact_date_value,
        "artifacts": {
            "logistic": {
                "metrics_path": str(logistic_artifacts.metrics_path),
                "predictions_path": str(logistic_artifacts.predictions_path),
            },
            "lightgbm": {
                "metrics_path": str(lightgbm_artifacts.metrics_path),
                "predictions_path": str(lightgbm_artifacts.predictions_path),
                "feature_importance_path": (
                    str(lightgbm_artifacts.feature_importance_path)
                    if lightgbm_artifacts.feature_importance_path is not None
                    else None
                ),
            },
        },
        "dataset": {
            "logistic_input_path": logistic_metrics.get("input_path"),
            "lightgbm_input_path": lightgbm_metrics.get("input_path"),
            "logistic_rows_used": logistic_metrics.get("rows_used"),
            "lightgbm_rows_used": lightgbm_metrics.get("rows_used"),
            "logistic_feature_count": logistic_metrics.get("feature_count"),
            "lightgbm_feature_count": lightgbm_metrics.get("feature_count"),
        },
        "metrics": {
            "validation": {
                "logistic": summarize_metrics(logistic_metrics, "validation"),
                "lightgbm": summarize_metrics(lightgbm_metrics, "validation"),
            },
            "test": {
                "logistic": summarize_metrics(logistic_metrics, "test"),
                "lightgbm": summarize_metrics(lightgbm_metrics, "test"),
            },
        },
        "confusion_matrices": {
            "labels": CLASS_LABELS,
            "validation": {
                "logistic": logistic_metrics.get("validation", {}).get(
                    "confusion_matrix",
                    confusion_from_predictions(logistic_predictions, "validation"),
                ),
                "lightgbm": lightgbm_metrics.get("validation", {}).get(
                    "confusion_matrix",
                    confusion_from_predictions(lightgbm_predictions, "validation"),
                ),
            },
            "test": {
                "logistic": logistic_metrics.get("test", {}).get(
                    "confusion_matrix",
                    confusion_from_predictions(logistic_predictions, "test"),
                ),
                "lightgbm": lightgbm_metrics.get("test", {}).get(
                    "confusion_matrix",
                    confusion_from_predictions(lightgbm_predictions, "test"),
                ),
            },
        },
        "prediction_agreement": compare_prediction_agreement(logistic_predictions, lightgbm_predictions),
        "lightgbm_top_feature_importance": load_feature_importance(lightgbm_artifacts.feature_importance_path),
    }

    for split in ("validation", "test"):
        logistic_accuracy = comparison["metrics"][split]["logistic"]["accuracy"]
        lightgbm_accuracy = comparison["metrics"][split]["lightgbm"]["accuracy"]
        if logistic_accuracy is not None and lightgbm_accuracy is not None:
            comparison["metrics"][split]["lightgbm_minus_logistic_accuracy"] = float(
                lightgbm_accuracy - logistic_accuracy
            )

    return comparison


def format_optional_float(value: float | int | None) -> str:
    """Format optional numeric values for text reports."""
    if value is None:
        return "n/a"
    return f"{float(value):.6f}"


def format_metric_table(comparison: dict[str, Any], split: str) -> list[str]:
    """Format key metrics for one split."""
    logistic = comparison["metrics"][split]["logistic"]
    lightgbm = comparison["metrics"][split]["lightgbm"]
    diff = comparison["metrics"][split].get("lightgbm_minus_logistic_accuracy")

    rows = [
        f"{split.title()} Metrics",
        "-" * (len(split) + 8),
        "metric, logistic_regression, lightgbm",
        f"rows, {logistic.get('rows')}, {lightgbm.get('rows')}",
        f"accuracy, {format_optional_float(logistic.get('accuracy'))}, {format_optional_float(lightgbm.get('accuracy'))}",
        f"precision_macro, {format_optional_float(logistic.get('precision_macro'))}, {format_optional_float(lightgbm.get('precision_macro'))}",
        f"recall_macro, {format_optional_float(logistic.get('recall_macro'))}, {format_optional_float(lightgbm.get('recall_macro'))}",
        f"f1_macro, {format_optional_float(logistic.get('f1_macro'))}, {format_optional_float(lightgbm.get('f1_macro'))}",
        f"f1_weighted, {format_optional_float(logistic.get('f1_weighted'))}, {format_optional_float(lightgbm.get('f1_weighted'))}",
        f"lightgbm_minus_logistic_accuracy, {format_optional_float(diff)}",
        "",
    ]
    return rows


def format_confusion_matrix(matrix: list[list[int]]) -> list[str]:
    """Format confusion matrix with fixed labels."""
    if not matrix:
        return ["n/a"]
    rows = ["labels: rows=true [-1, 0, 1], columns=predicted [-1, 0, 1]"]
    rows.extend(str(row) for row in matrix)
    return rows


def build_text_report(comparison: dict[str, Any]) -> str:
    """Build a human-readable model comparison report."""
    lines = [
        "LOB Model Comparison Report",
        "===========================",
        f"Artifact date: {comparison.get('artifact_date') or 'explicit paths'}",
        "",
        "Artifacts",
        "---------",
        f"Logistic metrics: {comparison['artifacts']['logistic']['metrics_path']}",
        f"Logistic predictions: {comparison['artifacts']['logistic']['predictions_path']}",
        f"LightGBM metrics: {comparison['artifacts']['lightgbm']['metrics_path']}",
        f"LightGBM predictions: {comparison['artifacts']['lightgbm']['predictions_path']}",
        f"LightGBM feature importance: {comparison['artifacts']['lightgbm']['feature_importance_path'] or 'n/a'}",
        "",
        "Dataset",
        "-------",
        f"Logistic input: {comparison['dataset']['logistic_input_path']}",
        f"LightGBM input: {comparison['dataset']['lightgbm_input_path']}",
        f"Rows used: logistic={comparison['dataset']['logistic_rows_used']}, lightgbm={comparison['dataset']['lightgbm_rows_used']}",
        f"Feature count: logistic={comparison['dataset']['logistic_feature_count']}, lightgbm={comparison['dataset']['lightgbm_feature_count']}",
        "",
    ]

    lines.extend(format_metric_table(comparison, "validation"))
    lines.extend(format_metric_table(comparison, "test"))

    lines.extend(["Confusion Matrices", "------------------"])
    for split in ("validation", "test"):
        lines.append(f"{split.title()} Logistic Regression")
        lines.extend(format_confusion_matrix(comparison["confusion_matrices"][split]["logistic"]))
        lines.append("")
        lines.append(f"{split.title()} LightGBM")
        lines.extend(format_confusion_matrix(comparison["confusion_matrices"][split]["lightgbm"]))
        lines.append("")

    lines.extend(["Prediction Agreement", "--------------------"])
    agreement = comparison["prediction_agreement"]
    lines.append(f"Overlapping rows: {agreement['overlap_rows']}")
    for split, split_summary in agreement.get("by_split", {}).items():
        lines.append(
            f"{split}: rows={split_summary['rows']}, "
            f"agreement_rate={split_summary['agreement_rate']:.6f}, "
            f"logistic_accuracy={split_summary['logistic_accuracy_on_overlap']:.6f}, "
            f"lightgbm_accuracy={split_summary['lightgbm_accuracy_on_overlap']:.6f}, "
            f"both_correct={split_summary['both_correct']}, "
            f"both_wrong={split_summary['both_wrong']}, "
            f"logistic_only_correct={split_summary['logistic_only_correct']}, "
            f"lightgbm_only_correct={split_summary['lightgbm_only_correct']}"
        )
    lines.append("")

    lines.extend(["Top LightGBM Feature Importance", "-----------------------------"])
    feature_importance = comparison["lightgbm_top_feature_importance"]
    if feature_importance:
        lines.append("rank, feature, importance_gain, importance_split")
        for rank, row in enumerate(feature_importance[:20], start=1):
            lines.append(
                f"{rank}, {row['feature']}, "
                f"{float(row['importance_gain']):.6f}, {int(row['importance_split'])}"
            )
    else:
        lines.append("n/a")
    lines.append("")

    return "\n".join(lines)


def save_reports(comparison: dict[str, Any], report_path: Path, json_path: Path) -> None:
    """Save text and JSON model comparison reports."""
    report_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(build_text_report(comparison), encoding="utf-8")
    json_path.write_text(json.dumps(comparison, indent=2), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reports-dir", type=Path, default=DEFAULT_REPORTS_DIR)
    parser.add_argument("--predictions-dir", type=Path, default=DEFAULT_PREDICTIONS_DIR)
    parser.add_argument("--output-path", type=Path, default=DEFAULT_REPORT_PATH)
    parser.add_argument("--json-output-path", type=Path, default=DEFAULT_JSON_PATH)
    parser.add_argument("--logistic-metrics-path", type=Path, default=None)
    parser.add_argument("--logistic-predictions-path", type=Path, default=None)
    parser.add_argument("--lightgbm-metrics-path", type=Path, default=None)
    parser.add_argument("--lightgbm-predictions-path", type=Path, default=None)
    parser.add_argument("--lightgbm-feature-importance-path", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        if any(
            [
                args.logistic_metrics_path,
                args.logistic_predictions_path,
                args.lightgbm_metrics_path,
                args.lightgbm_predictions_path,
            ]
        ):
            logistic_artifacts, lightgbm_artifacts, date_part = explicit_artifacts(args)
        else:
            logistic_artifacts, lightgbm_artifacts, date_part = discover_artifact_pairs(
                reports_dir=args.reports_dir,
                predictions_dir=args.predictions_dir,
            )

        comparison = build_comparison(logistic_artifacts, lightgbm_artifacts, date_part)
        save_reports(comparison, args.output_path, args.json_output_path)

        print(f"Saved comparison report: {args.output_path}", flush=True)
        print(f"Saved comparison JSON: {args.json_output_path}", flush=True)
    except (FileNotFoundError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr, flush=True)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
