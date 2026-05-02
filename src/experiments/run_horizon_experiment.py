"""Run horizon experiments for LightGBM target/backtest comparison."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.backtest import simple_backtest
from src.models import train_lightgbm
from src.targets import build_targets


DEFAULT_SYMBOL = "BTCUSDT"
DEFAULT_FEATURES_DIR = PROJECT_ROOT / "data" / "processed" / "features"
DEFAULT_MODEL_DATASET_DIR = PROJECT_ROOT / "data" / "processed" / "model_dataset"
DEFAULT_MODEL_DIR = PROJECT_ROOT / "models"
DEFAULT_REPORTS_DIR = PROJECT_ROOT / "reports"
DEFAULT_FIGURES_DIR = DEFAULT_REPORTS_DIR / "figures"
DEFAULT_HORIZONS = [5, 10, 30, 60, 120]
DEFAULT_THRESHOLD_ABS = 0.5
DEFAULT_SPLIT = "validation"
DEFAULT_SIGNAL_THRESHOLD = 0.75
DEFAULT_DIRECTION_MARGIN = 0.00
DEFAULT_NEUTRAL_MAX_PROBABILITY = 0.30
DEFAULT_FULL_COST_FEE_BPS = 1.0
DEFAULT_FULL_COST_SLIPPAGE_BPS = 1.0
SUMMARY_COLUMNS = [
    "horizon_seconds",
    "threshold_abs",
    "split",
    "cost_mode",
    "model_dataset_path",
    "predictions_path",
    "number_of_trades",
    "long_trades",
    "short_trades",
    "no_trade_rows",
    "no_trade_percentage",
    "total_gross_return_bps",
    "total_net_return_bps",
    "total_cost_bps",
    "average_gross_return_bps",
    "average_net_return_bps",
    "average_cost_per_trade_bps",
    "gross_return_per_trade_minus_cost_per_trade_bps",
    "hit_rate",
    "max_drawdown_bps",
    "turnover",
    "validation_accuracy",
    "validation_macro_f1",
    "validation_neutral_recall",
    "validation_up_precision",
    "validation_down_precision",
]


def threshold_token(threshold_abs: float) -> str:
    """Format an absolute target threshold for filenames."""
    text = f"{threshold_abs:.6f}".rstrip("0").rstrip(".")
    return f"th{text.replace('-', 'm').replace('.', 'p')}"


def feature_path_for_dataset(dataset_name: str, features_dir: Path, symbol: str) -> Path:
    """Resolve the feature file for an experiment dataset token."""
    path = features_dir / f"{symbol}_features_{dataset_name}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Feature file not found: {path}")
    return path


def horizon_dataset_path(
    dataset_name: str,
    horizon_seconds: int,
    threshold_abs: float,
    output_dir: Path,
    symbol: str,
) -> Path:
    """Build a horizon-specific model dataset path."""
    token = threshold_token(threshold_abs)
    return output_dir / f"{symbol}_model_dataset_{dataset_name}_h{horizon_seconds}_{token}.parquet"


def build_horizon_model_dataset(
    feature_path: Path,
    output_path: Path,
    horizon_seconds: int,
    threshold_abs: float,
    timestamp_tolerance_seconds: float,
) -> Path:
    """Build and save a horizon-specific model dataset."""
    features = build_targets.read_features(feature_path)
    dataset = build_targets.build_model_dataset(
        features,
        threshold_abs=threshold_abs,
        horizon_seconds=horizon_seconds,
        timestamp_tolerance_seconds=timestamp_tolerance_seconds,
    )
    build_targets.write_model_dataset(dataset, output_path)
    return output_path


def validation_metrics(metrics_path: Path) -> dict[str, float]:
    """Extract validation classification metrics used in horizon ranking."""
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    validation = metrics.get("validation", {})
    classification_report = validation.get("classification_report", {})
    return {
        "validation_accuracy": float(validation.get("accuracy", 0.0)),
        "validation_macro_f1": float(validation.get("f1_macro", 0.0)),
        "validation_neutral_recall": float(classification_report.get("0", {}).get("recall", 0.0)),
        "validation_up_precision": float(classification_report.get("1", {}).get("precision", 0.0)),
        "validation_down_precision": float(classification_report.get("-1", {}).get("precision", 0.0)),
    }


def backtest_output_stem(
    predictions_path: Path,
    split: str,
    signal_threshold: float,
    direction_margin: float,
    neutral_max_probability: float,
    cost_mode: str,
) -> str:
    """Build a backtest artifact stem that includes the model, horizon, thresholds, split, and cost mode."""
    return (
        f"{simple_backtest.artifact_stem(predictions_path)}_"
        f"{split}_"
        f"{simple_backtest.threshold_suffix(signal_threshold, direction_margin, neutral_max_probability)}_"
        f"{cost_mode}"
    )


def run_backtest_case(
    predictions_path: Path,
    model_dataset_path: Path,
    horizon_seconds: int,
    threshold_abs: float,
    split: str,
    signal_threshold: float,
    direction_margin: float,
    neutral_max_probability: float,
    fee_bps: float,
    slippage_bps: float,
    reports_dir: Path,
    figures_dir: Path,
    model_metrics: dict[str, float],
) -> dict[str, Any]:
    """Run one validation backtest case and return one summary row."""
    predictions = simple_backtest.read_parquet(predictions_path)
    model_dataset = simple_backtest.read_parquet(model_dataset_path)
    cost_mode = simple_backtest.determine_cost_mode(fee_bps, slippage_bps)
    backtest, trades = simple_backtest.run_backtest(
        predictions=predictions,
        model_dataset=model_dataset,
        split=split,
        signal_threshold=signal_threshold,
        direction_margin=direction_margin,
        neutral_max_probability=neutral_max_probability,
        horizon_seconds=horizon_seconds,
        fee_bps=fee_bps,
        slippage_bps=slippage_bps,
    )
    backtest_metrics = simple_backtest.summarize_backtest(backtest, trades)
    config = {
        "predictions_path": str(predictions_path),
        "model_dataset_path": str(model_dataset_path),
        "split": split,
        "signal_threshold": signal_threshold,
        "direction_margin": direction_margin,
        "neutral_max_probability": neutral_max_probability,
        "cost_mode": cost_mode,
        "horizon_seconds": horizon_seconds,
        "threshold_abs": threshold_abs,
        "fee_bps": fee_bps,
        "slippage_bps": slippage_bps,
    }
    stem = backtest_output_stem(
        predictions_path=predictions_path,
        split=split,
        signal_threshold=signal_threshold,
        direction_margin=direction_margin,
        neutral_max_probability=neutral_max_probability,
        cost_mode=cost_mode,
    )
    simple_backtest.save_backtest_outputs(
        stem=stem,
        backtest=backtest,
        trades=trades,
        metrics=backtest_metrics,
        config=config,
        reports_dir=reports_dir,
        figures_dir=figures_dir,
    )

    return {
        "horizon_seconds": horizon_seconds,
        "threshold_abs": threshold_abs,
        "split": split,
        "cost_mode": cost_mode,
        "model_dataset_path": str(model_dataset_path),
        "predictions_path": str(predictions_path),
        **{column: backtest_metrics.get(column, 0.0) for column in SUMMARY_COLUMNS if column in backtest_metrics},
        **model_metrics,
    }


def run_horizon(
    horizon_seconds: int,
    feature_path: Path,
    dataset_name: str,
    threshold_abs: float,
    timestamp_tolerance_seconds: float,
    split: str,
    signal_threshold: float,
    direction_margin: float,
    neutral_max_probability: float,
    full_cost_fee_bps: float,
    full_cost_slippage_bps: float,
    symbol: str,
    model_dataset_dir: Path,
    model_dir: Path,
    predictions_dir: Path,
    reports_dir: Path,
    figures_dir: Path,
    random_state: int,
) -> list[dict[str, Any]]:
    """Run target build, training, and validation backtests for one horizon."""
    print(f"\n=== Horizon {horizon_seconds}s ===", flush=True)
    model_dataset_path = horizon_dataset_path(
        dataset_name=dataset_name,
        horizon_seconds=horizon_seconds,
        threshold_abs=threshold_abs,
        output_dir=model_dataset_dir,
        symbol=symbol,
    )
    build_horizon_model_dataset(
        feature_path=feature_path,
        output_path=model_dataset_path,
        horizon_seconds=horizon_seconds,
        threshold_abs=threshold_abs,
        timestamp_tolerance_seconds=timestamp_tolerance_seconds,
    )
    print(f"Saved horizon model dataset: {model_dataset_path}", flush=True)

    training = train_lightgbm.train_lightgbm_file(
        input_path=model_dataset_path,
        model_dir=model_dir,
        predictions_dir=predictions_dir,
        metrics_dir=reports_dir,
        random_state=random_state,
    )
    print(f"Saved horizon predictions: {training.predictions_path}", flush=True)
    model_metrics = validation_metrics(training.metrics_path)

    rows = []
    for fee_bps, slippage_bps in [(0.0, 0.0), (full_cost_fee_bps, full_cost_slippage_bps)]:
        row = run_backtest_case(
            predictions_path=training.predictions_path,
            model_dataset_path=model_dataset_path,
            horizon_seconds=horizon_seconds,
            threshold_abs=threshold_abs,
            split=split,
            signal_threshold=signal_threshold,
            direction_margin=direction_margin,
            neutral_max_probability=neutral_max_probability,
            fee_bps=fee_bps,
            slippage_bps=slippage_bps,
            reports_dir=reports_dir,
            figures_dir=figures_dir,
            model_metrics=model_metrics,
        )
        print(
            f"{row['cost_mode']}: trades={row['number_of_trades']} "
            f"avg_net_bps={row['average_net_return_bps']:.6f} "
            f"avg_gross_bps={row['average_gross_return_bps']:.6f}",
            flush=True,
        )
        rows.append(row)
    return rows


def save_summary(rows: list[dict[str, Any]], reports_dir: Path) -> tuple[Path, Path]:
    """Save experiment summary as CSV and JSON."""
    reports_dir.mkdir(parents=True, exist_ok=True)
    summary = pd.DataFrame(rows)
    for column in SUMMARY_COLUMNS:
        if column not in summary.columns:
            summary[column] = pd.NA
    summary = summary[SUMMARY_COLUMNS]

    csv_path = reports_dir / "horizon_experiment_summary.csv"
    json_path = reports_dir / "horizon_experiment_summary.json"
    summary.to_csv(csv_path, index=False)
    json_path.write_text(json.dumps(summary.to_dict(orient="records"), indent=2), encoding="utf-8")
    return csv_path, json_path


def best_row(frame: pd.DataFrame, column: str, cost_mode: str | None = None, min_trades: int | None = None) -> pd.Series | None:
    """Return the best row for one ranking criterion."""
    subset = frame.copy()
    if cost_mode is not None:
        subset = subset.loc[subset["cost_mode"] == cost_mode]
    if min_trades is not None:
        subset = subset.loc[subset["number_of_trades"] >= min_trades]
    subset = subset.dropna(subset=[column])
    if subset.empty:
        return None
    return subset.sort_values(column, ascending=False).iloc[0]


def print_best(label: str, row: pd.Series | None, metric: str) -> None:
    """Print a ranking result."""
    if row is None:
        print(f"{label}: no eligible horizon", flush=True)
        return
    print(
        f"{label}: h{int(row['horizon_seconds'])} "
        f"({metric}={float(row[metric]):.6f}, trades={int(row['number_of_trades'])}, "
        f"cost_mode={row['cost_mode']})",
        flush=True,
    )


def print_rankings(rows: list[dict[str, Any]]) -> None:
    """Print validation ranking results."""
    summary = pd.DataFrame(rows)
    print("\nHorizon Ranking", flush=True)
    print("---------------", flush=True)
    print_best(
        "Best horizon by average_net_return_bps under full-cost mode",
        best_row(summary, "average_net_return_bps", cost_mode="cost_adjusted"),
        "average_net_return_bps",
    )
    print_best(
        "Best horizon by average_gross_return_bps under zero-cost mode",
        best_row(summary, "average_gross_return_bps", cost_mode="zero_cost"),
        "average_gross_return_bps",
    )
    print_best(
        "Best horizon by gross_return_per_trade_minus_cost_per_trade_bps",
        best_row(summary, "gross_return_per_trade_minus_cost_per_trade_bps", cost_mode="cost_adjusted"),
        "gross_return_per_trade_minus_cost_per_trade_bps",
    )
    print_best(
        "Best horizon with at least 50 trades",
        best_row(summary, "average_net_return_bps", cost_mode="cost_adjusted", min_trades=50),
        "average_net_return_bps",
    )

    cost_adjusted = summary.loc[summary["cost_mode"] == "cost_adjusted"]
    profitable_after_average_cost = (
        cost_adjusted["average_gross_return_bps"] > cost_adjusted["average_cost_per_trade_bps"]
    ).any()
    if not profitable_after_average_cost:
        print(
            "WARNING: no horizon has average_gross_return_bps above average_cost_per_trade_bps.",
            flush=True,
        )


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-name", type=str, default="merged_last_2_updated")
    parser.add_argument("--threshold-abs", type=float, default=DEFAULT_THRESHOLD_ABS)
    parser.add_argument("--horizons", type=int, nargs="+", default=DEFAULT_HORIZONS)
    parser.add_argument("--split", choices=["validation", "test"], default=DEFAULT_SPLIT)
    parser.add_argument("--signal-threshold", type=float, default=DEFAULT_SIGNAL_THRESHOLD)
    parser.add_argument("--direction-margin", type=float, default=DEFAULT_DIRECTION_MARGIN)
    parser.add_argument("--neutral-max-probability", type=float, default=DEFAULT_NEUTRAL_MAX_PROBABILITY)
    parser.add_argument("--fee-bps", type=float, default=DEFAULT_FULL_COST_FEE_BPS)
    parser.add_argument("--slippage-bps", type=float, default=DEFAULT_FULL_COST_SLIPPAGE_BPS)
    parser.add_argument("--timestamp-tolerance-seconds", type=float, default=None)
    parser.add_argument("--symbol", type=str, default=DEFAULT_SYMBOL)
    parser.add_argument("--features-dir", type=Path, default=DEFAULT_FEATURES_DIR)
    parser.add_argument("--model-dataset-dir", type=Path, default=DEFAULT_MODEL_DATASET_DIR)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--predictions-dir", type=Path, default=DEFAULT_MODEL_DATASET_DIR)
    parser.add_argument("--reports-dir", type=Path, default=DEFAULT_REPORTS_DIR)
    parser.add_argument("--figures-dir", type=Path, default=DEFAULT_FIGURES_DIR)
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    """Run the full horizon experiment."""
    args = parse_args()
    if any(horizon <= 0 for horizon in args.horizons):
        raise ValueError("--horizons must contain positive integers")
    if args.threshold_abs < 0:
        raise ValueError("--threshold-abs must be nonnegative")

    target_config = build_targets.load_target_config()
    timestamp_tolerance_seconds = (
        args.timestamp_tolerance_seconds
        if args.timestamp_tolerance_seconds is not None
        else target_config.timestamp_tolerance_seconds
    )
    feature_path = feature_path_for_dataset(args.dataset_name, args.features_dir, args.symbol)
    print(f"Using feature file: {feature_path}", flush=True)
    print(f"Using validation split: {args.split}", flush=True)
    print(f"Threshold abs: {args.threshold_abs}", flush=True)
    print(f"Horizons: {args.horizons}", flush=True)

    rows: list[dict[str, Any]] = []
    for horizon_seconds in args.horizons:
        rows.extend(
            run_horizon(
                horizon_seconds=horizon_seconds,
                feature_path=feature_path,
                dataset_name=args.dataset_name,
                threshold_abs=args.threshold_abs,
                timestamp_tolerance_seconds=timestamp_tolerance_seconds,
                split=args.split,
                signal_threshold=args.signal_threshold,
                direction_margin=args.direction_margin,
                neutral_max_probability=args.neutral_max_probability,
                full_cost_fee_bps=args.fee_bps,
                full_cost_slippage_bps=args.slippage_bps,
                symbol=args.symbol,
                model_dataset_dir=args.model_dataset_dir,
                model_dir=args.model_dir,
                predictions_dir=args.predictions_dir,
                reports_dir=args.reports_dir,
                figures_dir=args.figures_dir,
                random_state=args.random_state,
            )
        )

    csv_path, json_path = save_summary(rows, args.reports_dir)
    print(f"\nSaved horizon experiment CSV: {csv_path}", flush=True)
    print(f"Saved horizon experiment JSON: {json_path}", flush=True)
    print_rankings(rows)


if __name__ == "__main__":
    main()
