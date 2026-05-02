"""Run a simple probability-driven backtest for model predictions."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import dataclass
import json
import os
from pathlib import Path
import sys
from typing import Any

MATPLOTLIB_CONFIG_DIR = Path("/private/tmp/lob_project_matplotlib")
XDG_CACHE_DIR = Path("/private/tmp/lob_project_cache")
MATPLOTLIB_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
XDG_CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MATPLOTLIB_CONFIG_DIR))
os.environ.setdefault("XDG_CACHE_HOME", str(XDG_CACHE_DIR))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "CONFIG" / "config.yaml"
DEFAULT_MODEL_DATASET_DIR = PROJECT_ROOT / "data" / "processed" / "model_dataset"
DEFAULT_REPORTS_DIR = PROJECT_ROOT / "reports"
DEFAULT_FIGURES_DIR = DEFAULT_REPORTS_DIR / "figures"
DEFAULT_HORIZON_SECONDS = 5
DEFAULT_SIGNAL_THRESHOLD = 0.75
DEFAULT_DIRECTION_MARGIN = 0.00
DEFAULT_NEUTRAL_MAX_PROBABILITY = 0.30
DEFAULT_FEE_BPS = 1.0
DEFAULT_SLIPPAGE_BPS = 1.0
VALIDATION_SELECTED_NOTE = (
    "These thresholds were selected from validation sweep as the best current configuration, "
    "but the strategy is still not profitable because average gross return per trade is below "
    "average round-trip cost."
)


@dataclass(frozen=True)
class BacktestArtifacts:
    """Saved backtest artifact paths."""

    report_path: Path
    metrics_path: Path
    equity_curve_path: Path
    equity_curve_plot_path: Path
    trades_path: Path


@contextmanager
def suppress_native_stderr():
    """Temporarily suppress native library stderr noise during Parquet and plot I/O."""
    sys.stderr.flush()
    original_stderr_fd = os.dup(2)
    try:
        with open(os.devnull, "w", encoding="utf-8") as devnull:
            os.dup2(devnull.fileno(), 2)
            yield
    finally:
        os.dup2(original_stderr_fd, 2)
        os.close(original_stderr_fd)


def read_parquet(path: Path) -> pd.DataFrame:
    """Read a Parquet file quietly."""
    with suppress_native_stderr():
        return pd.read_parquet(path)


def write_parquet(frame: pd.DataFrame, path: Path) -> None:
    """Write a Parquet file atomically."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(".tmp.parquet")
    with suppress_native_stderr():
        frame.to_parquet(tmp_path, index=False)
    tmp_path.replace(path)


def load_config(config_path: Path = DEFAULT_CONFIG_PATH) -> dict[str, Any]:
    """Load YAML config, returning an empty config when unavailable."""
    if not config_path.exists():
        return {}
    loaded = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    return loaded if isinstance(loaded, dict) else {}


def backtest_config_value(config: dict[str, Any], key: str, default: Any) -> Any:
    """Read one backtest config value."""
    backtest_config = config.get("backtest", {})
    if not isinstance(backtest_config, dict):
        return default
    return backtest_config.get(key, default)


def find_newest_predictions(model_dataset_dir: Path = DEFAULT_MODEL_DATASET_DIR) -> Path:
    """Find the newest saved model prediction file."""
    files = sorted(model_dataset_dir.glob("*_predictions.parquet"), key=lambda path: path.stat().st_mtime)
    if not files:
        raise FileNotFoundError(f"No prediction Parquet files found in {model_dataset_dir}")
    return files[-1]


def infer_model_dataset_path(predictions_path: Path, model_dataset_dir: Path = DEFAULT_MODEL_DATASET_DIR) -> Path:
    """Infer the matching model dataset path from a predictions filename."""
    stem = predictions_path.stem
    if not stem.endswith("_predictions"):
        raise ValueError(f"Prediction filename must end with _predictions.parquet: {predictions_path}")

    prediction_stem = stem.removesuffix("_predictions")
    parts = prediction_stem.split("_")
    if len(parts) < 3:
        raise ValueError(f"Cannot infer model dataset date from {predictions_path.name}")

    date_part = parts[-1]
    symbol = parts[0]
    dataset_path = model_dataset_dir / f"{symbol}_model_dataset_{date_part}.parquet"
    if not dataset_path.exists():
        raise FileNotFoundError(f"Could not find matching model dataset: {dataset_path}")
    return dataset_path


def artifact_stem(predictions_path: Path) -> str:
    """Return output artifact stem from a prediction filename."""
    return predictions_path.stem.removesuffix("_predictions")


def format_threshold_component(value: float) -> str:
    """Format a probability-style value as a compact artifact suffix component."""
    return f"{int(round(value * 100)):03d}"


def threshold_suffix(
    signal_threshold: float,
    direction_margin: float,
    neutral_max_probability: float,
) -> str:
    """Build threshold suffix used for backtest outputs."""
    return (
        f"threshold{format_threshold_component(signal_threshold)}_"
        f"margin{format_threshold_component(direction_margin)}_"
        f"neutral{format_threshold_component(neutral_max_probability)}"
    )


def determine_cost_mode(fee_bps: float, slippage_bps: float) -> str:
    """Return the backtest cost mode implied by CLI fee/slippage inputs."""
    return "zero_cost" if fee_bps == 0.0 and slippage_bps == 0.0 else "cost_adjusted"


def create_signals(
    predictions: pd.DataFrame,
    signal_threshold: float = DEFAULT_SIGNAL_THRESHOLD,
    direction_margin: float = DEFAULT_DIRECTION_MARGIN,
    neutral_max_probability: float = DEFAULT_NEUTRAL_MAX_PROBABILITY,
) -> pd.DataFrame:
    """Create long/short/no-trade signals from model probabilities."""
    required = {"timestamp", "prob_up", "prob_down", "prob_neutral"}
    missing = sorted(required - set(predictions.columns))
    if missing:
        raise ValueError(f"Prediction file is missing required columns: {missing}")
    if not 0 <= signal_threshold <= 1:
        raise ValueError("signal threshold must be between 0 and 1")
    if direction_margin < 0:
        raise ValueError("direction margin must be nonnegative")
    if not 0 <= neutral_max_probability <= 1:
        raise ValueError("neutral max probability must be between 0 and 1")

    output = predictions.copy()
    output["timestamp"] = pd.to_datetime(output["timestamp"], utc=True, errors="coerce")
    output["prob_up"] = pd.to_numeric(output["prob_up"], errors="coerce")
    output["prob_down"] = pd.to_numeric(output["prob_down"], errors="coerce")
    output["prob_neutral"] = pd.to_numeric(output["prob_neutral"], errors="coerce")
    output = output.dropna(subset=["timestamp", "prob_up", "prob_down", "prob_neutral"])

    output["signal"] = 0
    neutral_allowed = output["prob_neutral"] <= neutral_max_probability
    long_mask = (
        (output["prob_up"] >= signal_threshold)
        & ((output["prob_up"] - output["prob_down"]) >= direction_margin)
        & neutral_allowed
    )
    short_mask = (
        (output["prob_down"] >= signal_threshold)
        & ((output["prob_down"] - output["prob_up"]) >= direction_margin)
        & neutral_allowed
    )
    output.loc[long_mask, "signal"] = 1
    output.loc[short_mask, "signal"] = -1
    return output.sort_values("timestamp").reset_index(drop=True)


def prepare_market_data(model_dataset: pd.DataFrame) -> pd.DataFrame:
    """Prepare mid-price and spread data used for realized 5-second PnL."""
    required = {"timestamp", "mid_price", "spread"}
    missing = sorted(required - set(model_dataset.columns))
    if missing:
        raise ValueError(f"Model dataset is missing required columns: {missing}")

    market = model_dataset[["timestamp", "mid_price", "spread"]].copy()
    market["timestamp"] = pd.to_datetime(market["timestamp"], utc=True, errors="coerce")
    market["mid_price"] = pd.to_numeric(market["mid_price"], errors="coerce")
    market["spread"] = pd.to_numeric(market["spread"], errors="coerce")
    market = market.dropna(subset=["timestamp", "mid_price", "spread"])
    market = market.sort_values("timestamp").drop_duplicates("timestamp", keep="last")
    market["spread_bps"] = market["spread"] / market["mid_price"] * 10_000
    return market.reset_index(drop=True)


def attach_future_prices(
    signals: pd.DataFrame,
    market: pd.DataFrame,
    horizon_seconds: int = DEFAULT_HORIZON_SECONDS,
) -> pd.DataFrame:
    """Attach current and future mid-prices using exact timestamp matching."""
    if horizon_seconds <= 0:
        raise ValueError("horizon_seconds must be positive")

    market_current = market.rename(
        columns={
            "mid_price": "entry_mid_price",
            "spread": "entry_spread",
            "spread_bps": "entry_spread_bps",
        }
    )
    merged = signals.merge(market_current, on="timestamp", how="left")
    merged["exit_timestamp"] = merged["timestamp"] + pd.Timedelta(seconds=horizon_seconds)

    market_future = market[["timestamp", "mid_price"]].rename(
        columns={"timestamp": "exit_timestamp", "mid_price": "exit_mid_price"}
    )
    merged = merged.merge(market_future, on="exit_timestamp", how="left")
    return merged.dropna(subset=["entry_mid_price", "entry_spread_bps", "exit_mid_price"]).reset_index(drop=True)


def apply_non_overlapping_hold_rule(frame: pd.DataFrame, horizon_seconds: int) -> pd.DataFrame:
    """Allow at most one open position during each hold window."""
    output = frame.sort_values("timestamp").reset_index(drop=True).copy()
    output["trade_taken"] = False
    next_available_timestamp: pd.Timestamp | None = None

    for index, row in output.iterrows():
        timestamp = row["timestamp"]
        if row["signal"] == 0:
            continue
        if next_available_timestamp is not None and timestamp < next_available_timestamp:
            continue
        output.at[index, "trade_taken"] = True
        next_available_timestamp = timestamp + pd.Timedelta(seconds=horizon_seconds)

    return output


def run_backtest(
    predictions: pd.DataFrame,
    model_dataset: pd.DataFrame,
    split: str = "test",
    signal_threshold: float = DEFAULT_SIGNAL_THRESHOLD,
    direction_margin: float = DEFAULT_DIRECTION_MARGIN,
    neutral_max_probability: float = DEFAULT_NEUTRAL_MAX_PROBABILITY,
    horizon_seconds: int = DEFAULT_HORIZON_SECONDS,
    fee_bps: float = DEFAULT_FEE_BPS,
    slippage_bps: float = DEFAULT_SLIPPAGE_BPS,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run the simple probability-driven backtest."""
    if fee_bps < 0:
        raise ValueError("fee_bps must be nonnegative")
    if slippage_bps < 0:
        raise ValueError("slippage_bps must be nonnegative")

    signals = create_signals(
        predictions,
        signal_threshold=signal_threshold,
        direction_margin=direction_margin,
        neutral_max_probability=neutral_max_probability,
    )
    if split != "all":
        if "split" not in signals.columns:
            raise ValueError("Prediction file has no split column; use --split all or provide split-aware predictions")
        signals = signals.loc[signals["split"] == split].copy()

    market = prepare_market_data(model_dataset)
    backtest = attach_future_prices(signals, market, horizon_seconds=horizon_seconds)
    backtest = apply_non_overlapping_hold_rule(backtest, horizon_seconds=horizon_seconds)

    round_trip_fee_bps = 2 * fee_bps
    round_trip_slippage_bps = 2 * slippage_bps
    backtest["gross_return_bps"] = (
        backtest["signal"] * (backtest["exit_mid_price"] / backtest["entry_mid_price"] - 1.0) * 10_000
    )
    backtest["spread_cost_bps"] = 0.0
    backtest["fee_cost_bps"] = np.where(backtest["trade_taken"], round_trip_fee_bps, 0.0)
    backtest["slippage_cost_bps"] = np.where(backtest["trade_taken"], round_trip_slippage_bps, 0.0)
    backtest["total_cost_bps"] = backtest["fee_cost_bps"] + backtest["slippage_cost_bps"]
    backtest["net_return_bps"] = np.where(
        backtest["trade_taken"],
        backtest["gross_return_bps"] - backtest["total_cost_bps"],
        0.0,
    )
    backtest["gross_return_bps"] = np.where(backtest["trade_taken"], backtest["gross_return_bps"], 0.0)
    backtest["equity_curve_bps"] = backtest["net_return_bps"].cumsum()

    trades = backtest.loc[backtest["trade_taken"]].copy().reset_index(drop=True)
    return backtest, trades


def summarize_backtest(backtest: pd.DataFrame, trades: pd.DataFrame) -> dict[str, Any]:
    """Compute simple backtest metrics."""
    no_trade_rows = int((backtest["signal"] == 0).sum()) if "signal" in backtest.columns else 0
    no_trade_percentage = float(no_trade_rows / len(backtest) * 100) if len(backtest) else 0.0
    if trades.empty:
        return {
            "rows": int(len(backtest)),
            "number_of_trades": 0,
            "long_trades": 0,
            "short_trades": 0,
            "no_trade_rows": no_trade_rows,
            "no_trade_percentage": no_trade_percentage,
            "total_gross_return_bps": 0.0,
            "total_net_return_bps": 0.0,
            "total_cost_bps": 0.0,
            "average_gross_return_bps": 0.0,
            "average_cost_per_trade_bps": 0.0,
            "average_net_return_bps": 0.0,
            "gross_return_per_trade_minus_cost_per_trade_bps": 0.0,
            "hit_rate": 0.0,
            "max_drawdown_bps": 0.0,
            "turnover": 0,
        }

    equity = backtest["equity_curve_bps"]
    running_peak = equity.cummax()
    drawdown = equity - running_peak
    average_gross_return_bps = float(trades["gross_return_bps"].mean())
    average_cost_per_trade_bps = float(trades["total_cost_bps"].mean())

    return {
        "rows": int(len(backtest)),
        "number_of_trades": int(len(trades)),
        "long_trades": int((trades["signal"] == 1).sum()),
        "short_trades": int((trades["signal"] == -1).sum()),
        "no_trade_rows": no_trade_rows,
        "no_trade_percentage": no_trade_percentage,
        "total_gross_return_bps": float(trades["gross_return_bps"].sum()),
        "total_net_return_bps": float(trades["net_return_bps"].sum()),
        "total_cost_bps": float(trades["total_cost_bps"].sum()),
        "average_gross_return_bps": average_gross_return_bps,
        "average_cost_per_trade_bps": average_cost_per_trade_bps,
        "average_net_return_bps": float(trades["net_return_bps"].mean()),
        "gross_return_per_trade_minus_cost_per_trade_bps": float(
            average_gross_return_bps - average_cost_per_trade_bps
        ),
        "hit_rate": float((trades["net_return_bps"] > 0).mean()),
        "max_drawdown_bps": float(drawdown.min()),
        "turnover": int(len(trades)),
    }


def save_equity_curve_plot(equity_curve: pd.DataFrame, output_path: Path) -> None:
    """Save an equity curve PNG."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with suppress_native_stderr():
        figure, axis = plt.subplots(figsize=(12, 5))
        axis.plot(equity_curve["timestamp"], equity_curve["equity_curve_bps"], linewidth=1)
        axis.set_title("Backtest Equity Curve")
        axis.set_xlabel("Timestamp")
        axis.set_ylabel("Net Return (bps)")
        axis.grid(True, alpha=0.3)
        figure.autofmt_xdate()
        figure.tight_layout()
        figure.savefig(output_path, dpi=150)
        plt.close(figure)


def build_text_report(metrics: dict[str, Any], config: dict[str, Any], artifacts: BacktestArtifacts) -> str:
    """Build a human-readable backtest report."""
    lines = [
        "LOB Simple Backtest Report",
        "==========================",
        f"Prediction file: {config['predictions_path']}",
        f"Model dataset: {config['model_dataset_path']}",
        f"Split: {config['split']}",
        f"Horizon seconds: {config['horizon_seconds']}",
        f"Signal threshold: {config['signal_threshold']}",
        f"Direction margin: {config['direction_margin']}",
        f"Neutral max probability: {config['neutral_max_probability']}",
        f"Cost mode: {config['cost_mode']}",
        f"Fee bps per side: {config['fee_bps']}",
        f"Slippage bps per side: {config['slippage_bps']}",
        "",
        "Important Note",
        "--------------",
        VALIDATION_SELECTED_NOTE,
        "",
        "Metrics",
        "-------",
    ]
    for key, value in metrics.items():
        if isinstance(value, float):
            lines.append(f"{key}: {value:.6f}")
        else:
            lines.append(f"{key}: {value}")
    lines.extend(
        [
            "",
            "Artifacts",
            "---------",
            f"Metrics JSON: {artifacts.metrics_path}",
            f"Equity curve parquet: {artifacts.equity_curve_path}",
            f"Equity curve plot: {artifacts.equity_curve_plot_path}",
            f"Trades parquet: {artifacts.trades_path}",
            "",
        ]
    )
    return "\n".join(lines)


def save_backtest_outputs(
    stem: str,
    backtest: pd.DataFrame,
    trades: pd.DataFrame,
    metrics: dict[str, Any],
    config: dict[str, Any],
    reports_dir: Path,
    figures_dir: Path,
) -> BacktestArtifacts:
    """Save report, metrics, trades, and equity curve artifacts."""
    artifacts = BacktestArtifacts(
        report_path=reports_dir / f"{stem}_backtest_report.txt",
        metrics_path=reports_dir / f"{stem}_backtest_metrics.json",
        equity_curve_path=reports_dir / f"{stem}_equity_curve.parquet",
        equity_curve_plot_path=figures_dir / f"{stem}_equity_curve.png",
        trades_path=reports_dir / f"{stem}_trades.parquet",
    )

    equity_curve = backtest[["timestamp", "equity_curve_bps"]].copy()
    write_parquet(equity_curve, artifacts.equity_curve_path)
    write_parquet(trades, artifacts.trades_path)
    save_equity_curve_plot(equity_curve, artifacts.equity_curve_plot_path)

    artifacts.metrics_path.parent.mkdir(parents=True, exist_ok=True)
    artifacts.metrics_path.write_text(
        json.dumps({"config": config, "metrics": metrics}, indent=2),
        encoding="utf-8",
    )
    artifacts.report_path.write_text(build_text_report(metrics, config, artifacts), encoding="utf-8")
    return artifacts


def parse_args() -> argparse.Namespace:
    config = load_config()
    default_signal_threshold = float(
        backtest_config_value(config, "signal_threshold", DEFAULT_SIGNAL_THRESHOLD)
    )
    default_direction_margin = float(
        backtest_config_value(config, "direction_margin", DEFAULT_DIRECTION_MARGIN)
    )
    default_neutral_max_probability = float(
        backtest_config_value(config, "neutral_max_probability", DEFAULT_NEUTRAL_MAX_PROBABILITY)
    )
    default_fee_bps = float(backtest_config_value(config, "fee_bps", DEFAULT_FEE_BPS))
    default_slippage_bps = float(backtest_config_value(config, "slippage_bps", DEFAULT_SLIPPAGE_BPS))

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", "--predictions-path", dest="predictions_path", type=Path, default=None)
    parser.add_argument("--dataset", "--model-dataset-path", dest="model_dataset_path", type=Path, default=None)
    parser.add_argument("--model-dataset-dir", type=Path, default=DEFAULT_MODEL_DATASET_DIR)
    parser.add_argument("--reports-dir", type=Path, default=DEFAULT_REPORTS_DIR)
    parser.add_argument("--figures-dir", type=Path, default=DEFAULT_FIGURES_DIR)
    parser.add_argument("--split", choices=["validation", "test", "all"], default="test")
    parser.add_argument("--signal-threshold", type=float, default=default_signal_threshold)
    parser.add_argument("--direction-margin", type=float, default=default_direction_margin)
    parser.add_argument("--neutral-max-probability", type=float, default=default_neutral_max_probability)
    parser.add_argument("--horizon-seconds", type=int, default=DEFAULT_HORIZON_SECONDS)
    parser.add_argument("--fee-bps", type=float, default=default_fee_bps)
    parser.add_argument("--slippage-bps", type=float, default=default_slippage_bps)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    predictions_path = args.predictions_path or find_newest_predictions(args.model_dataset_dir)
    model_dataset_path = args.model_dataset_path or infer_model_dataset_path(predictions_path, args.model_dataset_dir)
    cost_mode = determine_cost_mode(args.fee_bps, args.slippage_bps)

    predictions = read_parquet(predictions_path)
    model_dataset = read_parquet(model_dataset_path)
    backtest, trades = run_backtest(
        predictions=predictions,
        model_dataset=model_dataset,
        split=args.split,
        signal_threshold=args.signal_threshold,
        direction_margin=args.direction_margin,
        neutral_max_probability=args.neutral_max_probability,
        horizon_seconds=args.horizon_seconds,
        fee_bps=args.fee_bps,
        slippage_bps=args.slippage_bps,
    )
    metrics = summarize_backtest(backtest, trades)
    config = {
        "predictions_path": str(predictions_path),
        "model_dataset_path": str(model_dataset_path),
        "split": args.split,
        "signal_threshold": args.signal_threshold,
        "direction_margin": args.direction_margin,
        "neutral_max_probability": args.neutral_max_probability,
        "cost_mode": cost_mode,
        "horizon_seconds": args.horizon_seconds,
        "fee_bps": args.fee_bps,
        "slippage_bps": args.slippage_bps,
    }
    output_stem = (
        f"{artifact_stem(predictions_path)}_"
        f"{args.split}_"
        f"{threshold_suffix(args.signal_threshold, args.direction_margin, args.neutral_max_probability)}_"
        f"{cost_mode}"
    )
    artifacts = save_backtest_outputs(
        stem=output_stem,
        backtest=backtest,
        trades=trades,
        metrics=metrics,
        config=config,
        reports_dir=args.reports_dir,
        figures_dir=args.figures_dir,
    )

    print(f"Saved backtest report: {artifacts.report_path}", flush=True)
    print(f"Saved equity curve: {artifacts.equity_curve_path}", flush=True)
    print(f"Saved equity plot: {artifacts.equity_curve_plot_path}", flush=True)
    print(f"Trades: {metrics['number_of_trades']}", flush=True)
    print(f"Total net return bps: {metrics['total_net_return_bps']:.6f}", flush=True)


if __name__ == "__main__":
    main()
