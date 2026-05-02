"""Sweep probability thresholds for simple LOB trading signals."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import dataclass
import json
import os
from pathlib import Path
import re
import sys
from typing import Any

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_REPORTS_DIR = PROJECT_ROOT / "reports"
DEFAULT_HORIZON_SECONDS = 5
DEFAULT_FEE_BPS = 1.0
DEFAULT_SLIPPAGE_BPS = 1.0
SIGNAL_THRESHOLDS = [0.55, 0.60, 0.65, 0.70, 0.75, 0.80]
DIRECTION_MARGINS = [0.00, 0.05, 0.10, 0.15, 0.20]
NEUTRAL_MAX_PROBABILITIES = [0.30, 0.40, 0.50, 0.60]


@dataclass(frozen=True)
class SweepArtifacts:
    """Saved threshold sweep artifact paths."""

    csv_path: Path
    json_path: Path


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


def read_parquet(path: Path) -> pd.DataFrame:
    """Read a Parquet file quietly."""
    with suppress_native_stderr():
        return pd.read_parquet(path)


def sanitize_name(value: str) -> str:
    """Return a filesystem-friendly artifact name component."""
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_")


def dataset_name_from_path(dataset_path: Path) -> str:
    """Extract dataset token from model dataset filename."""
    stem = dataset_path.stem
    if "_model_dataset_" in stem:
        return stem.split("_model_dataset_", maxsplit=1)[1]
    return stem


def model_name_from_predictions_path(predictions_path: Path, dataset_name: str) -> str:
    """Extract model name from prediction filename."""
    stem = predictions_path.stem
    if stem.endswith("_predictions"):
        stem = stem.removesuffix("_predictions")

    prefix = "BTCUSDT_"
    if stem.startswith(prefix):
        stem = stem[len(prefix) :]

    dataset_suffix = f"_{dataset_name}"
    if stem.endswith(dataset_suffix):
        stem = stem[: -len(dataset_suffix)]

    return stem or predictions_path.stem.removesuffix("_predictions")


def sweep_output_paths(
    predictions_path: Path,
    dataset_path: Path,
    split: str,
    reports_dir: Path = DEFAULT_REPORTS_DIR,
) -> SweepArtifacts:
    """Build threshold sweep output paths."""
    dataset_name = sanitize_name(dataset_name_from_path(dataset_path))
    model_name = sanitize_name(model_name_from_predictions_path(predictions_path, dataset_name))
    stem = f"threshold_sweep_{model_name}_{dataset_name}_{split}"
    return SweepArtifacts(
        csv_path=reports_dir / f"{stem}.csv",
        json_path=reports_dir / f"{stem}.json",
    )


def prepare_predictions(predictions: pd.DataFrame, split: str) -> pd.DataFrame:
    """Prepare prediction probabilities for one split."""
    required = {"timestamp", "split", "prob_down", "prob_neutral", "prob_up"}
    missing = sorted(required - set(predictions.columns))
    if missing:
        raise ValueError(f"Prediction file is missing required columns: {missing}")

    output = predictions.copy()
    output["timestamp"] = pd.to_datetime(output["timestamp"], utc=True, errors="coerce")
    for column in ["prob_down", "prob_neutral", "prob_up"]:
        output[column] = pd.to_numeric(output[column], errors="coerce")
    output = output.dropna(subset=["timestamp", "split", "prob_down", "prob_neutral", "prob_up"])
    output = output.loc[output["split"] == split].copy()
    if output.empty:
        raise ValueError(f"No prediction rows found for split={split}")
    return output.sort_values("timestamp", kind="mergesort").reset_index(drop=True)


def prepare_market_data(model_dataset: pd.DataFrame) -> pd.DataFrame:
    """Prepare mid-price and spread data for realized 5-second PnL."""
    required = {"timestamp", "mid_price", "spread"}
    missing = sorted(required - set(model_dataset.columns))
    if missing:
        raise ValueError(f"Model dataset is missing required columns: {missing}")

    market = model_dataset[["timestamp", "mid_price", "spread"]].copy()
    market["timestamp"] = pd.to_datetime(market["timestamp"], utc=True, errors="coerce")
    market["mid_price"] = pd.to_numeric(market["mid_price"], errors="coerce")
    market["spread"] = pd.to_numeric(market["spread"], errors="coerce")
    market = market.dropna(subset=["timestamp", "mid_price", "spread"])
    market = market.sort_values("timestamp", kind="mergesort").drop_duplicates("timestamp", keep="last")
    market["spread_bps"] = market["spread"] / market["mid_price"] * 10_000
    return market.reset_index(drop=True)


def create_threshold_signals(
    predictions: pd.DataFrame,
    signal_threshold: float,
    direction_margin: float,
    neutral_max_probability: float,
) -> pd.DataFrame:
    """Create long/short/no-trade signals using threshold, margin, and neutral cap."""
    output = predictions.copy()
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
    return output


def attach_future_prices(
    signals: pd.DataFrame,
    market: pd.DataFrame,
    horizon_seconds: int = DEFAULT_HORIZON_SECONDS,
) -> pd.DataFrame:
    """Attach current and future mid-prices using exact timestamp matching."""
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
    output = frame.sort_values("timestamp", kind="mergesort").reset_index(drop=True).copy()
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


def run_one_configuration(
    predictions: pd.DataFrame,
    market: pd.DataFrame,
    split: str,
    signal_threshold: float,
    direction_margin: float,
    neutral_max_probability: float,
    fee_bps: float,
    slippage_bps: float,
    horizon_seconds: int = DEFAULT_HORIZON_SECONDS,
) -> dict[str, Any]:
    """Backtest one signal-threshold configuration."""
    signals = create_threshold_signals(
        predictions,
        signal_threshold=signal_threshold,
        direction_margin=direction_margin,
        neutral_max_probability=neutral_max_probability,
    )
    backtest = attach_future_prices(signals, market, horizon_seconds=horizon_seconds)
    backtest = apply_non_overlapping_hold_rule(backtest, horizon_seconds=horizon_seconds)

    round_trip_fee_bps = 2 * fee_bps
    round_trip_slippage_bps = 2 * slippage_bps
    backtest["gross_return_bps"] = (
        backtest["signal"] * (backtest["exit_mid_price"] / backtest["entry_mid_price"] - 1.0) * 10_000
    )
    backtest["spread_cost_bps"] = np.where(backtest["trade_taken"], backtest["entry_spread_bps"], 0.0)
    backtest["fee_cost_bps"] = np.where(backtest["trade_taken"], round_trip_fee_bps, 0.0)
    backtest["slippage_cost_bps"] = np.where(backtest["trade_taken"], round_trip_slippage_bps, 0.0)
    backtest["total_cost_bps"] = backtest["spread_cost_bps"] + backtest["fee_cost_bps"] + backtest["slippage_cost_bps"]
    backtest["net_return_bps"] = np.where(
        backtest["trade_taken"],
        backtest["gross_return_bps"] - backtest["total_cost_bps"],
        0.0,
    )
    backtest["gross_return_bps"] = np.where(backtest["trade_taken"], backtest["gross_return_bps"], 0.0)
    backtest["equity_curve_bps"] = backtest["net_return_bps"].cumsum()

    trades = backtest.loc[backtest["trade_taken"]].copy()
    number_of_trades = int(len(trades))
    rows = int(len(backtest))
    no_trade_rows = int((backtest["signal"] == 0).sum())
    no_trade_percentage = float(no_trade_rows / rows * 100) if rows else 0.0

    if number_of_trades == 0:
        return {
            "split": split,
            "signal_threshold": signal_threshold,
            "direction_margin": direction_margin,
            "neutral_max_probability": neutral_max_probability,
            "number_of_trades": 0,
            "long_trades": 0,
            "short_trades": 0,
            "no_trade_rows": no_trade_rows,
            "no_trade_percentage": no_trade_percentage,
            "total_gross_return_bps": 0.0,
            "total_net_return_bps": 0.0,
            "total_cost_bps": 0.0,
            "average_gross_return_bps": 0.0,
            "average_net_return_bps": 0.0,
            "hit_rate": 0.0,
            "max_drawdown_bps": 0.0,
            "turnover": 0,
            "average_cost_per_trade_bps": 0.0,
            "gross_return_per_trade_minus_cost_per_trade_bps": 0.0,
        }

    equity = backtest["equity_curve_bps"]
    drawdown = equity - equity.cummax()
    total_gross_return_bps = float(trades["gross_return_bps"].sum())
    total_net_return_bps = float(trades["net_return_bps"].sum())
    total_cost_bps = float(trades["total_cost_bps"].sum())
    average_gross_return_bps = float(trades["gross_return_bps"].mean())
    average_net_return_bps = float(trades["net_return_bps"].mean())
    average_cost_per_trade_bps = float(trades["total_cost_bps"].mean())

    return {
        "split": split,
        "signal_threshold": signal_threshold,
        "direction_margin": direction_margin,
        "neutral_max_probability": neutral_max_probability,
        "number_of_trades": number_of_trades,
        "long_trades": int((trades["signal"] == 1).sum()),
        "short_trades": int((trades["signal"] == -1).sum()),
        "no_trade_rows": no_trade_rows,
        "no_trade_percentage": no_trade_percentage,
        "total_gross_return_bps": total_gross_return_bps,
        "total_net_return_bps": total_net_return_bps,
        "total_cost_bps": total_cost_bps,
        "average_gross_return_bps": average_gross_return_bps,
        "average_net_return_bps": average_net_return_bps,
        "hit_rate": float((trades["net_return_bps"] > 0).mean()),
        "max_drawdown_bps": float(drawdown.min()),
        "turnover": number_of_trades,
        "average_cost_per_trade_bps": average_cost_per_trade_bps,
        "gross_return_per_trade_minus_cost_per_trade_bps": float(
            average_gross_return_bps - average_cost_per_trade_bps
        ),
    }


def run_threshold_sweep(
    predictions: pd.DataFrame,
    model_dataset: pd.DataFrame,
    split: str,
    fee_bps: float,
    slippage_bps: float,
) -> pd.DataFrame:
    """Run the full threshold/margin/neutral-probability sweep."""
    split_predictions = prepare_predictions(predictions, split=split)
    market = prepare_market_data(model_dataset)
    rows: list[dict[str, Any]] = []

    for signal_threshold in SIGNAL_THRESHOLDS:
        for direction_margin in DIRECTION_MARGINS:
            for neutral_max_probability in NEUTRAL_MAX_PROBABILITIES:
                rows.append(
                    run_one_configuration(
                        predictions=split_predictions,
                        market=market,
                        split=split,
                        signal_threshold=signal_threshold,
                        direction_margin=direction_margin,
                        neutral_max_probability=neutral_max_probability,
                        fee_bps=fee_bps,
                        slippage_bps=slippage_bps,
                    )
                )

    return pd.DataFrame(rows)


def sorted_sweep_results(results: pd.DataFrame) -> pd.DataFrame:
    """Sort sweep results by the requested selection view."""
    return results.sort_values(
        ["average_net_return_bps", "average_gross_return_bps", "number_of_trades"],
        ascending=[False, False, True],
    ).reset_index(drop=True)


def save_sweep_results(
    results: pd.DataFrame,
    predictions_path: Path,
    dataset_path: Path,
    split: str,
    fee_bps: float,
    slippage_bps: float,
    reports_dir: Path = DEFAULT_REPORTS_DIR,
) -> SweepArtifacts:
    """Save sweep results as CSV and JSON."""
    artifacts = sweep_output_paths(predictions_path, dataset_path, split, reports_dir=reports_dir)
    artifacts.csv_path.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(artifacts.csv_path, index=False)
    payload = {
        "config": {
            "predictions": str(predictions_path),
            "dataset": str(dataset_path),
            "split": split,
            "fee_bps": fee_bps,
            "slippage_bps": slippage_bps,
            "signal_thresholds": SIGNAL_THRESHOLDS,
            "direction_margins": DIRECTION_MARGINS,
            "neutral_max_probabilities": NEUTRAL_MAX_PROBABILITIES,
            "selection_sort": [
                "average_net_return_bps desc",
                "average_gross_return_bps desc",
                "number_of_trades asc",
            ],
        },
        "results": results.to_dict(orient="records"),
        "top_10": sorted_sweep_results(results).head(10).to_dict(orient="records"),
    }
    artifacts.json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return artifacts


def print_top_10(results: pd.DataFrame) -> None:
    """Print top 10 configurations using the requested sort order."""
    top = sorted_sweep_results(results).head(10)
    columns = [
        "signal_threshold",
        "direction_margin",
        "neutral_max_probability",
        "number_of_trades",
        "no_trade_percentage",
        "average_gross_return_bps",
        "average_cost_per_trade_bps",
        "average_net_return_bps",
        "total_cost_bps",
        "hit_rate",
    ]
    print("Top 10 threshold configurations:", flush=True)
    print(top[columns].to_string(index=False), flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--split", choices=["validation", "test"], default="validation")
    parser.add_argument("--fee-bps", type=float, default=DEFAULT_FEE_BPS)
    parser.add_argument("--slippage-bps", type=float, default=DEFAULT_SLIPPAGE_BPS)
    parser.add_argument("--reports-dir", type=Path, default=DEFAULT_REPORTS_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    predictions = read_parquet(args.predictions)
    model_dataset = read_parquet(args.dataset)
    results = run_threshold_sweep(
        predictions=predictions,
        model_dataset=model_dataset,
        split=args.split,
        fee_bps=args.fee_bps,
        slippage_bps=args.slippage_bps,
    )
    sorted_results = sorted_sweep_results(results)
    artifacts = save_sweep_results(
        results=sorted_results,
        predictions_path=args.predictions,
        dataset_path=args.dataset,
        split=args.split,
        fee_bps=args.fee_bps,
        slippage_bps=args.slippage_bps,
        reports_dir=args.reports_dir,
    )

    print(f"Saved threshold sweep CSV: {artifacts.csv_path}", flush=True)
    print(f"Saved threshold sweep JSON: {artifacts.json_path}", flush=True)
    print_top_10(sorted_results)


if __name__ == "__main__":
    main()
