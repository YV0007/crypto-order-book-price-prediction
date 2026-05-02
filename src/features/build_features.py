"""Build market microstructure features from cleaned top-20 order book data."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CLEANED_DIR = PROJECT_ROOT / "data" / "interim" / "cleaned_orderbook"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "processed" / "features"
DEFAULT_REPORT_PATH = PROJECT_ROOT / "reports" / "feature_build_report.txt"
DEFAULT_LEVELS = 20
DEPTH_WINDOWS = (1, 5, 10, 20)
DELTA_SOURCE_COLUMNS = ["mid_price", "spread", "imbalance_1", "imbalance_5", "imbalance_10", "imbalance_20"]
LAG_FEATURE_SPECS: dict[str, tuple[int, ...]] = {
    "imbalance_1": (1, 5, 10),
    "imbalance_5": (1, 5, 10),
    "imbalance_20": (1, 5),
    "weighted_imbalance": (1, 5, 10),
    "microprice_minus_mid": (1, 5, 10),
    "spread": (1, 5),
    "relative_spread": (1, 5),
}
ROLLING_MEAN_SPECS: dict[str, tuple[int, ...]] = {
    "imbalance_1": (5, 10, 30),
    "imbalance_5": (5, 10, 30),
    "imbalance_20": (5, 10, 30),
    "weighted_imbalance": (5, 10, 30),
    "microprice_minus_mid": (5, 10, 30),
    "spread": (5, 10),
    "relative_spread": (5, 10),
}
ROLLING_STD_SPECS: dict[str, tuple[int, ...]] = {
    "imbalance_1": (10, 30),
    "imbalance_5": (10, 30),
    "weighted_imbalance": (10, 30),
    "microprice_minus_mid": (10, 30),
    "spread": (10, 30),
    "relative_spread": (10, 30),
    "mid_return": (10, 30),
}
CHANGE_FEATURE_SPECS: dict[str, tuple[int, ...]] = {
    "imbalance_1": (5, 10),
    "imbalance_5": (5, 10),
    "imbalance_20": (5, 10),
    "weighted_imbalance": (5, 10),
    "microprice_minus_mid": (5, 10),
    "spread": (5,),
    "relative_spread": (5,),
    "bid_depth_1": (5,),
    "ask_depth_1": (5,),
    "bid_depth_5": (5,),
    "ask_depth_5": (5,),
    "bid_depth_20": (5,),
    "ask_depth_20": (5,),
    "weighted_bid_depth": (5,),
    "weighted_ask_depth": (5,),
}
MID_RETURN_LAGS = (1, 5, 10)
METADATA_COLUMNS = {"timestamp", "event_time", "exchange", "symbol", "session_id"}


@dataclass(frozen=True)
class FeatureBuildSummary:
    """Summary of the feature build run."""

    input_path: Path
    output_path: Path
    report_path: Path
    total_input_rows: int
    rows: int
    columns: int
    session_aware_features: bool
    unique_session_ids: int
    rows_dropped_due_to_lag_rolling_nans: int
    new_lagged_feature_count: int
    new_rolling_mean_feature_count: int
    new_rolling_std_feature_count: int
    new_change_feature_count: int
    total_final_feature_count: int
    largest_timestamp_gap_seconds: float
    gaps_gt_1_5_seconds: int
    gaps_gt_5_seconds: int
    gaps_gt_60_seconds: int


@dataclass(frozen=True)
class FeatureBuildDiagnostics:
    """Diagnostics captured while building features."""

    total_input_rows: int
    total_output_rows: int
    session_aware_features: bool
    unique_session_ids: int
    rows_dropped_due_to_lag_rolling_nans: int
    new_lagged_feature_count: int
    new_rolling_mean_feature_count: int
    new_rolling_std_feature_count: int
    new_change_feature_count: int
    total_final_feature_count: int
    largest_timestamp_gap_seconds: float
    gaps_gt_1_5_seconds: int
    gaps_gt_5_seconds: int
    gaps_gt_60_seconds: int


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


def find_newest_parquet(cleaned_dir: Path = DEFAULT_CLEANED_DIR) -> Path:
    """Find the newest cleaned Parquet file by modification time."""
    files = sorted(cleaned_dir.glob("*.parquet"), key=lambda path: path.stat().st_mtime)
    if not files:
        raise FileNotFoundError(f"No Parquet files found in {cleaned_dir}")
    return files[-1]


def feature_output_path(input_path: Path, output_dir: Path) -> Path:
    """Build the output feature filename from a cleaned input filename."""
    parts = input_path.stem.split("_", maxsplit=1)
    if len(parts) == 2:
        symbol, date_part = parts
        filename = f"{symbol}_features_{date_part}.parquet"
    else:
        filename = f"{input_path.stem}_features.parquet"
    return output_dir / filename


def price_columns(levels: int = DEFAULT_LEVELS) -> list[str]:
    """Return all price columns."""
    columns: list[str] = []
    for level in range(1, levels + 1):
        columns.extend([f"bid_price_{level}", f"ask_price_{level}"])
    return columns


def size_columns(levels: int = DEFAULT_LEVELS) -> list[str]:
    """Return all size columns."""
    columns: list[str] = []
    for level in range(1, levels + 1):
        columns.extend([f"bid_size_{level}", f"ask_size_{level}"])
    return columns


def required_columns(levels: int = DEFAULT_LEVELS) -> list[str]:
    """Return required columns for feature construction."""
    return ["timestamp", "event_time", "exchange", "symbol", *price_columns(levels), *size_columns(levels)]


def ensure_numeric_orderbook(frame: pd.DataFrame, levels: int = DEFAULT_LEVELS) -> pd.DataFrame:
    """Return a copy with timestamps and order book levels typed for calculations."""
    missing = sorted(set(required_columns(levels)) - set(frame.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    output = frame.copy()
    output["timestamp"] = pd.to_datetime(output["timestamp"], utc=True, errors="coerce")
    if "session_id" in output.columns:
        output["session_id"] = pd.to_numeric(output["session_id"], errors="coerce")
    for column in [*price_columns(levels), *size_columns(levels)]:
        output[column] = pd.to_numeric(output[column], errors="coerce")

    output = output.dropna(subset=["timestamp", *price_columns(levels), *size_columns(levels)])
    sort_columns = ["timestamp"]
    if "session_id" in output.columns:
        sort_columns = ["session_id", "timestamp"]
    output = output.sort_values(sort_columns, kind="mergesort").reset_index(drop=True)
    output["_feature_row_id"] = range(len(output))
    return output


def timestamp_gap_diagnostics(frame: pd.DataFrame) -> dict[str, float | int]:
    """Compute timestamp gap diagnostics from timestamp-sorted rows."""
    timestamps = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce").dropna().sort_values()
    gaps = timestamps.diff().dt.total_seconds().dropna()
    if gaps.empty:
        return {
            "largest_timestamp_gap_seconds": 0.0,
            "gaps_gt_1_5_seconds": 0,
            "gaps_gt_5_seconds": 0,
            "gaps_gt_60_seconds": 0,
        }

    return {
        "largest_timestamp_gap_seconds": float(gaps.max()),
        "gaps_gt_1_5_seconds": int((gaps > 1.5).sum()),
        "gaps_gt_5_seconds": int((gaps > 5.0).sum()),
        "gaps_gt_60_seconds": int((gaps > 60.0).sum()),
    }


def session_group_keys(frame: pd.DataFrame, max_gap_seconds: float = 1.5) -> pd.Series | None:
    """Build internal group keys that respect session_id and large timestamp gaps."""
    if "session_id" not in frame.columns:
        return None

    gaps = frame.groupby("session_id", sort=False, dropna=False)["timestamp"].diff().dt.total_seconds()
    continuity_breaks = gaps.gt(max_gap_seconds).fillna(False)
    continuity_segments = continuity_breaks.groupby(frame["session_id"], sort=False, dropna=False).cumsum().astype(int)
    session_values = frame["session_id"].astype("string").fillna("missing_session")
    return session_values + "_segment_" + continuity_segments.astype("string")


def safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    """Divide while returning NaN where the denominator is zero."""
    return numerator / denominator.replace(0, np.nan)


def grouped_shift(
    feature_frame: pd.DataFrame,
    column: str,
    periods: int,
    session_ids: pd.Series | None,
) -> pd.Series:
    """Shift a feature by rows, staying inside session boundaries when available."""
    if session_ids is not None:
        return feature_frame.groupby(session_ids, sort=False, dropna=False)[column].shift(periods)
    return feature_frame[column].shift(periods)


def grouped_diff(
    feature_frame: pd.DataFrame,
    column: str,
    session_ids: pd.Series | None,
) -> pd.Series:
    """Diff a feature by one row, staying inside session boundaries when available."""
    if session_ids is not None:
        return feature_frame.groupby(session_ids, sort=False, dropna=False)[column].diff()
    return feature_frame[column].diff()


def grouped_rolling(
    feature_frame: pd.DataFrame,
    column: str,
    window: int,
    aggregation: str,
    session_ids: pd.Series | None,
) -> pd.Series:
    """Apply a fixed-row rolling aggregation inside each session when available."""
    if session_ids is None:
        rolling = feature_frame[column].rolling(window=window, min_periods=window)
        if aggregation == "mean":
            return rolling.mean()
        if aggregation == "std":
            return rolling.std()
        raise ValueError(f"Unsupported rolling aggregation: {aggregation}")

    rolling = feature_frame.groupby(session_ids, sort=False, dropna=False)[column].rolling(
        window=window,
        min_periods=window,
    )
    if aggregation == "mean":
        result = rolling.mean()
    elif aggregation == "std":
        result = rolling.std()
    else:
        raise ValueError(f"Unsupported rolling aggregation: {aggregation}")
    return result.reset_index(level=0, drop=True).sort_index()


def depth_and_imbalance_features(frame: pd.DataFrame, levels: int = DEFAULT_LEVELS) -> dict[str, pd.Series]:
    """Build depth and imbalance feature columns for configured book depths."""
    columns: dict[str, pd.Series] = {}
    for depth in DEPTH_WINDOWS:
        if depth > levels:
            continue

        bid_cols = [f"bid_size_{level}" for level in range(1, depth + 1)]
        ask_cols = [f"ask_size_{level}" for level in range(1, depth + 1)]
        bid_depth_col = f"bid_depth_{depth}"
        ask_depth_col = f"ask_depth_{depth}"
        imbalance_col = f"imbalance_{depth}"

        columns[bid_depth_col] = frame[bid_cols].sum(axis=1)
        columns[ask_depth_col] = frame[ask_cols].sum(axis=1)
        total_depth = columns[bid_depth_col] + columns[ask_depth_col]
        columns[imbalance_col] = safe_divide(columns[bid_depth_col] - columns[ask_depth_col], total_depth)

    return columns


def weighted_depth_features(frame: pd.DataFrame, levels: int = DEFAULT_LEVELS) -> dict[str, pd.Series]:
    """Build inverse-level weighted depth and imbalance feature columns."""
    weighted_bid_depth = pd.Series(0.0, index=frame.index)
    weighted_ask_depth = pd.Series(0.0, index=frame.index)

    for level in range(1, levels + 1):
        weight = 1.0 / level
        weighted_bid_depth += weight * frame[f"bid_size_{level}"]
        weighted_ask_depth += weight * frame[f"ask_size_{level}"]

    total_weighted_depth = weighted_bid_depth + weighted_ask_depth
    return {
        "weighted_bid_depth": weighted_bid_depth,
        "weighted_ask_depth": weighted_ask_depth,
        "weighted_imbalance": safe_divide(
            weighted_bid_depth - weighted_ask_depth,
            total_weighted_depth,
        ),
    }


def microprice_features(frame: pd.DataFrame, mid_price: pd.Series) -> dict[str, pd.Series]:
    """Build top-of-book microprice feature columns."""
    size_sum = frame["bid_size_1"] + frame["ask_size_1"]
    numerator = frame["ask_price_1"] * frame["bid_size_1"] + frame["bid_price_1"] * frame["ask_size_1"]
    microprice = safe_divide(numerator, size_sum)
    return {
        "microprice": microprice,
        "microprice_minus_mid": microprice - mid_price,
    }


def delta_features(feature_frame: pd.DataFrame, session_ids: pd.Series | None = None) -> dict[str, pd.Series]:
    """Build backward-looking one-row delta columns without crossing sessions."""
    columns: dict[str, pd.Series] = {}
    for column in DELTA_SOURCE_COLUMNS:
        if column not in feature_frame.columns:
            continue
        columns[f"delta_{column}"] = grouped_diff(feature_frame, column, session_ids)
    return columns


def lagged_features(feature_frame: pd.DataFrame, session_ids: pd.Series | None = None) -> dict[str, pd.Series]:
    """Build session-aware lagged order book state features."""
    columns: dict[str, pd.Series] = {}
    for source_column, lags in LAG_FEATURE_SPECS.items():
        if source_column not in feature_frame.columns:
            continue
        for lag in lags:
            columns[f"{source_column}_lag_{lag}"] = grouped_shift(feature_frame, source_column, lag, session_ids)
    return columns


def rolling_mean_features(feature_frame: pd.DataFrame, session_ids: pd.Series | None = None) -> dict[str, pd.Series]:
    """Build session-aware rolling mean features."""
    columns: dict[str, pd.Series] = {}
    for source_column, windows in ROLLING_MEAN_SPECS.items():
        if source_column not in feature_frame.columns:
            continue
        for window in windows:
            columns[f"{source_column}_roll_mean_{window}"] = grouped_rolling(
                feature_frame,
                source_column,
                window,
                "mean",
                session_ids,
            )
    return columns


def rolling_std_features(feature_frame: pd.DataFrame, session_ids: pd.Series | None = None) -> dict[str, pd.Series]:
    """Build session-aware rolling standard deviation features."""
    columns: dict[str, pd.Series] = {}
    for source_column, windows in ROLLING_STD_SPECS.items():
        input_source = "mid_return_1" if source_column == "mid_return" else source_column
        if input_source not in feature_frame.columns:
            continue
        for window in windows:
            columns[f"{source_column}_roll_std_{window}"] = grouped_rolling(
                feature_frame,
                input_source,
                window,
                "std",
                session_ids,
            )
    return columns


def change_features(feature_frame: pd.DataFrame, session_ids: pd.Series | None = None) -> dict[str, pd.Series]:
    """Build session-aware multi-row change and return features."""
    columns: dict[str, pd.Series] = {}
    for source_column, windows in CHANGE_FEATURE_SPECS.items():
        if source_column not in feature_frame.columns:
            continue
        for window in windows:
            columns[f"{source_column}_change_{window}"] = (
                feature_frame[source_column] - grouped_shift(feature_frame, source_column, window, session_ids)
            )

    if "mid_price" in feature_frame.columns:
        for lag in MID_RETURN_LAGS:
            lagged_mid_price = grouped_shift(feature_frame, "mid_price", lag, session_ids)
            columns[f"mid_return_{lag}"] = safe_divide(feature_frame["mid_price"], lagged_mid_price) - 1.0
    return columns


def verify_session_history_boundaries(
    feature_frame: pd.DataFrame,
    first_session_row_ids: set[int],
) -> None:
    """Verify first rows of continuity groups are not present after history NaNs are dropped."""
    if not first_session_row_ids or "_feature_row_id" not in feature_frame.columns:
        return

    remaining_first_rows = first_session_row_ids & set(feature_frame["_feature_row_id"].astype(int))
    if remaining_first_rows:
        raise ValueError(
            "Session-aware feature verification failed: first row of at least one session still has history features"
        )


def build_features_with_diagnostics(
    frame: pd.DataFrame,
    levels: int = DEFAULT_LEVELS,
) -> tuple[pd.DataFrame, FeatureBuildDiagnostics]:
    """Build all MVP market microstructure features and session-aware diagnostics."""
    base = ensure_numeric_orderbook(frame, levels=levels)
    gap_stats = timestamp_gap_diagnostics(base)
    session_aware_features = "session_id" in base.columns
    session_keys = session_group_keys(base) if session_aware_features else None
    unique_session_ids = int(session_keys.nunique(dropna=True)) if session_keys is not None else 0
    if not session_aware_features:
        print("WARNING: session_id not found. Rolling/lagged features are calculated globally.", flush=True)

    feature_columns: dict[str, pd.Series] = {}
    feature_columns["mid_price"] = (base["bid_price_1"] + base["ask_price_1"]) / 2
    feature_columns["spread"] = base["ask_price_1"] - base["bid_price_1"]
    feature_columns["relative_spread"] = safe_divide(feature_columns["spread"], feature_columns["mid_price"])
    feature_columns.update(depth_and_imbalance_features(base, levels=levels))
    feature_columns.update(weighted_depth_features(base, levels=levels))
    feature_columns.update(microprice_features(base, feature_columns["mid_price"]))

    feature_frame = pd.DataFrame(feature_columns, index=base.index)

    delta_feature_columns = delta_features(feature_frame, session_ids=session_keys)
    lag_feature_columns = lagged_features(feature_frame, session_ids=session_keys)
    change_feature_columns = change_features(feature_frame, session_ids=session_keys)
    rolling_mean_feature_columns = rolling_mean_features(feature_frame, session_ids=session_keys)

    rolling_source_frame = pd.concat(
        [feature_frame, pd.DataFrame(change_feature_columns, index=base.index)],
        axis=1,
    )
    rolling_std_feature_columns = rolling_std_features(rolling_source_frame, session_ids=session_keys)

    history_feature_columns: dict[str, pd.Series] = {}
    history_feature_columns.update(delta_feature_columns)
    history_feature_columns.update(lag_feature_columns)
    history_feature_columns.update(rolling_mean_feature_columns)
    history_feature_columns.update(rolling_std_feature_columns)
    history_feature_columns.update(change_feature_columns)

    history_feature_names = list(history_feature_columns)
    features = pd.concat(
        [
            base,
            feature_frame,
            pd.DataFrame(history_feature_columns, index=base.index),
        ],
        axis=1,
    ).copy()

    first_session_row_ids: set[int] = set()
    if session_keys is not None:
        first_session_row_ids = set(
            base.groupby(session_keys, sort=False, dropna=False)["_feature_row_id"].first().astype(int)
        )

    before_drop = len(features)
    if history_feature_names:
        features = features.dropna(subset=history_feature_names).copy()
    rows_dropped_due_to_lag_rolling_nans = before_drop - len(features)
    verify_session_history_boundaries(features, first_session_row_ids)
    if history_feature_names and features[history_feature_names].isna().any().any():
        raise ValueError("Feature build failed: lagged, rolling, or change features still contain NaN values")

    features = features.drop(columns=["_feature_row_id"])
    features = features.reset_index(drop=True)
    total_final_feature_count = len([column for column in features.columns if column not in METADATA_COLUMNS])

    diagnostics = FeatureBuildDiagnostics(
        total_input_rows=len(frame),
        total_output_rows=len(features),
        session_aware_features=session_aware_features,
        unique_session_ids=unique_session_ids,
        rows_dropped_due_to_lag_rolling_nans=rows_dropped_due_to_lag_rolling_nans,
        new_lagged_feature_count=len(lag_feature_columns),
        new_rolling_mean_feature_count=len(rolling_mean_feature_columns),
        new_rolling_std_feature_count=len(rolling_std_feature_columns),
        new_change_feature_count=len(change_feature_columns),
        total_final_feature_count=total_final_feature_count,
        largest_timestamp_gap_seconds=float(gap_stats["largest_timestamp_gap_seconds"]),
        gaps_gt_1_5_seconds=int(gap_stats["gaps_gt_1_5_seconds"]),
        gaps_gt_5_seconds=int(gap_stats["gaps_gt_5_seconds"]),
        gaps_gt_60_seconds=int(gap_stats["gaps_gt_60_seconds"]),
    )
    return features, diagnostics


def build_features(frame: pd.DataFrame, levels: int = DEFAULT_LEVELS) -> pd.DataFrame:
    """Build all MVP market microstructure features from cleaned order book data."""
    features, _ = build_features_with_diagnostics(frame, levels=levels)
    return features


def read_cleaned_orderbook(input_path: Path) -> pd.DataFrame:
    """Load cleaned order book data."""
    with suppress_native_stderr():
        return pd.read_parquet(input_path)


def write_features(features: pd.DataFrame, output_path: Path) -> None:
    """Write feature data to Parquet."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_suffix(".tmp.parquet")
    with suppress_native_stderr():
        features.to_parquet(tmp_path, index=False)
    tmp_path.replace(output_path)


def build_feature_report(summary: FeatureBuildSummary) -> str:
    """Build a plain-text feature construction diagnostics report."""
    lines = [
        "LOB Feature Build Report",
        "========================",
        f"Input file path: {summary.input_path}",
        f"Output file path: {summary.output_path}",
        f"Input row count: {summary.total_input_rows}",
        f"Output row count: {summary.rows}",
        f"Rows dropped due to lag/rolling NaNs: {summary.rows_dropped_due_to_lag_rolling_nans}",
        f"Session-aware features used: {summary.session_aware_features}",
        f"Number of sessions: {summary.unique_session_ids}",
        f"Number of new lagged features: {summary.new_lagged_feature_count}",
        f"Number of new rolling mean features: {summary.new_rolling_mean_feature_count}",
        f"Number of new rolling std features: {summary.new_rolling_std_feature_count}",
        f"Number of new change features: {summary.new_change_feature_count}",
        f"Total final feature count: {summary.total_final_feature_count}",
        f"Largest timestamp gap seconds: {summary.largest_timestamp_gap_seconds:.6f}",
        f"Count of gaps > 1.5 seconds: {summary.gaps_gt_1_5_seconds}",
        f"Count of gaps > 5 seconds: {summary.gaps_gt_5_seconds}",
        f"Count of gaps > 60 seconds: {summary.gaps_gt_60_seconds}",
        f"Total output columns: {summary.columns}",
    ]
    return "\n".join(lines) + "\n"


def write_feature_report(summary: FeatureBuildSummary) -> None:
    """Write feature diagnostics report to disk."""
    summary.report_path.parent.mkdir(parents=True, exist_ok=True)
    summary.report_path.write_text(build_feature_report(summary), encoding="utf-8")


def build_features_file(
    input_path: Path,
    output_dir: Path,
    report_path: Path = DEFAULT_REPORT_PATH,
    levels: int = DEFAULT_LEVELS,
) -> FeatureBuildSummary:
    """Build features from one cleaned Parquet file."""
    frame = read_cleaned_orderbook(input_path)
    features, diagnostics = build_features_with_diagnostics(frame, levels=levels)
    output_path = feature_output_path(input_path, output_dir)
    write_features(features, output_path)
    summary = FeatureBuildSummary(
        input_path=input_path,
        output_path=output_path,
        report_path=report_path,
        total_input_rows=diagnostics.total_input_rows,
        rows=len(features),
        columns=len(features.columns),
        session_aware_features=diagnostics.session_aware_features,
        unique_session_ids=diagnostics.unique_session_ids,
        rows_dropped_due_to_lag_rolling_nans=diagnostics.rows_dropped_due_to_lag_rolling_nans,
        new_lagged_feature_count=diagnostics.new_lagged_feature_count,
        new_rolling_mean_feature_count=diagnostics.new_rolling_mean_feature_count,
        new_rolling_std_feature_count=diagnostics.new_rolling_std_feature_count,
        new_change_feature_count=diagnostics.new_change_feature_count,
        total_final_feature_count=diagnostics.total_final_feature_count,
        largest_timestamp_gap_seconds=diagnostics.largest_timestamp_gap_seconds,
        gaps_gt_1_5_seconds=diagnostics.gaps_gt_1_5_seconds,
        gaps_gt_5_seconds=diagnostics.gaps_gt_5_seconds,
        gaps_gt_60_seconds=diagnostics.gaps_gt_60_seconds,
    )
    write_feature_report(summary)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cleaned-dir", type=Path, default=DEFAULT_CLEANED_DIR)
    parser.add_argument("--input-path", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--report-path", type=Path, default=DEFAULT_REPORT_PATH)
    parser.add_argument("--levels", type=int, default=DEFAULT_LEVELS)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = args.input_path or find_newest_parquet(args.cleaned_dir)
    summary = build_features_file(
        input_path=input_path,
        output_dir=args.output_dir,
        report_path=args.report_path,
        levels=args.levels,
    )
    print(f"Saved feature file: {summary.output_path}", flush=True)
    print(f"Saved feature report: {summary.report_path}", flush=True)
    print(f"Rows: {summary.rows}", flush=True)
    print(f"Columns: {summary.columns}", flush=True)
    print(f"Session-aware features used: {summary.session_aware_features}", flush=True)
    print(f"Rows dropped due to lag/rolling NaNs: {summary.rows_dropped_due_to_lag_rolling_nans}", flush=True)
    print(f"New lagged features: {summary.new_lagged_feature_count}", flush=True)
    print(f"New rolling mean features: {summary.new_rolling_mean_feature_count}", flush=True)
    print(f"New rolling std features: {summary.new_rolling_std_feature_count}", flush=True)
    print(f"New change features: {summary.new_change_feature_count}", flush=True)


if __name__ == "__main__":
    main()
