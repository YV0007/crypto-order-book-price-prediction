"""Build timestamp-aware future mid-price direction targets."""

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
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "CONFIG" / "config.yaml"
DEFAULT_FEATURES_DIR = PROJECT_ROOT / "data" / "processed" / "features"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "processed" / "model_dataset"
DEFAULT_REPORTS_DIR = PROJECT_ROOT / "reports"
DEFAULT_SYMBOL = "BTCUSDT"
DEFAULT_HORIZON_SECONDS = 5
DEFAULT_TARGET_MODE = "absolute"
VALID_TARGET_MODES = {"absolute", "cost_aware"}
DEFAULT_THRESHOLD_ABS = 0.5
DEFAULT_THRESHOLD_CANDIDATES = [0.1, 0.2, 0.3, 0.5, 0.7, 1.0]
DEFAULT_COST_THRESHOLD_BPS = 4.0
DEFAULT_COST_BUFFER_BPS = 1.0
DEFAULT_COST_THRESHOLD_CANDIDATES = [1.0, 2.0, 4.0, 6.0, 8.0, 10.0]
DEFAULT_COST_BUFFER_CANDIDATES = [0.0, 1.0, 2.0]
DEFAULT_TIMESTAMP_TOLERANCE_SECONDS = 0.5
MAX_CONTINUITY_GAP_SECONDS = 1.5
PRICE_COL = "mid_price"


@dataclass(frozen=True)
class TargetConfig:
    """Target construction settings."""

    horizon_seconds: int = DEFAULT_HORIZON_SECONDS
    target_mode: str = DEFAULT_TARGET_MODE
    threshold_abs: float = DEFAULT_THRESHOLD_ABS
    threshold_candidates: list[float] | None = None
    cost_threshold_bps: float = DEFAULT_COST_THRESHOLD_BPS
    cost_buffer_bps: float = DEFAULT_COST_BUFFER_BPS
    timestamp_tolerance_seconds: float = DEFAULT_TIMESTAMP_TOLERANCE_SECONDS

    @property
    def candidates(self) -> list[float]:
        """Return configured threshold candidates."""
        return self.threshold_candidates or DEFAULT_THRESHOLD_CANDIDATES


@dataclass(frozen=True)
class TargetBuildSummary:
    """Summary of the target build run."""

    input_path: Path
    output_path: Path
    report_text_path: Path
    report_json_path: Path
    target_mode: str
    threshold_abs: float
    cost_threshold_bps: float
    cost_buffer_bps: float
    total_cost_aware_threshold_bps: float
    horizon_seconds: int
    timestamp_tolerance_seconds: float
    input_rows: int
    output_rows: int
    dropped_rows: int
    up_count: int
    neutral_count: int
    down_count: int
    class_percentages: dict[str, float]
    future_price_change_describe: dict[str, float]
    future_return_bps_describe: dict[str, float]


@dataclass(frozen=True)
class DiagnosticsArtifacts:
    """Saved threshold diagnostics report paths."""

    text_path: Path
    json_path: Path


@dataclass(frozen=True)
class CostAwareDiagnosticsArtifacts:
    """Saved cost-aware threshold diagnostics paths."""

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


def load_target_config(config_path: Path = DEFAULT_CONFIG_PATH) -> TargetConfig:
    """Load target settings from CONFIG/config.yaml, falling back to defaults."""
    values: dict[str, Any] = {}
    if config_path.exists():
        loaded = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
        if not isinstance(loaded, dict):
            raise ValueError(f"Config file must contain a mapping: {config_path}")
        target_values = loaded.get("target", {})
        if target_values is None:
            target_values = {}
        if not isinstance(target_values, dict):
            raise ValueError("CONFIG/config.yaml target section must be a mapping")
        values = target_values

    candidates = values.get("threshold_candidates", DEFAULT_THRESHOLD_CANDIDATES)
    if not isinstance(candidates, list):
        raise ValueError("target.threshold_candidates must be a list")

    config = TargetConfig(
        horizon_seconds=int(values.get("horizon_seconds", DEFAULT_HORIZON_SECONDS)),
        target_mode=str(values.get("target_mode", DEFAULT_TARGET_MODE)),
        threshold_abs=float(values.get("threshold_abs", DEFAULT_THRESHOLD_ABS)),
        threshold_candidates=[float(value) for value in candidates],
        cost_threshold_bps=float(values.get("cost_threshold_bps", DEFAULT_COST_THRESHOLD_BPS)),
        cost_buffer_bps=float(values.get("cost_buffer_bps", DEFAULT_COST_BUFFER_BPS)),
        timestamp_tolerance_seconds=float(
            values.get("timestamp_tolerance_seconds", DEFAULT_TIMESTAMP_TOLERANCE_SECONDS)
        ),
    )
    validate_target_config(config)
    return config


def validate_target_config(config: TargetConfig) -> None:
    """Validate target construction settings."""
    if config.horizon_seconds <= 0:
        raise ValueError("target.horizon_seconds must be positive")
    if config.target_mode not in VALID_TARGET_MODES:
        raise ValueError(f"target.target_mode must be one of {sorted(VALID_TARGET_MODES)}")
    if config.threshold_abs < 0:
        raise ValueError("target.threshold_abs must be nonnegative")
    if config.cost_threshold_bps < 0:
        raise ValueError("target.cost_threshold_bps must be nonnegative")
    if config.cost_buffer_bps < 0:
        raise ValueError("target.cost_buffer_bps must be nonnegative")
    if config.timestamp_tolerance_seconds < 0:
        raise ValueError("target.timestamp_tolerance_seconds must be nonnegative")
    if any(value < 0 for value in config.candidates):
        raise ValueError("target.threshold_candidates must be nonnegative")


def find_newest_parquet(features_dir: Path = DEFAULT_FEATURES_DIR) -> Path:
    """Find the newest feature Parquet file by modification time."""
    files = sorted(features_dir.glob("*.parquet"), key=lambda path: path.stat().st_mtime)
    if not files:
        raise FileNotFoundError(f"No Parquet files found in {features_dir}")
    return files[-1]


def feature_path_for_date(date_text: str, features_dir: Path, symbol: str = DEFAULT_SYMBOL) -> Path:
    """Return the feature path for a date/token such as 2026-04-30 or merged_last_2."""
    path = features_dir / f"{symbol}_features_{date_text}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Feature file not found for --date {date_text}: {path}")
    return path


def feature_path_for_dataset_name(dataset_name: str, features_dir: Path, symbol: str = DEFAULT_SYMBOL) -> Path:
    """Return the feature path for a dataset token such as merged_last_2_updated."""
    return feature_path_for_date(dataset_name, features_dir, symbol=symbol)


def date_token_from_feature_path(input_path: Path) -> str:
    """Extract the feature file date/token used for reports and output names."""
    stem = input_path.stem
    prefix = f"{DEFAULT_SYMBOL}_features_"
    if stem.startswith(prefix):
        return stem[len(prefix) :]
    if "_features_" in stem:
        return stem.split("_features_", maxsplit=1)[1]
    return stem


def resolve_input_path(args: argparse.Namespace) -> Path:
    """Resolve feature input path from --input-path, --dataset-name, --date, or newest file."""
    if args.input_path is not None:
        return args.input_path
    if args.dataset_name is not None:
        return feature_path_for_dataset_name(args.dataset_name, args.features_dir, symbol=args.symbol)
    if args.date is not None:
        return feature_path_for_date(args.date, args.features_dir, symbol=args.symbol)
    return find_newest_parquet(args.features_dir)


def format_number_token(value: float) -> str:
    """Format a numeric threshold as a filename-safe token."""
    text = f"{value:.6f}".rstrip("0").rstrip(".")
    return text.replace("-", "m").replace(".", "p")


def threshold_abs_token(threshold_abs: float) -> str:
    """Return filename token for an absolute price threshold."""
    return f"th{format_number_token(threshold_abs)}"


def cost_aware_threshold_token(total_threshold_bps: float) -> str:
    """Return filename token for a total cost-aware bps threshold."""
    return f"{format_number_token(total_threshold_bps)}bps"


def clean_output_suffix(output_suffix: str | None) -> str:
    """Return a safe optional output suffix component."""
    if output_suffix is None:
        return ""
    cleaned = output_suffix.strip().strip("_").replace(" ", "_")
    return cleaned


def target_settings_token(
    target_mode: str,
    horizon_seconds: int,
    threshold_abs: float,
    cost_threshold_bps: float,
    cost_buffer_bps: float,
    output_suffix: str | None = None,
) -> str:
    """Build the horizon/target settings suffix for model dataset outputs."""
    if target_mode == "absolute":
        settings = f"h{horizon_seconds}_{threshold_abs_token(threshold_abs)}"
    else:
        total_threshold_bps = cost_threshold_bps + cost_buffer_bps
        settings = f"h{horizon_seconds}_costaware_{cost_aware_threshold_token(total_threshold_bps)}"

    suffix = clean_output_suffix(output_suffix)
    return f"{settings}_{suffix}" if suffix else settings


def model_dataset_output_path(
    input_path: Path,
    output_dir: Path,
    target_mode: str = DEFAULT_TARGET_MODE,
    horizon_seconds: int = DEFAULT_HORIZON_SECONDS,
    threshold_abs: float = DEFAULT_THRESHOLD_ABS,
    cost_threshold_bps: float = DEFAULT_COST_THRESHOLD_BPS,
    cost_buffer_bps: float = DEFAULT_COST_BUFFER_BPS,
    output_suffix: str | None = None,
) -> Path:
    """Build the horizon-specific model dataset filename from a feature filename."""
    date_text = date_token_from_feature_path(input_path)
    settings = target_settings_token(
        target_mode=target_mode,
        horizon_seconds=horizon_seconds,
        threshold_abs=threshold_abs,
        cost_threshold_bps=cost_threshold_bps,
        cost_buffer_bps=cost_buffer_bps,
        output_suffix=output_suffix,
    )
    return output_dir / f"{DEFAULT_SYMBOL}_model_dataset_{date_text}_{settings}.parquet"


def diagnostics_output_paths(date_text: str, reports_dir: Path) -> DiagnosticsArtifacts:
    """Build diagnostics output paths for one date/token."""
    return DiagnosticsArtifacts(
        text_path=reports_dir / f"target_threshold_diagnostics_{date_text}.txt",
        json_path=reports_dir / f"target_threshold_diagnostics_{date_text}.json",
    )


def target_report_output_paths(date_text: str, settings_token: str, reports_dir: Path) -> DiagnosticsArtifacts:
    """Build target report output paths for one completed dataset."""
    stem = f"target_report_{date_text}_{settings_token}"
    return DiagnosticsArtifacts(
        text_path=reports_dir / f"{stem}.txt",
        json_path=reports_dir / f"{stem}.json",
    )


def cost_aware_diagnostics_output_paths(
    date_text: str,
    horizon_seconds: int,
    reports_dir: Path,
) -> CostAwareDiagnosticsArtifacts:
    """Build cost-aware diagnostics output paths."""
    stem = f"cost_aware_target_diagnostics_{date_text}_h{horizon_seconds}"
    return CostAwareDiagnosticsArtifacts(
        csv_path=reports_dir / f"{stem}.csv",
        json_path=reports_dir / f"{stem}.json",
    )


def read_features(input_path: Path) -> pd.DataFrame:
    """Load feature data."""
    with suppress_native_stderr():
        return pd.read_parquet(input_path)


def write_model_dataset(dataset: pd.DataFrame, output_path: Path) -> None:
    """Write the model dataset to Parquet."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_suffix(".tmp.parquet")
    with suppress_native_stderr():
        dataset.to_parquet(tmp_path, index=False)
    tmp_path.replace(output_path)


def prepare_features(frame: pd.DataFrame, price_col: str = PRICE_COL) -> pd.DataFrame:
    """Prepare feature rows for timestamp-aware target construction."""
    required = {"timestamp", price_col}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    output = frame.copy()
    output["timestamp"] = pd.to_datetime(output["timestamp"], utc=True, errors="coerce")
    output[price_col] = pd.to_numeric(output[price_col], errors="coerce")
    output = output.dropna(subset=["timestamp", price_col])

    if "session_id" in output.columns:
        output["session_id"] = pd.to_numeric(output["session_id"], errors="coerce")
        output = output.dropna(subset=["session_id"])
        output["session_id"] = output["session_id"].astype("int64")

    sort_columns = ["timestamp"]
    if "session_id" in output.columns:
        sort_columns = ["session_id", "timestamp"]
    output = output.sort_values(sort_columns, kind="mergesort")
    duplicate_columns = ["timestamp"]
    if "session_id" in output.columns:
        duplicate_columns = ["session_id", "timestamp"]
    output = output.drop_duplicates(subset=duplicate_columns, keep="last")
    output = output.sort_values("timestamp", kind="mergesort").reset_index(drop=True)
    output["_row_id"] = np.arange(len(output))
    return output


def continuity_group_keys(frame: pd.DataFrame, max_gap_seconds: float = MAX_CONTINUITY_GAP_SECONDS) -> pd.Series | None:
    """Build internal grouping keys that prevent future lookup across session gaps."""
    if "session_id" not in frame.columns:
        return None

    gaps = frame.groupby("session_id", sort=False, dropna=False)["timestamp"].diff().dt.total_seconds()
    continuity_breaks = gaps.gt(max_gap_seconds).fillna(False)
    continuity_segments = continuity_breaks.groupby(frame["session_id"], sort=False, dropna=False).cumsum().astype(int)
    session_values = frame["session_id"].astype("string").fillna("missing_session")
    return session_values + "_segment_" + continuity_segments.astype("string")


def _merge_future_for_group(
    group: pd.DataFrame,
    horizon_seconds: int,
    tolerance: pd.Timedelta,
    price_col: str,
) -> pd.DataFrame:
    """Attach future price for one timestamp-continuous group."""
    left = group[["_row_id", "timestamp"]].copy()
    left["_target_timestamp"] = left["timestamp"] + pd.Timedelta(seconds=horizon_seconds)
    left = left.sort_values("_target_timestamp", kind="mergesort")

    right = group[["timestamp", price_col]].rename(
        columns={"timestamp": "future_timestamp", price_col: "future_mid_price"}
    )
    right = right.sort_values("future_timestamp", kind="mergesort")

    merged = pd.merge_asof(
        left,
        right,
        left_on="_target_timestamp",
        right_on="future_timestamp",
        direction="nearest",
        tolerance=tolerance,
    )
    return merged[["_row_id", "future_timestamp", "future_mid_price"]]


def attach_future_prices(
    frame: pd.DataFrame,
    horizon_seconds: int = DEFAULT_HORIZON_SECONDS,
    timestamp_tolerance_seconds: float = DEFAULT_TIMESTAMP_TOLERANCE_SECONDS,
    price_col: str = PRICE_COL,
) -> pd.DataFrame:
    """Attach future mid-price using timestamp-aware lookup and session boundaries."""
    if horizon_seconds <= 0:
        raise ValueError("horizon_seconds must be positive")
    if timestamp_tolerance_seconds < 0:
        raise ValueError("timestamp_tolerance_seconds must be nonnegative")

    prepared = prepare_features(frame, price_col=price_col)
    if prepared.empty:
        prepared["future_timestamp"] = pd.NaT
        prepared["future_mid_price"] = np.nan
        prepared["future_price_change"] = np.nan
        prepared["future_return"] = np.nan
        prepared["future_return_bps"] = np.nan
        return prepared.drop(columns=["_row_id"], errors="ignore")

    tolerance = pd.Timedelta(seconds=timestamp_tolerance_seconds)

    group_keys = continuity_group_keys(prepared)
    if group_keys is not None:
        future_chunks = [
            _merge_future_for_group(group, horizon_seconds, tolerance, price_col)
            for _, group in prepared.groupby(group_keys, sort=False)
        ]
        future = (
            pd.concat(future_chunks, ignore_index=True)
            if future_chunks
            else pd.DataFrame(columns=["_row_id", "future_timestamp", "future_mid_price"])
        )
    else:
        future = _merge_future_for_group(prepared, horizon_seconds, tolerance, price_col)

    attached = prepared.merge(future, on="_row_id", how="left")
    attached = attached.sort_values("_row_id", kind="mergesort").reset_index(drop=True)
    attached["future_price_change"] = attached["future_mid_price"] - attached[price_col]
    attached["future_return"] = attached["future_price_change"] / attached[price_col]
    attached["future_return_bps"] = attached["future_return"] * 10_000
    attached = attached.drop(columns=["_row_id"])
    return attached


def add_absolute_target_from_future(frame: pd.DataFrame, threshold_abs: float) -> pd.DataFrame:
    """Add -1/0/1 absolute price-change target from already attached future columns."""
    if threshold_abs < 0:
        raise ValueError("threshold_abs must be nonnegative")
    required = {"future_mid_price", "future_price_change", "future_return", "future_return_bps"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Missing required future columns: {missing}")

    output = frame.dropna(subset=["future_mid_price", "future_price_change", "future_return", "future_return_bps"]).copy()
    output["target"] = np.select(
        [
            output["future_price_change"] > threshold_abs,
            output["future_price_change"] < -threshold_abs,
        ],
        [1, -1],
        default=0,
    ).astype("int8")
    return output.reset_index(drop=True)


def add_cost_aware_target_from_future(
    frame: pd.DataFrame,
    cost_threshold_bps: float,
    cost_buffer_bps: float,
) -> pd.DataFrame:
    """Add -1/0/1 target that only marks moves exceeding estimated round-trip costs."""
    if cost_threshold_bps < 0:
        raise ValueError("cost_threshold_bps must be nonnegative")
    if cost_buffer_bps < 0:
        raise ValueError("cost_buffer_bps must be nonnegative")
    required = {"future_mid_price", "future_price_change", "future_return", "future_return_bps"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Missing required future columns: {missing}")

    total_threshold_bps = cost_threshold_bps + cost_buffer_bps
    output = frame.dropna(subset=["future_mid_price", "future_price_change", "future_return", "future_return_bps"]).copy()
    output["target"] = np.select(
        [
            output["future_return_bps"] > total_threshold_bps,
            output["future_return_bps"] < -total_threshold_bps,
        ],
        [1, -1],
        default=0,
    ).astype("int8")
    return output.reset_index(drop=True)


def add_target_metadata(
    dataset: pd.DataFrame,
    target_mode: str,
    horizon_seconds: int,
    threshold_abs: float,
    cost_threshold_bps: float,
    cost_buffer_bps: float,
) -> pd.DataFrame:
    """Attach target construction metadata to every labeled row."""
    output = dataset.copy()
    output["horizon_seconds"] = int(horizon_seconds)
    output["target_mode"] = target_mode
    if target_mode == "absolute":
        output["threshold_abs"] = float(threshold_abs)
    else:
        output["cost_threshold_bps"] = float(cost_threshold_bps)
        output["cost_buffer_bps"] = float(cost_buffer_bps)
        output["total_cost_aware_threshold_bps"] = float(cost_threshold_bps + cost_buffer_bps)
    return output


def add_target_from_future(
    frame: pd.DataFrame,
    target_mode: str = DEFAULT_TARGET_MODE,
    threshold_abs: float = DEFAULT_THRESHOLD_ABS,
    cost_threshold_bps: float = DEFAULT_COST_THRESHOLD_BPS,
    cost_buffer_bps: float = DEFAULT_COST_BUFFER_BPS,
    horizon_seconds: int = DEFAULT_HORIZON_SECONDS,
) -> pd.DataFrame:
    """Add -1/0/1 target from already attached future price columns."""
    if target_mode not in VALID_TARGET_MODES:
        raise ValueError(f"target_mode must be one of {sorted(VALID_TARGET_MODES)}")
    if target_mode == "absolute":
        output = add_absolute_target_from_future(frame, threshold_abs=threshold_abs)
    else:
        output = add_cost_aware_target_from_future(
            frame,
            cost_threshold_bps=cost_threshold_bps,
            cost_buffer_bps=cost_buffer_bps,
        )
    return add_target_metadata(
        output,
        target_mode=target_mode,
        horizon_seconds=horizon_seconds,
        threshold_abs=threshold_abs,
        cost_threshold_bps=cost_threshold_bps,
        cost_buffer_bps=cost_buffer_bps,
    )


def build_model_dataset(
    frame: pd.DataFrame,
    threshold_abs: float,
    horizon_seconds: int = DEFAULT_HORIZON_SECONDS,
    timestamp_tolerance_seconds: float = DEFAULT_TIMESTAMP_TOLERANCE_SECONDS,
    target_mode: str = DEFAULT_TARGET_MODE,
    cost_threshold_bps: float = DEFAULT_COST_THRESHOLD_BPS,
    cost_buffer_bps: float = DEFAULT_COST_BUFFER_BPS,
) -> pd.DataFrame:
    """Build final model dataset with timestamp-aware target columns."""
    with_future = attach_future_prices(
        frame,
        horizon_seconds=horizon_seconds,
        timestamp_tolerance_seconds=timestamp_tolerance_seconds,
    )
    return add_target_from_future(
        with_future,
        target_mode=target_mode,
        threshold_abs=threshold_abs,
        cost_threshold_bps=cost_threshold_bps,
        cost_buffer_bps=cost_buffer_bps,
        horizon_seconds=horizon_seconds,
    )


def class_counts_and_percentages(target: pd.Series) -> tuple[dict[str, int], dict[str, float]]:
    """Return JSON-friendly class counts and percentages for -1, 0, 1."""
    total = len(target)
    counts = {str(label): int((target == label).sum()) for label in [-1, 0, 1]}
    percentages = {
        str(label): float(counts[str(label)] / total * 100) if total else 0.0
        for label in [-1, 0, 1]
    }
    return counts, percentages


def describe_series(series: pd.Series) -> dict[str, float]:
    """Return JSON-friendly describe statistics."""
    description = series.describe()
    return {str(key): float(value) for key, value in description.items()}


def target_distribution(dataset: pd.DataFrame) -> tuple[dict[str, int], dict[str, float]]:
    """Return target class counts and percentages."""
    return class_counts_and_percentages(dataset["target"])


def threshold_diagnostics(
    frame: pd.DataFrame,
    thresholds: list[float],
    horizon_seconds: int,
    timestamp_tolerance_seconds: float,
) -> dict[str, Any]:
    """Build target threshold diagnostics for multiple absolute thresholds."""
    with_future = attach_future_prices(
        frame,
        horizon_seconds=horizon_seconds,
        timestamp_tolerance_seconds=timestamp_tolerance_seconds,
    )
    usable = with_future.dropna(
        subset=["future_mid_price", "future_price_change", "future_return", "future_return_bps"]
    ).copy()
    dropped_rows = len(with_future) - len(usable)

    diagnostics: dict[str, Any] = {
        "input_rows": int(len(frame)),
        "prepared_rows": int(len(with_future)),
        "total_usable_rows": int(len(usable)),
        "dropped_rows_due_to_missing_valid_future_timestamp": int(dropped_rows),
        "horizon_seconds": int(horizon_seconds),
        "timestamp_tolerance_seconds": float(timestamp_tolerance_seconds),
        "session_aware": bool("session_id" in with_future.columns),
        "future_price_change_describe": describe_series(usable["future_price_change"]) if not usable.empty else {},
        "future_return_describe": describe_series(usable["future_return"]) if not usable.empty else {},
        "future_return_bps_describe": describe_series(usable["future_return_bps"]) if not usable.empty else {},
        "thresholds": [],
    }

    for threshold in thresholds:
        target = np.select(
            [
                usable["future_price_change"] > threshold,
                usable["future_price_change"] < -threshold,
            ],
            [1, -1],
            default=0,
        )
        counts, percentages = class_counts_and_percentages(pd.Series(target))
        diagnostics["thresholds"].append(
            {
                "threshold_abs": float(threshold),
                "total_usable_rows": int(len(usable)),
                "dropped_rows_due_to_missing_valid_future_timestamp": int(dropped_rows),
                "class_counts": counts,
                "class_percentages": percentages,
                "future_price_change_describe": diagnostics["future_price_change_describe"],
                "future_return_describe": diagnostics["future_return_describe"],
                "future_return_bps_describe": diagnostics["future_return_bps_describe"],
            }
        )

    return diagnostics


def cost_aware_threshold_diagnostics(
    frame: pd.DataFrame,
    cost_threshold_candidates: list[float],
    cost_buffer_candidates: list[float],
    horizon_seconds: int,
    timestamp_tolerance_seconds: float,
) -> list[dict[str, Any]]:
    """Build cost-aware class distribution diagnostics over cost and buffer thresholds."""
    with_future = attach_future_prices(
        frame,
        horizon_seconds=horizon_seconds,
        timestamp_tolerance_seconds=timestamp_tolerance_seconds,
    )
    usable = with_future.dropna(
        subset=["future_mid_price", "future_price_change", "future_return", "future_return_bps"]
    ).copy()
    dropped_rows = len(with_future) - len(usable)
    future_return_bps_stats = describe_series(usable["future_return_bps"]) if not usable.empty else {}

    rows: list[dict[str, Any]] = []
    for cost_threshold_bps in cost_threshold_candidates:
        for cost_buffer_bps in cost_buffer_candidates:
            total_threshold_bps = cost_threshold_bps + cost_buffer_bps
            target = np.select(
                [
                    usable["future_return_bps"] > total_threshold_bps,
                    usable["future_return_bps"] < -total_threshold_bps,
                ],
                [1, -1],
                default=0,
            )
            counts, percentages = class_counts_and_percentages(pd.Series(target))
            row: dict[str, Any] = {
                "horizon_seconds": int(horizon_seconds),
                "cost_threshold_bps": float(cost_threshold_bps),
                "cost_buffer_bps": float(cost_buffer_bps),
                "total_required_threshold_bps": float(total_threshold_bps),
                "input_rows": int(len(frame)),
                "usable_rows": int(len(usable)),
                "dropped_rows_due_to_missing_valid_future_timestamp": int(dropped_rows),
                "down_count": counts["-1"],
                "neutral_count": counts["0"],
                "up_count": counts["1"],
                "down_percentage": percentages["-1"],
                "neutral_percentage": percentages["0"],
                "up_percentage": percentages["1"],
            }
            for key, value in future_return_bps_stats.items():
                row[f"future_return_bps_{key}"] = value
            rows.append(row)
    return rows


def build_diagnostics_text(input_path: Path, diagnostics: dict[str, Any]) -> str:
    """Build a human-readable target threshold diagnostics report."""
    lines = [
        "LOB Target Threshold Diagnostics",
        "================================",
        f"Input feature file: {input_path}",
        f"Horizon seconds: {diagnostics['horizon_seconds']}",
        f"Timestamp tolerance seconds: {diagnostics['timestamp_tolerance_seconds']}",
        f"Session-aware lookup: {diagnostics['session_aware']}",
        f"Input rows: {diagnostics['input_rows']}",
        f"Prepared rows: {diagnostics['prepared_rows']}",
        f"Total usable rows: {diagnostics['total_usable_rows']}",
        "Dropped rows due to missing valid future timestamp: "
        f"{diagnostics['dropped_rows_due_to_missing_valid_future_timestamp']}",
        "",
        "Future Price Change Describe",
        "----------------------------",
    ]
    for key, value in diagnostics["future_price_change_describe"].items():
        lines.append(f"{key}: {value:.10f}")

    lines.extend(["", "Future Return Describe", "----------------------"])
    for key, value in diagnostics["future_return_describe"].items():
        lines.append(f"{key}: {value:.10f}")

    lines.extend(["", "Threshold Results", "-----------------"])
    for item in diagnostics["thresholds"]:
        lines.extend(
            [
                "",
                f"threshold_abs: {item['threshold_abs']}",
                f"total_usable_rows: {item['total_usable_rows']}",
                "dropped_rows_due_to_missing_valid_future_timestamp: "
                f"{item['dropped_rows_due_to_missing_valid_future_timestamp']}",
                "class_counts: "
                f"down={item['class_counts']['-1']}, "
                f"neutral={item['class_counts']['0']}, "
                f"up={item['class_counts']['1']}",
                "class_percentages: "
                f"down={item['class_percentages']['-1']:.4f}%, "
                f"neutral={item['class_percentages']['0']:.4f}%, "
                f"up={item['class_percentages']['1']:.4f}%",
            ]
        )

    return "\n".join(lines) + "\n"


def save_threshold_diagnostics(
    input_path: Path,
    date_text: str,
    diagnostics: dict[str, Any],
    reports_dir: Path = DEFAULT_REPORTS_DIR,
) -> DiagnosticsArtifacts:
    """Save threshold diagnostics as text and JSON."""
    artifacts = diagnostics_output_paths(date_text, reports_dir)
    artifacts.text_path.parent.mkdir(parents=True, exist_ok=True)
    artifacts.text_path.write_text(build_diagnostics_text(input_path, diagnostics), encoding="utf-8")
    artifacts.json_path.write_text(
        json.dumps({"input_path": str(input_path), **diagnostics}, indent=2),
        encoding="utf-8",
    )
    return artifacts


def save_cost_aware_diagnostics(
    date_text: str,
    horizon_seconds: int,
    diagnostics: list[dict[str, Any]],
    reports_dir: Path = DEFAULT_REPORTS_DIR,
) -> CostAwareDiagnosticsArtifacts:
    """Save cost-aware target diagnostics as CSV and JSON."""
    artifacts = cost_aware_diagnostics_output_paths(date_text, horizon_seconds, reports_dir)
    artifacts.csv_path.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(diagnostics)
    frame.to_csv(artifacts.csv_path, index=False)
    artifacts.json_path.write_text(json.dumps(diagnostics, indent=2), encoding="utf-8")
    return artifacts


def build_target_report_text(summary: TargetBuildSummary) -> str:
    """Build a human-readable report for one target dataset."""
    lines = [
        "LOB Target Build Report",
        "=======================",
        f"Input feature file: {summary.input_path}",
        f"Output model dataset file: {summary.output_path}",
        f"Target mode: {summary.target_mode}",
        f"Horizon seconds: {summary.horizon_seconds}",
        f"Timestamp tolerance seconds: {summary.timestamp_tolerance_seconds}",
    ]
    if summary.target_mode == "absolute":
        lines.append(f"Threshold abs: {summary.threshold_abs}")
    else:
        lines.extend(
            [
                f"Cost threshold bps: {summary.cost_threshold_bps}",
                f"Cost buffer bps: {summary.cost_buffer_bps}",
                f"Total cost-aware threshold bps: {summary.total_cost_aware_threshold_bps}",
            ]
        )
    lines.extend(
        [
            f"Total input rows: {summary.input_rows}",
            f"Usable rows: {summary.output_rows}",
            f"Dropped rows due to missing future timestamp: {summary.dropped_rows}",
            "",
            "Class Counts",
            "------------",
            f"-1: {summary.down_count}",
            f"0: {summary.neutral_count}",
            f"1: {summary.up_count}",
            "",
            "Class Percentages",
            "-----------------",
            f"-1: {summary.class_percentages['-1']:.6f}",
            f"0: {summary.class_percentages['0']:.6f}",
            f"1: {summary.class_percentages['1']:.6f}",
            "",
            "Future Price Change Describe",
            "----------------------------",
        ]
    )
    for key, value in summary.future_price_change_describe.items():
        lines.append(f"{key}: {value:.10f}")
    lines.extend(["", "Future Return Bps Describe", "--------------------------"])
    for key, value in summary.future_return_bps_describe.items():
        lines.append(f"{key}: {value:.10f}")
    return "\n".join(lines) + "\n"


def save_target_report(summary: TargetBuildSummary) -> None:
    """Save target report as text and JSON."""
    summary.report_text_path.parent.mkdir(parents=True, exist_ok=True)
    summary.report_text_path.write_text(build_target_report_text(summary), encoding="utf-8")
    payload = {
        "input_feature_file": str(summary.input_path),
        "output_model_dataset_file": str(summary.output_path),
        "target_mode": summary.target_mode,
        "horizon_seconds": summary.horizon_seconds,
        "timestamp_tolerance_seconds": summary.timestamp_tolerance_seconds,
        "threshold_abs": summary.threshold_abs if summary.target_mode == "absolute" else None,
        "cost_threshold_bps": summary.cost_threshold_bps if summary.target_mode == "cost_aware" else None,
        "cost_buffer_bps": summary.cost_buffer_bps if summary.target_mode == "cost_aware" else None,
        "total_cost_aware_threshold_bps": (
            summary.total_cost_aware_threshold_bps if summary.target_mode == "cost_aware" else None
        ),
        "total_input_rows": summary.input_rows,
        "usable_rows": summary.output_rows,
        "dropped_rows_due_to_missing_future_timestamp": summary.dropped_rows,
        "class_counts": {"-1": summary.down_count, "0": summary.neutral_count, "1": summary.up_count},
        "class_percentages": summary.class_percentages,
        "future_price_change_describe": summary.future_price_change_describe,
        "future_return_bps_describe": summary.future_return_bps_describe,
    }
    summary.report_json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def build_model_dataset_file(
    input_path: Path,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    reports_dir: Path = DEFAULT_REPORTS_DIR,
    target_mode: str = DEFAULT_TARGET_MODE,
    threshold_abs: float = DEFAULT_THRESHOLD_ABS,
    cost_threshold_bps: float = DEFAULT_COST_THRESHOLD_BPS,
    cost_buffer_bps: float = DEFAULT_COST_BUFFER_BPS,
    horizon_seconds: int = DEFAULT_HORIZON_SECONDS,
    timestamp_tolerance_seconds: float = DEFAULT_TIMESTAMP_TOLERANCE_SECONDS,
    output_suffix: str | None = None,
) -> TargetBuildSummary:
    """Build and save one model dataset from one feature file."""
    features = read_features(input_path)
    dataset = build_model_dataset(
        features,
        threshold_abs=threshold_abs,
        horizon_seconds=horizon_seconds,
        timestamp_tolerance_seconds=timestamp_tolerance_seconds,
        target_mode=target_mode,
        cost_threshold_bps=cost_threshold_bps,
        cost_buffer_bps=cost_buffer_bps,
    )
    output_path = model_dataset_output_path(
        input_path,
        output_dir,
        target_mode=target_mode,
        horizon_seconds=horizon_seconds,
        threshold_abs=threshold_abs,
        cost_threshold_bps=cost_threshold_bps,
        cost_buffer_bps=cost_buffer_bps,
        output_suffix=output_suffix,
    )
    write_model_dataset(dataset, output_path)

    counts, percentages = target_distribution(dataset)
    date_text = date_token_from_feature_path(input_path)
    settings = target_settings_token(
        target_mode=target_mode,
        horizon_seconds=horizon_seconds,
        threshold_abs=threshold_abs,
        cost_threshold_bps=cost_threshold_bps,
        cost_buffer_bps=cost_buffer_bps,
        output_suffix=output_suffix,
    )
    report_paths = target_report_output_paths(date_text, settings, reports_dir)
    return TargetBuildSummary(
        input_path=input_path,
        output_path=output_path,
        report_text_path=report_paths.text_path,
        report_json_path=report_paths.json_path,
        target_mode=target_mode,
        threshold_abs=threshold_abs,
        cost_threshold_bps=cost_threshold_bps,
        cost_buffer_bps=cost_buffer_bps,
        total_cost_aware_threshold_bps=cost_threshold_bps + cost_buffer_bps,
        horizon_seconds=horizon_seconds,
        timestamp_tolerance_seconds=timestamp_tolerance_seconds,
        input_rows=len(features),
        output_rows=len(dataset),
        dropped_rows=len(features) - len(dataset),
        up_count=counts["1"],
        neutral_count=counts["0"],
        down_count=counts["-1"],
        class_percentages=percentages,
        future_price_change_describe=describe_series(dataset["future_price_change"]) if not dataset.empty else {},
        future_return_bps_describe=describe_series(dataset["future_return_bps"]) if not dataset.empty else {},
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features-dir", type=Path, default=DEFAULT_FEATURES_DIR)
    parser.add_argument("--input-path", type=Path, default=None)
    parser.add_argument("--dataset-name", type=str, default=None)
    parser.add_argument("--date", type=str, default=None)
    parser.add_argument("--symbol", type=str, default=DEFAULT_SYMBOL)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--output-suffix", type=str, default=None)
    parser.add_argument("--reports-dir", type=Path, default=DEFAULT_REPORTS_DIR)
    parser.add_argument("--config-path", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--diagnostics", action="store_true")
    parser.add_argument("--cost-aware-diagnostics", action="store_true")
    parser.add_argument("--target-mode", choices=sorted(VALID_TARGET_MODES), default=None)
    parser.add_argument("--threshold-abs", type=float, default=None)
    parser.add_argument("--threshold-candidates", type=float, nargs="*", default=None)
    parser.add_argument("--cost-threshold-bps", type=float, default=None)
    parser.add_argument("--cost-buffer-bps", type=float, default=None)
    parser.add_argument("--cost-threshold-candidates", type=float, nargs="*", default=None)
    parser.add_argument("--cost-buffer-candidates", type=float, nargs="*", default=None)
    parser.add_argument("--horizon-seconds", type=int, default=None)
    parser.add_argument("--timestamp-tolerance-seconds", type=float, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_target_config(args.config_path)
    input_path = resolve_input_path(args)
    date_text = args.dataset_name or args.date or date_token_from_feature_path(input_path)
    target_mode = args.target_mode if args.target_mode is not None else config.target_mode
    horizon_seconds = args.horizon_seconds if args.horizon_seconds is not None else config.horizon_seconds
    tolerance_seconds = (
        args.timestamp_tolerance_seconds
        if args.timestamp_tolerance_seconds is not None
        else config.timestamp_tolerance_seconds
    )
    threshold_abs = args.threshold_abs if args.threshold_abs is not None else config.threshold_abs
    cost_threshold_bps = (
        args.cost_threshold_bps if args.cost_threshold_bps is not None else config.cost_threshold_bps
    )
    cost_buffer_bps = args.cost_buffer_bps if args.cost_buffer_bps is not None else config.cost_buffer_bps

    validate_target_config(
        TargetConfig(
            horizon_seconds=horizon_seconds,
            target_mode=target_mode,
            threshold_abs=threshold_abs,
            threshold_candidates=config.candidates,
            cost_threshold_bps=cost_threshold_bps,
            cost_buffer_bps=cost_buffer_bps,
            timestamp_tolerance_seconds=tolerance_seconds,
        )
    )

    if args.diagnostics:
        thresholds = args.threshold_candidates if args.threshold_candidates is not None else config.candidates
        features = read_features(input_path)
        diagnostics = threshold_diagnostics(
            features,
            thresholds=thresholds,
            horizon_seconds=horizon_seconds,
            timestamp_tolerance_seconds=tolerance_seconds,
        )
        artifacts = save_threshold_diagnostics(
            input_path=input_path,
            date_text=date_text,
            diagnostics=diagnostics,
            reports_dir=args.reports_dir,
        )
        print(f"Saved diagnostics report: {artifacts.text_path}", flush=True)
        print(f"Saved diagnostics JSON: {artifacts.json_path}", flush=True)
        for item in diagnostics["thresholds"]:
            print(
                "threshold_abs="
                f"{item['threshold_abs']}: "
                f"down={item['class_counts']['-1']} "
                f"neutral={item['class_counts']['0']} "
                f"up={item['class_counts']['1']}",
                flush=True,
        )
        return

    if args.cost_aware_diagnostics:
        cost_threshold_candidates = (
            args.cost_threshold_candidates
            if args.cost_threshold_candidates is not None
            else DEFAULT_COST_THRESHOLD_CANDIDATES
        )
        cost_buffer_candidates = (
            args.cost_buffer_candidates
            if args.cost_buffer_candidates is not None
            else DEFAULT_COST_BUFFER_CANDIDATES
        )
        if any(value < 0 for value in cost_threshold_candidates):
            raise ValueError("--cost-threshold-candidates must be nonnegative")
        if any(value < 0 for value in cost_buffer_candidates):
            raise ValueError("--cost-buffer-candidates must be nonnegative")
        features = read_features(input_path)
        diagnostics = cost_aware_threshold_diagnostics(
            features,
            cost_threshold_candidates=[float(value) for value in cost_threshold_candidates],
            cost_buffer_candidates=[float(value) for value in cost_buffer_candidates],
            horizon_seconds=horizon_seconds,
            timestamp_tolerance_seconds=tolerance_seconds,
        )
        artifacts = save_cost_aware_diagnostics(
            date_text=date_text,
            horizon_seconds=horizon_seconds,
            diagnostics=diagnostics,
            reports_dir=args.reports_dir,
        )
        print(f"Saved cost-aware diagnostics CSV: {artifacts.csv_path}", flush=True)
        print(f"Saved cost-aware diagnostics JSON: {artifacts.json_path}", flush=True)
        for item in diagnostics:
            print(
                f"cost_threshold_bps={item['cost_threshold_bps']} "
                f"cost_buffer_bps={item['cost_buffer_bps']} "
                f"total_required_threshold_bps={item['total_required_threshold_bps']}: "
                f"down={item['down_count']} neutral={item['neutral_count']} up={item['up_count']}",
                flush=True,
            )
        return

    summary = build_model_dataset_file(
        input_path=input_path,
        output_dir=args.output_dir,
        reports_dir=args.reports_dir,
        target_mode=target_mode,
        threshold_abs=threshold_abs,
        cost_threshold_bps=cost_threshold_bps,
        cost_buffer_bps=cost_buffer_bps,
        horizon_seconds=horizon_seconds,
        timestamp_tolerance_seconds=tolerance_seconds,
        output_suffix=args.output_suffix,
    )
    save_target_report(summary)

    print(f"Saved model dataset: {summary.output_path}", flush=True)
    print(f"Saved target report: {summary.report_text_path}", flush=True)
    print(f"Saved target report JSON: {summary.report_json_path}", flush=True)
    print(f"Target mode: {summary.target_mode}", flush=True)
    print(f"Horizon seconds: {summary.horizon_seconds}", flush=True)
    if summary.target_mode == "absolute":
        print(f"Selected threshold_abs: {summary.threshold_abs}", flush=True)
    else:
        print(f"Cost threshold bps: {summary.cost_threshold_bps}", flush=True)
        print(f"Cost buffer bps: {summary.cost_buffer_bps}", flush=True)
        print(f"Total cost-aware threshold bps: {summary.total_cost_aware_threshold_bps}", flush=True)
    print(f"Timestamp tolerance seconds: {summary.timestamp_tolerance_seconds}", flush=True)
    print(f"Rows: {summary.input_rows} features -> {summary.output_rows} labeled", flush=True)
    print(f"Dropped rows without valid future price: {summary.dropped_rows}", flush=True)
    print(
        f"Target counts: down={summary.down_count}, neutral={summary.neutral_count}, up={summary.up_count}",
        flush=True,
    )


if __name__ == "__main__":
    main()
