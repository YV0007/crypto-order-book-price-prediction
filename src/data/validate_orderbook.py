"""Validate the newest raw top-20 order book Parquet file."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import logging
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RAW_DIR = PROJECT_ROOT / "data" / "raw" / "orderbook_top20"
DEFAULT_REPORT_PATH = PROJECT_ROOT / "reports" / "data_quality_report.txt"
DEFAULT_LOG_FILE = PROJECT_ROOT / "logs" / "data_validation.log"
DEFAULT_LEVELS = 20


@dataclass(frozen=True)
class ValidationResult:
    """Container for validation report values."""

    input_path: Path
    row_count: int
    column_count: int
    required_column_count: int
    missing_required_columns: list[str]
    timestamp_dtype: str
    timestamp_parseable: bool
    start_timestamp: str
    end_timestamp: str
    duplicate_timestamps: int
    missing_values: int
    invalid_spreads: int
    negative_sizes: int
    min_spread: float | None
    mean_spread: float | None
    max_spread: float | None
    passed: bool


def required_columns(levels: int = DEFAULT_LEVELS) -> list[str]:
    """Return the required raw order book schema."""
    columns = ["timestamp", "event_time", "exchange", "symbol"]
    for level in range(1, levels + 1):
        columns.extend(
            [
                f"bid_price_{level}",
                f"bid_size_{level}",
                f"ask_price_{level}",
                f"ask_size_{level}",
            ]
        )
    return columns


def size_columns(levels: int = DEFAULT_LEVELS) -> list[str]:
    """Return all bid and ask size columns."""
    columns: list[str] = []
    for level in range(1, levels + 1):
        columns.extend([f"bid_size_{level}", f"ask_size_{level}"])
    return columns


def find_newest_parquet(raw_dir: Path = DEFAULT_RAW_DIR) -> Path:
    """Find the newest raw Parquet file by modification time."""
    files = sorted(raw_dir.glob("*.parquet"), key=lambda path: path.stat().st_mtime)
    if not files:
        raise FileNotFoundError(f"No Parquet files found in {raw_dir}")
    return files[-1]


def configure_logging(log_file: Path = DEFAULT_LOG_FILE) -> logging.Logger:
    """Configure validation logging to logs/data_validation.log."""
    log_file.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")

    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger


def safe_float(value: Any) -> float | None:
    """Return a float for report formatting, or None when unavailable."""
    if pd.isna(value):
        return None
    return float(value)


@contextmanager
def suppress_native_stderr():
    """Temporarily suppress native library stderr noise during Parquet reads."""
    sys.stderr.flush()
    original_stderr_fd = os.dup(2)
    try:
        with open(os.devnull, "w", encoding="utf-8") as devnull:
            os.dup2(devnull.fileno(), 2)
            yield
    finally:
        os.dup2(original_stderr_fd, 2)
        os.close(original_stderr_fd)


def read_parquet_file(input_path: Path) -> pd.DataFrame:
    """Read a Parquet file while keeping stdout reserved for PASS / NO PASS."""
    with suppress_native_stderr():
        return pd.read_parquet(input_path)


def validate_frame(frame: pd.DataFrame, input_path: Path, levels: int = DEFAULT_LEVELS) -> ValidationResult:
    """Run validation checks on one raw order book DataFrame."""
    expected = required_columns(levels)
    missing_required = sorted(set(expected) - set(frame.columns))

    timestamp_parseable = "timestamp" in frame.columns
    parsed_timestamps = pd.Series(dtype="datetime64[ns, UTC]")
    timestamp_dtype = "missing"
    start_timestamp = "n/a"
    end_timestamp = "n/a"
    duplicate_timestamps = 0

    if "timestamp" in frame.columns:
        timestamp_dtype = str(frame["timestamp"].dtype)
        parsed_timestamps = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce")
        timestamp_parseable = not parsed_timestamps.isna().any()
        duplicate_timestamps = int(parsed_timestamps.duplicated().sum())
        valid_timestamps = parsed_timestamps.dropna()
        if not valid_timestamps.empty:
            start_timestamp = valid_timestamps.min().isoformat()
            end_timestamp = valid_timestamps.max().isoformat()

    invalid_spreads = 0
    min_spread: float | None = None
    mean_spread: float | None = None
    max_spread: float | None = None

    if {"bid_price_1", "ask_price_1"}.issubset(frame.columns):
        spread = frame["ask_price_1"] - frame["bid_price_1"]
        invalid_spreads = int((spread <= 0).sum())
        min_spread = safe_float(spread.min())
        mean_spread = safe_float(spread.mean())
        max_spread = safe_float(spread.max())

    present_size_columns = [column for column in size_columns(levels) if column in frame.columns]
    negative_sizes = 0
    if present_size_columns:
        negative_sizes = int((frame[present_size_columns] < 0).sum().sum())

    missing_values = int(frame.isna().sum().sum())

    passed = (
        len(frame) > 0
        and not missing_required
        and timestamp_parseable
        and duplicate_timestamps == 0
        and missing_values == 0
        and invalid_spreads == 0
        and negative_sizes == 0
    )

    return ValidationResult(
        input_path=input_path,
        row_count=len(frame),
        column_count=len(frame.columns),
        required_column_count=len(expected),
        missing_required_columns=missing_required,
        timestamp_dtype=timestamp_dtype,
        timestamp_parseable=timestamp_parseable,
        start_timestamp=start_timestamp,
        end_timestamp=end_timestamp,
        duplicate_timestamps=duplicate_timestamps,
        missing_values=missing_values,
        invalid_spreads=invalid_spreads,
        negative_sizes=negative_sizes,
        min_spread=min_spread,
        mean_spread=mean_spread,
        max_spread=max_spread,
        passed=passed,
    )


def format_float(value: float | None) -> str:
    """Format optional float values for a text report."""
    if value is None:
        return "n/a"
    return f"{value:.10f}"


def build_report(result: ValidationResult) -> str:
    """Build a human-readable data quality report."""
    status = "PASS" if result.passed else "NO PASS"
    missing_columns = ", ".join(result.missing_required_columns) if result.missing_required_columns else "none"

    lines = [
        "LOB Raw Order Book Data Quality Report",
        "======================================",
        f"Status: {status}",
        f"Input file: {result.input_path}",
        "",
        "Schema",
        "------",
        f"Required columns: {result.required_column_count}",
        f"Actual columns: {result.column_count}",
        f"Missing required columns: {missing_columns}",
        "",
        "Rows and Timestamps",
        "-------------------",
        f"Row count: {result.row_count}",
        f"Timestamp dtype: {result.timestamp_dtype}",
        f"Timestamp parseable: {result.timestamp_parseable}",
        f"Start timestamp: {result.start_timestamp}",
        f"End timestamp: {result.end_timestamp}",
        f"Duplicate timestamps: {result.duplicate_timestamps}",
        "",
        "Missing and Invalid Values",
        "--------------------------",
        f"Missing values: {result.missing_values}",
        f"Invalid spreads ask_price_1 <= bid_price_1: {result.invalid_spreads}",
        f"Negative sizes: {result.negative_sizes}",
        "",
        "Spread Summary",
        "--------------",
        f"Min spread: {format_float(result.min_spread)}",
        f"Mean spread: {format_float(result.mean_spread)}",
        f"Max spread: {format_float(result.max_spread)}",
        "",
    ]
    return "\n".join(lines)


def validate_parquet_file(input_path: Path, report_path: Path, levels: int, logger: logging.Logger) -> ValidationResult:
    """Load one Parquet file, validate it, and write the text report."""
    logger.info("Loading raw order book file: %s", input_path)
    frame = read_parquet_file(input_path)
    result = validate_frame(frame, input_path=input_path, levels=levels)

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(build_report(result), encoding="utf-8")

    logger.info("Validation status: %s", "PASS" if result.passed else "NO PASS")
    logger.info("Wrote data quality report: %s", report_path)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    parser.add_argument("--input-path", type=Path, default=None)
    parser.add_argument("--report-path", type=Path, default=DEFAULT_REPORT_PATH)
    parser.add_argument("--log-file", type=Path, default=DEFAULT_LOG_FILE)
    parser.add_argument("--levels", type=int, default=DEFAULT_LEVELS)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logger = configure_logging(args.log_file)
    input_path = args.input_path or find_newest_parquet(args.raw_dir)
    result = validate_parquet_file(input_path, args.report_path, args.levels, logger)

    print("PASS" if result.passed else "NO PASS", flush=True)


if __name__ == "__main__":
    main()
