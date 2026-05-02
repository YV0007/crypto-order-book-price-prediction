"""Merge the latest raw daily top-20 order book Parquet files."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
import logging
import os
from pathlib import Path
import re
import sys

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RAW_DIR = PROJECT_ROOT / "data" / "raw" / "orderbook_top20"
DEFAULT_LOG_FILE = PROJECT_ROOT / "logs" / "merge_raw_orderbook.log"
DEFAULT_SYMBOL = "BTCUSDT"
DAILY_RAW_FILE_PATTERN = re.compile(r"^(?P<symbol>[A-Z0-9]+)_(?P<date>\d{4}-\d{2}-\d{2})\.parquet$")


@dataclass(frozen=True)
class DailyRawFile:
    """Daily raw Parquet file metadata."""

    path: Path
    symbol: str
    date_text: str


@dataclass(frozen=True)
class MergeSummary:
    """Summary of a raw merge run."""

    output_path: Path
    input_paths: list[Path]
    input_rows: int
    output_rows: int
    duplicate_timestamps_removed: int


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


def configure_logging(log_file: Path = DEFAULT_LOG_FILE) -> logging.Logger:
    """Configure merge logging."""
    log_file.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger


def parse_daily_raw_file(path: Path) -> DailyRawFile | None:
    """Return metadata for a daily raw Parquet file, or None for non-daily files."""
    match = DAILY_RAW_FILE_PATTERN.match(path.name)
    if match is None:
        return None
    return DailyRawFile(path=path, symbol=match.group("symbol"), date_text=match.group("date"))


def discover_daily_raw_files(raw_dir: Path, symbol: str = DEFAULT_SYMBOL) -> list[DailyRawFile]:
    """Find daily raw files, excluding merged output files."""
    files: list[DailyRawFile] = []
    for path in raw_dir.glob("*.parquet"):
        daily_file = parse_daily_raw_file(path)
        if daily_file is not None and daily_file.symbol == symbol:
            files.append(daily_file)
    return files


def select_latest_files(files: list[DailyRawFile], last_n: int, sort_by: str) -> list[DailyRawFile]:
    """Select the latest N daily files."""
    if last_n <= 0:
        raise ValueError("--last-n must be greater than zero")
    if len(files) < last_n:
        raise FileNotFoundError(f"Requested {last_n} files, but found only {len(files)} matching daily raw files")

    if sort_by == "date":
        ordered = sorted(files, key=lambda file: file.date_text)
    elif sort_by == "mtime":
        ordered = sorted(files, key=lambda file: file.path.stat().st_mtime)
    else:
        raise ValueError("--sort-by must be either date or mtime")

    selected = ordered[-last_n:]
    return sorted(selected, key=lambda file: file.date_text)


def default_output_path(selected_files: list[DailyRawFile], raw_dir: Path) -> Path:
    """Build a timestamped output path for the merged raw file."""
    symbol = selected_files[-1].symbol
    first_date = selected_files[0].date_text
    last_date = selected_files[-1].date_text
    run_timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    filename = f"{symbol}_merged_last_{len(selected_files)}_{first_date}_to_{last_date}_{run_timestamp}.parquet"
    return raw_dir / filename


def read_parquet(path: Path) -> pd.DataFrame:
    """Read one Parquet file quietly."""
    with suppress_native_stderr():
        return pd.read_parquet(path)


def write_parquet(frame: pd.DataFrame, output_path: Path) -> None:
    """Write a Parquet file atomically."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_suffix(".tmp.parquet")
    with suppress_native_stderr():
        frame.to_parquet(tmp_path, index=False)
    tmp_path.replace(output_path)


def preferred_column_order(columns: list[str]) -> list[str]:
    """Keep raw order book columns in a stable, familiar order."""
    prefix = ["timestamp", "event_time", "exchange", "symbol"]
    if "session_id" in columns:
        prefix.append("session_id")
    ordered_prefix = [column for column in prefix if column in columns]
    remaining = [column for column in columns if column not in ordered_prefix]
    return [*ordered_prefix, *remaining]


def merge_raw_files(input_paths: list[Path]) -> tuple[pd.DataFrame, int, int]:
    """Merge raw files, sort by timestamp, and deduplicate timestamps."""
    frames: list[pd.DataFrame] = []
    input_rows = 0

    for file_order, path in enumerate(input_paths):
        frame = read_parquet(path)
        if "timestamp" not in frame.columns:
            raise ValueError(f"Missing timestamp column in {path}")

        frame = frame.copy()
        frame["_merge_file_order"] = file_order
        frame["_merge_row_order"] = range(len(frame))
        frames.append(frame)
        input_rows += len(frame)

    if not frames:
        raise ValueError("No input frames to merge")

    merged = pd.concat(frames, ignore_index=True, sort=False)
    merged["timestamp"] = pd.to_datetime(merged["timestamp"], utc=True, errors="coerce")
    invalid_timestamps = int(merged["timestamp"].isna().sum())
    if invalid_timestamps:
        raise ValueError(f"Found {invalid_timestamps} rows with invalid timestamps")

    merged = merged.sort_values(["timestamp", "_merge_file_order", "_merge_row_order"], kind="mergesort")
    merged = merged.drop_duplicates(subset=["timestamp"], keep="last")
    merged = merged.drop(columns=["_merge_file_order", "_merge_row_order"])
    merged = merged.loc[:, preferred_column_order(merged.columns.tolist())]
    merged = merged.reset_index(drop=True)

    duplicate_timestamps_removed = input_rows - len(merged)
    return merged, input_rows, duplicate_timestamps_removed


def merge_latest_raw_files(
    raw_dir: Path,
    last_n: int,
    symbol: str = DEFAULT_SYMBOL,
    sort_by: str = "date",
    output_path: Path | None = None,
    logger: logging.Logger | None = None,
) -> MergeSummary:
    """Merge the latest N daily raw Parquet files into one new raw file."""
    files = discover_daily_raw_files(raw_dir=raw_dir, symbol=symbol)
    selected_files = select_latest_files(files=files, last_n=last_n, sort_by=sort_by)
    selected_paths = [file.path for file in selected_files]

    final_output_path = output_path or default_output_path(selected_files=selected_files, raw_dir=raw_dir)
    if final_output_path.exists():
        raise FileExistsError(f"Output file already exists: {final_output_path}")

    if logger is not None:
        logger.info("Merging %s files into %s", len(selected_paths), final_output_path)
        for path in selected_paths:
            logger.info("Selected input: %s", path)

    merged, input_rows, duplicate_timestamps_removed = merge_raw_files(selected_paths)
    write_parquet(merged, final_output_path)

    summary = MergeSummary(
        output_path=final_output_path,
        input_paths=selected_paths,
        input_rows=input_rows,
        output_rows=len(merged),
        duplicate_timestamps_removed=duplicate_timestamps_removed,
    )

    if logger is not None:
        logger.info(
            "Merged %s input rows into %s output rows; removed %s duplicate timestamps",
            summary.input_rows,
            summary.output_rows,
            summary.duplicate_timestamps_removed,
        )

    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    parser.add_argument("--last-n", type=int, required=True, help="Number of latest daily raw files to merge.")
    parser.add_argument("--symbol", type=str, default=DEFAULT_SYMBOL)
    parser.add_argument(
        "--sort-by",
        choices=["date", "mtime"],
        default="date",
        help="Select latest files by filename date or filesystem modification time.",
    )
    parser.add_argument("--output-path", type=Path, default=None)
    parser.add_argument("--log-file", type=Path, default=DEFAULT_LOG_FILE)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logger = configure_logging(args.log_file)
    summary = merge_latest_raw_files(
        raw_dir=args.raw_dir,
        last_n=args.last_n,
        symbol=args.symbol,
        sort_by=args.sort_by,
        output_path=args.output_path,
        logger=logger,
    )

    print("Merged raw order book files:", flush=True)
    for path in summary.input_paths:
        print(f"- {path}", flush=True)
    print(f"Saved merged file: {summary.output_path}", flush=True)
    print(f"Rows: {summary.input_rows} input -> {summary.output_rows} merged", flush=True)
    print(f"Duplicate timestamps removed: {summary.duplicate_timestamps_removed}", flush=True)


if __name__ == "__main__":
    main()
