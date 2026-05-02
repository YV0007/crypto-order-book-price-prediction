"""Clean the newest raw top-20 order book Parquet file."""

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
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "CONFIG" / "config.yaml"
DEFAULT_RAW_DIR = PROJECT_ROOT / "data" / "raw" / "orderbook_top20"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "interim" / "cleaned_orderbook"
DEFAULT_REPORT_PATH = PROJECT_ROOT / "reports" / "cleaning_report.txt"
DEFAULT_LOG_FILE = PROJECT_ROOT / "logs" / "clean_orderbook.log"
DEFAULT_LEVELS = 20
DEFAULT_SPREAD_OUTLIER_METHOD = "remove"
DEFAULT_SPREAD_UPPER_QUANTILE = 0.99
DEFAULT_MAX_ALLOWED_SPREAD = 1.0
DEFAULT_WINSORIZE_SIZES = True
DEFAULT_WINSOR_LOWER_QUANTILE = 0.001
DEFAULT_WINSOR_UPPER_QUANTILE = 0.99


@dataclass(frozen=True)
class CleaningConfig:
    """Configuration for cleaning and outlier handling."""

    spread_outlier_method: str = DEFAULT_SPREAD_OUTLIER_METHOD
    spread_upper_quantile: float = DEFAULT_SPREAD_UPPER_QUANTILE
    max_allowed_spread: float = DEFAULT_MAX_ALLOWED_SPREAD
    winsorize_sizes: bool = DEFAULT_WINSORIZE_SIZES
    winsor_lower_quantile: float = DEFAULT_WINSOR_LOWER_QUANTILE
    winsor_upper_quantile: float = DEFAULT_WINSOR_UPPER_QUANTILE


@dataclass(frozen=True)
class WinsorizationStats:
    """Winsorization details for one liquidity column."""

    column: str
    lower_cap: float
    upper_cap: float
    capped_lower: int
    capped_upper: int


@dataclass(frozen=True)
class CleaningSummary:
    """Summary of the cleaning run."""

    input_path: Path
    output_path: Path
    report_path: Path
    raw_rows: int
    cleaned_rows: int
    removed_rows: int
    duplicate_timestamps_removed: int
    missing_required_values_removed: int
    invalid_spread_rows_removed: int
    spread_99_percentile: float
    spread_outlier_rows_removed: int
    negative_size_rows_removed: int
    winsorized_columns: list[str]
    winsor_lower_quantile: float
    winsor_upper_quantile: float
    winsorization_stats: list[WinsorizationStats]

    @property
    def removed_percentage(self) -> float:
        """Return percentage of raw rows removed."""
        if self.raw_rows == 0:
            return 0.0
        return self.removed_rows / self.raw_rows * 100


def required_columns(levels: int = DEFAULT_LEVELS) -> list[str]:
    """Return the expected raw top-20 order book columns."""
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


def numeric_orderbook_columns(levels: int = DEFAULT_LEVELS) -> list[str]:
    """Return all price and size columns."""
    columns: list[str] = []
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


def liquidity_winsorization_columns(frame: pd.DataFrame, levels: int = DEFAULT_LEVELS) -> list[str]:
    """Return liquidity columns allowed for winsorization."""
    candidates: list[str] = size_columns(levels)
    candidates.extend(
        column
        for column in frame.columns
        if column.startswith("bid_depth_") or column.startswith("ask_depth_")
    )
    candidates.extend(["weighted_bid_depth", "weighted_ask_depth"])
    seen: set[str] = set()
    columns: list[str] = []
    for column in candidates:
        if column in frame.columns and column not in seen:
            columns.append(column)
            seen.add(column)
    return columns


def find_newest_parquet(raw_dir: Path = DEFAULT_RAW_DIR) -> Path:
    """Find the newest raw Parquet file by modification time."""
    files = sorted(raw_dir.glob("*.parquet"), key=lambda path: path.stat().st_mtime)
    if not files:
        raise FileNotFoundError(f"No Parquet files found in {raw_dir}")
    return files[-1]


def configure_logging(log_file: Path = DEFAULT_LOG_FILE) -> logging.Logger:
    """Configure cleaning logs."""
    log_file.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger


def load_cleaning_config(config_path: Path = DEFAULT_CONFIG_PATH) -> CleaningConfig:
    """Load cleaning settings from CONFIG/config.yaml, falling back to defaults."""
    values: dict[str, Any] = {}
    if config_path.exists():
        loaded = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
        if not isinstance(loaded, dict):
            raise ValueError(f"Config file must contain a mapping: {config_path}")
        cleaning_values = loaded.get("cleaning", {})
        if cleaning_values is None:
            cleaning_values = {}
        if not isinstance(cleaning_values, dict):
            raise ValueError("CONFIG/config.yaml cleaning section must be a mapping")
        values = cleaning_values

    config = CleaningConfig(
        spread_outlier_method=str(values.get("spread_outlier_method", DEFAULT_SPREAD_OUTLIER_METHOD)),
        spread_upper_quantile=float(values.get("spread_upper_quantile", DEFAULT_SPREAD_UPPER_QUANTILE)),
        max_allowed_spread=float(values.get("max_allowed_spread", DEFAULT_MAX_ALLOWED_SPREAD)),
        winsorize_sizes=bool(values.get("winsorize_sizes", DEFAULT_WINSORIZE_SIZES)),
        winsor_lower_quantile=float(values.get("winsor_lower_quantile", DEFAULT_WINSOR_LOWER_QUANTILE)),
        winsor_upper_quantile=float(values.get("winsor_upper_quantile", DEFAULT_WINSOR_UPPER_QUANTILE)),
    )
    validate_cleaning_config(config)
    return config


def validate_cleaning_config(config: CleaningConfig) -> None:
    """Validate cleaning configuration values."""
    if config.spread_outlier_method != "remove":
        raise ValueError('Only spread_outlier_method="remove" is supported')
    if not 0 < config.spread_upper_quantile < 1:
        raise ValueError("spread_upper_quantile must be between 0 and 1")
    if config.max_allowed_spread <= 0:
        raise ValueError("max_allowed_spread must be positive")
    if not 0 <= config.winsor_lower_quantile < config.winsor_upper_quantile <= 1:
        raise ValueError("winsor quantiles must satisfy 0 <= lower < upper <= 1")


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


def read_raw_orderbook(input_path: Path) -> pd.DataFrame:
    """Load a raw order book Parquet file."""
    with suppress_native_stderr():
        return pd.read_parquet(input_path)


def write_cleaned_orderbook(frame: pd.DataFrame, output_path: Path) -> None:
    """Write cleaned order book data to Parquet."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_suffix(".tmp.parquet")
    with suppress_native_stderr():
        frame.to_parquet(tmp_path, index=False)
    tmp_path.replace(output_path)


def winsorize_liquidity_columns(
    frame: pd.DataFrame,
    columns: list[str],
    lower_quantile: float,
    upper_quantile: float,
) -> tuple[pd.DataFrame, list[WinsorizationStats]]:
    """Winsorize liquidity columns only, preserving raw price columns."""
    output = frame.copy()
    stats: list[WinsorizationStats] = []

    for column in columns:
        series = pd.to_numeric(output[column], errors="coerce")
        lower_cap = float(series.quantile(lower_quantile))
        upper_cap = float(series.quantile(upper_quantile))
        capped_lower = int((series < lower_cap).sum())
        capped_upper = int((series > upper_cap).sum())
        output[column] = series.clip(lower=lower_cap, upper=upper_cap)
        stats.append(
            WinsorizationStats(
                column=column,
                lower_cap=lower_cap,
                upper_cap=upper_cap,
                capped_lower=capped_lower,
                capped_upper=capped_upper,
            )
        )

    return output, stats


def clean_orderbook_frame(
    frame: pd.DataFrame,
    config: CleaningConfig,
    levels: int = DEFAULT_LEVELS,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Sort, clean, remove spread outliers, and winsorize liquidity columns."""
    expected_columns = required_columns(levels)
    missing_columns = sorted(set(expected_columns) - set(frame.columns))
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    cleaned = frame.copy()
    cleaned["_raw_order"] = range(len(cleaned))
    raw_rows = len(cleaned)

    cleaned["timestamp"] = pd.to_datetime(cleaned["timestamp"], utc=True, errors="coerce")
    cleaned["event_time"] = pd.to_datetime(cleaned["event_time"], utc=True, errors="coerce")

    for column in numeric_orderbook_columns(levels):
        cleaned[column] = pd.to_numeric(cleaned[column], errors="coerce")
    for column in liquidity_winsorization_columns(cleaned, levels=levels):
        cleaned[column] = pd.to_numeric(cleaned[column], errors="coerce")

    required_value_columns = ["timestamp", "event_time", *numeric_orderbook_columns(levels)]
    before_missing = len(cleaned)
    cleaned = cleaned.dropna(subset=required_value_columns)
    missing_required_values_removed = before_missing - len(cleaned)

    cleaned = cleaned.sort_values(["timestamp", "_raw_order"], kind="mergesort")
    before_duplicates = len(cleaned)
    cleaned = cleaned.drop_duplicates(subset=["timestamp"], keep="last")
    duplicate_timestamps_removed = before_duplicates - len(cleaned)

    cleaned["_spread"] = cleaned["ask_price_1"] - cleaned["bid_price_1"]

    before_invalid_spread = len(cleaned)
    valid_spread_mask = cleaned["_spread"] > 0
    cleaned = cleaned.loc[valid_spread_mask]
    invalid_spread_rows_removed = before_invalid_spread - len(cleaned)

    spread_99_percentile = float(cleaned["_spread"].quantile(config.spread_upper_quantile)) if not cleaned.empty else 0.0
    before_spread_outliers = len(cleaned)
    spread_outlier_mask = (cleaned["_spread"] <= spread_99_percentile) & (
        cleaned["_spread"] <= config.max_allowed_spread
    )
    cleaned = cleaned.loc[spread_outlier_mask]
    spread_outlier_rows_removed = before_spread_outliers - len(cleaned)

    present_size_columns = [column for column in size_columns(levels) if column in cleaned.columns]
    before_negative_sizes = len(cleaned)
    no_negative_sizes_mask = (cleaned[present_size_columns] >= 0).all(axis=1)
    cleaned = cleaned.loc[no_negative_sizes_mask]
    negative_size_rows_removed = before_negative_sizes - len(cleaned)

    winsor_columns = liquidity_winsorization_columns(cleaned, levels=levels) if config.winsorize_sizes else []
    winsor_stats: list[WinsorizationStats] = []
    if winsor_columns and not cleaned.empty:
        cleaned, winsor_stats = winsorize_liquidity_columns(
            cleaned,
            columns=winsor_columns,
            lower_quantile=config.winsor_lower_quantile,
            upper_quantile=config.winsor_upper_quantile,
        )

    cleaned = cleaned.drop(columns=["_raw_order", "_spread"])
    cleaned = cleaned.reset_index(drop=True)

    report_values = {
        "raw_rows": raw_rows,
        "duplicate_timestamps_removed": duplicate_timestamps_removed,
        "missing_required_values_removed": missing_required_values_removed,
        "invalid_spread_rows_removed": invalid_spread_rows_removed,
        "spread_99_percentile": spread_99_percentile,
        "spread_outlier_rows_removed": spread_outlier_rows_removed,
        "negative_size_rows_removed": negative_size_rows_removed,
        "winsorized_columns": winsor_columns,
        "winsorization_stats": winsor_stats,
    }
    return cleaned, report_values


def build_cleaning_report(summary: CleaningSummary, config: CleaningConfig) -> str:
    """Build a detailed plain-text cleaning report."""
    lines = [
        "LOB Order Book Cleaning Report",
        "==============================",
        f"Input file path: {summary.input_path}",
        f"Output file path: {summary.output_path}",
        f"Original row count: {summary.raw_rows}",
        f"Duplicate timestamps removed: {summary.duplicate_timestamps_removed}",
        f"Rows removed for missing required values: {summary.missing_required_values_removed}",
        f"Rows removed for invalid spread: {summary.invalid_spread_rows_removed}",
        f"Spread 99th percentile value: {summary.spread_99_percentile:.10f}",
        f"Configured spread upper quantile: {config.spread_upper_quantile}",
        f"Max allowed spread: {config.max_allowed_spread:.10f}",
        f"Rows removed for spread outliers: {summary.spread_outlier_rows_removed}",
        f"Rows removed for negative sizes: {summary.negative_size_rows_removed}",
        f"Final row count: {summary.cleaned_rows}",
        f"Total rows removed: {summary.removed_rows}",
        f"Percentage of rows removed: {summary.removed_percentage:.4f}%",
        "",
        "Winsorization",
        "-------------",
        f"Winsorize sizes enabled: {config.winsorize_sizes}",
        f"Lower quantile: {summary.winsor_lower_quantile}",
        f"Upper quantile: {summary.winsor_upper_quantile}",
        "Winsorized columns list:",
    ]
    if summary.winsorized_columns:
        lines.extend(f"- {column}" for column in summary.winsorized_columns)
    else:
        lines.append("- none")

    lines.extend(["", "Winsorized column details:"])
    if summary.winsorization_stats:
        for item in summary.winsorization_stats:
            lines.extend(
                [
                    f"- {item.column}",
                    f"  lower_cap: {item.lower_cap:.10f}",
                    f"  upper_cap: {item.upper_cap:.10f}",
                    f"  capped_lower: {item.capped_lower}",
                    f"  capped_upper: {item.capped_upper}",
                ]
            )
    else:
        lines.append("- none")

    return "\n".join(lines) + "\n"


def write_cleaning_report(summary: CleaningSummary, config: CleaningConfig, report_path: Path) -> None:
    """Write the cleaning report to disk."""
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(build_cleaning_report(summary, config), encoding="utf-8")


def clean_orderbook_file(
    input_path: Path,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    report_path: Path = DEFAULT_REPORT_PATH,
    config: CleaningConfig | None = None,
    levels: int = DEFAULT_LEVELS,
    logger: logging.Logger | None = None,
) -> CleaningSummary:
    """Clean one raw Parquet file and save it to the interim directory."""
    cleaning_config = config or CleaningConfig()
    if logger is not None:
        logger.info("Loading raw order book file: %s", input_path)

    raw = read_raw_orderbook(input_path)
    cleaned, report_values = clean_orderbook_frame(raw, config=cleaning_config, levels=levels)
    output_path = output_dir / input_path.name
    write_cleaned_orderbook(cleaned, output_path)

    summary = CleaningSummary(
        input_path=input_path,
        output_path=output_path,
        report_path=report_path,
        raw_rows=len(raw),
        cleaned_rows=len(cleaned),
        removed_rows=len(raw) - len(cleaned),
        duplicate_timestamps_removed=int(report_values["duplicate_timestamps_removed"]),
        missing_required_values_removed=int(report_values["missing_required_values_removed"]),
        invalid_spread_rows_removed=int(report_values["invalid_spread_rows_removed"]),
        spread_99_percentile=float(report_values["spread_99_percentile"]),
        spread_outlier_rows_removed=int(report_values["spread_outlier_rows_removed"]),
        negative_size_rows_removed=int(report_values["negative_size_rows_removed"]),
        winsorized_columns=list(report_values["winsorized_columns"]),
        winsor_lower_quantile=cleaning_config.winsor_lower_quantile,
        winsor_upper_quantile=cleaning_config.winsor_upper_quantile,
        winsorization_stats=list(report_values["winsorization_stats"]),
    )
    write_cleaning_report(summary, cleaning_config, report_path)

    if logger is not None:
        logger.info(
            "Cleaned %s rows from %s into %s",
            summary.removed_rows,
            input_path,
            output_path,
        )

    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    parser.add_argument("--input-path", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--report-path", type=Path, default=DEFAULT_REPORT_PATH)
    parser.add_argument("--config-path", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--log-file", type=Path, default=DEFAULT_LOG_FILE)
    parser.add_argument("--levels", type=int, default=DEFAULT_LEVELS)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logger = configure_logging(args.log_file)
    config = load_cleaning_config(args.config_path)
    input_path = args.input_path or find_newest_parquet(args.raw_dir)
    summary = clean_orderbook_file(
        input_path=input_path,
        output_dir=args.output_dir,
        report_path=args.report_path,
        config=config,
        levels=args.levels,
        logger=logger,
    )

    print("Cleaning completed", flush=True)
    print(f"Rows: {summary.raw_rows} raw -> {summary.cleaned_rows} cleaned", flush=True)
    print(f"Saved cleaning report: {summary.report_path}", flush=True)
    print(f"Saved cleaned file: {summary.output_path}", flush=True)


if __name__ == "__main__":
    main()
