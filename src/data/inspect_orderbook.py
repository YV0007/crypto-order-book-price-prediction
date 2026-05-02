"""Inspect the newest cleaned top-20 order book Parquet file."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import os
import sys
from dataclasses import dataclass
from pathlib import Path


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


MATPLOTLIB_CONFIG_DIR = Path("/private/tmp/lob_project_matplotlib")
XDG_CACHE_DIR = Path("/private/tmp/lob_project_cache")
MATPLOTLIB_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
XDG_CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MATPLOTLIB_CONFIG_DIR))
os.environ.setdefault("XDG_CACHE_HOME", str(XDG_CACHE_DIR))

with suppress_native_stderr():
    import matplotlib

    matplotlib.use("Agg")

    import matplotlib.pyplot as plt
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CLEANED_DIR = PROJECT_ROOT / "data" / "interim" / "cleaned_orderbook"
DEFAULT_FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"


@dataclass(frozen=True)
class InspectionSummary:
    """Summary statistics for cleaned order book inspection."""

    input_path: Path
    row_count: int
    start_timestamp: str
    end_timestamp: str
    mid_price_min: float
    mid_price_mean: float
    mid_price_max: float
    spread_min: float
    spread_mean: float
    spread_max: float
    mid_price_plot: Path
    spread_plot: Path


def find_newest_parquet(cleaned_dir: Path = DEFAULT_CLEANED_DIR) -> Path:
    """Find the newest cleaned Parquet file by modification time."""
    files = sorted(cleaned_dir.glob("*.parquet"), key=lambda path: path.stat().st_mtime)
    if not files:
        raise FileNotFoundError(f"No Parquet files found in {cleaned_dir}")
    return files[-1]


def load_cleaned_orderbook(input_path: Path) -> pd.DataFrame:
    """Load cleaned order book data."""
    with suppress_native_stderr():
        return pd.read_parquet(input_path)


def add_inspection_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """Add mid_price and spread columns for inspection."""
    required = {"timestamp", "bid_price_1", "ask_price_1"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    output = frame.copy()
    output["timestamp"] = pd.to_datetime(output["timestamp"], utc=True, errors="coerce")
    output["bid_price_1"] = pd.to_numeric(output["bid_price_1"], errors="coerce")
    output["ask_price_1"] = pd.to_numeric(output["ask_price_1"], errors="coerce")
    output = output.dropna(subset=["timestamp", "bid_price_1", "ask_price_1"])
    output = output.sort_values("timestamp").reset_index(drop=True)
    output["mid_price"] = (output["bid_price_1"] + output["ask_price_1"]) / 2
    output["spread"] = (output["ask_price_1"] - output["bid_price_1"]).round(4)
    return output


def save_line_plot(
    frame: pd.DataFrame,
    y_column: str,
    title: str,
    ylabel: str,
    output_path: Path,
) -> Path:
    """Save a timestamp line plot."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with suppress_native_stderr():
        figure, axis = plt.subplots(figsize=(12, 5))
        axis.plot(frame["timestamp"], frame[y_column], linewidth=1)
        axis.set_title(title)
        axis.set_xlabel("Timestamp")
        axis.set_ylabel(ylabel)
        axis.grid(True, alpha=0.3)
        figure.autofmt_xdate()
        figure.tight_layout()
        figure.savefig(output_path, dpi=150)
        plt.close(figure)
    return output_path


def inspect_orderbook_file(input_path: Path, figures_dir: Path = DEFAULT_FIGURES_DIR) -> InspectionSummary:
    """Compute inspection stats and save mid-price/spread plots."""
    frame = load_cleaned_orderbook(input_path)
    inspected = add_inspection_columns(frame)
    if inspected.empty:
        raise ValueError(f"No valid rows available for inspection in {input_path}")

    mid_price_plot = save_line_plot(
        inspected,
        y_column="mid_price",
        title="BTCUSDT Mid Price",
        ylabel="Mid Price",
        output_path=figures_dir / "mid_price.png",
    )
    spread_plot = save_line_plot(
        inspected,
        y_column="spread",
        title="BTCUSDT Spread",
        ylabel="Spread",
        output_path=figures_dir / "spread.png",
    )

    return InspectionSummary(
        input_path=input_path,
        row_count=len(inspected),
        start_timestamp=inspected["timestamp"].min().isoformat(),
        end_timestamp=inspected["timestamp"].max().isoformat(),
        mid_price_min=float(inspected["mid_price"].min()),
        mid_price_mean=float(inspected["mid_price"].mean()),
        mid_price_max=float(inspected["mid_price"].max()),
        spread_min=float(inspected["spread"].min()),
        spread_mean=float(inspected["spread"].mean()),
        spread_max=float(inspected["spread"].max()),
        mid_price_plot=mid_price_plot,
        spread_plot=spread_plot,
    )


def print_summary(summary: InspectionSummary) -> None:
    """Print inspection stats to the terminal."""
    print(f"Input file: {summary.input_path}", flush=True)
    print(f"Rows: {summary.row_count}", flush=True)
    print(f"Start timestamp: {summary.start_timestamp}", flush=True)
    print(f"End timestamp: {summary.end_timestamp}", flush=True)
    print(
        "Mid price min/mean/max: "
        f"{summary.mid_price_min:.6f} / {summary.mid_price_mean:.6f} / {summary.mid_price_max:.6f}",
        flush=True,
    )
    print(
        "Spread min/mean/max: "
        f"{summary.spread_min:.6f} / {summary.spread_mean:.6f} / {summary.spread_max:.6f}",
        flush=True,
    )
    print(f"Saved plot: {summary.mid_price_plot}", flush=True)
    print(f"Saved plot: {summary.spread_plot}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cleaned-dir", type=Path, default=DEFAULT_CLEANED_DIR)
    parser.add_argument("--input-path", type=Path, default=None)
    parser.add_argument("--figures-dir", type=Path, default=DEFAULT_FIGURES_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = args.input_path or find_newest_parquet(args.cleaned_dir)
    summary = inspect_orderbook_file(input_path=input_path, figures_dir=args.figures_dir)
    print_summary(summary)


if __name__ == "__main__":
    main()
