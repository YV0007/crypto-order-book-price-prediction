"""Collect Binance USD-M Futures BTCUSDT top-20 order book snapshots.

The collector connects to the Binance Futures partial-depth stream, keeps the
latest depth message in memory, samples one normalized snapshot every second,
marks continuity breaks with session_id, and writes daily Parquet files under
data/raw/orderbook_top20/.
"""

from __future__ import annotations

import argparse
import asyncio
from contextlib import contextmanager
import json
import logging
import os
import sys
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import websockets
from websockets.exceptions import ConnectionClosed
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
UTC = timezone.utc
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "CONFIG" / "config.yaml"
DEFAULT_SYMBOL = "BTCUSDT"
DEFAULT_EXCHANGE = "binance_usd_m_futures"
DEFAULT_LEVELS = 20
DEFAULT_SAMPLE_INTERVAL_SECONDS = 1.0
DEFAULT_FLUSH_EVERY_SECONDS = 60.0
DEFAULT_GAP_THRESHOLD_SECONDS = 1.5
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "raw" / "orderbook_top20"
DEFAULT_LOG_FILE = PROJECT_ROOT / "logs" / "collector.log"


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


def load_config(config_path: Path = DEFAULT_CONFIG_PATH) -> dict[str, Any]:
    """Load YAML config, returning an empty config when it is unavailable."""
    if not config_path.exists():
        return {}
    with config_path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)
    return config if isinstance(config, dict) else {}


def get_nested(config: dict[str, Any], keys: tuple[str, ...], default: Any) -> Any:
    """Read a nested config value."""
    value: Any = config
    for key in keys:
        if not isinstance(value, dict) or key not in value:
            return default
        value = value[key]
    return value


def configured_flush_every_seconds(config: dict[str, Any]) -> float:
    """Return the configured flush cadence in seconds."""
    value = get_nested(config, ("collector", "flush_every_seconds"), None)
    if value is None:
        value = get_nested(config, ("collection", "flush_every_seconds"), DEFAULT_FLUSH_EVERY_SECONDS)
    return float(value)


def build_stream_url(symbol: str, levels: int = DEFAULT_LEVELS) -> str:
    """Build the Binance USD-M Futures partial-depth stream URL."""
    stream_name = f"{symbol.lower()}@depth{levels}@100ms"
    return f"wss://fstream.binance.com/ws/{stream_name}"


def floor_to_second(value: datetime) -> datetime:
    """Floor a timezone-aware datetime to the nearest second."""
    return value.astimezone(UTC).replace(microsecond=0)


def next_sample_time(interval_seconds: float) -> datetime:
    """Return the next UTC sample timestamp."""
    now = datetime.now(tz=UTC)
    return floor_to_second(now) + timedelta(seconds=interval_seconds)


def expected_columns(levels: int = DEFAULT_LEVELS) -> list[str]:
    """Return the ordered schema for normalized order book snapshots."""
    columns = ["timestamp", "event_time", "exchange", "symbol", "session_id"]
    columns.extend(f"bid_price_{level}" for level in range(1, levels + 1))
    columns.extend(f"bid_size_{level}" for level in range(1, levels + 1))
    columns.extend(f"ask_price_{level}" for level in range(1, levels + 1))
    columns.extend(f"ask_size_{level}" for level in range(1, levels + 1))
    return columns


def unwrap_stream_message(message: dict[str, Any]) -> dict[str, Any]:
    """Support both raw and combined Binance stream message shapes."""
    data = message.get("data")
    if isinstance(data, dict):
        return data
    return message


def parse_price_levels(levels: Iterable[Iterable[Any]], expected_count: int, side: str) -> list[tuple[float, float]]:
    """Parse price/size levels from the Binance stream payload."""
    parsed: list[tuple[float, float]] = []
    for raw_level in levels:
        try:
            price, size = raw_level
            parsed.append((float(price), float(size)))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid {side} level: {raw_level}") from exc

    if len(parsed) < expected_count:
        raise ValueError(f"Expected at least {expected_count} {side} levels, received {len(parsed)}")

    return parsed[:expected_count]


def normalize_depth_message(
    message: dict[str, Any],
    timestamp: datetime,
    symbol: str,
    session_id: int,
    exchange: str = DEFAULT_EXCHANGE,
    levels: int = DEFAULT_LEVELS,
) -> dict[str, Any]:
    """Convert one Binance depth message into the raw project schema."""
    payload = unwrap_stream_message(message)
    event_time_ms = payload.get("E")
    bids_raw = payload.get("b") or payload.get("bids")
    asks_raw = payload.get("a") or payload.get("asks")

    if event_time_ms is None:
        raise ValueError("Depth message is missing event time field 'E'")
    if bids_raw is None or asks_raw is None:
        raise ValueError("Depth message is missing bid or ask levels")

    bids = parse_price_levels(bids_raw, expected_count=levels, side="bid")
    asks = parse_price_levels(asks_raw, expected_count=levels, side="ask")

    row: dict[str, Any] = {
        "timestamp": floor_to_second(timestamp),
        "event_time": pd.to_datetime(int(event_time_ms), unit="ms", utc=True).to_pydatetime(),
        "exchange": exchange,
        "symbol": symbol.upper(),
        "session_id": int(session_id),
    }

    for index, (bid_price, _) in enumerate(bids, start=1):
        row[f"bid_price_{index}"] = bid_price
    for index, (_, bid_size) in enumerate(bids, start=1):
        row[f"bid_size_{index}"] = bid_size
    for index, (ask_price, _) in enumerate(asks, start=1):
        row[f"ask_price_{index}"] = ask_price
    for index, (_, ask_size) in enumerate(asks, start=1):
        row[f"ask_size_{index}"] = ask_size

    return row


def rows_to_frame(rows: list[dict[str, Any]], levels: int) -> pd.DataFrame:
    """Convert buffered rows to a typed DataFrame with stable column order."""
    frame = pd.DataFrame(rows, columns=expected_columns(levels))
    return normalize_frame_schema(frame, levels)


def infer_session_ids_from_timestamps(
    timestamps: pd.Series,
    gap_threshold_seconds: float = DEFAULT_GAP_THRESHOLD_SECONDS,
    start_session_id: int = 1,
) -> pd.Series:
    """Infer session ids from timestamp gaps."""
    if timestamps.empty:
        return pd.Series(dtype="int64")

    sorted_timestamps = pd.to_datetime(timestamps, utc=True, errors="coerce")
    gaps = sorted_timestamps.diff().dt.total_seconds()
    session_offsets = (gaps > gap_threshold_seconds).fillna(False).cumsum()
    return (session_offsets + start_session_id).astype("int64")


def normalize_frame_schema(frame: pd.DataFrame, levels: int = DEFAULT_LEVELS) -> pd.DataFrame:
    """Normalize raw rows to the expected collector schema."""
    frame = frame.reindex(columns=expected_columns(levels))
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce")
    frame["event_time"] = pd.to_datetime(frame["event_time"], utc=True, errors="coerce")

    for column in expected_columns(levels):
        if column.startswith(("bid_price_", "bid_size_", "ask_price_", "ask_size_")):
            frame[column] = pd.to_numeric(frame[column], errors="coerce")

    parsed_session_id = pd.to_numeric(frame["session_id"], errors="coerce")
    if parsed_session_id.isna().any():
        frame = frame.sort_values("timestamp", kind="mergesort").reset_index(drop=True)
        parsed_session_id = pd.to_numeric(frame["session_id"], errors="coerce")
        inferred_session_id = infer_session_ids_from_timestamps(frame["timestamp"])
        frame["session_id"] = parsed_session_id.fillna(inferred_session_id).astype("int64")
    else:
        frame["session_id"] = parsed_session_id.astype("int64")

    return frame


def deduplicate_and_sort(frame: pd.DataFrame, levels: int = DEFAULT_LEVELS) -> pd.DataFrame:
    """Sort rows by timestamp and keep the last row per timestamp."""
    normalized = normalize_frame_schema(frame, levels)
    normalized["_row_order"] = range(len(normalized))
    normalized = normalized.sort_values(["timestamp", "_row_order"], kind="mergesort")
    normalized = normalized.drop_duplicates(subset=["timestamp"], keep="last")
    normalized = normalized.drop(columns=["_row_order"]).reset_index(drop=True)
    normalized["session_id"] = normalized["session_id"].astype("int64")
    return normalized.reindex(columns=expected_columns(levels))


def latest_saved_state(
    output_dir: Path,
    symbol: str,
    levels: int = DEFAULT_LEVELS,
    gap_threshold_seconds: float = DEFAULT_GAP_THRESHOLD_SECONDS,
) -> tuple[int, datetime | None]:
    """Return the latest session id and timestamp already saved on disk."""
    files = sorted(output_dir.glob(f"{symbol.upper()}_*.parquet"))
    if not files:
        return 1, None

    state_frames: list[pd.DataFrame] = []
    for path in files:
        with suppress_native_stderr():
            frame = pd.read_parquet(path)
        if "timestamp" not in frame.columns:
            continue
        state_frame = frame[[column for column in ["timestamp", "session_id"] if column in frame.columns]].copy()
        if "session_id" not in state_frame.columns:
            state_frame["session_id"] = pd.NA
        state_frames.append(state_frame)

    if not state_frames:
        return 1, None

    combined = pd.concat(state_frames, ignore_index=True)
    combined["timestamp"] = pd.to_datetime(combined["timestamp"], utc=True, errors="coerce")
    combined = combined.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    if combined.empty:
        return 1, None

    if combined["session_id"].isna().all():
        combined["session_id"] = infer_session_ids_from_timestamps(
            combined["timestamp"],
            gap_threshold_seconds=gap_threshold_seconds,
        )
    else:
        combined["session_id"] = pd.to_numeric(combined["session_id"], errors="coerce")
        inferred = infer_session_ids_from_timestamps(
            combined["timestamp"],
            gap_threshold_seconds=gap_threshold_seconds,
        )
        combined["session_id"] = combined["session_id"].fillna(inferred).astype("int64")

    last_row = combined.iloc[-1]
    return int(last_row["session_id"]), last_row["timestamp"].to_pydatetime()


@dataclass
class SessionTracker:
    """Track continuity of saved snapshots and assign session ids."""

    session_id: int = 1
    last_saved_timestamp: datetime | None = None
    gap_threshold_seconds: float = DEFAULT_GAP_THRESHOLD_SECONDS
    logger: logging.Logger = field(default_factory=lambda: logging.getLogger(__name__))

    def assign(self, timestamp: datetime) -> int:
        """Assign the current session id, incrementing after continuity breaks."""
        timestamp = floor_to_second(timestamp)
        if self.last_saved_timestamp is not None:
            gap_seconds = (timestamp - self.last_saved_timestamp).total_seconds()
            if gap_seconds > self.gap_threshold_seconds:
                previous_timestamp = self.last_saved_timestamp
                self.session_id += 1
                message = (
                    "Continuity break detected: "
                    f"previous={previous_timestamp.isoformat()}, "
                    f"current={timestamp.isoformat()}, "
                    f"gap_seconds={gap_seconds:.3f}, "
                    f"new_session_id={self.session_id}"
                )
                self.logger.warning(message)
                print(f"WARNING: {message}", flush=True)

        self.last_saved_timestamp = timestamp
        return self.session_id


@dataclass
class DailyParquetWriter:
    """Buffer sampled snapshots and flush them into daily Parquet files."""

    output_dir: Path
    symbol: str
    levels: int = DEFAULT_LEVELS
    flush_every_seconds: float = DEFAULT_FLUSH_EVERY_SECONDS
    logger: logging.Logger = field(default_factory=lambda: logging.getLogger(__name__))
    rows: list[dict[str, Any]] = field(default_factory=list)
    last_flush_at: datetime = field(default_factory=lambda: datetime.now(tz=UTC))

    def add(self, row: dict[str, Any]) -> list[Path]:
        """Add one normalized row and flush when the flush cadence is reached."""
        self.rows.append(row)
        now = datetime.now(tz=UTC)
        if (now - self.last_flush_at).total_seconds() >= self.flush_every_seconds:
            return self.flush()
        return []

    def flush(self) -> list[Path]:
        """Write buffered rows to daily Parquet files and clear the buffer."""
        if not self.rows:
            self.last_flush_at = datetime.now(tz=UTC)
            return []

        self.output_dir.mkdir(parents=True, exist_ok=True)
        frame = rows_to_frame(self.rows, self.levels)
        written_paths: list[Path] = []

        for date_value, daily_frame in frame.groupby(frame["timestamp"].dt.date, sort=True):
            output_path = self.output_dir / f"{self.symbol.upper()}_{date_value.isoformat()}.parquet"
            combined = self._append_existing(output_path, daily_frame)
            tmp_path = output_path.with_suffix(".tmp.parquet")
            with suppress_native_stderr():
                combined.to_parquet(tmp_path, index=False)
            tmp_path.replace(output_path)
            written_paths.append(output_path)
            self.logger.info("Wrote %s buffered rows to %s", len(daily_frame), output_path)

        self.rows.clear()
        self.last_flush_at = datetime.now(tz=UTC)
        return written_paths

    def _append_existing(self, output_path: Path, new_rows: pd.DataFrame) -> pd.DataFrame:
        if output_path.exists():
            with suppress_native_stderr():
                existing = pd.read_parquet(output_path)
            combined = pd.concat([existing, new_rows], ignore_index=True)
        else:
            combined = new_rows

        return deduplicate_and_sort(combined, self.levels)


def configure_logging(log_file: Path) -> logging.Logger:
    """Configure console and file logging for the collector."""
    log_file.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger


async def collect_orderbook_top20(
    symbol: str,
    websocket_url: str,
    output_dir: Path,
    sample_interval_seconds: float,
    levels: int,
    flush_every_seconds: float,
    limit_samples: int | None,
    reconnect_delay_seconds: float,
    gap_threshold_seconds: float,
    logger: logging.Logger,
) -> None:
    """Collect sampled order book snapshots until stopped or limit is reached."""
    initial_session_id, last_saved_timestamp = latest_saved_state(
        output_dir=output_dir,
        symbol=symbol,
        levels=levels,
        gap_threshold_seconds=gap_threshold_seconds,
    )
    session_tracker = SessionTracker(
        session_id=initial_session_id,
        last_saved_timestamp=last_saved_timestamp,
        gap_threshold_seconds=gap_threshold_seconds,
        logger=logger,
    )
    writer = DailyParquetWriter(
        output_dir=output_dir,
        symbol=symbol,
        levels=levels,
        flush_every_seconds=flush_every_seconds,
        logger=logger,
    )
    total_samples = 0

    logger.info(
        "Collector starting: symbol=%s, stream=%s, initial_session_id=%s, last_saved_timestamp=%s, "
        "flush_every_seconds=%s",
        symbol,
        websocket_url,
        initial_session_id,
        last_saved_timestamp.isoformat() if last_saved_timestamp else None,
        flush_every_seconds,
    )

    try:
        while limit_samples is None or total_samples < limit_samples:
            try:
                async with websockets.connect(
                    websocket_url,
                    ping_interval=20,
                    ping_timeout=20,
                    close_timeout=5,
                    max_queue=1_000,
                ) as websocket:
                    logger.info("Connected to %s", websocket_url)
                    print(f"Connected to {websocket_url}", flush=True)

                    latest_message: dict[str, Any] | None = None
                    sample_at = next_sample_time(sample_interval_seconds)

                    while limit_samples is None or total_samples < limit_samples:
                        now = datetime.now(tz=UTC)

                        if latest_message is None and now >= sample_at:
                            sample_at = floor_to_second(now) + timedelta(seconds=sample_interval_seconds)

                        if latest_message is not None and now >= sample_at:
                            sample_timestamp = floor_to_second(now)
                            session_id = session_tracker.assign(sample_timestamp)
                            row = normalize_depth_message(
                                latest_message,
                                timestamp=sample_timestamp,
                                symbol=symbol,
                                session_id=session_id,
                                levels=levels,
                            )
                            written_paths = writer.add(row)
                            total_samples += 1
                            print(
                                "Sampled "
                                f"{total_samples} rows at {row['timestamp'].isoformat()} "
                                f"(session_id={row['session_id']})",
                                flush=True,
                            )
                            for path in written_paths:
                                print(f"Flushed buffered snapshots to {path}", flush=True)

                            sample_at = sample_timestamp + timedelta(seconds=sample_interval_seconds)

                            if limit_samples is not None and total_samples >= limit_samples:
                                break

                        if limit_samples is not None and total_samples >= limit_samples:
                            break

                        timeout_seconds = max((sample_at - datetime.now(tz=UTC)).total_seconds(), 0.001)
                        try:
                            raw_message = await asyncio.wait_for(websocket.recv(), timeout=timeout_seconds)
                        except TimeoutError:
                            continue

                        latest_message = json.loads(raw_message)

            except (ConnectionClosed, OSError, asyncio.TimeoutError, json.JSONDecodeError, ValueError) as exc:
                logger.exception("Collector error: %s", exc)
                written_paths = writer.flush()
                for path in written_paths:
                    print(f"Flushed buffered snapshots to {path}", flush=True)
                if limit_samples is not None and total_samples >= limit_samples:
                    break
                print(f"Collector error: {exc}. Reconnecting in {reconnect_delay_seconds} seconds.", flush=True)
                await asyncio.sleep(reconnect_delay_seconds)

    finally:
        written_paths = writer.flush()
        for path in written_paths:
            print(f"Flushed buffered snapshots to {path}", flush=True)
        logger.info("Collector stopped after %s samples", total_samples)


def parse_args() -> argparse.Namespace:
    config = load_config()
    default_flush_every_seconds = configured_flush_every_seconds(config)

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--symbol", default=DEFAULT_SYMBOL)
    parser.add_argument("--websocket-url", default=None)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--sample-interval-seconds", type=float, default=DEFAULT_SAMPLE_INTERVAL_SECONDS)
    parser.add_argument("--levels", type=int, default=DEFAULT_LEVELS)
    parser.add_argument("--flush-every-seconds", type=float, default=default_flush_every_seconds)
    parser.add_argument("--flush-rows", type=float, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--limit-samples", type=int, default=None)
    parser.add_argument("--reconnect-delay-seconds", type=float, default=5.0)
    parser.add_argument("--gap-threshold-seconds", type=float, default=DEFAULT_GAP_THRESHOLD_SECONDS)
    parser.add_argument("--log-file", type=Path, default=DEFAULT_LOG_FILE)
    args = parser.parse_args()

    if args.flush_rows is not None:
        args.flush_every_seconds = float(args.flush_rows)

    return args


def main() -> None:
    args = parse_args()
    websocket_url = args.websocket_url or build_stream_url(args.symbol, args.levels)
    logger = configure_logging(args.log_file)

    try:
        asyncio.run(
            collect_orderbook_top20(
                symbol=args.symbol,
                websocket_url=websocket_url,
                output_dir=args.output_dir,
                sample_interval_seconds=args.sample_interval_seconds,
                levels=args.levels,
                flush_every_seconds=args.flush_every_seconds,
                limit_samples=args.limit_samples,
                reconnect_delay_seconds=args.reconnect_delay_seconds,
                gap_threshold_seconds=args.gap_threshold_seconds,
                logger=logger,
            )
        )
    except KeyboardInterrupt:
        logger.info("Collector interrupted manually with KeyboardInterrupt")
        print("Collector stopped manually.", flush=True)


if __name__ == "__main__":
    main()
