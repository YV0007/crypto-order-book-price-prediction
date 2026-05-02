<<<<<<< HEAD
# BTCUSDT Limit Order Book ML MVP

Python-only research pipeline for BTCUSDT limit order book machine learning.

This project tests whether short-horizon BTCUSDT mid-price direction can be predicted from Binance Futures top-20 order book structure. It is a quant research MVP, not a production trading or HFT execution engine.

## MVP Objective

Build an end-to-end, reproducible research pipeline that:

- collects BTCUSDT top-20 order book snapshots from Binance Futures
- validates and cleans raw order book data
- engineers market microstructure features
- builds timestamp-aware short-horizon targets
- trains Logistic Regression and LightGBM baselines
- evaluates models with chronological validation
- runs a simple probability-threshold backtest with configurable costs

The current MVP conclusion is deliberately conservative: the pipeline works end to end, and the zero-cost validation backtest shows weak positive signal, but cost-adjusted performance is negative because average gross return per trade is below estimated transaction costs.

## Data Source

- Exchange: Binance USD-M Futures
- Symbol: BTCUSDT
- WebSocket stream: `wss://fstream.binance.com/ws/btcusdt@depth20@100ms`
- Sampling: one normalized top-20 snapshot per second
- Storage format: Parquet

Raw data is not included in this repository. Users need to collect their own Binance WebSocket data before running the full pipeline.

## Pipeline

1. Data collection: save normalized top-20 snapshots to daily Parquet files.
2. Raw merge: combine recent daily files for multi-day experiments.
3. Validation: check schema, timestamps, missing values, spreads, and sizes.
4. Cleaning: remove invalid rows and spread outliers; winsorize size columns.
5. Inspection: create mid-price and spread plots.
6. Feature engineering: build spread, depth, imbalance, weighted imbalance, microprice, lag, rolling, volatility, and change features.
7. Target construction: build timestamp-aware 5-second absolute-threshold labels.
8. Modeling: train Logistic Regression and LightGBM using chronological splits.
9. Evaluation: compare metrics, confusion matrices, prediction agreement, and feature importance.
10. Backtest: convert probabilities into long/short/no-trade signals and compare zero-cost vs cost-adjusted results.

## Project Structure

```text
CONFIG/
  config.yaml
  config_example.yaml

src/
  data/
    collect_orderbook_top20.py
    merge_raw_orderbook.py
    validate_orderbook.py
    clean_orderbook.py
    inspect_orderbook.py
  features/
    build_features.py
  targets/
    build_targets.py
  models/
    train_logistic.py
    train_lightgbm.py
    evaluate_model.py
  backtest/
    simple_backtest.py
    threshold_sweep.py
  experiments/
    run_horizon_experiment.py

data/       ignored generated data
models/     ignored trained model files
reports/    ignored generated reports and figures
logs/       ignored runtime logs
```

## Setup

Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Review configuration:

```bash
cp CONFIG/config_example.yaml CONFIG/config.yaml
```

The included `CONFIG/config.yaml` uses relative paths and contains no secrets. If you customize it for your machine, keep credentials and private paths out of Git.

## Example Commands

Collect data until stopped manually:

```bash
.venv/bin/python src/data/collect_orderbook_top20.py
```

Merge the latest two daily raw files:

```bash
.venv/bin/python src/data/merge_raw_orderbook.py \
  --last-n 2 \
  --output-path data/raw/orderbook_top20/BTCUSDT_merged_last_2_final_mvp.parquet
```

Validate raw data:

```bash
.venv/bin/python src/data/validate_orderbook.py \
  --input-path data/raw/orderbook_top20/BTCUSDT_merged_last_2_final_mvp.parquet
```

Clean raw data:

```bash
.venv/bin/python src/data/clean_orderbook.py \
  --input-path data/raw/orderbook_top20/BTCUSDT_merged_last_2_final_mvp.parquet
```

Build features:

```bash
.venv/bin/python src/features/build_features.py \
  --input-path data/interim/cleaned_orderbook/BTCUSDT_merged_last_2_final_mvp.parquet
```

Build final MVP targets:

```bash
.venv/bin/python src/targets/build_targets.py \
  --dataset-name merged_last_2_final_mvp \
  --target-mode absolute \
  --horizon-seconds 5 \
  --threshold-abs 0.5 \
  --output-suffix final_mvp
```

Train Logistic Regression:

```bash
.venv/bin/python src/models/train_logistic.py \
  --input-path data/processed/model_dataset/BTCUSDT_model_dataset_merged_last_2_final_mvp_h5_th0p5_final_mvp.parquet
```

Train LightGBM:

```bash
.venv/bin/python src/models/train_lightgbm.py \
  --input-path data/processed/model_dataset/BTCUSDT_model_dataset_merged_last_2_final_mvp_h5_th0p5_final_mvp.parquet
```

Run a validation backtest with fee and slippage costs:

```bash
.venv/bin/python src/backtest/simple_backtest.py \
  --predictions data/processed/model_dataset/BTCUSDT_lightgbm_no_raw_prices_balanced_merged_last_2_final_mvp_h5_th0p5_final_mvp_predictions.parquet \
  --dataset data/processed/model_dataset/BTCUSDT_model_dataset_merged_last_2_final_mvp_h5_th0p5_final_mvp.parquet \
  --split validation \
  --signal-threshold 0.75 \
  --direction-margin 0.00 \
  --neutral-max-probability 0.30 \
  --horizon-seconds 5 \
  --fee-bps 1.0 \
  --slippage-bps 1.0
```

## GitHub Artifact Policy

Generated data, models, reports, logs, and local environments are intentionally excluded from GitHub.

Ignored examples:

- `.venv/`
- `data/raw/`
- `data/interim/`
- `data/processed/`
- `models/`
- `reports/`
- `logs/`
- `*.parquet`
- `*.joblib`

This keeps the repository lightweight and avoids publishing heavy files or local runtime artifacts. To reproduce results, collect fresh Binance WebSocket data and rerun the pipeline.

## Current MVP Result

The final MVP pipeline is functional. The zero-cost validation backtest showed a weak positive signal, but the cost-adjusted backtest was negative because average gross return per trade was below estimated round-trip transaction costs.

This means the project is a valid research baseline, not a deployable strategy.

## Next Improvements

- collect more continuous multi-day data
- run walk-forward validation
- test longer prediction horizons
- calibrate model probabilities
- add more realistic spread, maker/taker, latency, and fill assumptions
- compare tabular baselines with sequence models after the data pipeline is stable
=======
# crypto-order-book-price-prediction
Machine learning project for short-term BTC price movement prediction using limit order book data, order book imbalance, spread, depth, and market microstructure features.
>>>>>>> e2f85a67dde862adf0419664a31f5ec1f315982d
