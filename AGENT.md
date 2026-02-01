# Exp-Weave Agent Notes

## Purpose
- Cross-sectional (CS) forecasting + policy backtest pipeline over panel time series.
- Primary decision target: **Sharpe_gross** on walk‑forward (WF) with **real-time (observed_only=false)**.
- Horizon/label changes are the main lever; policy tweaks are secondary.

## Data & Environment
- Dataset: `data/yahoo_panel_big.npz`
- Base configs live in `configs/`
- Default Python: `.venv/bin/python` if present, else `python3`.
- GPU training expects `training.device: cuda` (for Colab/L4).

## Protocol Invariants (keep fixed unless explicitly changed)
- `observed_only: false`
- `future_obs_mode: nearest` for real‑time horizon testing.
- Universe filter **fold_train_only** (`universe_active_end = train_end_t` per fold).
- Keep `split_purge` / `split_embargo` consistent across runs.
- Keep `min_time_ic_count`, `min_future_obs`, `min_past_obs` consistent for a sweep.

## Key Scripts
- **Transformer WF**: `scripts/run_walk_forward.py`
  - Produces per‑fold artifacts + `summary.csv`, `wf_summary.json`.
- **Baseline WF** (Ridge/ElasticNet/GBDT): `scripts/run_walk_forward_baseline.py`
  - Same protocol; model only changes.
- **Policy backtest**: `scripts/mu_value_weighted_backtest.py`
  - Outputs `metrics.csv` with `pnl`, `gross`, `vol_mkt`, `turnover`.

## Configs (nearest)
- `configs/CS_L1_W10_OOS_REAL6.yaml`
- `configs/CS_L1_W10_OOS_REAL12.yaml`
- `configs/CS_L1_W10_OOS_REAL24_NEAREST.yaml`

## Horizon Sweep Template (Transformer)
```
python scripts/run_walk_forward.py \
  --base_config <config.yaml> \
  --policy_config configs/backtest_return_dynCap2.yaml \
  --out_dir runs/<tag> \
  --fold_size 64 \
  --n_folds 2 \
  --seeds 7,13 \
  --skip_existing
```

## Baseline Sweep Template (Ridge / GBDT)
```
python scripts/run_walk_forward_baseline.py \
  --base_config <config.yaml> \
  --policy_config configs/backtest_return_dynCap2.yaml \
  --out_dir runs/<tag> \
  --fold_size 64 \
  --n_folds 2 \
  --seeds 7,13 \
  --model ridge|gbdt \
  --skip_existing
```
- GBDT backend: LightGBM if available; else XGBoost.

## Outputs to Report
- `summary.csv`:
  - `sharpe_gross`, `turnover_mean`, `dominance` (top5 share),
  - coverage (`ic_count_mean/p10`, `active_count_mean/p10`)
  - `score_mean/score_std` (baseline only).
- `wf_summary.json`:
  - `coverage_summary`, `time_units.effective_horizon_hours`.

## Decision Rules (typical)
- **GO** if majority of folds `Sharpe_gross > 0` **and** aggregate not negative.
- If baselines also negative → likely weak signal at that horizon/label.
- If baselines positive but transformer negative → training/objective issue.
