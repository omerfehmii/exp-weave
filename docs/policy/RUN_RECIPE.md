# RUN RECIPE — DynCap-2 Policy

This file documents the exact commands used to reproduce the DynCap-2 results.

## Prereqs
- Data: `data/yahoo_panel_big.npz`
- Preds: `runs/ens_w10_preds.npz`
- Config: `runs/backtest_return.yaml`

## Repro snapshot (recorded)
- **Code commit (dyn cap switch):** `9d60171`
- **Results commit (dyn cap results):** `9d54f58`
- **Repo HEAD at last update:** `b317c5a`
- **Data hash (SHA256):** `b8330c6a3543f9c99911971ab7c34bb224b9ffec10b272c06e9ec432d57b14a8`
- **Preds hash (SHA256):** `a81ad8e1b6fbd729ebd77ada08ce731f2f13ab26abdf576f8a27d0a9c16ee607`

If running locally, use your venv python:
```
.venv/bin/python ...
```
On Colab, use `python`.

## 0) Build the return config
```
cat > runs/backtest_return.yaml <<'YAML'
data:
  path: data/yahoo_panel_big.npz
  freq: hourly
  L: 240
  H: 24
  quantiles: [0.1, 0.5, 0.9]
  target_dim: 1
  past_feat_dim: 6
  future_feat_dim: 6
  train_frac: 0.7
  val_frac: 0.15
  step: 24
  scale_x: true
  scale_y: true
  target_mode: return
  target_log_eps: 1.0e-06
YAML
```

## 1) Baseline (heuristic, static cap)
```
python scripts/mu_value_weighted_backtest.py \
  --config runs/backtest_return.yaml \
  --preds runs/ens_w10_preds.npz \
  --h 24 --use_cs --ret_cs \
  --disp_metric std --disp_hist_window 200 \
  --disp_scale_q_low 0.15 --disp_scale_q_high 0.40 \
  --disp_scale_floor 0.25 --disp_scale_power 2.0 \
  --disagree_q_low 0.4 --disagree_q_high 0.7 \
  --disagree_hist_window 200 --disagree_scale 0.3 \
  --consistency_min 0.05 --consistency_scale 0.6 \
  --gate_combine avg --gate_avg_weights "0.45,0.35,0.20" \
  --ema_halflife_min 2 --ema_halflife_max 8 \
  --ema_disp_lo 0.02 --ema_disp_hi 0.10 \
  --min_hold 2 --turnover_budget 0.15 \
  --pos_cap 0.05 --gross_target 1.0 \
  --walk_folds 12 \
  --out_csv runs/opt_base_tb15_12fold.csv \
  --out_metrics runs/opt_base_tb15_12fold_metrics.csv
```

## 2) Optimizer A2 (static cap)
```
python scripts/mu_value_weighted_backtest.py \
  --config runs/backtest_return.yaml \
  --preds runs/ens_w10_preds.npz \
  --h 24 --use_cs --ret_cs \
  --disp_metric std --disp_hist_window 200 \
  --disp_scale_q_low 0.15 --disp_scale_q_high 0.40 \
  --disp_scale_floor 0.25 --disp_scale_power 2.0 \
  --disagree_q_low 0.4 --disagree_q_high 0.7 \
  --disagree_hist_window 200 --disagree_scale 0.3 \
  --consistency_min 0.05 --consistency_scale 0.6 \
  --gate_combine avg --gate_avg_weights "0.45,0.35,0.20" \
  --ema_halflife_min 2 --ema_halflife_max 8 \
  --ema_disp_lo 0.02 --ema_disp_hi 0.10 \
  --min_hold 2 --turnover_budget 0.15 \
  --pos_cap 0.04 --gross_target 1.0 \
  --optimize --opt_lambda 1.0 --opt_kappa 0.0 \
  --opt_steps 20 --opt_risk_window 240 --opt_dollar_neutral \
  --walk_folds 12 \
  --out_csv runs/opt_A2_lam1p0_cap0p04_tb15_12fold.csv \
  --out_metrics runs/opt_A2_lam1p0_cap0p04_tb15_12fold_metrics.csv
```

## 3) Top-10 cap sweep (static)
```
python scripts/mu_value_weighted_backtest.py \
  --config runs/backtest_return.yaml \
  --preds runs/ens_w10_preds.npz \
  --h 24 --use_cs --ret_cs \
  --disp_metric std --disp_hist_window 200 \
  --disp_scale_q_low 0.15 --disp_scale_q_high 0.40 \
  --disp_scale_floor 0.25 --disp_scale_power 2.0 \
  --disagree_q_low 0.4 --disagree_q_high 0.7 \
  --disagree_hist_window 200 --disagree_scale 0.3 \
  --consistency_min 0.05 --consistency_scale 0.6 \
  --gate_combine avg --gate_avg_weights "0.45,0.35,0.20" \
  --ema_halflife_min 2 --ema_halflife_max 8 \
  --ema_disp_lo 0.02 --ema_disp_hi 0.10 \
  --min_hold 2 --turnover_budget 0.15 \
  --pos_cap 0.04 --gross_target 1.0 \
  --optimize --opt_lambda 1.0 --opt_kappa 0.0 \
  --opt_steps 20 --opt_risk_window 240 --opt_dollar_neutral \
  --topn_cap 0.15 --topn_n 10 \
  --walk_folds 12 \
  --out_csv runs/opt_A2_top10cap015.csv \
  --out_metrics runs/opt_A2_top10cap015_metrics.csv

python scripts/mu_value_weighted_backtest.py \
  --config runs/backtest_return.yaml \
  --preds runs/ens_w10_preds.npz \
  --h 24 --use_cs --ret_cs \
  --disp_metric std --disp_hist_window 200 \
  --disp_scale_q_low 0.15 --disp_scale_q_high 0.40 \
  --disp_scale_floor 0.25 --disp_scale_power 2.0 \
  --disagree_q_low 0.4 --disagree_q_high 0.7 \
  --disagree_hist_window 200 --disagree_scale 0.3 \
  --consistency_min 0.05 --consistency_scale 0.6 \
  --gate_combine avg --gate_avg_weights "0.45,0.35,0.20" \
  --ema_halflife_min 2 --ema_halflife_max 8 \
  --ema_disp_lo 0.02 --ema_disp_hi 0.10 \
  --min_hold 2 --turnover_budget 0.15 \
  --pos_cap 0.04 --gross_target 1.0 \
  --optimize --opt_lambda 1.0 --opt_kappa 0.0 \
  --opt_steps 20 --opt_risk_window 240 --opt_dollar_neutral \
  --topn_cap 0.12 --topn_n 10 \
  --walk_folds 12 \
  --out_csv runs/opt_A2_top10cap012.csv \
  --out_metrics runs/opt_A2_top10cap012_metrics.csv
```

## 4) Dynamic cap (DynCap-2, hysteresis, p90 shock)
```
python scripts/mu_value_weighted_backtest.py \
  --config runs/backtest_return.yaml \
  --preds runs/ens_w10_preds.npz \
  --h 24 --use_cs --ret_cs \
  --disp_metric std --disp_hist_window 200 \
  --disp_scale_q_low 0.15 --disp_scale_q_high 0.40 \
  --disp_scale_floor 0.25 --disp_scale_power 2.0 \
  --disagree_q_low 0.4 --disagree_q_high 0.7 \
  --disagree_hist_window 200 --disagree_scale 0.3 \
  --consistency_min 0.05 --consistency_scale 0.6 \
  --gate_combine avg --gate_avg_weights "0.45,0.35,0.20" \
  --ema_halflife_min 2 --ema_halflife_max 8 \
  --ema_disp_lo 0.02 --ema_disp_hi 0.10 \
  --min_hold 2 --turnover_budget 0.15 \
  --pos_cap 0.04 --gross_target 1.0 \
  --optimize --opt_lambda 1.0 --opt_kappa 0.0 \
  --opt_steps 20 --opt_risk_window 240 --opt_dollar_neutral \
  --topn_cap 0.15 --topn_cap_low 0.12 --topn_n 10 \
  --topn_dyn_q_hi 0.85 --topn_dyn_q_lo 0.70 \
  --shock_metric p90 --shock_hist_window 200 \
  --walk_folds 12 \
  --out_csv runs/opt_A2_dynCap_hyst_p90.csv \
  --out_metrics runs/opt_A2_dynCap_hyst_p90_metrics.csv
```

## 5) Block bootstrap CI
```
python scripts/bootstrap_ci.py --pnl_csv runs/opt_A2_dynCap_hyst_p90.csv --method block --block_len 5
```

## Repro control checklist
- `git rev-parse HEAD` == `b317c5a`
- data SHA256 matches
- preds SHA256 matches
- output CSV filenames match
- KPI tolerance within range (e.g., Sharpe ±0.02, dominance ±0.02)
- if mismatch: check random seed / BLAS / solver determinism

## Notes
- Reported Sharpe is **per-step** (per 24h bar), not annualized.
- For annualized Sharpe (daily bars): `Sharpe_annual ≈ Sharpe_step * sqrt(252)`.
