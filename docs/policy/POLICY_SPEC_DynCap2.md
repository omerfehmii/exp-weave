# POLICY SPEC — DynCap-2 (Prod/Research Candidate)

## Summary
This policy is the current prod/research candidate for cross-sectional trading using:
- L1-w10 rank-only model with 5-seed ensemble
- Cross-sectional scoring (p_cs) with D+U+S strong gating
- Optimizer-based portfolio construction (A2)
- Dynamic Top-10 concentration cap with shock-based hysteresis

Primary goal: stabilize CS edge by controlling concentration/tail dependence and reducing fold dominance.

---

## Model
- **Training objective:** L1-w10 rank-only
- **Ensemble:** 5 seeds (ensemble inference)

---

## Scoring
- **Signal:** `p_cs` (cross-section demean + zscore)
- **Gate:** D+U+S strong

---

## Gate configuration
### Dispersion (D)
- `disp_scale_q_low = 0.15`
- `disp_scale_q_high = 0.40`
- `disp_scale_floor = 0.25`
- `disp_scale_power = 2.0`

### Disagreement (U)
- `disagree_q_low = 0.40`
- `disagree_q_high = 0.70`
- `disagree_hist_window = 200`
- `disagree_scale = 0.30`

### Consistency (S)
- `consistency_min = 0.05`
- `consistency_scale = 0.60`

### Gate combine
- Weighted average combine:
  - `gate_avg_weights = [0.45, 0.35, 0.20]`  # D, U, S
- Gate applied to **gross exposure target** (NOT per-asset weights pre-normalization).

---

## Optimizer (A2 line)
- **Risk aversion:** `lambda = 1.0`
- **Per-name cap:** `pos_cap = 0.04`
- **Turnover budget:** `TB = 0.15`
- **Solver steps:** `steps = 20`
- **Risk window:** `risk_window = 240`
- **Constraint:** dollar-neutral

---

## Dynamic Top-10 concentration cap (DynCap-2)
- **Normal mode:** `top10_cap = 0.15`
- **Risk-off mode:** `top10_cap = 0.12`
- **Shock metric:** p90 shock
- **Shock history window:** `shock_hist_window = 200`
- **Hysteresis thresholds:**
  - Enter risk-off if `shock > q85`  → cap = 0.12
  - Exit risk-off if `shock < q70`   → cap = 0.15

---

## Expected KPI (12-fold WF)
- mean = **0.2601**
- Sharpe = **0.277**
- dominance = **0.301**
- turnover = **0.281**
- CI95 = **[0.1006, 0.4897]**
- top-5 day contribution = **0.789**

---

## Repo artifacts
- **Code commit:** `9d60171` (dynamic cap switch)
- **Results commit:** `9d54f58`
- `opt_A2_dynCap_hyst_p90.csv`
- `opt_A2_dynCap_hyst_p90_metrics.csv`

---

## Optional mini-polish (not required)
If risk-off ratio (~40%) needs to be reduced:
- Hysteresis tweak: `q88/q72`
- Or increase risk-off cap to `0.13`
Run 1–2 confirmation experiments only (avoid overfitting).

---

## Change protocol (parameter updates)
- Require 12-fold walk-forward + block CI; CI lower bound must stay positive
- dominance < 0.32
- top-5 day contribution < 0.85
- median fold Sharpe must not decline

---

## Notes / Rationale
- Rank-only objective is required to produce CS signal; multi-task caused gradient conflict and degraded CS metrics.
- Optimizer reduces variance and improves Sharpe; Top-10 cap reduces tail dependence and fold dominance.
- Dynamic cap with hysteresis further reduces tail dependence while preserving/improving Sharpe.
