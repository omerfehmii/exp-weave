# POLICY MONITORING — DynCap-2

## Purpose
Operational checklist for monitoring the DynCap-2 policy in production.

---

## Core metrics (log daily/weekly)
- **Sharpe (rolling)**: per-step and annualized
- **Mean PnL (rolling)**
- **Dominance (rolling)**: top fold / total PnL proxy
- **Top-5 day contribution** (rolling)
- **Turnover** (mean + p90)
- **Top-10 |w| sum** (mean + p90)
- **HHI (sum w^2)** (mean + p90)
- **Risk-off ratio** (fraction of time in cap=0.12)
- **cap_used_mean / cap_used_p90**
- **gate_scale_mean** and **shock_scale_mean**

---

## Suggested alerts (heuristics)
- **Dominance > 0.35** for 2+ consecutive windows
- **Top-5 contribution > 0.90** (tail dependence returning)
- **Turnover** outside target band (0.15–0.35)
- **Risk-off ratio > 0.60** (over-conservative regime)
- **CI lower bound ≤ 0** for 2+ rolling windows
- **Top-10 |w| sum p90 > 0.20** (concentration creep)

---

## Suggested actions
1) **If dominance / tail rises**
   - Tighten cap: switch to `top10_cap_low = 0.12` more frequently
   - Raise shock thresholds slightly (q88/q72)
2) **If Sharpe drops while turnover stable**
   - Reduce risk-off ratio (increase risk-off cap to 0.13)
3) **If turnover spikes**
   - Increase TB modestly or tighten min_hold
4) **If risk-off ratio too high**
   - Increase `topn_dyn_q_hi` or reduce shock sensitivity

---

## Sanity checks
- Ensure **cap_used_mean** is between 0.12–0.15
- Ensure **pca_exposure_abs** is near zero only when PCA-neutral is enabled
- Verify shock metric (p90) is populated and finite

---

## Notes
- Sharpe values in logs are per-step (24h). Report annualized separately if needed.
- Dominance is sensitive to small sample windows; interpret jointly with CI.
