# Pilot Run: 100-Object LOOO Evaluation

**Date:** 2026-03-13
**Purpose:** Validate the LOOO methodology and sigma calibration approach on a small sample.

---

## Data

| File | Path |
|------|------|
| Observations | `data/looo_sample/mpc_observations.parquet` |
| Orbits | `data/looo_sample/mpc_orbits.parquet` |
| Fetch metadata | `data/looo_sample/sample_metadata.json` |

**100 objects, 61,812 observations, 731 unique observatories.**

> **Note on reproducibility:** The sample was fetched via an ad-hoc BigQuery query using
> `ORDER BY RAND()` (non-deterministic). The saved parquet files *are* the reproducible
> artifact — re-running the fetch will produce a different random sample. The exact provids
> are listed in `sample_metadata.json`.

---

## Runs

### Run 1 — Raw MPC sigmas (constant fallback)

**Output:** `data/looo_results_100/`
**Command equivalent:**
```bash
python scripts/02_run_looo.py \
    --input-dir data/looo_sample \
    --output-dir data/looo_results_100 \
    --run-id pilot_const_sigma \
    --propagator assist \
    --sigma-model const \
    --min-obs-remaining 6 \
    --min-arc-length 7.0 \
    --max-held-out-fraction 0.8 \
    --max-processes 6
```
**Config:** `data/looo_results_100/run_config.json`
**Result:** 53,864 held-out observations. chi2/obs highly inflated (10–100×) for most stations due to missing sigmas filled with a near-zero constant.

### Run 2 — Veres 2017 sigma model ✓ (canonical result)

**Output:** `data/looo_results/20260314T055433Z/`
**Command:**
```bash
python scripts/02_run_looo.py \
    --input-dir data/looo_sample \
    --output-dir data/looo_results \
    --propagator assist \
    --sigma-model veres2017 \
    --min-obs-remaining 6 \
    --min-arc-length 7.0 \
    --max-held-out-fraction 0.8 \
    --max-processes 6
```
**Config:** `data/looo_results/20260314T055433Z/run_config.json`
**Result:** 53,864 held-out observations. chi2/obs in 1–5 range for most modern survey stations. 92 of 100 objects produced results (8 failed eligibility or DC non-convergence).

---

## Analysis

### Analysis of Run 2 (Veres sigma model)

**Output:** `data/looo_analysis/veres2017_100obj/`
**Command:**
```bash
python scripts/03_analyze.py \
    --input-dir data/looo_results/20260314T055433Z \
    --output-dir data/looo_analysis \
    --run-id veres2017_100obj \
    --max-chi2 100.0 \
    --top-n 60
```

**Key outputs:**
- `observatory_stats.parquet` — per-observatory bias, RMS, chi2/obs
- `catalog_stats.parquet` — per-(observatory, catalog) breakdown
- `observatory_summary.txt` — human-readable table

---

## Key Findings

- **92/100 objects** produced usable LOOO results
- **Well-calibrated stations** (chi2/obs 1–3): M22, W68, R17, C40, J51, T05, T08, G96, F52
- **Systematic biases** (>0.2"): 704 (+0.32"/+0.32", USNOA2), 699 (+0.24"/+0.22"), 926 (+1.09" RA)
- **Outlier stations** needing further investigation: 557, I41, 926, J95
- Veres 2017 sigma model reduced chi2 by 10–30× for stations with missing reported uncertainties
- Methodology validated; ready to scale to larger sample

---

## Environment

- `adam_core` 0.5.5
- `adam-assist` (ASSIST/REBOUND N-body propagator)
- Python 3.12, PDM
- GCP project: `moeyens-thor-dev`
- BQ datasets: `mpc_sbn_aurora`, `mpc_sbn_aurora_views`
- Branch: `kk/claude/mpc-eval`
