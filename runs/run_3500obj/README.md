# Run: 3,500-Object LOOO Evaluation

**Date:** 2026-03-16 / 2026-03-17
**Purpose:** Scale LOOO methodology to a larger sample to derive empirical per-observatory,
per-catalog sigma estimates. Builds on the 100-object pilot (`runs/pilot_100obj/`).

---

## Data

| File | Path |
|------|------|
| Observations | `data/looo_sample_3500/mpc_observations.parquet` |
| Orbits | `data/looo_sample_3500/mpc_orbits.parquet` |
| Fetch metadata | `data/looo_sample_3500/sample_metadata.json` |

**3,500 objects, 1,383,049 observations, deterministic sample (seed=42).**

Fetch command:
```bash
python scripts/01_fetch_mpc_sample.py \
    --project moeyens-thor-dev \
    --dataset-id mpc_sbn_aurora \
    --views-dataset-id mpc_sbn_aurora_views \
    --sample-type numbered \
    --n-objects 3500 \
    --min-obs 20 \
    --output-dir data/looo_sample_3500 \
    --seed 42
```

> **Reproducibility:** Uses `FARM_FINGERPRINT(CONCAT(provid, CAST(42 AS STRING)))` for
> deterministic ordering. Re-running the fetch script with the same arguments on the same
> BigQuery snapshot will produce identical output. The exact provids are listed in
> `sample_metadata.json`.

---

## LOOO Run

**Run ID:** `20260316T190152Z`
**Output:** `data/looo_results/20260316T190152Z/`
**Wall clock:** ~8h15m (12:01–20:17 UTC-8, 2026-03-16), 6 cores
**Config:** `data/looo_results/20260316T190152Z/run_config.json`

```bash
python scripts/02_run_looo.py \
    --input-dir data/looo_sample_3500 \
    --output-dir data/looo_results \
    --run-id 20260316T190152Z \
    --propagator assist \
    --sigma-model veres2017 \
    --min-obs-remaining 6 \
    --min-arc-length 7.0 \
    --max-held-out-fraction 0.8 \
    --max-processes 6
```

**Result:** 3,386 of 3,500 objects produced results (114 failed eligibility or DC
non-convergence). 1,211,865 held-out observations across 719 observatories.

> **Note on interruption:** The run was interrupted by a laptop shutdown at ~3,495/3,500
> objects and resumed via checkpointing. All per-object results are preserved in
> `data/looo_results/20260316T190152Z/checkpoints/`.

---

## Analysis

**Output:** `data/looo_analysis/veres2017_3500obj/`

```bash
python scripts/03_analyze.py \
    --input-dir data/looo_results/20260316T190152Z \
    --output-dir data/looo_analysis \
    --run-id veres2017_3500obj \
    --max-chi2 100.0 \
    --top-n 80
```

**Key outputs:**
- `observatory_stats.parquet` — per-observatory bias, RMS, chi2/obs
- `catalog_stats.parquet` — per-(observatory, catalog) breakdown
- `observatory_summary.txt` — human-readable table

---

## Key Findings

### Scale
- **3,386/3,500 objects** produced usable LOOO results
- **1,211,865 held-out observations** across **719 observatories**
- **~8.25 hours** wall clock on 6 cores (~1.4 core-min/object, below pilot estimate of 1.8)

### Well-calibrated stations (chi2/obs 0.7–2.0)
F52, M22, O18, W68, 691, R17, 644, T09, 705, 461, W91

### Systematic biases (>0.2" in RA or Dec)
| Station | Bias RA" | Bias Dec" | Notes |
|---------|----------|-----------|-------|
| 704 | +0.219 | +0.428 | USNOA2 catalog, well-known |
| 699 | +0.155 | +0.361 | USNOB1/USNOA2 mix |
| 703 | -0.095 | +0.122 | Mixed catalogs |
| W84 | +0.138 | +0.005 | 2MASS catalog |
| 809 | +0.411 | +0.101 | Large RA bias |

### Stations requiring investigation
| Station | chi2/obs | Issue |
|---------|----------|-------|
| F51 | 177.9 | RMS_RA=2.4" — large RA scatter across 2MASS/Gaia1/Gaia3E |
| G96 | 14.2 | RMS_RA=1.26" vs RMS_Dec=0.32" — asymmetric scatter |
| N86 | ~3×10⁸ | Numerical overflow — flag and exclude |
| 675 | 43.9 | RMS_RA=5.1" — very large scatter |
| M01 | 27.2 | Needs catalog breakdown |
| Y00 | 25.4 | RMS ~1.1" both axes |

### Notable catalog findings
- **F51/2MASS**: chi2=240 vs **F51/Gaia3E**: chi2=38 — Gaia reduces scatter substantially
  but F51 RA scatter persists across all catalogs (likely a systematic in the telescope/pipeline)
- **G96/UCAC4**: chi2=2.2 vs **G96/Gaia2**: chi2=9.9 — UCAC4 appears better calibrated
  for G96 than Gaia2 (unexpected; may reflect epoch/proper-motion effects)
- **X05/Gaia3**: chi2=12 — inflated despite modern catalog
- **W84/nan** (no catalog recorded): chi2=0.04, RMS~0.11" — the uncatalogued observations
  are the best-behaved group; likely older high-quality hand-measured positions

### Sigma model note
Veres 2017 sigmas were used as fill-ins when `rmsra`/`rmsdec` were missing. The chi2/obs
values above reflect calibration *relative to Veres*. To derive independent empirical sigmas,
use the raw RMS residual columns (`rms_ra_arcsec`, `rms_dec_arcsec`) from
`observatory_stats.parquet` and `catalog_stats.parquet` — these are independent of the
fill-in sigma choice.

### Known biases: propagator drift over long time baselines

Even ASSIST (N-body with relativistic corrections) accumulates position errors when
propagating over years-to-decades timescales.  This introduces a systematic bias into sigma
estimates that cannot be fully removed by filtering alone:

- **Non-gravitational forces** (Yarkovsky A2, radiation pressure) are not modeled.  For
  objects with significant acceleration, the propagated position can be wrong by arcseconds
  over a few years.  This position error gets absorbed into the residuals of whichever station
  happened to observe at that epoch, inflating that station's apparent scatter.
- **Chaotic close-approach orbits** amplify sensitivity to initial-condition uncertainty in
  ways that no deterministic propagator can resolve.
- **Epoch asymmetry**: stations whose observations are predominantly old (e.g., 691 with
  USNOA2 observations from the 1990s) will appear to have larger scatter partly because the
  propagated reference orbit is less accurate at those epochs, not purely because of
  astrometric error.

**Mitigations applied in this run:**
- `--object-weighted` flag: per-object statistics are averaged first (each object weighted
  equally), preventing a single pathological object from dominating station statistics.
- `--max-object-chi2 50`: objects whose mean chi2 across all stations exceeds 50 are excluded
  as likely non-gravitational-force candidates (comets, high-Yarkovsky NEOs).
- 47 objects were filtered this way, reducing the pool from 3,386 to ~3,339 for
  object-weighted statistics.

These mitigations reduce but do not eliminate the bias.  Stations with old observations
should be interpreted as lower bounds on astrometric quality; the true scatter may be
somewhat smaller than reported.

---

## Environment

- `adam_core` 0.5.5
- `adam-assist` (ASSIST/REBOUND N-body propagator)
- Python 3.12, PDM
- GCP project: `moeyens-thor-dev`
- BQ datasets: `mpc_sbn_aurora`, `mpc_sbn_aurora_views`
- Branch: `kk/claude/mpc-eval`
