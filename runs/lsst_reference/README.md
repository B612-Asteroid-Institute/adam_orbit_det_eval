# Run: LSST Reference-Orbit Evaluation (Pilot)

**Date:** 2026-03-17
**Purpose:** Calibrate other-observatory sigma estimates against a reference orbit derived
exclusively from X05 (Rubin Observatory / LSST) observations.  This is the inverse of LOOO:
instead of holding one station out and refitting with the rest, the orbit is fit on X05 only
and every other station is evaluated against it.

---

## Methodology

### Reference-orbit pipeline (`scripts/04_run_lsst_reference.py`)

1. For each object, run DC using **only X05 observations**.
2. Predict sky-plane positions at every non-X05 station's observation times using the
   resulting orbit.
3. Record residuals with the same `LOOOResult` schema so `03_analyze.py` works unchanged.

Column semantics are remapped relative to LOOO:

| Column | Meaning in this context |
|--------|------------------------|
| `n_obs_remaining` | X05 observations used for the DC fit |
| `arc_length_remaining_days` | arc length of the X05-only observation set |
| `hold_in_reduced_chi2` | reduced chi2 of the X05-only fit (same for all evaluated stations on an object) |
| `held_out_fraction` | n_stn_obs / (n_ref_obs + n_stn_obs) |

---

## Data

| File | Path |
|------|------|
| Observations | `data/looo_sample_lsst100/mpc_observations.parquet` |
| Orbits | `data/looo_sample_lsst100/mpc_orbits.parquet` |
| Fetch metadata | `data/looo_sample_lsst100/sample_metadata.json` |

**65 objects, 30,509 total observations.**
X05 observation range: 240–506 per object (top objects by X05 count, ≥10 other-station obs).

Fetch command:
```bash
python scripts/01_fetch_lsst_sample.py \
    --project moeyens-thor-dev \
    --dataset-id mpc_sbn_aurora \
    --views-dataset-id mpc_sbn_aurora_views \
    --reference-stn X05 \
    --n-objects 100 \
    --min-ref-obs 20 \
    --min-other-obs 10 \
    --output-dir data/looo_sample_lsst100
```

> Only 65 of 100 requested objects passed both the `--min-ref-obs 20` and `--min-other-obs 10`
> filters, reflecting LSST's recent deployment and the limited cross-observatory follow-up
> of LSST-selected targets so far.

---

## Run

**Run ID:** `lsst_ref_assist`
**Output:** `data/lsst_reference_results/lsst_ref_assist/`
**Wall clock:** ~15s (65 objects × 1 DC fit each, 6 cores)
**Config:** `data/lsst_reference_results/lsst_ref_assist/run_config.json`

```bash
python scripts/04_run_lsst_reference.py \
    --input-dir data/looo_sample_lsst100 \
    --output-dir data/lsst_reference_results \
    --propagator assist \
    --sigma-model veres2017 \
    --reference-stn X05 \
    --min-ref-obs 20 \
    --max-processes 6 \
    --run-id lsst_ref_assist
```

**Result:** 65/65 objects processed, 2,890 evaluated observation rows across 35 observatories.

---

## Analysis

**Output:** `data/lsst_reference_analysis/lsst_ref_assist/`

```bash
python scripts/03_analyze.py \
    --input-dir data/lsst_reference_results/lsst_ref_assist \
    --output-dir data/lsst_reference_analysis \
    --run-id lsst_ref_assist \
    --max-chi2 100.0 \
    --min-obs-per-stn 3 \
    --top-n 60
```

> `data/lsst_reference_results/lsst_ref_assist/looo_results.parquet` is a symlink to
> `reference_orbit_results.parquet` to satisfy `03_analyze.py`'s expected filename.

---

## Key Findings

### X05-only DC fit quality

The reference orbit fits themselves are excellent:

| Metric | Value |
|--------|-------|
| Median hold-in reduced chi2 | **0.84** |
| 90th percentile | 1.11 |
| Maximum | 1.30 |
| X05 arc length | **6–12 days** |
| X05 obs per object | 240–506 |

### Critical issue: timing mismatch

**The X05 arcs are only 6–12 days long.** LSST began operations in late 2025; even the most
densely-observed objects have arcs of less than two weeks. The other stations in this sample
have historical observations spanning 2013–2022 — years before these arcs.

Contemporaneous coverage (observations within ±30 days of the X05 window):

| Station | Total obs | Within ±30 days | Frac |
|---------|-----------|-----------------|------|
| F51 | 1,227 | 63 | 5% |
| F52 | 402 | 33 | 8% |
| G96 | 563 | 12 | 2% |
| V00 | 61 | 8 | 13% |
| O18 | 23 | 4 | 17% |
| 691, 703, W84, … | many | **0** | 0% |

Projecting a 12-day arc back 3,000–8,000 days produces huge position errors even with an
accurate N-body propagator.  This is **not a calibration failure** — it is an orbit
extrapolation error caused by insufficient arc length.

### Results: contemporaneous stations only

The one station with meaningful contemporaneous coverage shows the method works correctly:

| Station | N obs | chi2/obs | RMS_RA" | RMS_Dec" | Notes |
|---------|-------|----------|---------|----------|-------|
| O18 | 23 | **1.14** | 0.15 | 0.11 | Contemporaneous; well-calibrated |
| T08 | 8 | 2.42 | 1.1 | 0.36 | Single object |
| F52/Gaia3E | 175 | 5,197 | 10.5 | 2.6 | Mix of contemp. + old obs |
| F51 | 1,227 | 314,996 | 106 | 27 | Dominated by 2013–2022 obs |

O18's chi2 ≈ 1.1 confirms the reference-orbit approach is working as designed when
observations are contemporaneous.

### Why the overall results are not usable for calibration

A 6–12 day orbit arc cannot reliably extrapolate to observations from prior years.
Residuals from non-contemporaneous observations reflect orbit extrapolation uncertainty,
not observing-station astrometric error.  Reporting sigma estimates from this run would
misattribute propagation error as station-level calibration bias.

---

## Comparison with LOOO (3,500-object run)

In the LOOO run, each orbit is fit on **all stations except the held-out one**, giving a
well-constrained, multi-year arc for extrapolation.  That approach is robust to the
timing mismatch because the hold-in set always includes many contemporaneous and historical
observations from other stations.

The reference-orbit approach is fundamentally more sensitive to arc length: a short X05 arc
cannot constrain orbital elements well enough to predict positions at remote epochs.

| | LOOO (3,500 obj) | Ref-orbit (65 obj, pilot) |
|--|------------------|--------------------------|
| X05 chi2/obs | ~10 | N/A (X05 is the reference) |
| F51 chi2/obs | 1.65 (obj-weighted) | 314,996 (arc extrap.) |
| Arc used for fit | Full MPC arc | 6–12 day X05 arc |
| Cross-epoch validity | Yes | No (short arc) |

---

## Known Biases in the Overall Sigma Methodology

The following biases apply to both the LOOO and reference-orbit pipelines and should be
considered when interpreting sigma estimates from any run in this project.

### 1. Propagator drift over long time baselines

Even ASSIST (N-body with relativistic corrections) accumulates position errors when
propagating over years-to-decades timescales:

- **Close planetary encounters** introduce sensitivity to initial conditions and ephemeris
  uncertainty that no deterministic propagator can fully capture.
- **Non-gravitational forces** (radiation pressure, Yarkovsky A2 drift) are not modeled by
  default.  For objects with significant Yarkovsky acceleration, the propagated position can
  be wrong by arcseconds over a few years — and this error will be absorbed into the
  residuals of whichever station happened to observe at that epoch.
- **Impact**: stations with predominantly old observations will appear to have larger scatter
  than stations with recent observations, even if their astrometry is equally good.
  This creates a spurious apparent bias against older data.
- **Mitigation**: the `--max-object-chi2` filter in `03_analyze.py` (e.g. `--max-object-chi2
  50`) removes objects where the mean chi2 across all stations is elevated — a proxy for
  objects with poorly-modeled non-gravitational forces or chaotic orbits.  The
  `--object-weighted` flag prevents a single problematic object from dominating station
  statistics.  These are partial mitigations; propagator drift bias cannot be fully
  eliminated without a priori non-gravitational force estimates.

### 2. Sample selection bias (newly-discovered vs. well-known objects)

Objects selected for many observations from a specific station are not representative of
the broader population.  In particular, objects with many LSST observations are by
definition recently discovered or recently re-observed — their short arcs amplify
propagator sensitivity.

### 3. Veres 2017 sigma circularity

Chi2 values computed using Veres 2017 fill-ins are circular for stations with missing
reported uncertainties.  Use the raw `rms_ra_arcsec` / `rms_dec_arcsec` columns as the
primary unbiased sigma estimate.

---

## When to Re-run

Re-run this pipeline when X05 has built up arcs of **≥3 months** on objects that also have
contemporaneous observations from F51, G96, F52, and other major surveys.  The expected
timeline is mid-to-late 2026.

The full infrastructure is in place:

| Script | Purpose |
|--------|---------|
| `scripts/01_fetch_lsst_sample.py` | Fetch top-N X05 objects from BigQuery |
| `scripts/04_run_lsst_reference.py` | Run reference-orbit evaluation pipeline |
| `scripts/03_analyze.py` | Compute per-observatory / per-catalog statistics |
| `scripts/05_generate_report.py` | Generate shareable HTML report + sigma CSV |

Suggested re-run command (no code changes needed):
```bash
python scripts/01_fetch_lsst_sample.py \
    --reference-stn X05 --n-objects 100 --min-ref-obs 50 \
    --output-dir data/looo_sample_lsst100_v2

python scripts/04_run_lsst_reference.py \
    --input-dir data/looo_sample_lsst100_v2 \
    --output-dir data/lsst_reference_results \
    --propagator assist --sigma-model veres2017 \
    --reference-stn X05 --min-ref-obs 50 \
    --max-processes 6 --run-id lsst_ref_v2
```

---

## Environment

- `adam_core` 0.5.5
- `adam-assist` (ASSIST/REBOUND N-body propagator)
- Python 3.12, PDM
- GCP project: `moeyens-thor-dev`
- BQ datasets: `mpc_sbn_aurora`, `mpc_sbn_aurora_views`
- Branch: `kk/claude/mpc-eval`
