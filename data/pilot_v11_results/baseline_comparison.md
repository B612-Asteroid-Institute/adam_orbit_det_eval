# Pilot v11 vs 3,500-object reference: baseline-correctness comparison

**Date:** 2026-05-01
**Bead:** `beads_agent_setup-ojo` (does NOT close on this run — see "Verdict")
**Branch:** `kk/mpc-scale-bias-catalog`
**Pilot output:** `data/pilot_v11_results/`
**Reference catalog:** `data/bias_catalog/3500obj/`
**Pilot catalog (built for this comparison):** `data/bias_catalog/pilot_v11/`

## How the comparison was set up

The original briefing instructed loading `pilot_v11/observatory_stats.parquet`
and `bias_catalog/3500obj/observatory_stats.parquet` and checking CI overlap.
Two issues with that as written:

1. The reference dir has no `observatory_stats.parquet`; it has
   `bias_table.parquet` (built by `scripts/17_generate_bias_table.py`).
2. The pilot's `observatory_stats.parquet` (from
   `scripts/13_collect_cloud_results.py` → `compute_observatory_stats(merged_results)`)
   uses **observation-weighted** aggregation with **no CIs**, whereas the
   reference uses **object-weighted** aggregation with 2,000-resample bootstrap CIs.
   These products are not directly comparable.

To get apples-to-apples comparison, the canonical bias-table builder was rerun
on the pilot output:

```
.venv/bin/python scripts/17_generate_bias_table.py \
    --looo-results data/pilot_v11_results/merged_looo_results.parquet \
    --output-dir data/bias_catalog/pilot_v11 \
    --validate-anchors
```

Same script, same defaults, same bootstrap (n=2000, seed=42), same filters
(`max_chi2=100`, `max_object_chi2=50`, `min_obs_per_group=10`,
`min_objects_per_group=3`) used to produce the reference catalog.
`--observations` was not passed (only powers `sigma_model_source` and
`obs_epoch_*` metadata fields; not used in the anchor comparison).

The pilot bias table contains **29 observatory-level rows + 2 (obs, program_code) rows**,
versus 246 observatory rows in the reference.

## Pilot v11 vs reference: sample-size context

The pilot was supposed to enable a baseline-correctness check against the
3,500-object reference, but the two object sets are largely disjoint:

| | pilot v11 | reference 3500 |
|---|---:|---:|
| unique objects | 1,349 | 3,500 |
| LOOO rows | 44,236 | (input parquet 1.38M obs) |
| observatory rows in bias table | 29 | 246 |

Object-set overlap:
- intersection: **976** objects (~28% of reference, ~72% of pilot)
- pilot-only: 373
- reference-only: 2,524

The bead description says pilot input is "existing 3,500-object sample at
`data/looo_sample_3500/`", but the pilot output objects do not match that
sample. This is one of two reasons the comparison cannot meet the original
exit criteria.

## Anchor delta table (13 anchors)

Bias values are observatory-level (program_code null), object-weighted means,
arcsec. CIs are 95% bootstrap. "Sign" is OK if the pilot and reference biases
have the same sign and both are ≥ 0.02" in magnitude (sign of a near-zero bias
is not informative; marked NZ). "CI ovlp" is whether the pilot's [low, high]
overlaps the reference's.

| stn | present | ref RA | pilot RA | Δ RA | sign RA | CI RA | ref Dec | pilot Dec | Δ Dec | sign Dec | CI Dec | pilot n_obj | ref n_obj |
|-----|---|---:|---:|---:|:---:|:---:|---:|---:|---:|:---:|:---:|---:|---:|
| 704 | both | +0.212 | +0.302 | +0.090 | OK | YES | +0.434 | +0.691 | +0.258 | OK | YES | 3 | 917 |
| 699 | **PILOT-MISSING** | +0.154 | — | — | — | — | +0.355 | — | — | — | — | 0 | 611 |
| 703 | both | -0.114 | -0.319 | -0.205 | OK | YES | +0.145 | +0.337 | +0.192 | OK | YES | 18 | 1,300 |
| W84 | both | +0.076 | +0.129 | +0.053 | OK | YES | +0.070 | +0.079 | +0.009 | OK | YES | 68 | 702 |
| 809 | **PILOT-MISSING** | +0.257 | — | — | — | — | +0.112 | — | — | — | — | 0 | 75 |
| F52 | both | +0.000 | -0.011 | -0.011 | NZ | YES | +0.026 | +0.017 | -0.009 | NZ | YES | 376 | 3,020 |
| M22 | **PILOT-MISSING** | +0.001 | — | — | — | — | +0.043 | — | — | — | — | 1 | 495 |
| R17 | **PILOT-MISSING** | -0.022 | — | — | — | — | +0.045 | — | — | — | — | 2 | 203 |
| T09 | both | -0.064 | +0.370 | +0.434 | **FAIL** | **NO** | +0.075 | -0.071 | -0.146 | **FAIL** | YES | 34 | 273 |
| W68 | **PILOT-MISSING** | +0.019 | — | — | — | — | +0.034 | — | — | — | — | 1 | 505 |
| 691 | both | -0.134 | -0.290 | -0.156 | OK | YES | +0.177 | +0.126 | -0.051 | OK | YES | 131 | 2,037 |
| 644 | **PILOT-MISSING** | +0.101 | — | — | — | — | +0.343 | — | — | — | — | 4* | 797 |
| 705 | **PILOT-MISSING** | +0.130 | — | — | — | — | +0.118 | — | — | — | — | 3* | 34 |

*\* present in raw pilot LOOO data but filtered out by `max_object_chi2<=50` and/or `min_objects_per_group=3`.*

## Why the 7 missing anchors are missing

| stn | raw pilot obs | raw pilot objects | reason filtered |
|-----|---:|---:|---|
| 699 | 0 | 0 | sample didn't include any object observed by 699 |
| 809 | 0 | 0 | same |
| M22 | 162 | 1 | fails `min_objects_per_group≥3` |
| R17 | 96 | 2 | fails `min_objects_per_group≥3` |
| W68 | 148 | 1 | fails `min_objects_per_group≥3` |
| 644 | 27 | 4 | borderline; chi2 + object-chi2 filter drops below threshold |
| 705 | 40 | 3 | borderline; same |

This is a sample-coverage artifact of the smaller, mostly-disjoint pilot object
set, not a worker-side bug. The reference catalog's 13/13 anchor coverage came
from a 3,500-object sample specifically curated to populate these stations.

## Comparable-anchor agreement (6 of 13 present in both)

- CI overlap on RA: **5 / 6**
- CI overlap on Dec: **6 / 6**
- CI overlap on both: **5 / 6**
- Sign-agree on RA (excluding 1 near-zero): **4 / 5** informative
- Sign-agree on Dec (excluding 1 near-zero): **4 / 5** informative
- Anchors with neither sign disagreement nor CI failure: **5 / 6**

The lone catastrophic disagreement is **T09** (RA sign FAIL, RA CI does not
overlap, Dec sign FAIL but Dec CI does overlap). T09 has only 34 objects in
the pilot vs 273 in the reference, so the pilot point estimate is dominated by
a few objects. The pilot RA CI is `[-0.045, +0.957]` — the bootstrap is
clearly resolving the underprovisioning by widening the CI dramatically — but
even so the lower edge sits just above the reference's upper edge of
`-0.052"`, so they don't quite touch.

For the 5 well-behaved anchors, pilot biases are systematically larger in
magnitude than reference biases (e.g., 691: -0.290" pilot vs -0.134" ref;
703: -0.319" pilot vs -0.114" ref). Bias magnitude scales with the
distribution of objects observed, so this is consistent with the pilot
sampling a different mix of objects rather than a numerical regression.

## Residual-tail check (regression flag)

Briefing exit criterion: *"max |residual_ra_arcsec| should be < ~60" after the
smoke-gate fix… If you see another catastrophic tail, stop — that's a regression."*

Pilot v11 raw `merged_looo_results.parquet` (44,236 rows):

```
|residual_RA|   p50=0.224   p90=26.156   p99=28,791   max=261,228
|residual_Dec|  p50=0.183   p90= 8.889   p99= 9,169   max= 96,321
chi2            p50=2.040   p90=16,469   p99= 2.6e10  max= 6.6e16
hold_in_rchi2   p50=0.708   p90= 1.180   p99= 7.5e9   max= 1.3e14

|RA|>60":  3,901 / 44,236  =  8.82%
|RA|<1":  35,587 / 44,236  = 80.45%
```

Compare to v10 numbers in the bead's prior notes (`max=261,227`, `p99=26,791`).
The v11 max is essentially identical to v10. The `hw1` fix made
`hold_in_reduced_chi2` and `hold_in_fit_success` 100% non-null (verified), but
the catastrophic-residual numerical issue itself was not addressed by that
fix.

### Cross-check: reference 3,500-obj raw `looo_results.parquet`

To rule out "the local pipeline has the same tail and we just hadn't
characterized it," ran the same percentiles on
`data/looo_results/20260316T190152Z/looo_results.parquet` (the LOOO output
that produced the reference bias_table):

```
N rows: 1,211,865
|residual_RA|   p50=0.151   p90=0.725   p99=  3.821   max=545.503
|residual_Dec|  p50=0.143   p90=0.666   p99=  2.350   max=201.413
chi2            p50=0.934   p90=8.309   p99=200.278   max=1.29e+10

|RA|>60":     394 / 1,211,865  = 0.0325%
|RA|>10":   5,145 / 1,211,865  = 0.4246%
|RA|< 1": 1,139,745 / 1,211,865 = 94.05%
```

Pilot v11 vs reference, per-cut ratio:

| cut | reference | pilot v11 | ratio |
|---|---:|---:|---:|
| max |RA| | 545.5" | 261,228" | **479×** |
| p99 |RA| | 3.8" | 28,791" | **~7,500×** |
| p90 |RA| | 0.73" | 26.2" | **~36×** |
| % rows >60" | 0.0325% | 8.82% | **~270×** |
| max chi² | 1.3e10 | 6.6e16 | **~5,000,000×** |

This is overwhelming evidence that the cloud worker's per-object hold-in fits
are pathological at a rate the local pipeline does not produce. The tail is
not a numerical inevitability of the LOOO method — it is a cloud-only
anomaly. (The local pipeline does have a thin tail of its own — 394 rows
>60" out of 1.2M — but it's three orders of magnitude rarer per row and two
orders of magnitude smaller in maximum.)

Important: the bias-table builder's `max_chi2=100` and `max_object_chi2=50`
filters cleanly excise the catastrophic tail before bootstrap (44,236 → 41,699
→ 30,246 rows used in the bootstrap). So the bias values reported above are
**not contaminated by the tail**. The downstream pipeline does the right
thing. But the rate of bad fits is a real regression in the cloud per-object
pipeline that needs an independent investigation — see the child bead
spawned from ojo.1.

## Verdict — ojo NOT closed

Three independent reasons the bead's exit criteria are not met:

1. **Catastrophic residual tail unchanged** from v10 (max |RA| ≈ 261,000")
   trips the briefing's literal "stop on catastrophic tail" rule.
2. **6 of 13 anchors absent** from the pilot bias table; cannot reach
   "≥10/13 anchors agree directionally and have overlapping CIs" by
   construction.
3. **Pilot v11 is on a mostly-disjoint object set** from the 3,500-obj
   reference (only 976 of 3,500 reference objects appear in the pilot output),
   making any point-estimate comparison method-bound rather than a
   correctness check.

What does work:
- Cloud worker now populates `hold_in_reduced_chi2` and `hold_in_fit_success`
  (hw1 fix verified — 100% non-null, 100% fit_success=True).
- Of the 6 anchors that can be compared, 5 have overlapping CIs on both
  axes; 4 have directionally-consistent point estimates (5 if one near-zero
  is counted).
- Bias values are not contaminated by the catastrophic tail —
  `max_object_chi2≤50` excises it correctly during bootstrap.

## Tail diagnosis (root cause, by experiment)

### Hypothesis ranking

| # | Hypothesis | Evidence verdict |
|---|---|---|
| 1 | **FindOrb cold-start under-determined-fit divergence** (the cloud worker uses `--orbit-fitter findorb` by default; the reference predates the orbit_fitter knob and used scipy `fit_least_squares` warm-started with the MPC seed orbit) | **STRONG — primary cause** |
| 2 | **MPCORB.DAT missing in the v11 image** (FindOrb has known-orbit lookup via `get_orbit_from_mpcorb_sof()` but no catalog file to read) | SUPPORTING — explains the long-arc dense well-known-asteroid 100%-catastrophic cluster |
| 3 | **Propagator / ephemeris / time-scale / leap-second mismatch** | RULED OUT |
| 4 | **Rejected-obs handling (bead 589)** | MINOR — possibly contributes to the 37-vs-40 held-out-row delta on `2020 ML22`, but does not explain the chi² magnitudes |

### Witnesses and the decisive 4-way comparison

Three witnesses picked from `data/pilot_v11_results/merged_looo_results.parquet`
that (a) appear in `data/looo_sample_3500/mpc_observations.parquet` (pilot ∩
reference) and (b) have at least one pilot held-out row with |residual_RA| > 10,000".
Top three by max|RA|:

- **2020 ML22** — max|RA|_pilot = 261,228" (40 obs total, 11–32 held-in obs, 17.9–3770 day held-in arc range, dominant station F51)
- **2014 NH86** — max|RA|_pilot = 139,227" (40 obs total, 8–32 held-in, 31–4066 d, dominant F51)
- **2020 TP96** — max|RA|_pilot = 109,363" (26 obs total, 7–19 held-in, 16–2693 d, dominant F51)

In all three, residual |RA| is Spearman-correlated **−0.7** with `arc_remaining_days`
and `n_obs_remaining`, and **+0.7** with `|delta_q_au|`: the catastrophic rows
are exactly the holdouts that leave a **short held-in arc with few obs** (the
F51 holdouts, since F51 is the dominant station). Pilot's |Δq| in these cases is
**0.07–0.09 AU** — a giant orbital-element drift — vs reference's `1e−8 to 1e−4 AU`.
That's 6–7 orders of magnitude difference. The orbit *itself* is wrong, not the
propagation of it.

Decisive comparison, same observations across all four runs (chi²/obs for
the F51 holdout — the worst case in each witness):

| Witness / holdout | Reference (`20260316T190152Z`, scipy implicit, MPC seed warm-start) | Local scipy today (`fit_least_squares`, MPC seed warm-start, `adam_fo 6cfec10`, ASSIST) | Local FindOrb today (`--orbit-fitter findorb`, cold-start, same `adam_fo 6cfec10`, same ASSIST) | Pilot v11 cloud (FindOrb cold-start, image `pilot-v11-20260429`) |
|---|---:|---:|---:|---:|
| 2014 NH86 / F51 | max\|RA\|=0.37" | chi²/obs = **1.66** | chi²/obs = **2.36e+11** | catastrophic (max\|RA\| = 139,227") |
| 2020 ML22 / F51 | max\|RA\|=159" | chi²/obs = **227,086** | chi²/obs = **8.86e+11** | catastrophic (max\|RA\| = 261,228") |
| 2020 TP96 / F51 | max\|RA\|=110" | chi²/obs = **80,227** | chi²/obs = **1.29e+11** | catastrophic (max\|RA\| = 109,363") |

Conclusion: **local post-hw1 FindOrb on these observations reproduces the
cloud catastrophe** to within an order of magnitude on chi². Local scipy
warm-started with the MPC seed orbit produces fits 6–11 orders of magnitude
better than FindOrb cold-start *on the same observations, in the same
environment, with the same propagator*. The bug is not cloud-specific.

(Local scipy is moderately worse than the reference run on `2020 ML22` and
`2020 TP96` — chi² of 227k and 80k vs reference's ~1e5 / ~10 — but in the same
order, not the 6–11 orders that separate FindOrb from scipy. Some smaller
secondary regression between Mar 16 and today, presumably in ASSIST or
sigma-model defaults, is worth documenting separately but does not explain the
catastrophic tail.)

### Property clustering on the broader catastrophic set (2,875 rows of 44,236)

| | catastrophic-rate |
|---|---|
| arc_remaining 7–30 d | 21.2% |
| arc_remaining 30–90 d | 5.6% |
| arc_remaining 1–3 yr | 4.9% |
| arc_remaining 10y+ | 8.2% (and **88%** of all catastrophic rows live here, on long-arc dense well-known objects) |
| n_obs_remaining 5–10 | 37.7% |
| n_obs_remaining 100+ | 41.4% (long-arc dense) |
| station T08 / T05 / M22 / W68 / R17 | **100%** catastrophic on every row |
| station 704 | 96%; station P07 | 97% |
| station F51 / G96 (heavy stations) | 1–2% |

Two distinct catastrophic clusters consistent with hypothesis 1 + 2:

- **Short-arc, under-determined fits** (Cluster B): 5–30 day held-in arcs, 5–10
  obs. FindOrb cold-starts via Gauss/Vaisala on an under-determined geometry,
  lands in the wrong basin, returns a wildly wrong orbit. Witnesses 2014 NH86 / 2020 ML22 / 2020 TP96 sit here.
- **Long-arc dense, all-rows catastrophic** (Cluster A): well-observed
  asteroids (3,000+ obs over 30+ years: `1981 QE2`, `1995 UX`, `1999 CJ16`,
  `2001 DD22`). Every fit is broken — `hold_in_reduced_chi2` itself is huge
  (median 4.7e9 on catastrophic rows). FindOrb cannot converge from a generic
  initial guess on thousands of obs over decades; the absent `MPCORB.DAT`
  means there is no known-orbit warm-start to get it close.

Stations hitting 100% catastrophic rates (T08, T05, M22, W68, R17) all happen
to be involved in holdouts on the long-arc dense cluster, where every fit
fails regardless of which station is held out.

### Environment diff (pilot v11 image vs local)

| | v11 image (`pilot-v11-20260429`) | local workspace |
|---|---|---|
| Python | 3.11.15 | 3.12 |
| `erfa` | 2.0.1.5 | (unchecked, not the bug) |
| `astropy` | 7.2.0 | (unchecked) |
| Leap seconds expiry | 2026-12-28 (current) | (current) |
| DE440 ephemeris | `/usr/local/lib/python3.11/site-packages/naif_de440/de440.bsp` | same package present in `.venv` |
| DE441 small-bodies n16 | `/usr/local/.../jpl_small_bodies_de441_n16/sb441-n16.bsp` | present in `.venv` |
| Find_Orb commit | `294bd5d` | same (built locally Feb 24) |
| `adam_fo` (installed) | post-hw1 (has `evaluate_orbits` + `_TwoBodyPropagator` + `success=True`) | same (`6cfec10`) |
| `adam_orbit_det_eval` | post-hw1 (`62c351e`, bandaid removed, propagator threaded into FindOrbOrbitFitter) | same |
| **`MPCORB.DAT` / `mpcorb.sof`** | **ABSENT** (only `mpcorb.hdr`, the format-doc header, is shipped) | **ABSENT** (same — local FindOrb also can't warm-start from a catalog) |

The environment diff is essentially null. The relevant difference is purely
methodological: which fitter is configured by default for the LOOO pipeline.

### Notes on the briefing's coordinator hint

The coordinator's earlier message said "post-hw1 adam_fo at commit 6cfec10"
and "pre-hw1 adam_fo (commit c80bc5a)". The `c80bc5a` reference is correct
(parent of the merge `e9e4d2d`). `6cfec10` exists only in the local workspace
`adam_fo` and is the post-hw1 commit; the v11 image has it installed
(`/usr/local/lib/python3.11/site-packages/adam_fo/find_orb_orbit_fitter.py`)
even though `/app/sources/adam_fo/.git` only knows up to `e9e4d2d` (the image
build vendors the post-hw1 source under `/app/adam_fo` and pip-installs from
there). I initially misread this and thought the image was missing the hw1
fix; that was wrong, retracted, and corrected here.

### What this means for ojo

The catastrophic tail is **not a regression to chase in the cloud worker**.
It is the (intended-as-default-but-perhaps-not-fully-vetted) consequence of
switching from scipy `fit_least_squares` warm-started with the MPC seed
orbit to FindOrb cold-started from observations alone. The change happened
between the reference run (Mar 16) and the orbit-fitter knob landing
(`scripts/12_run_looo_cloud_shard.py:39 default="findorb"`).

Coordinator decisions that follow from this diagnosis:

1. Either change the cloud-worker default back to `--orbit-fitter scipy`
   for a baseline-correctness comparison run, or
2. Ship `MPCORB.DAT` (or equivalent `mpcorb.sof` packed orbit catalog) into
   the production image and verify FindOrb actually consults it for
   warm-starts, or
3. Modify the `OrbitFitter` interface to accept a warm-start `reference_orbit`
   argument and pass it through (the `OrbitFitter` ABC and `initial_fit()`
   signature in `adam_fo/find_orb_orbit_fitter.py:230-234` currently take only
   `(object_id, observations)`).

None of those are in scope for ojo or this diagnostic.

A separate, smaller item worth filing: local scipy today gives ~10–100×
worse chi² than the Mar 16 reference on `2020 ML22` / `2020 TP96` (sub-orders-of-magnitude,
not the FindOrb gap). Likely a `fit_least_squares` / ASSIST / sigma-model
default change between Mar and Apr 2026. Only worth chasing once the FindOrb
issue is resolved.

## Artifacts

- This file: `data/pilot_v11_results/baseline_comparison.md`
- Pilot bias table: `data/bias_catalog/pilot_v11/bias_table.{parquet,csv}`
- Pilot validation report: `data/bias_catalog/pilot_v11/validation_report.txt`
- Pilot config: `data/bias_catalog/pilot_v11/bias_table_config.json`
