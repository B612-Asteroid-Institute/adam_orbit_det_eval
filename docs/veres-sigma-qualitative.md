# LOOO per-station sigmas vs Veres 2017 Table 1 — qualitative check

Bead: `ei3` (qualitative companion to the bias-side investigation in `5z2`).

## Source

- Vereš, P., Farnocchia, D., Chesley, S. R., Chamberlin, A. B. 2017.
  "Statistical Analysis of Astrometric Errors for the Most Productive Asteroid
  Surveys." *Icarus* 296, 139–149 (arXiv:1703.03479). Per-station RMS values
  taken from **Table 1** of the paper (the 13 most productive CCD surveys, RMS
  of multi-apparition asteroid residuals after FCCT14 catalog debiasing as
  preprocessing).
- Our values: 3,500-object reference run
  (`data/looo_results/20260316T190152Z/looo_results.parquet`), aggregated into
  `data/bias_catalog/3500obj/bias_table.parquet`. Per-station `rms_ra_arcsec`
  and `rms_dec_arcsec` columns. Aggregation filter is the legacy
  `max_hold_in_reduced_chi2 = 100.0` (this catalog predates bead `7bt`'s
  multi-tier `bias_filter`; rerunning with the stricter 7bt defaults would
  trim the catastrophic-tail rows that pull RMS up, which would shift the
  comparison slightly toward Veres but is out of scope here).

Existing project Veres data found and read: `src/adam_orbit_det_eval/utils.py`
(per-catalog sigma fill-in for missing `rmsra`/`rmsdec`, derived from Veres
2017 Table 1 catalog-level averages plus a few station-level overrides);
`data/looo_analysis/veres2017_3500obj/observatory_summary.txt` (April 2026
prior aggregation of the same 3,500-obj results, used here as a cross-check
on the bias-table RMS columns — the two sources agree to within rounding).

## Comparison

Stations Veres Table 1 covers (post-FCCT14-debiased multi-apparition
orbit-fit residuals, JPL pipeline):

| stn | ours σ_RA (") | Veres σ_RA (") | ours σ_Dec (") | Veres σ_Dec (") | qualitative |
|-----|---------------|----------------|----------------|-----------------|-------------|
| F51 (Pan-STARRS1)  | 0.144 | 0.120 | 0.136 | 0.120 | ours ~20% larger on both axes |
| G96 (Mt. Lemmon)   | 0.335 | 0.310 | 0.310 | 0.280 | ours ~8–11% larger — close |
| 703 (Catalina)     | 0.770 | 0.690 | 0.716 | 0.670 | ours ~12% / 7% larger — close |
| 704 (LINEAR)       | 0.751 | 0.670 | 0.827 | 0.660 | RA ~12% larger, Dec ~25% larger |
| 691 (Spacewatch)   | 0.414 | 0.370 | 0.364 | 0.340 | ours ~12% / 7% larger — close |
| 644 (NEAT)         | 0.370 | 0.300 | 0.473 | 0.360 | RA ~23% larger, Dec ~31% larger |
| 699 (LONEOS)       | 0.730 | 0.650 | 0.730 | 0.590 | RA ~12% larger, Dec ~24% larger |

Stations not in Veres Table 1 (no comparison possible — included for
visibility of the coverage gap):

| stn | ours σ_RA (") | Veres σ_RA (") | ours σ_Dec (") | Veres σ_Dec (") | note |
|-----|---------------|----------------|----------------|-----------------|------|
| F52 (Pan-STARRS2)  | 0.113 | — | 0.111 | — | Pan-STARRS2 came online after Veres; no coverage |
| W84 (LCO CTIO)     | 0.144 | — | 0.113 | — | LCO is in Veres Table 4 weights only (σ=0.5"), no Table-1 RMS row |
| T09 (LCO Maui K2)  | 0.165 | — | 0.153 | — | Veres Table 4 lists T09 with σ=0.1" Gaia weight; no Table-1 RMS row |

## Findings

**Headline: broadly aligned.** On every Veres-covered anchor station our LOOO
σ values agree with Veres Table 1 in magnitude and rank order. Our values run
consistently larger than Veres — between 7% and 31% per (station, axis),
mean of the seven station-RA ratios is ~1.14 and mean of the seven
station-Dec ratios is ~1.16. Direction of the offset is uniform: never
smaller than Veres, never by more than a factor of ~1.3.

**Plausible causes of the consistent positive offset:**
1. *LOOO held-out vs orbit-fit residuals.* Veres residuals are
   observed-minus-fit from a multi-apparition JPL fit that conditioned on the
   observation; LOOO residuals are observed-minus-prediction from a fit
   that excluded the observation. The held-out residual is the orbit-fit
   residual *plus* the prediction error from a fit that did not see the
   point — strictly larger in expectation. A ~10–20% inflation is consistent
   with the order of magnitude for short-arc LOOO held-out cases.
2. *FCCT14 debiasing skipped.* Veres preprocesses with Farnocchia et al. 2015
   star-catalog debiasing tables; our LOOO does not. Residual catalog
   systematics (especially in the older `703`, `704`, `699`, `644` data) leak
   into the RMS and inflate it.
3. *Sample selection.* Veres uses multi-apparition asteroids from the full
   MPC catalog as of 2016-02-09; we use a 3,500-object pilot subset selected
   for LOOO eligibility (sufficient cross-observatory coverage, etc.). Sample
   composition differences explain a few percent.
4. *Aggregation filter.* The 3,500-obj catalog was aggregated under
   `max_hold_in_reduced_chi2 = 100.0`. The newer bead-7bt filter (default
   `max_chi2 = 10`) trims catastrophic-tail rows that we keep; rerunning
   under the 7bt defaults would reduce our RMS and tighten the agreement,
   probably more so for the visibly-noisier `644` and `704` Dec rows.

**No anchor station shows a >2× disagreement.** The largest single ratio is
`644 Dec` at ~1.31× Veres. None of the seven Veres-covered anchors crosses
the ei3 stop condition's 2× threshold.

**Recommendation: move on.** The sigma model in `utils.py` is qualitatively
consistent with what we measure. No follow-up bead needed for sigma
correctness on the basis of this check. If the full MPC run (bead `22h`)
shows a similar ~1.1–1.2× offset against Veres, that is expected and
explainable; if the offset grows to ~2× or flips sign, *that* would warrant
investigation.

## Caveats

- **Methodology gap is not negligible** (LOOO held-out vs JPL orbit-fit, no
  FCCT14 preprocessing on our side). The qualitative agreement here means
  "same order of magnitude, same rank ordering, consistent direction," not
  "the two pipelines measure the same quantity." For a tight quantitative
  match we would need to either rerun our pipeline with FCCT14 preprocessing
  or restrict to multi-apparition orbit-fit residuals (not held-out).
- **Veres Table 1 covers 13 stations**; our 13-station anchor list overlaps
  in only 7 (`704`, `G96`, `F51`, `703`, `691`, `644`, `699`). Anchors like
  `F52`, `T09`, `M22`, `R17`, `W68`, `W84`, `705`, `809` have no Veres-Table-1
  RMS reference.
  For those, the only external sanity check is the sigma fill-in already in
  `utils.py` and the bias-side comparison (currently blocked on bead `5z2`'s
  re-scoping after the FCCT14 finding).
- **Catalog drift.** Veres used data through 2016-02; our 3,500-obj reference
  spans observations through ~2026. Stations whose pipeline / catalog mix
  changed substantially since 2016 (notably Pan-STARRS, Mt. Lemmon, Catalina
  switching to Gaia) may have *better* astrometry now than Veres measured —
  which would push our RMS *below* Veres if it dominated, but the held-out
  inflation more than compensates and our RMS remains slightly above.
- **The 3,500-obj bias_table predates bead 7bt.** The bias_filter from 7bt
  was validated to drop the catastrophic-tail rows that inflate per-station
  RMS. Rerunning the aggregation under the 7bt defaults is a small task
  (`scripts/17_generate_bias_table.py` + filter swap) but is not part of
  this bead.
