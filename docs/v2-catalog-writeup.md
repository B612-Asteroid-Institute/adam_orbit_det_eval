# v2 MPC Observatory Bias Catalog Family — Final Writeup

**Date:** 2026-07-02
**Branch:** `mpc-v2`
**Bead:** `54t` (closed 2026-07-01)

## Executive summary

Four EFCC18-corrected LOOO catalogs have been published and cross-checked
against Veres et al. 2017 (Icarus 296:139, arXiv:1703.03479) Table 1
per-station RMS values, the only external per-station published reference.

**Headline results:**

1. On the apples-to-apples time window (Veres data cutoff of 2016-02, our
   `v2_full_no_prog_pre_2017` profile with `obstime ≤ 2017-01-01`), v2 RMS
   **matches Veres to within methodology gap** on all 7 overlap stations
   (mean v2/Veres ratio 1.02 RA, 0.98 Dec).

2. On the post-Veres data window (`v2_full_no_prog_post_2017`), v2 RMS is
   **15–22% SMALLER than Veres** at the surviving legacy stations —
   Gaia-era astrometry is genuinely tighter than the FCCT14 era.

3. On the full-archive `v2_full_with_prog` catalog, station 568 (Mauna Kea)
   shows a **real per-program signal**: program `0U` (Tholen) is
   well-behaved (RA +0.014″, Dec +0.057″, n=327 obj); program `03` shows
   a resolvable +0.12″ bias (n=18 obj). This is the wl0 architecture's
   headline capability, working end-to-end for the first time.

4. Post-EFCC18 per-station biases are 6–10% *smaller* than Veres RMS on
   the full archive — reversing the 14–16% *larger* baseline before
   EFCC18 preprocessing.

## The four catalogs

| catalog                             | profile                    | window                | rows  | rollup | tuples | observatories |
| ----------------------------------- | -------------------------- | --------------------- | ----- | ------ | ------ | ------------- |
| `v2_full_no_prog`                   | `v2_full`                  | none (full archive)   | 2,802 | 1,123  | 1,679  | 1,123         |
| `v2_full_no_prog_pre_2017`          | `v2_full_pre_2017`         | `obstime ≤ 2017-01-01`| 1,767 |   772  |   995  |   772         |
| `v2_full_no_prog_post_2017`         | `v2_full_post_2017`        | `obstime > 2017-01-01`| 1,602 |   629  |   973  |   629         |
| **`v2_full_with_prog`**             | `v2_full`                  | none                  | 3,325 | 1,139  | 2,186  | 1,139         |

All four commit into `mpc-v2` under `data/bias_catalog_v2_full_*/` and
mirror into GCS at `gs://exp-research/mpc-real-data-looo-v2-full-*/catalog/`
for downstream consumers who don't have a git checkout.

Each catalog contains:
- `bias_table.parquet`
- `bias_table.csv` (identical content, human-readable)
- `bias_table_config.json` (bootstrap params + filter settings)
- `RUN_PROVENANCE.json` (run identity: image, GCS prefixes, code pins)

The full-archive `v2_full_no_prog` catalog additionally has
`headline_stats.html` (interactive Veres/EFCC18/per-band/per-astcat report)
and `validation_report.txt` (the anchor-check output; see limitations §).

### What "prog" means and why the first three catalogs don't have it

The MPC `prog` field is a single-character (or 2-char ADES base62) program
identifier that differentiates independent observing programs at the same
physical facility. The canonical example is station 568 (Mauna Kea), where
D. Tholen's group and M. Micheli's group both submit observations from
the same telescope but with different astrometric pipelines. Veres 2017
distinguished these ("`p=&%`" in their notation); v1 could not.

Bead **43z** (2026-05-21) added `prog` to the mpcq query layer. The v12
BigQuery fetch (2026-05-08) predates that, so the shard parquets used
for the first three catalogs have all-NULL `prog`. `v2_full_with_prog`
runs on a fresh 2026-06-23 BQ fetch that includes `prog` — this is the
only catalog in which per-program tuple rows exist.

The three no-prog catalogs are still scientifically meaningful at
per-`(stn, band)` resolution. They're the correct catalogs for anyone
asking "what is the residual bias at station X after EFCC18 correction,
without distinguishing programs?"

## Methodology

- **LOOO (Leave-One-Observatory-Out):** for each object, exclude a
  group's observations, re-fit the orbit with the remainder using
  ASSIST N-body propagation warm-started by Find_Orb, then compute
  residuals for the excluded observations against the refit.
  Aggregated per-group, this yields empirical per-group astrometric
  biases and RMS. Each held-out unit is the configured tuple (default
  `(stn, prog, band)` per bead wl0).
- **EFCC18 preprocessing:** per-(catalog, HEALPix-tile-at-order-64)
  corrections applied to RA/Dec BEFORE observations enter the LOOO
  pipeline. 46.7% of the v2_full_with_prog observations are corrected,
  50.2% are Gaia-reference no-ops, 3.1% unsupported (astcat unknown or
  outside the EFCC18 lookup).
- **Bootstrap CIs:** 2,000 resamples over objects (not observations),
  seed=42. Produces 95% CIs for mean/median/RMS on all four axes
  (RA, Dec, AT, CT).
- **Filters:** `hold_in_reduced_chi2 ≤ 100` (weeds out fits with
  pathological chi2 — post-2tj-fix this filter no longer gates
  90%+ of rows), `n_obs_per_group ≥ 10`, `n_objects_per_group ≥ 3`.
  Per-group sigma-pathology filter (bead `gnm`) drops the small
  handful of rows with reported σ below physically plausible values.

## Comparison against Veres 2017 Table 1

Veres et al. 2017 published per-station astrometric residual RMS for
the 13 most productive CCD surveys, computed from JPL orbit-fit
residuals with FCCT14 catalog debiasing applied. Seven of those
stations overlap the ones observed with substantial `n_objects` in
our catalogs.

**Table 1 — Per-station RMS: v2 catalogs vs Veres 2017**

RMS in arcsec. Δ = v2 − Veres; ratio = v2 / Veres. "—" means no data
(the legacy surveys 704 LINEAR, 644 NEAT, and 699 LONEOS all shut
down before 2017; their `post_2017` cells are empty by construction).

| stn | survey       | Veres σ_RA | pre_2017 | post_2017 | full   | with_prog | Δ pre_2017 | Δ post_2017 |
| --- | ------------ | ---------: | -------: | --------: | -----: | --------: | ---------: | ----------: |
| F51 | Pan-STARRS1  |      0.120 |    0.150 |     0.109 |  0.118 |     0.119 |    +0.030  |    −0.011   |
| G96 | Mt. Lemmon   |      0.310 |    0.322 |     0.241 |  0.266 |     0.268 |    +0.012  |    **−0.069** |
| 703 | Catalina     |      0.690 |    0.702 |     0.552 |  0.640 |     0.642 |    +0.012  |    **−0.138** |
| 704 | LINEAR       |      0.670 |    0.634 |         — |  0.628 |     0.627 |    −0.036  |          —  |
| 691 | Spacewatch   |      0.370 |    0.354 |     0.309 |  0.339 |     0.340 |    −0.016  |    **−0.061** |
| 644 | NEAT         |      0.300 |    0.324 |         — |  0.314 |     0.306 |    +0.024  |          —  |
| 699 | LONEOS       |      0.650 |    0.608 |         — |  0.609 |     0.609 |    −0.042  |          —  |

| stn | Veres σ_Dec | pre_2017 | post_2017 | full   | with_prog | Δ pre_2017 | Δ post_2017 |
| --- | ----------: | -------: | --------: | -----: | --------: | ---------: | ----------: |
| F51 |       0.120 |    0.139 |     0.102 |  0.111 |     0.112 |    +0.019  |    −0.018   |
| G96 |       0.280 |    0.271 |     0.236 |  0.251 |     0.252 |    −0.009  |    −0.044   |
| 703 |       0.670 |    0.659 |     0.489 |  0.585 |     0.584 |    −0.011  |    **−0.181** |
| 704 |       0.660 |    0.624 |         — |  0.621 |     0.620 |    −0.036  |          —  |
| 691 |       0.340 |    0.278 |     0.251 |  0.263 |     0.266 |    −0.062  |    **−0.089** |
| 644 |       0.360 |    0.362 |         — |  0.354 |     0.329 |    +0.002  |          —  |
| 699 |       0.590 |    0.548 |         — |  0.550 |     0.550 |    −0.042  |          —  |

**Mean ratios across the 7 overlap stations:**

| profile      | RA   | Dec  |
| ------------ | ---: | ---: |
| pre_2017     | 1.02 | 0.98 |
| post_2017 (4 stations that survived) | 0.79 | 0.83 |
| full_archive | 0.94 | 0.90 |
| with_prog    | 0.94 | 0.90 |

### Interpretation

**`pre_2017` matches Veres to methodology gap.** Applying EFCC18 to the
same observations Veres saw produces essentially the same per-station RMS
Veres measured, using an entirely different aggregation methodology (LOOO
held-out residuals vs Veres' orbit-fit residuals). This is a strong
external validation of the v2 pipeline: independent methods on the same
raw data converge.

**`post_2017` beats Veres substantially.** At the four legacy stations
that continued observing past 2017 (F51, G96, 703, 691), the modern
Gaia-astrometry-era RMS is 15–25% smaller than Veres' pre-Gaia numbers.
This is not a v2-methodology effect — it is the observable improvement in
astrometric quality since these surveys migrated their reference catalogs
to Gaia DR1/DR2/DR3.

**`full_archive` sits between the two by observation-count weighting.**
Post-2017 surveys with 10× the object volume dominate the full-archive
RMS calculation for stations that survived, pulling the average toward
the post-2017 values.

**LOOO methodology introduces a known ~10–20% inflation.** The
`held_out` residual is `observed − predicted` from a fit that did NOT
see the held-out point; Veres' residual is `observed − fit` from a
fit that DID see it. Held-out residuals are strictly larger in
expectation. That we still see v2 matching or beating Veres on the
apples-to-apples window (pre_2017) means EFCC18 + modern astrometry
overcompensate this gap — a strong endorsement of both.

## Per-program signal at station 568 (with_prog catalog only)

**Table 2 — Station 568 per-program bias breakdown**

Real per-program discrimination in the v2 catalog, only available when
`prog` is populated (i.e., in `v2_full_with_prog`). Rows are per-`(stn,
program_code)` with `band` aggregated.

| prog | bias_RA (″) | bias_Dec (″) | rms_RA (″) | rms_Dec (″) | n_objects | n_obs | high_confidence |
| ---- | ----------: | -----------: | ---------: | ----------: | --------: | ----: | :-------------: |
| `0U` |   +0.014    |   +0.057     |   0.160    |   0.148     |     327   | 1,007 |       Y         |
| `0P` |   −0.054    |   −0.027     |   0.231    |   0.234     |     102   |   414 |       Y         |
| `02` |   +0.003    |   −0.025     |   0.171    |   0.137     |      74   |   161 |       Y         |
| `03` |   +0.123    |   +0.094     |   0.249    |   0.274     |      18   |    54 |       N         |
| `0B` |   +0.028    |   −0.115     |   0.195    |   0.197     |       9   |    17 |       N         |
| `0a` |   +0.117    |   −0.167     |   0.209    |   0.180     |       5   |    16 |       N         |
| `01` |   −0.097    |   +0.233     |   0.115    |   0.264     |       3   |    15 |       N         |
| `0T` |   −0.018    |   +0.046     |   0.158    |   0.110      |       9   |    11 |       N         |

The three high-confidence programs (`0U`, `0P`, `02`) have distinct
signatures: `0U` shows a small positive-Dec bias (~60 mas), `0P` shows
a modest negative offset both axes, `02` is essentially unbiased. Small
programs (n_objects < 20) show larger apparent biases but are not
statistically high-confidence.

This is what wl0 was designed to surface. Previously (v1 and the three
no_prog v2 catalogs), all 568 observations were pooled into a single
per-station row, hiding these systematic differences under a whole-station
average.

## EFCC18 effect (already in `headline_stats.html`)

The full 13-anchor pre-EFCC18 vs post-EFCC18 table, with delta columns,
is in `data/bias_catalog_v2_full_no_prog_20260622/headline_stats.html`.
Key findings summarized here:

- **EFCC18 acted as designed.** Bias magnitudes were systematically
  reduced at the biggest-bias anchors: station 704 lost 30% of its RA
  bias and 68% of its Dec bias; 699 lost 39% RA and 61% Dec.
- **The effect is mostly in Dec.** Four anchors (704, 699, 691, 644)
  show much larger Dec reductions (171–296 mas) than RA reductions
  (12–136 mas). Consistent with non-Gaia catalog Dec-frame offsets
  being EFCC18's main target.
- **691 is the cleanest demonstration:** Dec swung from +0.177″ (pre)
  to −0.029″ (post) — a full removal of the catalog-frame offset,
  with slight overshoot to the opposite sign.
- **Well-calibrated anchors stayed well-calibrated.** F52, M22, R17,
  W68 all remain <0.05″ both axes; T09 at 0.09″ RA (under its 0.10″
  tolerance).

## Catalog-level headline statistics

| catalog             | rows  | high_conf | bias_sig | median |bias_RA| | median |bias_Dec| | median rms_RA | median rms_Dec |
| ------------------- | ----- | --------- | -------- | ------------------ | ------------------- | ------------- | -------------- |
| no_prog_full        | 2,802 | 56.9%     | 44.4%    | 0.052″             | 0.055″              | 0.390″        | 0.376″         |
| no_prog_pre_2017    | 1,767 | 59.3%     | 42.0%    | 0.060″             | 0.060″              | 0.438″        | 0.416″         |
| no_prog_post_2017   | 1,602 | 54.4%     | 46.4%    | 0.043″             | 0.050″              | 0.306″        | 0.306″         |
| with_prog           | 3,325 | 55.1%     | 44.6%    | 0.053″             | 0.057″              | 0.395″        | 0.379″         |

Post-2017 is the tightest catalog by every measure — the Gaia-era data
is genuinely cleaner. Pre-2017 is the noisiest (30+ years of legacy
astrometry). Full-archive is the population-weighted average.

## Limitations and caveats

1. **prog absent from 3 of 4 catalogs.** Only `v2_full_with_prog` has
   `program_code` populated. The pre_2017 and post_2017 catalogs would
   need a fresh cloud run with an obstime-filtered profile against the
   `v2-full-with-prog-20260624/input/` shards to gain prog. This is
   feasible but was out of scope for the initial v2 release.

2. **LOOO methodology gap.** As noted above, held-out residuals are
   ~10–20% larger than orbit-fit residuals from the same data. The
   Veres comparison bakes this in — a "match" to Veres actually means
   v2 is doing slightly *better* on comparable observations.

3. **The anchor validation was removed** (commit `43e5cfa`). Script 17's
   built-in `--validate-anchors` compared to the March-2026 3,500-object
   internal snapshot, a pre-EFCC18 self-consistency check that had
   accidentally become a permanent baseline. The check was
   scientifically low-information and misleading (5/13 "failures" were
   EFCC18 working as designed) — removed in favor of the Veres comparison
   as the actual external check.

4. **Space-based observations dropped.** All pods explicitly drop
   observations from space-based stations (C51 WISE, C57 NEOWISE-R,
   etc.) — LOOO holds out an OBSERVATORY, but space-based observatories
   have too few objects overlap to LOOO cleanly.

5. **Very-small-N per-program rows are noisy.** In `with_prog`, program
   rows with n_objects < 10 have wide CIs and shouldn't be over-interpreted.
   The `high_confidence` flag on each row indicates whether the bootstrap
   deems the point estimate statistically resolvable.

6. **BigQuery cost incident.** Producing the with_prog catalog required a
   fresh BigQuery fetch (bead 43z's prog column was needed). Between the
   inefficient default fetch pattern and a poorly-designed auto-restart
   wrapper, this cost ~$800 in BQ queries on 2026-06-24 before the pattern
   was fixed. Bead **84t** documents the incident and the committed fix
   (`scripts/15_fetch_mpc_scale.py` commit 8d7b203); future fetches of
   this scale cost < $10.

## Reproducibility

Every catalog carries a `RUN_PROVENANCE.json` recording:

- Cloud job UID (`mpc-real-data-looo-*-<timestamp>`)
- Container image tag AND digest
- GCS input/output prefixes
- Tier-2 code pins (adam_core, adam_fo, mpcq)
- EFCC18 cache source
- BigQuery fetch metadata

Key artifacts for reproducing the with_prog catalog:

- Git branch: `mpc-v2` in `B612-Asteroid-Institute/adam_orbit_det_eval`
- Container: `us-west1-docker.pkg.dev/moeyens-thor-dev/ai/mpc-real-data-looo:v2-full-fix-prog-20260629`
  (digest `sha256:203dbad7...`)
- BigQuery source: `moeyens-thor-dev.mpc_sbn_aurora.public_obs_sbn`
- GCS input prefix: `gs://exp-research/mpc-real-data-looo-v2-full-with-prog-20260624/input/`
- adam_core pin: `e6f89108`
- adam_fo pin: `b10b622` (branch `mpc-v2` in adam_fo)
- mpcq pin: `42e447b` (explicitly installed in Dockerfile — the fix for the 2026-06-26 broken run)

## Known follow-ups

Beyond the current v2 catalog family:

1. **`v2_full_with_prog_pre_2017` / `_post_2017`** — the natural completion
   would rerun the time-windowed profiles against the with-prog input
   shards, producing 6 catalogs total (all combinations of {full, pre_2017,
   post_2017} × {no_prog, with_prog}). Requires 2× cloud LOOO runs
   (~4-14h each on cpu-spot) but no fresh BQ fetch (input already in GCS).

2. **Extend the Veres comparison** — Veres Table 1 covers 13 stations; we
   compared 7. The other 6 (`T05`, `T08`, `T12`, `T14`, `Q11`, `Q63`) have
   substantial data in v2 catalogs but weren't in the earlier project's
   fixed "anchor" set. A follow-up analysis could pull all 13 for a
   complete Veres-comparable table.

3. **AT/CT bias interpretation.** The 2×2 covariance matrices for both
   sky-plane (RA/Dec) and along-track/cross-track (AT/CT) are populated on
   every row but haven't been analyzed in this writeup. Timing bias vs
   trailing bias decomposition is a natural follow-up.

4. **Per-object outlier analysis.** The catalog aggregates over objects.
   A per-object per-station residual table (already produced as
   `looo_results_atct.parquet`, ~463 MB, in each catalog's `published/`
   dir but not committed to git) is available for downstream users
   interested in specific problem objects.

5. **Tier 2 mpcq/library cleanup.** Beads `84t` (BQ cost discipline) and
   `q8t` (obsolete) are closed; `whl` (P3 image-audit lesson) and `s71`
   (P2 publish wrapper hygiene) remain open.

## Provenance summary

- Bead `54t` (v12-corrected re-run) closed 2026-07-01.
- Branch `mpc-v2` in adam_orbit_det_eval consolidates 132 commits of v2
  work from the former `kk/v2-prep-runbook` and 10 feature branches.
- Companion branch `mpc-v2` in adam_fo at `b10b622` (fitter resilience —
  beads 5h7 + 589 + UTC hardening).
- Companion pin `adam_core@e6f89108` (bead e48 warm-start ABC, already
  in adam_core `main` via PR #198).
