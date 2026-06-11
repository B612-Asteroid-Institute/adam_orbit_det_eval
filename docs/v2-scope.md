# v2 Catalog Scope

**Status:** open, blocked on v1 publication wave (fqv, ird, ctq — all closed; ready to schedule v2).
**Tracker:** bead `54t` (canonical record; this doc is a readable snapshot).
**Last updated:** 2026-06-02.

> This is the file equivalent of `bd show 54t`. Edit the bead, not the doc, when scope changes — then refresh this file.

---

## Headline

Run ONE final corrected re-run of the LOOO pipeline bundling all accumulated improvements identified during v1 analysis. Produce a second catalog (`v2-corrected`) alongside the published raw v1. The two together let downstream consumers see both "raw MPC astrometry bias" and "bias remaining after known catalog systematics are removed."

**Do not execute** until v1 publication is shipped. v2 is the final freeze before the v1+v2 catalog pair is declared done.

---

## Core improvement: EFCC18 catalog-debiasing as preprocessing

Apply EFCC18 (Eggl, Farnocchia, Chamberlin, Chesley 2018, BIAS_VERSION 3.0, `debias_2018.tgz`) corrections to MPC astrometry **before** the LOOO pipeline runs.

EFCC18 publishes per-(catalog, HEALPix-tile-at-order-64) bias offsets and proper-motion-corrected drift terms. LOOO is structurally blind to catalog systematics shared across the residual network (bd memory `looo-catalog-systematic-blindspot`); preprocessing decomposes catalog vs intrinsic station bias.

Correction formula (per observation):
```
corrected_RA  = observed_RA  - (dRA  + (t-2000) * pmRA  / 1000) / cos(DEC)
corrected_DEC = observed_DEC - (dDEC + (t-2000) * pmDEC / 1000)
```
Lookup keyed on `(mpc_catalog_flag, healpix.ang2pix(64, ra, dec, nest=False))`. Data on disk: `/Users/kathleenkiker/claude_cli/debias_2018.tgz`.

---

## What to bundle into v2

The corrected re-run is the only place to fold in all v1 follow-ups so the cloud cycle is paid once.

1. **EFCC18 preprocessing** — the headline; this bead's core.
2. **ITF status='I' drop** — already in commit 6b78d9f on `kk/mpc-scale-bias-catalog` (fetch-script level).
3. **ITF re-fetch / re-run of the 12 contaminated objects** (bead j7p) — execute as part of this run.
4. **mode='OCC' drop at fetch level** — currently planned as a publication-time filter; moving it upstream to the BQ query is cleaner.
5. **Source-data sanity filters from ie2** — `obstime > 0`, `ra in [0, 360)`.
6. **Veres fallback coverage extension** — if ie2's audit surfaced gaps.
7. **AT/CT publish wrapper** — atomic chain of scripts 16 → 17 → 19 so v2 cannot ship with empty AT/CT columns (the d5b incident on v1).

## What to produce

Two outputs side-by-side:
- `data/mpc_scale_results_<date>/raw/` — re-run raw output, equivalent to v1 plus the ITF/OCC/sanity filters.
- `data/mpc_scale_results_<date>/corrected/` — EFCC18-corrected residuals.

Plus a `delta_bias_table.parquet`:
- `station, n_obs, n_objects` (same in both)
- `bias_raw_*, bias_corrected_*, delta_bias_*` for RA, Dec, AT, CT, and rms_*
- 95% CIs on each (reuse bead `ird`'s bootstrap machinery).

## Interpretation framework

- `delta_bias → 0` — catalog-systematic-induced bias, correctly identified and removed. Cross-validated.
- `delta_bias` **unchanged** — non-catalog station systematic (timing, trailing, calibration, instrumental). Real station-level bias.
- `delta_bias` **grows** — red flag. Either EFCC18 removed real signal at that station, or our pipeline has a sign/frame/coordinate bug.

The corrected catalog is the apples-to-apples comparison target for Veres 2017 Table 1 RMS (bead 5z2). The raw catalog stays publishable as "biases observers see if they don't apply catalog debiasing."

## Out of scope

- Do not modify the orbit-fitter or LOOO core; this is a preprocessing wrapper.
- Do not change anything about reported per-observation sigmas (`rmsra`/`rmsdec`); the no-clamp/no-floor rule still holds for v2's `raw` and `corrected` catalogs. (See "Iterative sigma override" below for the separate exception.)
- Do not invalidate or retract the raw v1 catalog; it remains publishable in its own right.

---

# Design additions (v2 column-level changes, schema)

## Per-station 2×2 residual covariance (2026-05-26)

Schema addition to `bias_table.parquet` and `bias_catalog` deliverables:

| existing | added |
|---|---|
| `bias_ra, bias_dec, rms_ra, rms_dec, n_obs, n_objects, CIs` | `resid_var_ra, resid_var_dec, resid_cov_ra_dec, n` |

One-pass aggregation over the per-obs residual table that already exists in `merged_looo_results` — no new computation cost, no fitter changes.

**Why:** the OD experiments workstream found that constructing per-obs covariance from the outer product of the published mean-bias vector `b·bᵀ` produces a near-degenerate covariance for survey-dominated short arcs (PS1/PS2/H21) and pulls fits off JPL at a 20% rate on a held-out NEO cohort. The mean-bias outer product encodes the *direction* of the mean, not the *scatter* of per-obs residuals.

**Acceptance:**
- New columns populated for every station with `n_obs ≥` small-sample cutoff.
- `resid_var_ra` and `resid_var_dec` match `rms_ra²` and `rms_dec²` to numerical precision (cross-check).
- `resid_cov_ra_dec` sign-checked against a known-correlated and a known-independent test station.

The corresponding fix to the OD application pattern (replace `b·bᵀ` with `diag(σ_b_RA², σ_b_Dec²)` from the CI half-widths) is owned by the OD experiments coordinator and out of scope for 54t.

## Correlated-vs-random variance decomposition per station (OD intake, 2026-05-27)

The single most valuable addition surfaced by the OD program. Short-arc damage is *geometric* — a station's bias hurts short arcs because it is systematic/correlated within an arc, not random per-obs. v1 publishes mean + RMS but says nothing about how much of the residual is coherent across a night/arc vs independent.

**Proposal:** publish, per station, within-arc-correlated variance vs per-obs-random variance. Mechanism options to evaluate:
- Variogram on per-night residual series per station.
- Night-binned residual analysis: `variance(per-obs)` vs `variance(per-night-mean)`.
- ANOVA-style decomposition: total residual variance = within-night + between-night.

**Schema addition:** `var_correlated_ra, var_correlated_dec, var_random_ra, var_random_dec, correlation_scale_hours`.

**Acceptance:** decomposition computed for every station with ≥5 multi-obs nights; cross-check that `var_correlated + var_random ≈ rms²` to within sampling noise.

## Per-(station, astcat) bias + covariance (OD intake, 2026-05-27)

Bias is largely a star-catalog systematic. v1 keys by station only. An obs reduced against Gaia DR3 deserves different treatment than USNO-B1 from the same station. `astcat` is already in `mpcq.MPCObservations`.

Pipeline decision required before scoping: hold out per-(station, astcat, prog), or hold out per-station and aggregate per-(station, astcat) at the bias-table step. Probably the latter for tractability.

**Acceptance:** bias_table rows keyed by `(station, astcat)`; ≥20 stations have ≥2 astcat entries with `n_obs ≥ 100`.

## Per-(station, band/filter) bias + covariance (OD intake, 2026-05-27)

v1 integrates bias over all bands at a station. Different bands carry physically distinct centroid biases from atmospheric DCR (largest in blue at high airmass), filter-dependent PSF, per-band detector response. Multi-band stations smear published bias.

**Proposal:** extend astcat conditioning to per-(station, band) or per-(station, astcat, band). `band` is in `mpcq.MPCObservations.band`. Same pipeline question as astcat.

**Acceptance:** bias_table rows keyed by `(station, band)`; ≥15 stations have ≥2 band entries with `n_obs ≥ 100`. For multi-band stations (PS1 g/r/i/z/y), expect meaningful per-band differences.

## AT/CT covariance as first-class (OD intake, 2026-05-27)

Timing bias (pure AT) is physically distinct from centroiding bias; the OD experiments leaned on the RA/Dec rollup and underused this. Mirror the per-station 2×2 in the AT/CT decomposition: `var_at, var_ct, cov_at_ct`. Small extension of the AT/CT publish wrapper already in scope.

## Graded per-station confidence (OD intake, 2026-05-27)

Current `high_confidence` filter (`n_obs ≥ 100, n_objects ≥ 20`) drops exactly the long-tail stations that observe short-arc NEOs.

**Proposal:** keep `high_confidence` as a label for backwards compatibility, add a continuous confidence column (effective sample size, or CI-width relative to bias magnitude). Lets partial info be used with appropriate uncertainty downstream instead of zeroing it out. No data drops on publication.

## Epoch-resolved (or epoch-tagged) bias (OD intake, 2026-05-27)

Less ambitious version of "per-time-period LOOO" below. Populate `obs_epoch_start` / `obs_epoch_end` (currently blank in v1). Add a non-stationarity flag per station (KS on pre-2017 vs post-2017 residuals → significant deviation flags). Useful intermediate even if per-time LOOO is deferred.

**Acceptance:** `obs_epoch_start/end` populated for every station; `non_stationary` flag (boolean + p-value) on every bias_table row.

---

# Design notes (v2 pipeline-level)

## LOOO group keys + time filter — configurable architecture (RESOLVED, 2026-06-08)

*Implementation: bead wl0 ships the configurable group-key plumbing (Axis 1 — the `--group-by` flag, tuple hold-out unit, and `compute_bias_table(group_by=...)` per-tuple + per-station rollup rows). Axis 2 (the parametric obstime filter + named run-profile config layer) is sibling bead tcu, which builds on wl0.*

The v2 pipeline takes two configuration axes that together define a "run profile":

**Axis 1 — Group keys (hold-out level + aggregation level).** Each of {`stn`, `prog`, `band`, `astcat`} is independently togglable. The configured set becomes both:
- The **LOOO hold-out unit**: each unique tuple is held out as a unit per object (e.g. with `[stn, prog, band]`, observations from station 568 in program "2" through filter "r" are held out together, separately from station 568 program "_" through filter "i"). This is more granular than v1's per-station hold-out.
- The **bias_table aggregation key**: residual rows are grouped by the same tuple. Configurable group keys in `bias_table.py` accept any subset.

**Axis 2 — Observation time filter (parametric).** A pre-LOOO step filters observations by `obstime_min` / `obstime_max`. Single mechanism, reused for all time-window variants (pre/post-2017 now, finer cuts later).

### astcat — decision and rationale

**astcat is NOT a default hold-out or aggregation dimension. It is available as an optional diagnostic.**

EFCC18 preprocessing (the headline v2 change) corrects mean catalog bias at per-(catalog, HEALPix-tile-at-order-64) resolution before LOOO runs. czs's coverage audit (bead `ie2` follow-up + `czs` Veres extension) confirmed ~98% of v1 observations are either EFCC18-corrected (52.4%) or Gaia-reference no-ops (45.8%); only ~2% genuinely uncorrectable. Post-EFCC18, the catalog signal has already been pulled out — using `astcat` as an LOOO hold-out dimension would mostly be measuring noise on a much smaller sample, at the cost of substantially more refits per object.

**What we keep instead:** an off-by-default toggle. If a pilot result shows residual catalog structure (e.g. EFCC18 coverage on the actual v2 input shard turns out worse than 98%, or per-(stn, astcat) interactions survive EFCC18's factorization), flip `astcat` on. The architecture supports it without code changes.

### Default profile + named variants

| Profile | Group keys | Time filter | Notes |
|---|---|---|---|
| **v2_full** | `stn, prog, band` | none | Primary v2 catalog. astcat off. |
| **v2_full_pre_2017** | `stn, prog, band` | `obstime ≤ 2017-01-01` | Apples-to-apples vs Veres 2017 Table 1. |
| **v2_full_post_2017** | `stn, prog, band` | `obstime > 2017-01-01` | "What changed since Veres" per station. |

Each profile produces a separate published catalog (`data/mpc_scale_results_<date>_<profile>/`).

### Future extensibility

Time cuts are parametric — add new profiles (per-year, per-quarter, pre/post specific known events like CCD swaps) by adding config entries, no pipeline change. Per-program-code curated-station lists and per-time-period curated-station lists (sections below) can layer on top of this by overriding the default group-key set.

### Replaces / resolves

- The "pipeline decision required" notes in the per-(station, astcat) and per-(station, band) sections are resolved here. Per-band is in by default; per-astcat is configurable, off by default.
- The "Pre-2017 / post-2017 split runs" section below is now implemented as named profiles on this single pipeline rather than as separate code paths.

## Sigma-pathology handling — two-prong (REQUIRED, 2026-05-15)

Surfaced by zw0 investigation. The `max_object_mean_chi2=50.0` filter at `src/adam_orbit_det_eval/looo/bias_table.py:537` uses chi² built from per-obs reported sigmas. N86 reports nanoarcsec sigmas → chi² ~10⁸; a single N86 obs on a shared object pushes `mean_chi²` > 50 and indicts that object at every station that observed it. Net effect on v1: 373,050 non-N86 rows dropped across 780 stations, 4–15% loss at every Veres anchor.

**Both prongs required, not either/or:**

**A. Sigma-pathology detection + substitution (upstream — applies before LOOO):**
- Audit all stations. Threshold: median reported sigma < 0.01″ (10 mas) AND `n_obs ≥ 100`.
- Flagged stations: treat reported `rmsra`/`rmsdec` as MISSING; the existing `compute_sigmas_with_fallback` substitutes the Veres 2017 per-station fallback.
- Rationale: extends "fallback when missing" to "fallback when demonstrably broken." NOT a clamp/floor on otherwise-valid sigmas — treating garbage as garbage. Consistent with chi²-is-signal because substitution fires only when reports are physically impossible (sub-10-mas precision is not real).
- Per-run audit output (list of flagged stations + median sigma) is a catalog artifact.

**B. Sigma-independent object-mean filter (downstream — at aggregation):**
- Replace chi² with `|residual_arcsec| / 1″` as the quality proxy in `max_object_mean_chi2`. Sigma-independent; preserves intent (catch non-gravitational-force objects).
- Defense-in-depth: even if a future pathological station slips through (A), the aggregation no longer cares about reported sigmas.

**Out of scope:** clamping/flooring otherwise-valid reported sigmas. The chi²-is-signal framing stands for v1's raw output. Substitution only at physical-impossibility threshold, not statistical disagreement with Veres.

*Note: bead gnm dropped `max_object_mean_chi2` to None entirely on v1 as its option-b fix. Prong B above is the v2 replacement that brings the filter back in sigma-independent form.*

## Per-program-code LOOO (curated stations)

Today's setup: an object's residuals at "703-A" are predicted by an orbit fitted with that station's "703-B" observations still included. Intra-station program differences are smeared into the orbit fit.

**Proposal:** for a curated shortlist of stations with documented per-program differences, run LOOO holding out each `(station, program_code)` pair as a unit.

**Upstream prerequisite:** bead 43z (closed 2026-05-21) shipped the program_code column-source fix. The pipeline now carries canonical MPC `prog` (98 values, ~90% NULL) rather than `trksub`.

**Identification:** needs a curated list. Sources: 43z-corrected `program_code_stats` sorted by intra-station bias-variance-across-prog; survey DR notices; MPC vfcc17 examples (568 p=2 Tholen vs p=_ Micheli; 309 p=&% LCO).

**Scope:** targeted (~10–20 stations) is post-processing on existing residuals. Catalog-wide requires a pipeline change to make `program_code` part of the LOOO holdout key.

**Acceptance:** ≥10 stations have ≥3 prog values each with `n_obs ≥ 100` in per-(station × prog) aggregation.

## Per-time-period LOOO (curated stations)

Same problem in the time dimension. Stations get CCD swaps, pipeline rewrites, calibration changes.

**Proposal:** for a curated shortlist of stations with documented behavior changes at known epochs, split observations at the change epoch and treat each window as a separate LOOO unit. Report per-(station, time-window).

**Identification:** observatory communications, or breakpoint detection in v1 per-station residual time series.

**Scope:** targeted is post-processing; catalog-wide is a pipeline change.

## Pre-2017 / post-2017 split runs (Veres comparison)

Veres 2017 Table 1 RMS values were derived from data up to ~2017. v1 covers the full MPC archive through 2026. A direct v1-vs-Veres comparison conflates "method differences" with "time-period differences."

**Proposal:** run two fully separated LOOO catalogs using the v2 corrected pipeline:
- **pre-2017:** observations with epoch ≤ 2017-01-01 → apples-to-apples vs Veres.
- **post-2017:** observations with epoch > 2017-01-01 → "what changed since Veres" per station.

Delta between pre and post per station is the headline scientific output. Especially valuable where v1 diverges from Veres (e.g. D29's improvement).

**Scope:** two additional cloud-cycle runs on the v2 image. Roughly doubles wall-clock vs the single v2 run. Cleanest framed as a **sibling bead** that depends on v2's image build, not a bundle.

## Iterative sigma override from v1 empirical values

**Proposal:** in v2, override per-obs reported sigmas with v1-measured per-station empirical noise for stations where the reported sigma is materially smaller than what v1 measured. Mechanism options: floor, replace, or selective (only where `reported < f × v1_rms_*`).

**Framing tension:** this is precisely the clamp/floor the chi²-is-signal rule prohibits for v1. Defensible framing for v2: "v1 stands as the under-reporting measurement; v2 uses v1's published noise model as input." Both catalogs coexist.

**Implication:** produce a SEPARATE catalog (`data/mpc_scale_results_<date>/corrected_iterative/`), NOT replace the main v2 corrected output. Three catalogs side-by-side: raw / EFCC18-corrected / EFCC18 + iterative-sigma-override.

**Decisions owed:**
- Mechanism: floor vs replace vs selective?
- Apply to all stations or only those flagged by an extended sigma-pathology audit ("under-reports by > N×")?
- Iteration budget: stop at v2, or plan a v3 convergence check?
- Catalog naming and downstream-consumer messaging — the three-catalog story needs an interpretation-doc paragraph.

---

# Process gate: OD-utility regression as v2 acceptance criterion (OD intake, 2026-05-27)

The whole OD program's lesson is that internal LOOO consistency ≠ OD improvement. v2 should be gated on a held-out OD check — or v2 ships and rediscovers the same null result.

**Proposal:** before declaring v2 catalog "published," run a small OD-utility regression on a held-out NEO cohort (the OD coordinator's si3 cohort or equivalent) under two weighting variants:
- **v2_diag_inflate** — subtract per-station bias, inflate diagonal by CI half-widths from the v2 catalog.
- **v2_correlated** — same plus the correlated-variance component from the variance-decomposition addition above.

Both must beat or match the `no_bias` baseline on Δr/σ-to-JPL with ≤4% regression rate on short-arc controls.

**Owner:** coordinate with the OD experiments workspace (`/Users/kathleenkiker/od_experiments_setup/`) to reuse their cohort + variant infrastructure. They have the harness; v2 just plugs in new column choices.

---

# Pre-flight checks (from a6w runbook)

## Quota reconciliation (action before submit)

- v1 spec footprint: 98 CPU / 224 GiB (cpu=7 × parallelism=14, mem=16Gi × 14).
- Documented `cpu-ondemand` tenant quota: 25 CPU / 100 GiB.
- v1 admitted only via Kueue cohort borrowing — fragile if other workloads compete.

**Action:** verify current `cpu-ondemand` quota and cohort headroom **live** before v2 submit. If posture has changed, either request a quota raise, reduce parallelism, or sequence v2 around other consumers.

## JOURNAL.md absent (informational)

Pre-v1 cluster timeline was reconstructed from beads, not a primary `JOURNAL.md` log. Consider starting one for v2 if cluster ops decisions accumulate. Not a blocker.

---

# Program-code state (43z, resolved)

The v1 `program_code` mislabel (trksub vs canonical MPC `prog`) was resolved by bead **43z** (closed 2026-05-21). Three commits on `kk/mpc-scale-bias-catalog`:
- `87265f6` — pipeline reads `prog` after mpcq plumbing.
- `ffc36a5` — mpcq pinned to upstream `42e447b` (includes `prog` in `OBSERVATION_COLUMNS_ADES`).
- `8701dec` — `refresh_program_code.py` produced sidecar parquets; `trksub` retained as separate column for tracklet-level forensics.

**For v2:** no additional program_code work required. Ensure the 43z commits land in the v2 image and the v2 fetch path retains `prog`. Acceptance gate: `program_code_stats` has ≤200 rows (not 30,842).

---

# Execution scope

## Preprocessing

- `scripts/preprocess_efcc18.py` — load `debias_2018.tgz`, build `(catalog, healpix_tile)` lookup, apply correction to a sharded observation parquet.
- `data/efcc18/{bias.parquet, tiles.parquet, catalog_codes.csv}` with provenance header.

## Pipeline

- One image-build cycle (e.g. `pilot-v13-<date>`) with the preprocessing wired into the cloud shard runner before LOOO.
- One submit on the validated `cpu-ondemand` spec (`cpu=7, mem=16Gi, parallelism=14`). Expected wall time ~5–6h based on v1 baseline.

## Validation gates

- Apply Cloud Image Validation Rules from CLAUDE.md (build-time end-to-end smoke `RUN` step + interactive image audit) before submit.
- Verify EFCC18 lookup hits a reasonable fraction of observations (catalog flag coverage).
- 13-anchor-station spot-check on corrected catalog: residual tail should be tighter than raw v1 for catalog-systematic-dominated stations.

---

# Recommended sequencing

| Cluster | Items | Rationale |
|---|---|---|
| **Aggregation-layer (single PR in v2 publish wrapper)** | 2×2 covariance, AT/CT covariance, correlated-vs-random decomposition | Pure column-add, no pipeline change. |
| **Pipeline / fetch changes** | per-astcat, per-band conditioning | Confirm scope with v2 executor before kickoff. |
| **Publish-layer relaxation** | graded confidence | Small. |
| **Process gate** | OD-utility regression on held-out NEOs | At end of v2 cycle, before declaring published. |

---

# Dependencies

**Depends on (all closed):**
- `1i5` — Bump pinned mpc_obscodes to match cloud image baseline.
- `ctq` — Rewrite `compute_program_code_stats` with `pyarrow.compute.group_by`.
- `fqv` — v1 publication hygiene.
- `ird` — Per-observatory bias table with confidence intervals.

**Blocks:**
- `5z2` — Compare full MPC per-station residual RMS against Veres 2017 Table 1.

**Absorbs:** `j7p` (ITF re-fetch — execute here, not as a standalone splice).

---

# References

- bd memory `looo-catalog-systematic-blindspot` — full reasoning for why LOOO is blind to shared catalog systematics.
- bd memory `external-bias-reference` — why FCCT14/EFCC18 are not directly comparable to LOOO output.
- `/Users/kathleenkiker/claude_cli/debias_2018.tgz` — EFCC18 data on disk.
- `docs/mpc-scale-run-runbook.md` — v1 operational runbook (bead a6w).
- `docs/findorb-warm-start.md` — FindOrb reference-orbit warm-start docs.
- `docs/mpc-bias-catalog-interpretation.md` — downstream consumer interpretation guide.
