# MPC Bias Catalog — Interpretation Guide

This note accompanies the published v12 LOOO catalog under
`data/mpc_scale_results_20260510/` (specifically
`merged_looo_results_published.parquet` and
`observatory_stats_published.parquet`). It documents the methodology choices
that a downstream reader needs in order to interpret the per-station bias and
scatter numbers correctly, and the publication-hygiene filters applied to
the v12 catalog before release.

Companion artefacts:
- `publication_hygiene_audit.json` — exact row counts before and after each
  filter (machine-readable).
- `bias_filter_stats.json` — aggregation-time bad-fit filter audit
  (bead 7bt).
- `collection_report.json` — shard / object / row counts from the cloud run.

## 1. The catalog does not consume reported sigmas

The LOOO method derives bias and scatter from the *empirical* distribution
of `(observed − predicted)` residuals computed by holding out an observatory
and refitting the orbit on the remaining observations. The hold-in fit
does use the reported `rmsra`/`rmsdec` (with a Veres 2017 fallback for
missing values) because some kind of weighting is required to do
least-squares; but the per-station bias and scatter numbers in the catalog
are *not weighted by those sigmas*. The aggregations operate on raw
residuals.

This is intentional. The point of the study is to derive per-station
calibration numbers that do **not** depend on the observatory's own
self-report of its astrometric uncertainty. Some stations under-report
their errors, sometimes by an order of magnitude; the empirical residual
distribution is the ground truth we are after.

## 2. Reduced χ² ≫ 1 against reported σ is the *signal*, not a bug

The `chi2` column in `merged_looo_results_published.parquet` is computed as
`(residual_ra / sigma_ra)² + (residual_dec / sigma_dec)²` where the sigmas
are the *reported* per-observation values (with Veres-2017 fallback). For
stations whose reported sigmas are well-calibrated, the per-observation
χ² should be ~2 on average (two degrees of freedom — RA and Dec). When the
empirical residual RMS is much larger than the reported σ, the reduced χ²
will be ≫ 1.

A reader who interprets reduced χ² > 1 as "the fit is bad" will reach the
wrong conclusion. For this catalog, **reduced χ² > 1 is empirical evidence
that the station is under-reporting its astrometric uncertainty**. We do
not "fix" that by clamping or flooring sigmas — doing so would discard the
very signal the study is built to surface.

The authoritative noise estimate per station is the empirical residual RMS
(`rms_ra_arcsec`, `rms_dec_arcsec` in `observatory_stats_published.parquet`,
and the `rms_ra_arcsec` / `rms_dec_arcsec` columns of `bias_table.parquet`).
The reported-sigma-based χ² columns are kept for diagnostic comparison; they
are not the recommended weighting source for downstream consumers.

### Anchor stations: empirical vs reported sigma

The table below summarises, for the 13 anchor stations used in the
`scripts/17_generate_bias_table.py --validate-anchors` check, how the
empirical residual RMS compares to the median reported `rmsra` /
`rmsdec` for that station in the v12 source data. Stations marked
`n/a` have no reported per-observation sigmas in the source MPC data (the
ADES `rmsra`/`rmsdec` columns are not historically populated for some
stations / epochs).

| stn  | empirical RMS RA (″) | median reported `rmsra` (″) | RA factor (empirical / reported) | empirical RMS Dec (″) | median reported `rmsdec` (″) | Dec factor |
|------|---------------------:|----------------------------:|---------------------------------:|----------------------:|-----------------------------:|-----------:|
| 704  | 0.69                 | n/a                          | n/a                              | 0.50                  | n/a                          | n/a       |
| 699  | 0.64                 | n/a                          | n/a                              | 0.54                  | n/a                          | n/a       |
| 703  | 0.58                 | 0.385                        | 1.52                             | 0.53                  | 0.388                        | 1.37      |
| W84  | 0.16                 | 0.136                        | 1.18                             | 0.13                  | 0.139                        | 0.94      |
| 809  | 1.21                 | n/a                          | n/a                              | 1.17                  | n/a                          | n/a       |
| F52  | 0.10                 | 0.160                        | 0.64                             | 0.10                  | 0.153                        | 0.66      |
| M22  | 0.29                 | 0.389                        | 0.73                             | 0.29                  | 0.389                        | 0.75      |
| R17  | 0.20                 | 0.300                        | 0.66                             | 0.19                  | 0.299                        | 0.65      |
| T09  | 0.21                 | 0.057                        | 3.73                             | 0.17                  | 0.057                        | 2.98      |
| W68  | 0.28                 | 0.345                        | 0.80                             | 0.28                  | 0.345                        | 0.81      |
| 691  | 0.43                 | 0.100                        | 4.26                             | 0.35                  | 0.100                        | 3.47      |
| 644  | 0.37                 | 0.500                        | 0.75                             | 0.48                  | 0.500                        | 0.96      |
| 705  | 0.33                 | n/a                          | n/a                              | 0.30                  | n/a                          | n/a       |

Three observations from the table:

- **T09 and 691 under-report by factors of 3–4×.** Their LOOO residuals will
  exhibit reduced χ² ≫ 1 against the reported sigmas. This is the signal,
  not noise.
- **F52, M22, R17, 703 over-report by 30–50% on one or both axes.** Reduced
  χ² < 1 for these stations is expected and is also a real measurement,
  not "good calibration".
- **704, 699, 809, 705 have no reported per-obs sigmas.** Their empirical
  RMS is the *only* noise estimate available. Anyone consuming these
  stations' values from a reported-sigma-based weighting scheme will be
  using a fallback (Veres 2017 in our pipeline; whatever the consumer
  chooses elsewhere).

These factors are informational and intentionally not a "calibration
target". The catalog publishes the empirical residual distribution per
station; downstream consumers are free to derive their own weighting
schemes from it.

## 3. Publication-hygiene filters applied to v12

`merged_looo_results_published.parquet` and `observatory_stats_published.parquet`
are the result of applying three publication-time filters to the
post-bias-filter merged catalog (bead `fqv`). The audit trail with exact
row counts is in `publication_hygiene_audit.json`.

### 3.1 mode='OCC' drop

Occultation-timing records (`mode='OCC'` in MPC's ADES schema) are a
fundamentally different astrometric measurement and must not contribute to
a per-station optical-residual bias catalog. In the v12 source data there
are 103 mode='OCC' observations, all attributed to station 275 (a
space-based / synthetic code). The LOOO pipeline excludes space-based
stations during OD preparation
(`adam_orbit_det_eval.utils.get_spacebased_stns`), so none of the 103 OCC
rows reach the merged catalog. The filter is applied unconditionally
anyway as a hard guard against future inputs that route occultation rows
through the pipeline differently — for the v12 catalog it is a no-op
(0 rows dropped).

### 3.2 Unknown observatory codes

The published merged catalog drops rows whose `stn` is not present in the
adam_core observatory code table used by the local validation environment
(`adam_core.observers.OBSERVATORY_CODES`). This filter is a live check
against the installed `mpc_obscodes` table, not a hard-coded list.

For the v12 catalog the local pin has been bumped to
`mpc_obscodes>=2026.3.25` (bead `1i5`), matching the snapshot baked into
the cloud image (`pilot-v12-20260507`). With that pin all 1,228 observatory
codes present in the merged input resolve locally and the unknown-code
filter drops **0 rows** from the v12 published catalog.

Prior to the pin bump the filter dropped 89 rows attributed to 4 codes
across 6 distinct objects (O58, S42, W26, X26) — codes that were added
to MPC after the previously pinned `mpc_obscodes==2026.3.12` release. The
published v12 catalog now includes these rows:

| stn  | rows | distinct objects |
|------|-----:|-----------------:|
| O58  | 21   | 3                |
| S42  | 21   | 3                |
| W26  | 17   | 3                |
| X26  | 30   | 3                |

None of the 13 anchor stations is affected. The four newly-included codes
all fail the small-sample cutoff (§3.3) and therefore do not appear in
`observatory_stats_published.parquet`. In `bias_table.parquet` S42 alone
clears the looser bias-table cutoff (`n_obs ≥ 10` AND `n_objects ≥ 3`
after the per-row quality filter); O58, W26, and X26 fall below the
cutoff once the per-row chi² filter is applied.

### 3.3 Small-sample cutoff

`observatory_stats_published.parquet` drops rows where `n_obs < 100` or
`n_objects < 20`. Per-station numbers below these thresholds are
statistically noisy: their bootstrap CIs span a factor of several, so the
point estimates and CIs they expose are not publication-grade. The cutoff
takes the v12 observatory_stats from 1,228 stations to **544 stations**
(684 dropped — including the four newly-recovered codes O58, S42, W26,
X26 that were unknown before the `mpc_obscodes` pin bump in §3.2). The
row-level `merged_looo_results_published.parquet` keeps every surviving
row regardless of station n_obs, so downstream consumers who *want*
small-sample stations can re-aggregate from the row level with their own
thresholds.

The cutoff filters by AND of (n_obs ≥ 100, n_objects ≥ 20), not OR. A
station with 200 observations across only 5 objects would still fail —
that case suggests the per-object scatter is dominating the per-station
average, and a 5-object resample is not enough to put a meaningful CI on
the mean.

The choice to *drop* failing stations rather than mark them
`published=False` is deliberate: a consumer should be able to read the
file and trust every row.

### 3.4 Verification against anchor stations

The 13 anchor stations from the 3,500-object reference run
(`data/bias_catalog/3500obj/`) are all present in the published v12 stats
with identical n_obs, n_objects, and per-station RMS values as the
pre-hygiene v12 catalog (the four newly-recovered codes O58/S42/W26/X26
are not among the anchors, so the pin bump and the unknown-code filter
are both no-ops for them). Running
`scripts/17_generate_bias_table.py --validate-anchors` on
`merged_looo_results_published.parquet` produces the same anchor pass
count as on the pre-hygiene merged input — hygiene does not regress
anchor-station behaviour.

The pass count itself (6/13 against the 3,500-obj reference tolerances)
is not 13/13 because the anchor expected values were calibrated against
the 3,500-obj run and v12 point estimates differ slightly with the larger
object set. That is a separate calibration question; the published v12
catalog supersedes the 3,500-obj reference as the bias source of record.

### 3.5 No object-level quality filter at aggregation (v1)

`compute_bias_table` previously defaulted to dropping objects whose
mean held-out reduced χ² exceeded 50 (`max_object_mean_chi2=50.0`). The
intent was to suppress non-gravitational-force contamination (comets,
high-Yarkovsky NEOs) at the aggregation step.

That filter is disabled in v1 (`max_object_mean_chi2=None`) because the
χ² it consumes is built from MPC-reported per-observation sigmas, and a
small number of stations report pathologically small sigmas (N86 reports
nanoarcsec-scale σ in some submissions). On any object N86 also observed,
N86's term in the per-object mean χ² dominates and drives the object's
mean χ² to ~10⁸, indicting the object across *every* station that
observed it. Reconciliation under the previous default dropped ~373,000
rows across 780 stations from the per-station rollup — almost all of
them clean data — purely because N86 happened to share an object with
that station. Of the 261 objects globally flagged by the old filter,
only ~28 were legitimately high-χ²; ~233 were collateral damage from
N86's sigma reporting. (See `bias_catalog_published/reconciliation_544_vs_520.txt`
and bead `zw0` for the full root-cause trace.)

Consequence for v1 readers:

- The 100/20 small-sample cutoff (§3.3) is the only object-level
  quality gate at the publication-hygiene boundary. Bias values are
  empirical residual means and are independent of the reported sigmas,
  so removing the σ-dependent filter does not corrupt the bias estimate
  itself — it restores rows that were unfairly hidden.
- N86 now appears in the published `bias_table` with `bias_ra`,
  `bias_dec`, and AT/CT values in a perfectly reasonable arcsec-scale
  range, but with `chi2_per_obs` in the 10⁷–10⁸ range. **That huge
  χ² is honest reporting of N86's broken sigma stream, not a numerical
  problem with N86's bias estimate.** Downstream consumers who want
  to suppress non-gravitational-force objects should apply an
  object-level filter that does *not* depend on the reported σ — see
  v2 below.
- For program-level (per-(stn, program_code)) rows the same caveat
  applies. A handful of (stn, program_code) groups will have
  similarly inflated `chi2_per_obs` and should be read with the same
  caveat: the bias values are fine; the χ² column tells you about the
  observer's σ reports.

v2 (bead `54t`) replaces this filter with a σ-independent
residual-magnitude proxy (drop objects whose held-out residual
magnitude exceeds a robust catalog-wide percentile). That avoids
the sigma-contamination failure mode while still suppressing
genuine non-gravitational-force objects from the aggregation. v1
ships without that replacement and without the broken original; the
denominator is the same 544 stations as `observatory_stats_published`.

## 4. Pointers

- Bead `5z2` — comparison of the published catalog against FCCT14 reference
  weights. The v12 catalog should be compared against the publication-
  hygiene-cleaned siblings, not the raw merged output.
- `src/adam_orbit_det_eval/looo/publication_hygiene.py` — the filter
  implementation.
- `scripts/18_apply_publication_hygiene.py` — the reproducible CLI that
  produced the `_published` parquets.
- `src/adam_orbit_det_eval/looo/bias_filter.py` — the aggregation-time
  bad-fit filter (bead 7bt) that runs *before* publication hygiene.
- `src/adam_orbit_det_eval/looo/bias_table.py` /
  `scripts/17_generate_bias_table.py` — the bootstrap bias-table generator
  used for FCCT14-style downstream comparison.
