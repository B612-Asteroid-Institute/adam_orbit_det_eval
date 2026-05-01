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

Important: the bias-table builder's `max_chi2=100` and `max_object_chi2=50`
filters cleanly excise the catastrophic tail before bootstrap (44,236 → 41,699
→ 30,246 rows used in the bootstrap). So the bias values reported above are
**not contaminated by the tail**. The downstream pipeline does the right
thing. But by the literal exit criterion, this run trips the regression flag.

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

## Recommendations to coordinator

Per bead's standing instructions ("If anchors disagree badly: do not close.
Report findings, file a child bead under ojo describing the disagreement, and
stop. Do not rebuild images or rerun the pilot — that's a coordinator
decision."), the next steps are coordinator decisions:

1. Decide whether the catastrophic residual tail (max |RA| ≈ 261,000") is a
   regression to address before the full cloud run, or an acceptable
   downstream-filtered artifact of the LOOO+evaluate_orbits path.
2. Decide whether to re-pilot on the actual 3,500-object input set
   (`data/looo_sample_3500/`) so the baseline-correctness comparison is
   well-posed (current pilot input was apparently sourced differently and
   produces a mostly-disjoint object set).
3. Either way, file a child bead documenting (1)+(2) and what evidence would
   be sufficient to close ojo.

## Artifacts

- This file: `data/pilot_v11_results/baseline_comparison.md`
- Pilot bias table: `data/bias_catalog/pilot_v11/bias_table.{parquet,csv}`
- Pilot validation report: `data/bias_catalog/pilot_v11/validation_report.txt`
- Pilot config: `data/bias_catalog/pilot_v11/bias_table_config.json`
