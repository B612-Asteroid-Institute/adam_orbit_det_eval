# Does the v1 station-bias catalog improve orbit determination? — Program summary

**Bead:** `od_experiments_setup-h46` · **Branch:** `kk/od-bias-experiments` · **Date:** 2026-05-26
**Program:** `zgf → 9d7 → 9f2 → cph → qsd → 7en → si3 → 8d9` (8 beads)
**Verdict in one line:** No uniform application of the v1 per-station bias catalog improves orbit determination on generic NEOs without regressing short arcs. The only consistent discrepant-set improver is the legacy `v1_subtract`, and it is also the consistent short-arc regressor. Every *principled* winner was refuted by the next, harder cohort — the program ends NO-GO with a clear, narrow operational niche and a concrete next lever.

> **Synthesis note.** This report is synthesis-only: it reads the gitignored per-bead `data/` outputs, the per-bead `REPORT.md` files, and Kathleen's curated project memories. No fits were run. All headline medians and regression counts below were **recomputed from the parquets** and reproduce each bead's published numbers (conventions in §3 and the appendix). Where a bead's own `REPORT.md` disagrees with the user's curated memory, this report **sides with the memory** and flags the tension (see §4.8 and §5.4 — the 8d9 GO/NO-GO reversal).

---

## 1. Question & motivation

The MPC Observatory Bias Study (v1 catalog) publishes a per-station astrometric bias estimate — a mean RA/Dec offset for each observatory, derived by a Leave-One-Observatory-Out (LOOO) procedure over thousands of objects. The catalog *characterizes* systematic error. This program asks the **downstream** question:

> **Does applying the v1 per-station bias numbers at fit time actually improve orbit determination, measured as closeness to the JPL/SBDB nominal orbit?**

The arc was deliberately staged from cheap to expensive evidence:

1. **Single-object pilots (zgf, 9f2)** on 2024 YR4 — is there *any* signal worth chasing?
2. **Population assembly (9d7)** — build a discrepant cohort where bias could matter.
3. **Single-object triage (qsd)** — which of many candidate "levers" deserve population compute?
4. **Population sweeps (cph, 7en)** — do the promising levers hold at scale?
5. **Hold-out validation (si3)** — does the population winner generalise to a fresh cohort?
6. **Remediation (8d9)** — can a targeted fix repair the failure mode the hold-out exposed?

The two-sided tension that drives the whole program: a station bias is *real* signal, so **subtracting** it should help where it dominates; but the catalog reports a *mean* offset with finite uncertainty, so subtracting it as ground truth **over-trusts** it where the bias is comparable to per-observation noise. The first instinct (codified early as a feedback rule) was therefore "use the bias as a σ-floor, never subtract." A central finding of this program is that this rule had to be **revised**: subtraction turned out to be the only thing that consistently moves discrepant fits toward JPL, and the σ-floor was the *worst* principled lever. The framework matured from "don't subtract" to "subtraction is the strongest lever but only safe in a narrow long-arc/discrepant niche — and no σ-based dressing fixes its short-arc damage."

---

## 2. The v1 catalog as an OD input (primer)

**What it is.** A per-station table produced by the LOOO pipeline at `data/mpc_scale_results_20260510/`. For each observatory it reports a mean RA/Dec bias (arcsec), the bias significance, a bootstrap confidence interval, and per-station residual-scatter statistics. Two products are relevant here:

- **`high_confidence_bias_table` (HC rollup)** — RA/Dec mean bias per station, restricted to stations with **n_obs ≥ 100 AND n_objects ≥ 20** (the "HC filter"). **544 stations** survive. This is the table every experiment below consumes.
- **`bias_table` AT/CT** — the same biases projected into along-track / cross-track of each object's sky motion (used by the `v1_at_ct_floor` variant in 7en).

**Sign convention.** `bias_ra_arcsec`, `bias_dec_arcsec` are the station's mean (observed − reference) offset. The **subtract** mode removes it: `RA_used = RA_obs − bias_ra`. The **σ-floor** family never touches RA/Dec; it only inflates the per-observation uncertainty.

**`bias_significant`.** A boolean: the station's bias CI excludes zero (the offset is statistically distinguishable from noise). Used by the `drop_bias_significant` filter and as a candidate gate for selective application.

**The HC filter** selects well-sampled stations — but "well-sampled" correlates strongly with "modern high-volume survey" (Pan-STARRS F51/F52, Catalina G96/703, ATLAS, etc.). This matters: the stations the catalog knows *best* (tightest CIs) are exactly the ones that dominate modern short arcs — a fact that turns out to be the program's central failure mechanism (§5.2).

**EFCC18 coverage caveat.** EFCC18 (star-catalog debiasing) is the published reference debiasing source, but it covers only catalogs that predate it. On modern NEOs its coverage is **0–3%** (2024 YR4: 15/492 obs = 3%; 2025/2026 designations: 0%), rising to 5–72% only for older long-arc objects whose observations used legacy star catalogs. EFCC18 is therefore a **near-no-op** on exactly the objects of greatest interest, and any v1-only-vs-JPL comparison confounds station systematics with star-catalog systematics (JPL applies EFCC18; v1 was fit against pre-EFCC18 residuals).

---

## 3. Methods toolbox — variant taxonomy

Every experiment is a matrix of **bias-application variants** evaluated against the JPL nominal. All variants share the Veres-2017 σ fill-in for missing MPC uncertainties; `no_bias` and `veres_only` are coded identically (kept as the conventional anchor). Per the standing feedback rule, **`v1_subtract` is retained as a labelled legacy/reference anchor in every matrix** so the principled-vs-legacy gap is always visible.

| Class | Variant | Mathematical form | Touches RA/Dec? |
|---|---|---|---|
| **baseline** | `no_bias` / `veres_only` | MPC σ, Veres-2017 fill-in; no bias | no |
| **position-modify** | `v1_subtract` *(legacy/ref)* | `RA_used = RA_obs − bias_ra` (dec likewise) | **yes** |
| | `efcc18_only` | subtract EFCC18 per-(tile,catalog,epoch) correction | yes |
| | `v1_subtract_sem_inflated` (8d9) | subtract bias **and** diagonal `σ² = σ_base² + σ_b²`, `σ_b = (ci_hi−ci_lo)/2/1.96` | yes |
| **σ-modify** | `v1_sigma_floor` | `σ_used = max(σ_reported, \|bias\|)` | no |
| | `v1_RSS_additive` | `σ_used = √(σ_reported² + bias²)` | no |
| | `v1_performance_weighted` | `σ_used = σ_reported · √max(χ²_per_obs_stn, 1)` | no |
| | `v1_covar_inflation` | `Cov += outer((b_ra,b_dec),(b_ra,b_dec))` — **off-diagonal** RA·Dec term | no |
| | `v1_bayes_shrinkage` | shrink bias toward 0 by CI width, then floor σ | no |
| | `v1_at_ct_floor` | σ-floor applied in along-track/cross-track frame | no |
| | `veres_v1_max_floor` | `σ_used = max(σ_MPC, σ_Veres-lookup)` (force Veres model) | no |
| | `uniform_sigma` | flat `σ = 0.5″` for all obs (control) | no |
| **obs-filter** | `drop_non_HC_stations` | drop obs from stations absent from HC table | drops rows |
| | `drop_bias_significant` | drop obs from stations with `bias_significant=True` | drops rows |
| | `drop_high_rms_stations` | drop obs from stations with high residual RMS | drops rows |
| **outlier-reject** | `v1_chi2_outlier_reject` | iterative per-obs χ² rejection using v1 σ | drops rows |
| **stack** | `v1_sigma_floor + efcc18` | σ-floor **and** EFCC18 debiasing together | yes |

**The program's culminating variant** — `v1_subtract_sem_inflated` (8d9) — is the principled attempt to *keep* subtraction's discrepant-set win while *avoiding* its short-arc damage: subtract the bias (so the win survives where σ_b is small relative to baseline σ), but inflate σ diagonally by **how well we know the offset** (σ_b, the CI half-width), so that on poorly-measured stations the inflation dominates and the variant degrades to ≈ `no_bias`. The deliberate contrast with `v1_covar_inflation` is the **diagonal-only** σ inflation: no off-diagonal `b_ra·b_dec` term, because si3 had shown the off-diagonal term breaks survey-dominated short arcs. (As §4.8 shows, the premise failed — the well-measured survey stations have the *smallest* σ_b, so the inflation never engages where it's needed.)

**Metric conventions (validated against every published REPORT).**
- *Discrepant / stratum median* = median `dr_over_sigma` (Δr to JPL in units of JPL position σ) over **converged** fits, restricted to non-pathological χ²∈[0.3, 3] for 7en/si3/8d9. cph predates the pathological flag and reported over all converged fits (this reproduces its 1.61× headline).
- *Control regression* = a control object whose variant `dr_over_sigma` exceeds **2× its per-object baseline** (`veres_only` for cph/7en, `no_bias` for si3/8d9), over all converged fits (no χ² filter).
- *Improvement factor* = baseline median / variant median; >1 means closer to JPL.

---

## 4. Chronological narrative

### 4.1 zgf / 9f2 — 2024 YR4 single-object pilot (`17026ca`, `8ddbcc1`)

The pilot fit 2024 YR4 (492 obs, 62 stations, 88-day arc; 46 stations in the HC table covering 84% of obs) under the first variants and compared the resulting orbit to JPL. **The surprise that set the program's direction:** the only variant that moved the fit *toward* JPL was `v1_subtract` (‖Δr‖ 4.38e-7 AU vs 1.16e-6 baseline, **2.65×**). Every *principled* variant was a no-op or worse — `v1_sigma_floor` **0.67×** (away from JPL), `efcc18_only` 1.02× (EFCC18 covers only 3% of YR4 obs), the stack 0.71×. This directly contradicted the standing "use bias as σ-floor, never subtract" rule and motivated retaining `v1_subtract` as the labelled anchor in everything downstream. **n=1 caveat:** one well-observed object is not a verdict; the point was that there was a signal worth chasing.

### 4.2 9d7 — discrepancy population assembly (`083b636`)

To test bias where it could plausibly matter, 9d7 built a stratified NEO sample (66 attempted, 64 converged) and ranked by Δr/σ vs JPL. **14 objects were flagged discrepant** (Δr/σ > 3): a mix of impact-monitor and long-arc objects (Apophis, Bennu, A898 PA at Δr/σ ≈ 2700, …) whose no-bias fits were statistically incompatible with JPL. Crucially, this set was **hand-selected as the worst-fit tail** — a framing whose limits si3 later exposed. The discrepant set's dominant stations (703, G96, F51, F52, 704, H21, …) are all in the HC table, so the bias correction has numbers to apply. *(9d7's hold-in χ² was pathological — median 4.4e5 — from a `diag_nan=1e-9` covariance path; cph's uniform Veres-2017 σ policy fixed this, median 0.98.)*

### 4.3 qsd — 11-variant YR4 triage (`33ff24c`)

Before spending population compute, qsd triaged 11 levers on YR4 (ranking by ‖Δr‖ to JPL). Results, with χ²_in for context:

| rank | variant | ‖Δr‖ (AU) | × vs no_bias | χ²_in |
|---|---|---|---|---|
| 1 | `v1_subtract` (legacy) | 4.38e-7 | 2.65× | 1.25 |
| 2 | **`v1_performance_weighted`** | 5.38e-7 | **2.16×** | 0.44 |
| 3 | `drop_bias_significant` | 7.65e-7 | 1.52× | 0.77 |
| 4 | `v1_RSS_additive` | 9.39e-7 | 1.24× | 0.91 |
| … | `no_bias` | 1.16e-6 | 1.00× | 1.28 |
| 10 | `uniform_sigma` | 1.74e-6 | 0.67× | 0.16 |
| 11 | **`v1_sigma_floor`** | 1.74e-6 | **0.67×** | 6.85 |

**The key surprise:** `v1_sigma_floor` — the principled *default* under the old rule — was the **worst** principled variant, with χ²_in = 6.85 indicating it was *overweighting* noisy stations rather than down-weighting them. The catalog has real signal, but **floor-by-bias-magnitude is the wrong functional form.** The top principled lever, `v1_performance_weighted` (σ scaled by each station's measured residual scatter √χ²_per_obs), came within 23% of legacy subtract. qsd fanned `v1_performance_weighted`, `drop_bias_significant`, `v1_RSS_additive`, `drop_non_HC_stations` out to 7en, and flagged that the σ-floor rule itself might need revising once population evidence landed.

### 4.4 cph — 24-object population matrix (`359c92c`)

cph ran a 6-variant matrix on 9d7's 14 discrepant NEOs + 10 short-arc controls (uniform Veres-2017 σ; hold-in χ² median 0.98 — the diag_nan pathology is dead). Headline on the discrepant set (median Δr/σ, no_bias = 41.22):

| variant | median Δr/σ | × vs no_bias | control regressions |
|---|---|---|---|
| `v1_subtract` (legacy) | **25.66** | **1.61×** | **4 / 10** |
| `v1_sigma_floor` | 41.05 | 1.00× | 1 / 10 |
| `efcc18_only` | 42.11 | 0.98× | 0 / 10 |
| `v1_sigma_floor + efcc18` | 42.03 | 0.98× | 1 / 10 |

`v1_subtract` was **the only variant moving the discrepant set toward JPL** (1.61×) — and the only one regressing controls (4/10, all short-arc 2025/2026 designations: 2026 DX 9.68×, 2025 QZ10 3.87×, 2026 HZ3 3.09×, 2026 AC3 2.65×). This crystallised the **core mechanism**: subtraction helps when there are enough observations to overwhelm the noise in the bias estimate (long-arc discrepants), and hurts when the bias is comparable to per-obs σ (short arcs). The principled variants were flat; the stack double-corrected (worse than σ-floor alone — overlapping signal on the same biased stations). EFCC18's 0% coverage on 2025/2026 NEOs was confirmed as the limiting factor for modern objects.

### 4.5 7en — 149-object, 17-variant wide sweep (`aaa7243`)

7en fanned to a stratified 149-object population (impact_monitor 34, long_arc 35, main_belt 35, short_arc 45; cph's 24 force-included) across 17 variants — the program's widest design. The question: can a *principled* variant close most of `v1_subtract`'s discrepant-set gap **without** re-inheriting its short-arc regressions?

| variant | 7en discrepant Δr/σ | × vs no_bias | short-arc regressions /45 |
|---|---|---|---|
| `v1_subtract` (legacy) | 25.66 | 1.61× | 9 |
| `v1_performance_weighted` | 28.55 | 1.44× | 8 |
| **`v1_covar_inflation`** | **29.22** | **1.41×** | **2** |
| `v1_sigma_floor` | 41.18 | 1.00× | 1 |

**`v1_covar_inflation` looked like the answer:** 1.41× on discrepants (matching `v1_performance_weighted`, within 14% of legacy) while regressing only **2/45** controls — the first principled variant to thread the needle. `v1_performance_weighted` reproduced its YR4 win on discrepants but inherited subtract's short-arc fragility (8/45). 7en also catalogued several variants out of contention (§5.2): `drop_bias_significant` catastrophic (6% median obs survival, 139/149 objects >50% dropped), `uniform_sigma` underconstrained (41/149 χ²-pathological), `veres_v1_max_floor` best on MBAs but worst on impact-monitor NEOs. 7en recommended a hold-out stress test of `v1_covar_inflation`.

### 4.6 si3 — 100-object fresh hold-out (`7e9b9de`) — **the refutation**

si3 ran the 4 surviving variants on a fresh 100-NEO cohort with **zero overlap** with cph/7en. Verdict: **NO-GO on `v1_covar_inflation`**, for two distinct reasons.

1. **No discrepant-set improvement at scale.** The validation cohort's "discrepant subset" (impact_monitor + long_arc, n=69) has `no_bias` median Δr/σ = **0.76** — already a sub-σ fit to JPL. 7en's 41.22 was that high only because cph **hand-selected the 14 worst-fit objects**. On random NEOs there is *no gap to close*: `v1_covar_inflation` 0.756, `no_bias` 0.755, `v1_subtract` 0.698, `v1_performance_weighted` 0.798 — all within sampling noise. **7en's headline win was an artifact of cohort selection.**
2. **Control regressions 5× worse.** On 30 fresh short-arc controls, `v1_covar_inflation` regressed **20% (6/30)** vs 7en's 4% (2/45); `v1_subtract` 33% (10/30); `v1_performance_weighted` 30% (9/30).

The diagnostic pinned the mechanism: 7en's two regressed controls (2026 DX, 2025 UA3) share stations **F51 (Pan-STARRS 1), F52 (Pan-STARRS 2), H21**. The `outer(b,b)` covariance inflation creates a correlated RA·Dec covariance that, when PS1/PS2 dominate a short arc, over-weights one residual direction and pulls the fit off JPL. The validation cohort's short-arc stratum is dominated by the same survey footprint — hence the regressions multiply. **Confirmed station-specific failure mode**, not sampling noise. si3's durable lesson: **always pair a discrepant-subset score with a control regression rate on random NEOs; never declare a winner from a hand-selected discrepant set alone.**

### 4.7 8d9 — `v1_subtract_sem_inflated` remediation (`97a3a60`)

8d9 tested the targeted fix on a joint cohort (cph 24 + si3 100 = 124 objects, 5 variants). The variant: subtract the bias (keep the win) + **diagonal** σ inflation by σ_b = CI half-width / 1.96 (degrade to no_bias where the offset is poorly known), **no off-diagonal term** (avoid covar_inflation's coupling). Two targets:

- **(a) cph discrepant set — PASS.** `v1_subtract_sem_inflated` median Δr/σ = **28.06**, vs `v1_subtract` 25.66 (9.4% behind — inside the 10% target) and best of the principled variants (beats covar 29.22, perf_wt 28.55). The subtraction win survives.
- **(b) si3 short-arc controls — FAIL.** It regresses **12/30** controls — **identical to `v1_subtract` (12/30)**, and worse than `v1_covar_inflation` (7/30) and `v1_performance_weighted` (8/30).

**Verdict: NO-GO.** The σ_b lever does not engage. The σ_b distribution is p50 = 0.030″, p95 = 0.10″ — all *below* the 0.15–0.3″ baseline σ. The survey stations that dominate short arcs and cause the regressions (F51/F52, 691, 807, G96) are the catalog's **best-measured** → tightest CIs → smallest σ_b (≈0.01–0.03″). So the inflation is negligible exactly where it was needed, the subtraction runs at full strength, and on 29/30 short-arc controls `sem_inflated` lands closer to `v1_subtract` than to `no_bias` — it *is* `v1_subtract` in this regime. The durable reframing: **v1_subtract's short-arc damage is geometric, not an offset-uncertainty problem.** Subtracting a real, well-measured bias from a short arc shifts the whole arc coherently off JPL; σ_b (uncertainty *of the offset*) is smallest precisely where the subtraction does the most harm, so it is the wrong lever.

### 4.8 ⚠ Tension flagged: 8d9's REPORT.md says GO; the memory (and recomputation) say NO-GO

**8d9's own `data/sem_inflated_sweep/REPORT.md` concludes "GO".** It reaches that verdict because its HEADLINE (b) control check ran on a **degenerate control set of n=1** ("0 regressed out of 1 control"), where *every* variant — including `v1_subtract` (which si3 showed regresses 33%) — trivially shows 0 regressions. Kathleen's curated `project_8d9_sem_inflated_result.md` recomputed the regressions over the **full n=30 short-arc stratum** and reversed the verdict to **NO-GO (12/30 = v1_subtract)**. This report **sides with the memory**, and independently confirms it: recomputing si3's exact regression definition over 8d9's full n=30 stratum yields `v1_subtract_sem_inflated` = 12, `v1_subtract` = 12, `v1_covar_inflation` = 7, `v1_performance_weighted` = 8 (see `results_summary.parquet`, cohort `8d9_si3_controls`). The figures and tables here use the **corrected n=30** numbers. *This is the single most important discrepancy in the program — a GO verdict that flips to NO-GO once the control set is evaluated correctly — and it does not appear in any single bead's published REPORT.*

---

## 5. Cross-experiment synthesis

### 5.1 Variant × cohort ranking

Median Δr/σ on discrepant cohorts; control-regression counts on short-arc cohorts; MBA = 7en main-belt median. "—" = variant not run in that bead. (Full machine-readable version in `results_summary.parquet`.)

| variant | YR4 Δr/σ¹ | cph_disc | cph reg /10 | 7en_disc | 7en reg /45 | si3_disc | si3 reg /30 | MBA | 8d9_disc | 8d9 reg /30 |
|---|---|---|---|---|---|---|---|---|---|---|
| `no_bias` | 24.40 | 41.22 | 0 | 41.22 | 0 | 0.755 | 0 | 0.81 | 41.22 | 0 |
| `v1_subtract` *(legacy)* | **9.20** | **25.66** | 4 | **25.66** | 9 | 0.698 | 10 | 1.02 | **25.66** | 12 |
| `v1_sigma_floor` | 36.59 | 41.05 | 1 | 41.18 | 1 | — | — | 0.81 | — | — |
| `efcc18_only` | 23.97 | 42.11 | 0 | 42.11 | 0 | — | — | 0.83 | — | — |
| `v1_sigma_floor+efcc18` | 34.50 | 42.03 | 1 | 42.03 | 1 | — | — | 0.86 | — | — |
| `v1_RSS_additive` | 19.73 | — | — | 41.60 | 2 | — | — | 0.81 | — | — |
| `v1_performance_weighted` | 11.30 | — | — | 28.55 | 8 | 0.798 | 9 | 3.73 | 28.55 | 8 |
| `v1_covar_inflation` | — | — | — | 29.22 | 2 | 0.756 | 6 | 0.79 | 29.22 | 7 |
| `v1_subtract_sem_inflated` | — | — | — | — | — | — | — | — | **28.06** | **12** |
| `drop_non_HC_stations` | 22.23 | — | — | 40.96 | 0 | — | — | 0.81 | — | — |
| `drop_bias_significant` | 16.06 | — | — | 328.06 | 18 | — | — | 456.84 | — | — |
| `drop_high_rms_stations` | — | — | — | 44.84 | 3 | — | — | 0.77 | — | — |
| `uniform_sigma` | 36.46 | — | — | 39.82 | 13 | — | — | 18.39 | — | — |
| `v1_bayes_shrinkage` | — | — | — | 41.56 | 2 | — | — | 0.79 | — | — |
| `v1_at_ct_floor` | — | — | — | 41.23 | 3 | — | — | 0.80 | — | — |
| `v1_chi2_outlier_reject` | — | — | — | 41.48 | 2 | — | — | 0.82 | — | — |
| `veres_v1_max_floor` | — | — | — | 58.54 | 11 | — | — | 0.75 | — | — |

¹ YR4 is a single object; lower Δr/σ = better (the ordering matches qsd's ‖Δr‖ ranking — `v1_subtract` best, `v1_sigma_floor` worst).

**The two structural facts the table makes visible** (and `figures/fig1_winner_flips_across_cohorts.png`, `fig2_control_regression_rates.png`):

- **The principled winner flips across cohorts.** qsd → `v1_performance_weighted`; 7en → `v1_covar_inflation`; si3 → *nobody* (all tie no_bias); 8d9 → none survives the control test. No principled variant leads two consecutive cohorts.
- **`v1_subtract` is the only consistent leader on hand-selected discrepants (1.61× on cph/7en/8d9, 2.65× on YR4) AND the consistent regressor on short-arc controls (4/10, 9/45, 10/30, 12/30).** It is simultaneously the best and the most damaging — the defining tension of the program.

### 5.2 Failure modes catalogued

| # | Failure mode | Bead(s) | Mechanism | Evidence |
|---|---|---|---|---|
| F1 | **`outer(b,b)` over-coupling** | si3 | off-diagonal RA·Dec covariance over-weights one residual direction when PS1/PS2 dominate a short arc | 2026 DX, 2025 UA3 both regress; shared stations **F51, F52, H21**; si3 reg 20% vs 7en 4% |
| F2 | **Geometric coherent shift** | 8d9 | subtracting a well-measured bias shifts a short arc coherently off JPL; σ_b too small (best-measured stations) to damp it | `sem_inflated` ≈ `v1_subtract` on 29/30 controls; 12/30 regressions identical |
| F3 | **`uniform_sigma` underconstraint** | qsd, 7en | flat 0.5″ σ is too large; fits go unconstrained (χ²_in ≪ 1) | χ²_in = 0.16 on YR4; **41/149** χ²-pathological in 7en (highest of all variants) |
| F4 | **`drop_bias_significant` mass obs-drop** | 7en | at scale, most obs come from `bias_significant` survey stations | **6% median obs survival; 139/149 objects lose >50% of obs**; MBA Δr/σ 456.8 |
| F5 | **MBA–NEO σ-source split** | 7en | forcing Veres σ penalises modern surveys (Gaia-class σ < Veres default) | `veres_v1_max_floor` best on MBA (0.745) but worst on impact-monitor NEOs (0.70× vs no_bias); see `fig5_mba_neo_sigma_split.png` |
| F6 | **Hand-selected-discrepant overfit** | si3 | random NEOs already fit JPL (median Δr/σ 0.76); the "gap to close" is an artifact of selecting the worst-fit tail | 7en disc 41.22 → si3 disc 0.76 for no_bias; all variants tie within noise |

F1 and F2 are **two distinct mechanisms for the same symptom** (short-arc regression on PS1/PS2/H21-dominated arcs), visible side-by-side in `fig3_ps1ps2h21_failure_mode.png`: on 2026 DX, `v1_subtract` and `sem_inflated` both sit at Δr/σ ≈ 0.12 (≈9.6× no_bias — geometric shift), while `covar_inflation` sits lower at 0.037 (it *downweights* via the larger `|bias|` magnitude, paying instead with F1's off-diagonal degeneracy). This is why covar regresses *fewer* controls than subtract (7 vs 12) yet still fails — the two failure modes trade off but neither is eliminated.

### 5.3 The principled-vs-legacy gap arc

The gap between the best *principled* variant and the legacy `v1_subtract` anchor **closed, then reopened, then closed-then-collapsed**:

- **qsd (YR4):** gap closes to 23% (`v1_performance_weighted` 2.16× vs subtract 2.65×).
- **7en (149-obj):** gap closes to 14% (`v1_covar_inflation` 1.41× vs 1.61×) *and* with far fewer control regressions (2 vs 9) — looked like a genuine principled win.
- **si3 (hold-out):** gap is **meaningless** — both principled and legacy tie no_bias on a cohort with no room to improve, and all variants regress controls at 20–33%.
- **8d9 (remediation):** the discrepant gap closes again (`sem_inflated` 0.91× of subtract) but the control gap **collapses to zero** — `sem_inflated` regresses identically to subtract. The principled variant became the legacy variant.

### 5.4 Why the principled winners kept getting refuted

Three compounding reasons, each a transferable lesson:

1. **Cheap cohorts overfit.** YR4 (n=1) and the hand-selected 14-object discrepant set are not representative. A lever tuned to look good there (`v1_performance_weighted` on YR4, `v1_covar_inflation` on cph's discrepants) had no reason to generalise — and didn't (si3).
2. **The "discrepant" framing manufactured the signal.** Selecting the worst-fit tail guarantees a gap that *any* aggressive correction can partially close; on random NEOs (si3) the gap evaporates. The headline improvement factors were partly measuring cohort selection, not catalog quality.
3. **The failure is structural, not parametric.** Every σ-based dressing (floor, RSS, covar, SEM) either fails to engage (σ_b/floor too small on best-measured stations) or introduces a new pathology (off-diagonal coupling, underconstraint). The damage is geometric: any "apply station bias uniformly to all obs from a listed station" scheme coherently shifts short arcs. No reweighting of a uniform per-station offset escapes this.

---

## 6. Where this leaves the v1 catalog as an OD input

**Bottom line: do not adopt any v1 bias-application mode as a blanket OD default.** On the only generalisation test in the program (si3's fresh random-NEO cohort), no variant — principled or legacy — beat `no_bias` on the discrepant subset, and all regressed short-arc controls at 20–33%. The catalog characterises real systematics, but applying a uniform per-station offset at fit time is not, as tested, a net improvement for general orbit determination.

**Defensible mode-by-mode recommendation given current evidence:**

- **`v1_subtract` — use only as a narrow remediation tool, never as a default.** It is the only consistent discrepant-set improver (1.6× on long-arc discrepants). Restrict to fits that are *already* statistically incompatible with JPL (Δr/σ > 3) **and** have a long arc / high obs count, where there are enough observations to overwhelm the bias-estimate noise. **Never apply to short arcs** (it coherently shifts them off — 4/10, 9/45, 10/30, 12/30 control regressions across cohorts). An arc-length / obs-count gate is mandatory.
- **`v1_sigma_floor` (the old "principled default") — do not use.** It was the worst principled lever (qsd χ²_in = 6.85, *overweighting* noisy stations) and a near-no-op on discrepants. Floor-by-bias-magnitude is the wrong functional form. *This supersedes the original "use bias as a σ-floor" feedback rule.*
- **`v1_covar_inflation` and `v1_subtract_sem_inflated` — do not use.** Both refuted: covar by F1 (off-diagonal coupling, doesn't generalise), sem_inflated by F2 (σ_b too small to engage; reduces to subtract). Retire the "uncertainty-of-offset" inflation family (`bayes_shrinkage`, `sem_inflated`) — catalog CIs are too tight relative to baseline σ to move the regression axis.
- **`veres_v1_max_floor` — use for MBAs only, never for modern-survey NEOs.** Best MBA variant (0.745) but worst on impact monitors; forcing Veres σ penalises Gaia-class astrometry.
- **`drop_bias_significant` — drop from all future matrices.** Catastrophic at scale (6% obs survival).
- **EFCC18 — not a substitute on modern NEOs** (0–3% coverage); only relevant for older long-arc objects.

**Open questions / what would settle them:**

1. **Arc-length-gated subtraction** (the untested lever the 8d9 memory flags as promising): apply `v1_subtract` only when arc > N days, else fall back to `no_bias`/σ-floor. This targets F2 (the geometric failure) directly. *Would settle:* whether subtraction's long-arc win can be captured without the short-arc damage. **Recommended next experiment.**
2. **Per-epoch / per-night bias decorrelation** — the structural alternative to "uniform per-station offset." *Would settle:* whether the coherent-short-arc-shift is intrinsic to any per-station scheme or specific to applying a single static offset.
3. **A true discrepant-only cohort** (random NEOs filtered to no_bias Δr/σ > 3, fresh from cph/7en/si3) — to test whether `v1_subtract`'s remediation niche generalises beyond hand-picked objects. *Would settle:* whether recommendation #1's niche is real or another selection artifact.
4. **AT/CT decomposition** is largely untested as a weighting input (`v1_at_ct_floor` was a no-op in 7en at 41.23); whether timing-vs-trailing-resolved biases help remains open.

The catalog's value as an OD input is, on this evidence, **diagnostic** (flag which stations and which fits are bias-affected) rather than **corrective** (mechanically fix the fit). The corrective path requires either arc-aware gating or per-epoch decorrelation, neither yet tested.

---

## 7. Reproducibility appendix

All `data/` paths are **gitignored locators** (not committed); they live in this worktree at `adam_orbit_det_eval/`.

| Bead | Commit | Role | Output files (under `data/`) |
|---|---|---|---|
| `zgf` | `17026ca` | YR4 single-object pilot (v1_subtract) | `yr4_experiment/{observations,no_bias_orbit,v1_subtract_orbit,jpl_orbit,comparison_*}.parquet` |
| `9d7` | `083b636` | discrepancy population builder | `od_discrepancy_population/{discrepancy_ranking.parquet,failures.json,REPORT.md}` |
| `9f2` | `8ddbcc1` | YR4 6-variant + EFCC18 hook | `yr4_experiment/REPORT.md`, `comparison_summary.parquet` |
| `cph` | `359c92c` | 24-obj 6-variant matrix | `od_discrepancy_population/{variant_comparison.parquet (144),REPORT_variants.md}` |
| `qsd` | `33ff24c` | YR4 11-variant triage | `yr4_experiment/{comparison_summary_expanded.parquet (11),REPORT.md}` |
| `7en` | `aaa7243` | 149-obj 17-variant sweep | `wide_variant_sweep/{variant_comparison.parquet (2533),population_manifest.parquet,REPORT.md}` |
| `si3` | `7e9b9de` | 100-obj hold-out (4 variants) | `validation_sweep/{variant_comparison.parquet (400),cohort_manifest.parquet,REPORT.md}` |
| `8d9` | `97a3a60` | SEM-inflated remediation (5 variants) | `sem_inflated_sweep/{variant_comparison.parquet (620),joint_manifest.parquet,REPORT.md}` |
| `h46` | *(this commit)* | program summary | `reports/v1_bias_catalog_od_evaluation/{REPORT.md,results_summary.parquet,figures/*.png}` |

**Deliverables in this report directory:**
- `REPORT.md` — this narrative.
- `results_summary.parquet` — 131 rows, one per (cohort, variant); columns `cohort_name, variant, n_objects, median_dr_over_sigma, p95_dr_over_sigma, median_chi2_in, n_control_regressions, n_chi2_pathological, bead_id, commit_sha`. 13 cohorts: `yr4`, `cph_{discrepant,controls}`, `7en_{discrepant,controls,mba,impact_monitor,long_arc}`, `si3_{discrepant,controls}`, `8d9_{cph_discrepant,si3_controls,joint}`.
- `figures/` — `fig1_winner_flips_across_cohorts.png`, `fig2_control_regression_rates.png`, `fig3_ps1ps2h21_failure_mode.png`, `fig4_chronological_leader.png`, `fig5_mba_neo_sigma_split.png`.
- `_scripts/figures.py` — regenerates the parquet and figures from the per-bead `data/` parquets (synthesis only; runs no fits).

> **Caveat on 8d9 numbers.** The `8d9_si3_controls` regression counts in `results_summary.parquet` (and every 8d9 control figure here) are **recomputed over the full n=30 short-arc stratum**, which reproduces the user's `project_8d9` memory NO-GO verdict. They intentionally **differ** from 8d9's published `data/sem_inflated_sweep/REPORT.md`, whose HEADLINE (b) "0 regressions / GO" rests on a degenerate n=1 control set (see §4.8).
