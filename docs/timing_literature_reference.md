# Timing Error Detection in Asteroid Astrometry: Literature Review and Simulation Results

**Status:** Living reference document
**Last updated:** 2026-04-06
**Context:** Leave-One-Observatory-Out (LOOO) pipeline evaluation for 1-second timing error detection

---

## 1. Background and Motivation

Astrometric observations submitted to the Minor Planet Center include a reported UTC timestamp for each measurement. A timing error — a systematic offset between the true exposure midpoint and the reported time — produces an along-track (AT) displacement in the measured position proportional to the object's sky-plane rate of motion:

```
AT displacement = rate × Δt
```

where rate is in arcsec/s and Δt is the timing offset in seconds. Because this displacement is purely along the direction of motion, it projects into right ascension and declination in a rate- and direction-dependent way, making it look different from a simple RA or Dec bias.

Timing errors can arise from multiple sources: incorrect NTP synchronization, FITS header bugs that record shutter open rather than midpoint, incorrect exposure-midpoint corrections, or time zone confusion. The most dramatic documented case is the Catalina Sky Survey (CSS), which operated for an extended period with a −12-second timing error caused by a FITS header bug. This was discovered through orbit determination failures — not through any systematic astrometric analysis — underscoring the difficulty of the problem.

The Leave-One-Observatory-Out (LOOO) pipeline holds out each MPC observatory's observations, refits an orbit from the remaining stations, and measures the mean AT residual for the held-out station. We have been evaluating whether 1-second timing errors are detectable this way across populations of ~100 MPC objects per bias type.

---

## 2. Key Literature

### 2.1 Papers Table

| Reference | Journal | Topic | Relevance |
|---|---|---|---|
| Farnocchia et al. 2022 | PSJ 3, 156 | IAWN campaign on 2019 XS; timing bias detection | Primary comparison: method and results |
| Farnocchia et al. 2023 | PSJ 4, 203 | IAWN campaign on 2005 LW3; timing bias follow-up | Confirms network-wide negative bias |
| Carpino, Milani & Chesley 2003 | Icarus 166, 248 | Error statistics; same-night correlation structure | Effective-N argument for sensitivity |
| Veres et al. 2017 | Icarus 296 | Rate-of-motion dependent error model (MPC weighting) | Implicit timing sensitivity; no explicit AT/CT |
| Stronati et al. 2023 | MNRAS 521 | NEA vs. MBA error stratification; AT/CT analysis | Station-level timing signatures; not isolated |

### 2.2 Farnocchia et al. 2022 — IAWN Campaign on 2019 XS

**Citation:** Farnocchia, D., et al. 2022. "Astrometric Errors Due to Timing Biases: An IAWN Campaign." *The Planetary Science Journal* 3, 156.

**Object and geometry:** 2019 XS made a close approach in 2021 December, reaching a peak sky-plane rate of 3.8 arcsec/s at closest approach. This is roughly three orders of magnitude faster than a typical MBA and ~200× faster than a fast NEO in the general MPC catalog.

**Dataset:** 957 observations from approximately 50 participating IAWN stations, observed over the close-approach window when rates were highest.

**Method:** AT/CT decomposition. The AT component (along the direction of motion) is sensitive to timing errors; the CT component (perpendicular) is not. With an externally-anchored orbit derived from radar ranging and pre-close-approach optical data, the AT residuals of each station are not absorbed into the orbit fit. This is critical — the reference orbit is held fixed, so there is no "parameter absorption" of the timing signal.

**Results:**
- Network-wide timing bias: −0.5 s (reported times were, on average, 0.5 seconds early relative to truth)
- Station Y00: −0.66 ± 0.05 s (6-sigma detection at the individual-station level)
- Several other stations showed 1–3σ hints; most showed no detectable bias
- The negative sign (reported times earlier than actual) is consistent across both campaigns and suggests a systematic origin, possibly in exposure-midpoint calculation conventions

**Why this works:** Per-observation SNR. With a rate of 3.8 arcsec/s, a 0.5-second timing error produces a 1.9-arcsec AT displacement per observation. With typical astrometric scatter of ~0.3 arcsec, that is SNR ≈ 6 per observation. Integrating over 957 observations yields a fleet-level SNR of roughly 6 × √957 ≈ 185. Detection is trivial.

### 2.3 Farnocchia et al. 2023 — IAWN Campaign on 2005 LW3

**Citation:** Farnocchia, D., et al. 2023. "Timing Bias Analysis from the 2022 IAWN Campaign on 2005 LW3." *The Planetary Science Journal* 4, 203.

**Object and geometry:** 2005 LW3 passed at a slower rate than 2019 XS — peak rate ~2.4 arcsec/s — but still 100–250× faster than typical catalog objects at the relevant observed epochs.

**Dataset:** 1046 observations from 82 stations.

**Results:** The network-wide negative bias persisted. Individual-station detections were fewer than in the 2019 XS campaign (consistent with the lower peak rate). The authors concluded that timing calibration should be treated as "an important future consideration" for MPC data quality, particularly as LSST-era precision increases.

**Notable:** This campaign used a less favorable geometry (slower rates, shorter peak window), and the results were correspondingly noisier. It demonstrates that even at 2.4 arcsec/s, a 1-second error is detectable with sufficient observations — but the threshold is steep.

### 2.4 Carpino, Milani & Chesley 2003 — Error Correlations

**Citation:** Carpino, M., Milani, A., & Chesley, S. R. 2003. "Error Statistics of Asteroid Optical Astrometry." *Icarus* 166, 248–270.

**Key finding for timing:** Observations taken on the same night by the same observatory share the same systematic errors — including timing errors, atmospheric refraction residuals, and catalog systematics. This means the effective number of independent observations for detecting a systematic bias is not N_obs but approximately N_nights (number of distinct observing nights).

**Implication for LOOO timing detection:** If a station has 10,000 observations spread across 1,000 nights (10 observations per night on average), the effective sample size for timing detection is ~1,000, not 10,000. Combined with low per-observation SNR (~0.03), the effective SNR for detection is:

```
SNR_eff = per_obs_SNR × √N_nights ≈ 0.03 × √1000 ≈ 0.95
```

This is below detection threshold even with a generous number of nights.

**Broader context:** Carpino et al. introduced the framework of debiasing astrometry by computing per-station, per-catalog residual statistics — the same framework later operationalized by the Chesley group and incorporated into the Veres 2017 MPC weighting scheme. Their correlation model is the standard reference for why naive obs-counting overstates sensitivity to systematic effects.

### 2.5 Veres et al. 2017 — Rate-Dependent Error Model

**Citation:** Veres, P., et al. 2017. "Improved Astrometric Error Model for Asteroid Surveys." *Icarus* 296, 139–149.

**Relevance:** The Veres weighting model assigns observation uncertainties as a function of the object's sky-plane rate of motion, catalog generation used, and observatory. Rate-dependent weights implicitly encode the fact that fast-moving objects have larger AT errors from any fixed timing offset — but the model does not decompose into AT/CT components or fit timing offsets explicitly. There is no Δt parameter in the Veres scheme.

**Gap:** The Veres model absorbs timing effects into the empirical uncertainty floor for each station. This means timing biases are partly hidden inside the weighting model rather than isolated. A rate-stratified AT analysis (plotting mean AT residual vs. rate per station) would be complementary to Veres weighting and could reveal timing offsets the current scheme obscures.

### 2.6 Stronati et al. 2023 — AT/CT Station Stratification

**Citation:** Stronati, L., et al. 2023. "Systematic Astrometric Errors in Near-Earth Asteroid and Main-Belt Asteroid Populations." *MNRAS* 521.

**Method:** Compared AT and CT residuals separately for NEA and MBA populations, stratified by observatory. Found stations where AT/CT ratios differ significantly between the two populations.

**Relevance:** The station-level AT/CT differences are qualitatively consistent with timing biases (which would appear primarily in AT and would differ between NEAs and MBAs because the populations have different rate distributions). However, the paper does not isolate timing as the cause — other systematics (catalog errors, differential refraction, trail smearing) could produce similar patterns. It represents the closest thing in the published literature to what our LOOO timing analysis attempts, but without the explicit timing injection and without the AT rate-slope diagnostic.

---

## 3. Signal Scaling and Detection Thresholds

### 3.1 Per-Observation SNR Formula

The fundamental sensitivity equation for timing detection is:

```
SNR_per_obs = (rate × Δt) / σ_obs
```

where:
- `rate` = sky-plane rate of motion in arcsec/s
- `Δt` = timing error in seconds
- `σ_obs` = per-observation astrometric uncertainty in arcsec

For fleet-level detection with N independent observations:

```
SNR_fleet = SNR_per_obs × √N
```

With the Carpino same-night correlation correction, replace N with N_nights.

### 3.2 Comparison Table

| Population | Rate (arcsec/s) | Δt (s) | σ_obs (arcsec) | SNR/obs | N for 3σ |
|---|---|---|---|---|---|
| IAWN 2019 XS (peak) | 3.8 | 0.5 | 0.3 | ~6 | ~0.25 |
| IAWN 2005 LW3 (peak) | 2.4 | 0.5 | 0.3 | ~4 | ~0.6 |
| Our NEOs | 0.021 | 1.0 | 0.3 | ~0.07 | ~1,800,000 |
| Our MBAs | 0.009 | 1.0 | 0.3 | ~0.03 | ~10,000,000 |

The "N for 3σ" column gives the number of fully independent observations required to reach a 3-sigma detection. For MBAs, this is ~10 million independent observations — roughly the entire MPC catalog for a single station over its lifetime, treated as fully independent. With the Carpino correction (N_nights < N_obs by a factor of ~10), this becomes even more pessimistic.

### 3.3 Rate Conversion

The simulations use rates in arcsec/day. To convert:

```
rate (arcsec/s) = rate (arcsec/day) / 86400
```

MBA mean rate in our simulations: 750 arcsec/day = 0.0087 arcsec/s
NEO mean rate in our simulations: 1800 arcsec/day = 0.0208 arcsec/s

---

## 4. Our Simulation Results

### 4.1 Setup

- Population size: 100 objects per run, drawn from real MPC orbits
- Bias types tested: TimingBias (1 s), ConstantBias, TrailingBias
- Observatories: 9 MPC stations (mix of survey and follow-up)
- Detection criterion: AT z-score > 3σ at fleet level or per-station level
- Pipeline: LOOO — hold out one observatory, refit orbit on complement, measure mean AT residual of held-out station

### 4.2 MBA Results (100 objects, mean rate ~750 arcsec/day)

| Metric | Value |
|---|---|
| Fleet AT z-score detections (timing) | 0 / 9 |
| Best corrected AT SNR (largest station, ~14K obs) | ~2σ |
| ConstantBias detections | 9 / 9 (>50σ) |
| TrailingBias detections | 9 / 9 |

### 4.3 NEO Results (100 objects, mean rate ~1800 arcsec/day)

| Metric | Value |
|---|---|
| Fleet AT z-score detections (timing) | 0 / 9 |
| Best corrected AT SNR (703 Spacewatch, ~31K obs) | ~2.6σ |
| ConstantBias detections | 9 / 9 (>50σ) |
| TrailingBias detections | 9 / 9 |

Note on NEO RA cancellation: For the NEO population, the RA component of the timing signal approximately cancels at the fleet level. Because NEOs are observed at all orbital phases, their east-west motion averages close to zero across the full dataset. The AT signal in RA (which depends on the component of motion along RA) washes out in the mean. This is an intrinsic limitation, not a pipeline artifact.

### 4.4 Baseline Floor Problem

The LOOO AT residuals for a clean (unbiased) simulation show a non-zero per-station mean of +0.15 to +0.65 arcsec depending on the station. This floor arises from orbit-fit correlations: when the orbit is refit on the complement set, it adjusts slightly to match those stations' residuals, leaving a systematic shift in the held-out station's residuals. This floor is 10–70× larger than the expected timing signal (~0.009–0.021 arcsec/obs mean AT shift for 1-second errors), and it overwhelms the signal in any fleet comparison that does not subtract the clean baseline.

With proper baseline correction (computed from the clean simulation):

- MBA timing: detectable at ~3.5σ for PanSTARRS (PS1, ~15K obs); not detectable for smaller stations
- NEO timing: best station ~2.6σ

These results use the simulated clean baseline as a reference, which is not available in a real-data application — making operational detection even harder.

### 4.5 Parameter Absorption Estimate

When the orbit is refit on the complement set, it partially absorbs the timing signal from the biased stations (since the biased observations contributed to the original orbit). We estimate this absorption at ~40–50% of the timing signal based on comparing simulated AT means with the expected AT displacement from the injected Δt. This absorption is an irreducible feature of cross-validation approaches that refit the model.

---

## 5. What LOOO Can and Cannot Detect

### 5.1 LOOO Works Well For

- **ConstantBias** (fixed RA or Dec offset): detectable at >50σ because the signal is the same sign for every observation regardless of rate or direction. No averaging cancellation. No rate-dependent SNR.
- **TrailingBias** (error correlated with trail length): 9/9 detection because trailing length is directly observable and the residual pattern is distinctive.
- **Systematic catalog offsets**: similarly detectable if they are constant within a station.

### 5.2 LOOO Cannot Reliably Detect

- **TimingBias at 1 second for MBAs or NEOs**: per-observation SNR is 0.03–0.07, requiring millions of independent observations for detection. The LOOO baseline floor adds additional masking.
- **Timing biases at slower-moving objects generally**: the signal scales with rate; anything below ~0.1 arcsec/s produces sub-detectable per-observation SNR.

---

## 6. Methods That Would Work Better

### 6.1 Rate-Stratified AT Analysis

Plot mean AT residual as a function of object sky-plane rate for each observatory. If a timing error Δt exists:

```
⟨AT⟩ = Δt × ⟨rate⟩
```

The slope of this regression is the timing offset directly. This does not require holding out a station or refitting an orbit — it uses the standard residuals from a global orbit solution. It is essentially what IAWN does, generalized to the full catalog at all rates rather than a single fast-flyby event.

This approach avoids the parameter absorption problem because the timing offset is estimated from the slope of the rate-residual relation, not from the absolute residual level (which would be absorbed).

### 6.2 Explicit Timing Parameter in OD

Add a per-station timing offset Δt_s as a free parameter in the orbit determination alongside the six orbital elements. The partial derivative of AT residual with respect to Δt_s is simply the rate at that observation epoch. Fitting Δt_s explicitly avoids absorption and gives a direct estimate with formal uncertainty.

This is the correct Bayesian approach and is more powerful than LOOO for timing specifically. It can be implemented as an extension to any existing least-squares OD system by appending Δt columns to the design matrix.

### 6.3 Focus on Close-Approaching NEOs

A single tracklet of a NEO at 0.1 arcsec/s (achievable for objects passing within ~0.1 AU) gives ~10× the per-observation SNR of a typical MBA. A few hundred such tracklets per station over a multi-year dataset could reach 3σ detection of 1-second timing errors. This is a natural complement to the IAWN campaign approach but using archival MPC data rather than dedicated campaigns.

### 6.4 Apply Carpino Correlation Model

Weight same-night observations as a block with effective weight 1/√N_tracklet rather than counting them as independent. This prevents naive overstating of statistical power and gives more honest detection thresholds when evaluating fleet-level timing sensitivity.

---

## 7. The CSS −12 Second Incident (Undocumented Case Study)

The Catalina Sky Survey operated for an extended period with a −12-second timing offset caused by a FITS header bug recording shutter-open time rather than the corrected exposure midpoint. This was discovered through orbit determination failures — newly discovered objects had orbits that were systematically inconsistent between CSS observations and follow-up from other stations, prompting investigation of the CSS timing pipeline.

This case is instructive in several ways:
1. A 12-second error at typical CSS survey rates (~300–500 arcsec/day ≈ 0.004–0.006 arcsec/s) produces an AT displacement of ~0.05–0.07 arcsec per observation — still below the typical σ_obs ≈ 0.3 arcsec. The error was detected through orbit consistency failures (multi-night arc inconsistencies), not through a systematic AT analysis.
2. This suggests that the operational threshold for timing error discovery via OD failures is somewhere between 1 second and 12 seconds for survey-rate objects, with 1 second being below the detection threshold for current methods.
3. A rate-stratified AT analysis of archival CSS data would likely have found the −12-second error more quickly and quantitatively.

---

## 8. Novel Aspects of Our LOOO Approach

No published paper uses leave-one-observatory-out cross-validation for timing bias detection across the MPC catalog. The approach is novel in its cross-validation framing. The main limitations identified:

1. **Parameter absorption (~40–50%)**: orbit refit on complement partially absorbs timing signal
2. **Baseline floor (+0.15–0.65 arcsec)**: clean LOOO residuals are not zero due to fit correlations
3. **Low per-obs SNR (0.03–0.07)**: fundamental physical limit from MBA/NEO rates
4. **NEO RA cancellation**: orbital phase averaging cancels RA component of timing signal
5. **Carpino effective-N reduction**: same-night observations not independent

The LOOO framework remains well-suited for ConstantBias and TrailingBias detection, which are the most operationally relevant bias types for MPC data quality monitoring.

---

## 9. Recommended Next Steps

1. **Implement rate-stratified AT diagnostic** as a standard output alongside LOOO pipeline — this is low-cost (uses existing residuals) and directly sensitive to timing.
2. **Prototype explicit Δt fitting** for a subset of stations with large observation counts — PanSTARRS (PS1/PS2), 703 Spacewatch, and G96 CSS are the most promising targets.
3. **Identify close-approach NEO tracklets** in the MPC catalog per station; even a few hundred high-rate tracklets can constrain station timing to <1 s.
4. **Apply Carpino weighting** to all fleet-level AT statistics to avoid over-optimistic sensitivity claims.

---

## 10. References

- Carpino, M., Milani, A., & Chesley, S. R. 2003. "Error Statistics of Asteroid Optical Astrometry." *Icarus* 166, 248–270.
- Farnocchia, D., et al. 2022. "Astrometric Errors Due to Timing Biases: An IAWN Campaign." *The Planetary Science Journal* 3, 156.
- Farnocchia, D., et al. 2023. "Timing Bias Analysis from the 2022 IAWN Campaign on 2005 LW3." *The Planetary Science Journal* 4, 203.
- Stronati, L., et al. 2023. "Systematic Astrometric Errors in Near-Earth Asteroid and Main-Belt Asteroid Populations." *MNRAS* 521.
- Veres, P., et al. 2017. "Improved Astrometric Error Model for Asteroid Surveys." *Icarus* 296, 139–149.
