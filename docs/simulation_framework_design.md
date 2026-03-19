# Simulation Framework Design
## Synthetic Truth-State Validation of LOOO Bias Detection

**Status:** Architecture complete, implementation pending
**Date:** 2026-03-19

---

## Motivation

The LOOO pipeline measures residuals from real observations and uses them to estimate
per-observatory astrometric biases and sigmas. Before trusting those results for OD
calibration, we need to demonstrate that the pipeline can correctly recover known
injected biases. This simulation framework provides that validation by:

1. Selecting well-observed objects whose MPC nominal orbit is taken as ground truth
2. Re-generating synthetic observations at the exact times and cadences of the real MPC
   data, using ephemeris evaluation rather than real reported positions
3. Assigning fake observatory codes (AA00–AAZZ, etc.) so real calibration errors are
   absent by construction
4. Adding controlled Gaussian noise with known sigma
5. Injecting known systematic biases of chosen type and magnitude per fake station
6. Running the unchanged LOOO pipeline on the synthetic data
7. Comparing recovered biases to injected truth

---

## Data Flow

```
Real MPCObservations (template)
  (obstime, real_stn, mag, object_id)
          │
          ▼
  ObservatoryMap         ← assigns fake codes, noise sigmas, bias models
          │
          ▼
  generate_ephemeris()   ← ASSIST propagator, uses real site coords for parallax
  → true_ra, true_dec (topocentric, arcsec)                    [CACHED: truth_ephemeris]
          │
          ├─ + GaussianNoise(σ_ra, σ_dec, seed)                [CACHED: noisy_baseline]
          │
          ├─ + BiasModel.apply(ra, dec, mag, obstime, ...)
          │
          ▼
  synthetic_ra, synthetic_dec
  + fake_stn_code, rmsra=σ_ra, rmsdec=σ_dec, astcat            [CACHED: sim_observations]
          │
          ▼
  MPCObservations parquet  (same schema as real data)
  truth_biases.csv         (injected bias per fake station)
          │
          ▼
  02_run_looo.py  (unchanged)                                   [CACHED: looo_results]
          │
          ▼
  03_analyze.py   (unchanged)                                   [CACHED: analysis]
          │
          ▼
  09_evaluate_sim_recovery.py
  → compare recovered bias vs injected truth
```

---

## Module Structure

```
src/adam_orbit_det_eval/
  simulation/
    __init__.py
    observatory_map.py     # FakeObservatory dataclass, ObservatoryMap
    bias_models.py         # BiasModel ABC + all implementations
    noise_model.py         # per-station Gaussian noise
    synthetic_obs.py       # ephemeris → synthetic MPCObservations
    dataset.py             # SimulationConfig, SimulationDataset
    evaluate.py            # recovery metrics, RecoveryReport

scripts/
  06_fetch_sim_sample.py        # fetch well-observed objects
  07_generate_sim_dataset.py    # generate synthetic obs parquet + truth table
  08_run_sim_pipeline.py        # run 02_run_looo.py + 03_analyze.py on synthetic data
  09_evaluate_sim_recovery.py   # compare injected vs recovered, write report

data/
  sim_sample/                   # fetched MPC observations + orbits (reusable)
  sim_products/
    <run_id>/
      truth_ephemeris/          # per-object cached true positions (reusable across bias variants)
      noisy_baseline/           # truth + noise, no bias (reusable across bias variants)
      datasets/
        <dataset_id>/           # per-bias-scenario synthetic obs + truth table
      looo_results/
        <dataset_id>/           # LOOO raw results (checkpointed)
      analysis/
        <dataset_id>/           # observatory_stats, catalog_stats
      recovery/
        <dataset_id>/           # recovery metrics vs truth
```

---

## Key Classes

### `observatory_map.py`

```python
@dataclass
class FakeObservatory:
    fake_code: str             # e.g. "AA01"
    real_code: str             # real MPC code → site coordinates + parallax
    noise_sigma_ra: float      # arcsec, 1-sigma Gaussian noise
    noise_sigma_dec: float     # arcsec
    biases: List[BiasModel]    # composable; empty list = clean reference station
    astcat: str = "Gaia3E"     # catalog code (affects Veres sigma lookup)

class ObservatoryMap:
    assignments: List[FakeObservatory]

    def get_real_code(self, fake_code: str) -> str
    def truth_table(self) -> pd.DataFrame  # injected bias per fake station
```

### `bias_models.py`

All models share this interface:

```python
class BiasModel(ABC):
    @abstractmethod
    def apply(
        self,
        ra: np.ndarray,                # degrees
        dec: np.ndarray,               # degrees
        obstime: np.ndarray,           # MJD
        mag: np.ndarray,               # nullable
        zenith_angle: np.ndarray,      # degrees, nullable
        parallactic_angle: np.ndarray, # degrees, nullable
        object_rate: np.ndarray,       # arcsec/hour, nullable
        field_ra: np.ndarray,          # field center RA, nullable
        field_dec: np.ndarray,         # field center Dec, nullable
    ) -> Tuple[np.ndarray, np.ndarray]  # (Δra*cos(dec), Δdec) in arcsec
```

Biases are composable via `CompoundBias(biases: List[BiasModel])`, which sums contributions.

### `synthetic_obs.py`

```python
def generate_synthetic_observations(
    truth_orbit: Orbits,
    obs_template: MPCObservations,
    observatory_map: ObservatoryMap,
    propagator: Propagator,
    noise_seed: int = 42,
    cache_dir: Optional[Path] = None,   # if set, loads/writes truth ephemeris cache
) -> MPCObservations
```

### `dataset.py`

```python
@dataclass
class SimulationConfig:
    run_id: str
    objects: List[str]
    observatory_map: ObservatoryMap
    propagator_class: Type[Propagator]
    noise_seed: int = 42
    looo_config: LOOOConfig = field(default_factory=LOOOConfig)

class SimulationDataset:
    def generate(
        self,
        obs_template: MPCObservations,
        truth_orbits: MPCOrbits,
        output_dir: Path,
        cache_dir: Optional[Path] = None,
    ) -> Path
    # Writes: mpc_observations.parquet, mpc_orbits.parquet,
    #         truth_biases.csv, sim_config.json
```

### `evaluate.py`

```python
@dataclass
class StationRecovery:
    fake_code: str
    bias_type: str
    injected_ra: float       # arcsec
    injected_dec: float      # arcsec
    recovered_ra: float      # mean residual from LOOO stats
    recovered_dec: float
    recovery_error_ra: float # injected - recovered
    recovery_error_dec: float
    n_obs: int
    detection_snr: float     # recovered_bias / noise_floor
    detected: bool           # |recovery_error| < threshold

def evaluate_recovery(
    observatory_stats: ObservatoryStats,
    truth_biases_csv: Path,
    threshold_arcsec: float = 0.05,
) -> pd.DataFrame            # one row per fake station
```

---

## Bias Model Catalog

### I. Astrometric Positional

| Class | Parameters |
|-------|-----------|
| `ConstantBias` | `delta_ra, delta_dec` (arcsec) |
| `FieldRotationBias` | `rotation_arcsec` — counter-rotating RA/Dec offset from field center |
| `PlateScaleBias` | `scale_error_ppm` — radial offset proportional to distance from field center |

### II. Catalog-Induced

| Class | Parameters |
|-------|-----------|
| `CatalogZeroPointBias` | `delta_ra, delta_dec` — entire catalog shifted |
| `CatalogEpochBias` | `epoch_error_years, pm_ra_median, pm_dec_median` — reference stars at wrong epoch |

### III. Timing

| Class | Parameters |
|-------|-----------|
| `TimingBias` | `delta_t_sec` — constant clock offset → along-track position shift |
| `ClockDrift` | `drift_sec_per_year, ref_mjd` — secular timing error |
| `ReportingTruncation` | `precision_deg` — round coordinates to simulated reporting precision |

### IV. Atmospheric / Environmental

| Class | Parameters |
|-------|-----------|
| `DCRBias` | `bandpass_nm, ref_wavelength_nm` — differential chromatic refraction from zenith/parallactic angle |
| `RefractionModelError` | `delta_pressure_hPa, delta_temp_K` — incorrect atmosphere assumed |

### V. Instrumental / Detector

| Class | Parameters |
|-------|-----------|
| `CTEBias` | `amplitude, readout_axis` — charge transfer efficiency trailing in readout direction |

### VI. Object-Rate Dependent

| Class | Parameters |
|-------|-----------|
| `TrailingBias` | `trailing_factor` — centroid pulled in direction of motion; scales with object rate |
| `MagnitudeDependentBias` | `slope_ra, slope_dec, ref_mag` — centroid error linear in magnitude |
| `ColorDependentBias` | `slope_ra, slope_dec, ref_color` — catalog color term error |

### VII. Time-Variable

| Class | Parameters |
|-------|-----------|
| `SeasonalBias` | `amplitude_ra, amplitude_dec, phase_days` — 365.25-day sinusoid |
| `StepChangeBias` | `bias_before, bias_after, change_mjd` — pipeline/hardware step change |
| `NightlyDrift` | `slope_ra, slope_dec` — linear drift within each observing night |

### VIII. Pipeline / Reporting

| Class | Parameters |
|-------|-----------|
| `ReportingTruncation` | `precision_deg` — coordinate rounding |
| `WrongSiteBias` | `coord_error_km` — site coordinates wrong by N km (parallax error) |

---

## Test Matrix

### Phase 1 — One bias type per station, well-separated

| Fake Code | Real Code | Bias Type | N_obs | Cadence |
|-----------|-----------|-----------|-------|---------|
| AA00 | F51 | None (reference) | 500 | Survey, 3-yr arc |
| AA01 | G96 | `ConstantBias` | 200 | Survey, 2-yr arc |
| AA02 | F52 | `TimingBias` | 200 | Survey, 2-yr arc |
| AA03 | 703 | `MagnitudeDependentBias` | 100 | Follow-up, 1-yr arc |
| AA04 | 691 | `CatalogEpochBias` | 80 | Old survey, 5-yr arc |
| AA05 | W84 | `SeasonalBias` | 150 | Survey, 2-yr arc |
| AA06 | 568 | `StepChangeBias` | 120 | Split arc, step at midpoint |
| AA07 | T09 | `DCRBias` | 100 | Low-dec survey, 1-yr arc |
| AA08 | V00 | `TrailingBias` | 80 | Fast-mover follow-up |

Test values per bias type: small (0.1σ), medium (0.5σ), large (2σ) where σ = station noise.

### Phase 2 — N_obs sensitivity

Repeat Phase 1 biases at N_obs ∈ {5, 10, 30, 100, 300} to characterize minimum detectable
bias as a function of sample size.

### Phase 3 — Compound biases

Stations with 2+ simultaneous bias types; tests whether the pipeline aliases them or
partially resolves them.

### Phase 4 — Adversarial cadence

Short arcs, sparse sampling, single-apparition objects; stress-tests eligibility filters.

---

## Key Design Decisions

1. **Observer position is always physically correct.** Biases are injected in RA/Dec space
   *after* the correct topocentric correction. This separates geometric (parallax) from
   astrometric (calibration) signals cleanly.

2. **rmsra/rmsdec always populated.** Synthetic obs always carry explicit uncertainties
   equal to the injected noise sigma. Veres 2017 fill-ins are never triggered, isolating
   bias detection from sigma-model uncertainty.

3. **Truth ephemeris is cached.** Running ASSIST for 100 objects × thousands of observations
   is the bottleneck. The true topocentric positions are computed once and cached per object.
   All bias/noise variants are derived from this cache without re-running the propagator.

4. **Noise seed is fixed and recorded.** Every run records its seed in `sim_config.json`.
   Re-running with the same seed is exactly reproducible.

5. **One truth orbit per object.** The MPC nominal orbit is the fixed truth. It is also used
   as the DC starting point in the LOOO pipeline, so DC convergence is guaranteed for clean
   (no-bias) stations.

6. **Noise sigmas come from our empirical estimates.** The RMS values from the 3,500-object
   LOOO run (object-weighted, `veres2017_3500obj_objweighted/`) serve as the noise model.
   This closes the loop: our calibration study directly feeds the simulation's realism.

---

## References

- LOOO methodology and 3,500-object results: `runs/run_3500obj/README.md`
- LSST reference orbit pilot (timing mismatch analysis): `runs/lsst_reference/README.md`
- Empirical sigma estimates: `reports/mpc_looo_3500obj/sigma_table.csv`
- Bias taxonomy: Task 1 output, documented in `docs/simulation_framework_design.md` (this file)
