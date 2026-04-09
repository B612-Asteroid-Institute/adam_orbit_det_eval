# Handoff: MPC-Scale Cloud Run Infrastructure
*Written April 2026 for agent continuity. Branch: `kk/mpc-scale-infra-plan`.*

For broader project context (simulation framework, real-data LOOO results, pipeline scripts),
see `docs/project_context_for_new_agent.md`. This document covers only the cloud
infrastructure work and the state of the in-progress MPC-scale job.

---

## What Was Built

The goal was to run the Phase 1 isolation study at MPC scale (3,500 objects, all 9 bias
types × 9 stations = 81 scenarios) using the exp-research cloud batch framework.

### New Files on `kk/mpc-scale-infra-plan`

| File | Purpose |
|------|---------|
| `src/adam_orbit_det_eval/isolation_study.py` | Extracted pipeline module shared by local CLI and cloud shard runner. Contains `PHASE1_STATIONS`, `BIAS_NAMES`, `get_bias_catalogue()`, `build_isolation_map()`, `scenario_id()`, `list_scenarios()`, `run_scenario()`. |
| `scripts/10_run_isolation_study.py` | Refactored as a thin CLI wrapper (imports from `isolation_study` module). Added `--scenario-parallelism` for concurrent local runs via `ThreadPoolExecutor`. |
| `scripts/12_run_looo_cloud_shard.py` | Cloud shard runner for exp-research indexed jobs. Reads `JOB_COMPLETION_INDEX`, maps to scenario via GCS manifest, syncs checkpoints for spot-preemption recovery, runs `run_scenario()`, uploads all outputs. Uses `google.cloud.storage` (no gsutil needed in pods). |
| `scripts/13_collect_cloud_results.py` | Post-run results collector. Downloads all per-scenario outputs from GCS, merges `recovery_report.csv` files into `combined_recovery.csv`, prints detection summary. |
| `infra/exp-research/mpc-scale-isolation-study.json` | exp-research job spec: 81 completions, 64 parallelism, 16 CPU / 32 GiB per pod, ASSIST propagator, `backoff_limit_per_index: 2` for spot retry. |
| `docs/mpc_scale_infra_plan.md` | Timing estimates, architecture rationale, run instructions. |
| `.gcloudignore` | Excludes `data/`, `logs/`, etc. from Cloud Build context (reduced context from 10.9 GB to ~100 MB). |

### Modified Files

| File | Change |
|------|--------|
| `src/adam_orbit_det_eval/looo/core.py` | Added `orbit_fitter: Optional[OrbitFitter] = None` parameter to `run_looo_for_object`. DC step branches: if fitter provided → `orbit_fitter.refine_fit()`; else → original `fit_least_squares()`. Added `_orbits_as_fitted()` helper to wrap `Orbits` → `FittedOrbits` for the fitter interface. |
| `src/adam_orbit_det_eval/looo/pipeline.py` | Threads `orbit_fitter` through `_worker` and `run_looo_pipeline`. |
| `src/adam_orbit_det_eval/simulation/bias_models.py` | All `apply()` methods accept `velocity_ra_unit`, `velocity_dec_unit`. `TimingBias`, `TrailingBias`, `ClockDrift` project shifts along actual sky-plane motion direction (was RA-only before). |
| `src/adam_orbit_det_eval/simulation/synthetic_obs.py` | `_compute_obs_geometry()` now returns velocity unit vectors from finite differences of truth ephemeris. Passed through to bias `apply()` calls. |
| `src/adam_orbit_det_eval/simulation/observatory_map.py` | `_ZeroBias.apply()` signature updated to match new interface. |
| `scripts/11_atct_analysis.py` | Added baseline correction: clean scenario mean AT/CT subtracted before computing detection SNR. Formula: `(bias_mean - clean_mean) / sqrt(sem_bias² + sem_clean²)`. |

---

## GCS Layout

```
gs://exp-research/mpc-isolation-study/
  input/
    scenarios.json              ← 81-scenario manifest (index → fake_code, bias_name)
    mpc_observations.parquet    ← 3,500 objects, ~1.38M observations
    mpc_orbits.parquet
  output/
    {scenario}/                 ← e.g. AA00_clean/, AA01_constant/, ...
      datasets/default/
        mpc_observations.parquet
        mpc_orbits.parquet
        truth_biases.csv
      looo_results/default/
        looo_results.parquet
        checkpoints/            ← mid-run; synced for spot recovery
      analysis/default/
        observatory_stats.parquet
      recovery/default/
        recovery_report.csv
        recovery_summary.txt
```

Both input files are already uploaded. The scenario manifest is uploaded.

---

## Cloud Job State (as of April 9, 2026)

**Docker image:** Built successfully.
```
us-west1-docker.pkg.dev/moeyens-thor-dev/ai/mpc-isolation-study:latest-20260409170834
```
Cloud Build ID: `7711a8d5-6b95-42ed-89e8-e2947cc7b0ce`

**Job submission:** Failed at the `kubectl apply` step because the local kubeconfig
did not have the `gke_moeyens-thor-dev_us-west1-a_exp-research` context.

**To submit the job** (image build will be skipped — already done):
```bash
# 1. Authenticate to the cluster
gcloud container clusters get-credentials exp-research \
    --region us-west1-a --project moeyens-thor-dev

# 2. Submit (will reuse cached image)
cd /Users/kathleenkiker/claude_cli/exp-research
./expctl submit -f ../adam_orbit_det_eval/infra/exp-research/mpc-scale-isolation-study.json
```

---

## How to Run the Full Sequence (from scratch)

```bash
# 0. Make sure you're on the right branch
cd /Users/kathleenkiker/claude_cli/adam_orbit_det_eval
git checkout kk/mpc-scale-infra-plan

# 1. Generate and upload the scenario manifest
.venv/bin/python3 scripts/12_run_looo_cloud_shard.py --generate-manifest 2>/dev/null \
    | gsutil cp - gs://exp-research/mpc-isolation-study/input/scenarios.json

# 2. Upload input parquets (already done — skip if unchanged)
gsutil -m cp data/looo_sample_3500/mpc_observations.parquet \
             data/looo_sample_3500/mpc_orbits.parquet \
             gs://exp-research/mpc-isolation-study/input/

# 3. Submit the job
cd /Users/kathleenkiker/claude_cli/exp-research
./expctl submit -f ../adam_orbit_det_eval/infra/exp-research/mpc-scale-isolation-study.json

# 4. Monitor
kubectl --context gke_moeyens-thor-dev_us-west1-a_exp-research \
    get pods -n research -l job-name=mpc-isolation-study -w

# 5. Collect results (once all 81 pods complete)
cd /Users/kathleenkiker/claude_cli/adam_orbit_det_eval
.venv/bin/python3 scripts/13_collect_cloud_results.py \
    --gcs-output-prefix gs://exp-research/mpc-isolation-study/output \
    --output-dir data/sim_products/mpc_scale_run
```

---

## Orbit Fitter Plug-and-Play

The LOOO pipeline now accepts an `orbit_fitter` argument at every level:

```python
from adam_orbit_det_eval.looo import run_looo_pipeline

# Default: fit_least_squares (scipy-backed, same as before)
run_looo_pipeline(..., orbit_fitter=None)

# FindOrb backend
from adam_fo import FindOrbOrbitFitter
run_looo_pipeline(..., orbit_fitter=FindOrbOrbitFitter())

# Native adam_core backend
from adam_core.orbit_determination.orbit_fitter import NativeOrbitFitter
run_looo_pipeline(..., orbit_fitter=NativeOrbitFitter())
```

The `run_scenario()` function in `isolation_study.py` also accepts `orbit_fitter=None` and
threads it through to `run_looo_pipeline`. Any `OrbitFitter` subclass (from `adam_core`) that
is picklable works.

---

## Expected Results

With 3,500 objects and ASSIST propagator, based on 100-object test runs:

| Metric | Estimate |
|--------|---------|
| Wall time per pod | ~4–6 hours |
| Total pod-hours | ~320–480 |
| Cost | ~$50–80 (spot pricing) |
| Output size | ~5–10 GB across all 81 scenarios |

Detection rates from the 100-object isolation study (baseline for comparison):

| Bias Type | Det Rate (RA) | Det Rate (AT) |
|-----------|--------------|--------------|
| constant  | 8/9 (89%)    | 8/9 (89%)   |
| timing    | 1/9 (11%)    | 5/9 (56%)   |
| trailing  | 0/9 (0%)     | 3/9 (33%)   |
| dcr       | 2/9 (22%)    | 2/9 (22%)   |
| seasonal  | 5/9 (56%)    | 5/9 (56%)   |
| step      | 4/9 (44%)    | 4/9 (44%)   |
| epoch     | 3/9 (33%)    | 3/9 (33%)   |

With 3,500 objects (~35× more), SNR scales as √35 ≈ 6×, so marginal cases should
become clearly detectable.
