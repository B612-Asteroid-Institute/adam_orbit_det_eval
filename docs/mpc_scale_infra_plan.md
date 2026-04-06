# MPC-Scale Isolation Study: Infrastructure Plan

**Branch:** `kk/mpc-scale-infra-plan`  
**Status:** Draft — pending team review

---

## 1. Scope: What is "MPC-Scale"?

The current isolation study runs on **100 synthetic asteroids** across **81 scenarios** (9 stations × 9 bias types), using real MPC observations as a template for ephemeris timing.

"MPC-scale" means increasing the asteroid sample to get statistically meaningful detection power per observatory. The relevant thresholds:

| Scale | Objects | Purpose |
|-------|---------|---------|
| Current | 100 | Prototype / feasibility |
| 1K | 1,000 | Pilot at realistic scale for major stations |
| 10K | 10,000 | Full statistical study for all 500C+ stations |
| 50K+ | 50,000+ | Production bias atlas, matches published MPC analyses |

For the Phase 1 study (9 stations), **3,500–10,000 objects** gives detection power comparable to the Veres 2017 model. For a full MPC-station sweep (all ~2,000 coded observatories), you need 50K+ objects.

The `data/looo_sample_3500/` directory already contains 3,500 objects — that is the near-term target.

---

## 2. Infrastructure Options

### Option A: Single Large Cloud VM

**Description:** n2-highmem-96 (96 vCPU, 768 GiB RAM) on GCP, run the existing script with `--max-processes 90`.

| Property | Assessment |
|----------|-----------|
| Setup time | Minutes — just `gcloud compute instances create` |
| Parallelism | 90 processes across 81 sequential scenarios |
| Time for 3,500 objects | ~3–4 hours (see estimates below) |
| Time for 50K objects | ~2–3 days |
| Fault tolerance | None — spot preemption kills the run |
| Checkpointing | Already built-in at object level, survives restart |
| Cost | ~$5–15/hr on spot; ~$30–60 total for 3.5K |
| Ops overhead | Low — ssh in, run script, come back |

**Verdict:** Best choice for the 3,500-object run. Simple, fast to set up, low risk.

---

### Option B: exp-research (GKE + Kueue + IndexedJob)

**Description:** Use the existing `exp-research` cluster (`moeyens-thor-dev`, `us-west1-a`) to run each of the 81 scenarios as a parallel indexed job shard. Each pod processes one complete scenario independently, writing checkpoints and outputs to GCS.

| Property | Assessment |
|----------|-----------|
| Setup time | 1–2 days of code changes + cluster config |
| Parallelism | 81 shards run simultaneously; N cores per shard |
| Time for 3,500 objects | ~30–90 min wall-clock (all scenarios in parallel) |
| Time for 50K objects | ~3–8 hours |
| Fault tolerance | Per-index retry (`backoff_limit_per_index: 2`); GCS checkpoints survive pod kill |
| Checkpointing | Needs GCS-backed checkpoint adapter (new code) |
| Cost | Spot nodes ~$0.05–0.10/core-hr; ~$20–80 for 3.5K; ~$200–600 for 50K |
| Ops overhead | Medium — requires job spec, GCS adapter, new cloud runner script |

**Verdict:** Necessary for 50K+ objects. Overkill for 3,500 unless you want to iterate fast.

---

### Option C: Dagster on K8s (adam_etl / adam-jobs stack)

**Description:** Define the isolation study as Dagster assets, run on the existing Celery-K8s executor.

| Property | Assessment |
|----------|-----------|
| Setup time | Several days — Dagster asset definitions, op graph, sensor triggers |
| Benefit | Dashboard visibility, asset lineage, scheduled retries, integration with existing ETL |
| Drawback | Heavy orchestration overhead for a research batch job; not designed for embarrassingly parallel one-off runs |
| When it makes sense | If this becomes a recurring production pipeline (e.g., monthly MPC bias atlas) |

**Verdict:** Skip for now. Revisit if the bias atlas becomes a recurring data product.

---

## 3. Recommendation by Phase

| Phase | Objects | Recommended Infra |
|-------|---------|-------------------|
| **Phase 1 (now)** | 3,500 | Single large VM (n2-highmem-96 spot) |
| **Phase 2** | 10,000–50,000 | exp-research indexed job |
| **Phase 3 (production)** | 50K+, recurring | Dagster on K8s |

---

## 4. Time Estimates

### Calibration

From existing run logs:
- 100 objects, ASSIST propagator, 6 workers: **~5 min/scenario** wall-clock
- Implies: ~19 core-seconds per object (ASSIST); ~1–2 core-seconds per object (two-body)

### Phase 1 — 3,500 objects, single VM (n2-highmem-96)

```
Per-scenario ASSIST:   3,500 × 19s / 90 workers ≈ 740s ≈ 12 min
81 scenarios sequential:                         ≈ 16 hours  ← too slow
```

**Better: parallelize scenarios, not just objects.**

The existing script runs scenarios sequentially. On a 96-core VM, we can run 9 scenarios simultaneously (one per station), each using 10 cores:

```
9 stations × 9 bias types = 81 scenarios
Run 9 at once (one per station group), each with 10 workers
Each station group: 9 bias scenarios sequential, each 12 min
Total: 9 × 12 min = ~1.8 hours
```

Or use GNU parallel / Python multiprocessing at the scenario level (see §6.1).

With full scenario parallelism (81 concurrent, 1 worker each on a 96-core VM):
```
All 81 scenarios at once, 1 core each:  3,500 × 19s / 1 = 66,500s ≈ 18 hours  ← worse
Best: balance scenario × object parallelism
```

**Recommended VM config for 3,500 objects:**
- Run ~16 scenarios in parallel, each with ~6 workers → 96 cores fully used
- 81 scenarios / 16 parallel ≈ 6 batches of scenarios
- Time: 6 × (3,500 × 19s / 6) ≈ 6 × 11,083s ≈ 18.5 hours
- **Practical estimate: 20–30 hours** (accounting for dataset gen, GCS I/O overhead, ASSIST warm-up)

With two-body propagator (faster, lower fidelity):
- **Practical estimate: 2–4 hours** for 3,500 objects, 81 scenarios

### Phase 2 — 10,000 objects, exp-research (16 cores/pod, 81 pods)

```
All 81 scenarios in parallel, 16 workers each:
Time per pod: 10,000 × 19s / 16 ≈ 11,875s ≈ 3.3 hours
+ 20 min startup/data loading
= ~3.5–4 hours wall-clock
```

### Phase 2 — 50,000 objects, exp-research (32 cores/pod, 81 pods)

```
50,000 × 19s / 32 ≈ 29,688s ≈ 8.2 hours per pod (= wall-clock, all pods parallel)
+ 30 min overhead
= ~9 hours wall-clock
```

---

## 5. Architecture for Phase 1 (3,500 objects, single VM)

### Immediate approach

Modify the scenario loop in `10_run_isolation_study.py` to run scenarios concurrently using a `ProcessPoolExecutor` at the scenario level, with a configurable `--scenario-parallelism` flag.

```
┌─────────────────────────────────────────────────────────┐
│  10_run_isolation_study.py --scenario-parallelism 16    │
│                                                         │
│  ProcessPoolExecutor(max_workers=16)                    │
│    ├── Scenario AA00_clean  (6 object-workers)          │
│    ├── Scenario AA00_constant (6 object-workers)        │
│    ├── Scenario AA01_clean  (6 object-workers)          │
│    ├── ...                                              │
│    └── Scenario AA08_trailing (6 object-workers)        │
└─────────────────────────────────────────────────────────┘
```

**Caveat:** ASSIST requires SPICE kernels to be loaded per-process. Nested ProcessPoolExecutors work but need careful pickling. The current worker model already handles this (each worker imports and instantiates its own propagator).

### VM provisioning (one-liner)

```bash
gcloud compute instances create looo-mpc-3500 \
  --project=moeyens-thor-dev \
  --zone=us-west1-a \
  --machine-type=n2-highmem-96 \
  --provisioning-model=SPOT \
  --instance-termination-action=STOP \
  --image-family=ubuntu-2204-lts \
  --image-project=ubuntu-os-cloud \
  --boot-disk-size=200GB \
  --boot-disk-type=pd-ssd \
  --metadata=startup-script='apt-get install -y python3.11 git'
```

---

## 6. Code Changes Required

### 6.1 Scenario-level parallelism in `10_run_isolation_study.py`

Add `--scenario-parallelism` (int, default 1) that wraps the scenario loop in a `ThreadPoolExecutor` or `ProcessPoolExecutor`. Each scenario process then calls `run_scenario()` with `max_processes=N` object workers.

**Caution:** Total worker count = scenario_parallelism × max_processes. On a 96-core VM:
- `--scenario-parallelism 16 --max-processes 6` = 96 workers total

```python
# Sketch — not final
from concurrent.futures import ProcessPoolExecutor as PE

with PE(max_workers=args.scenario_parallelism) as pool:
    futs = {
        pool.submit(run_scenario, fc, bn, ..., max_processes=args.max_processes): (fc, bn)
        for fc, bn in scenarios
    }
    for fut in as_completed(futs):
        rows = fut.result()
        ...
```

### 6.2 GCS-backed checkpoints (for Phase 2 / exp-research)

Current checkpoints go to local `looo_results/default/checkpoints/`. For cloud pods, we need GCS:

```python
# New module: adam_orbit_det_eval/looo/gcs_checkpoint.py
class GCSCheckpointStore:
    def __init__(self, gcs_prefix: str):
        ...
    def exists(self, object_id: str) -> bool:
        ...
    def write(self, object_id: str, table: pa.Table) -> None:
        ...
    def load_completed_ids(self) -> set[str]:
        ...
```

The `run_looo_pipeline()` signature already accepts `output_path: Path`; adding a `checkpoint_store: Optional[CheckpointStore] = None` parameter would be backward-compatible.

### 6.3 Cloud runner script for exp-research (Phase 2)

New script `scripts/12_run_looo_cloud_shard.py`:
- Reads `JOB_COMPLETION_INDEX` from environment
- Reads scenario assignment from a manifest JSON (stored in GCS)
- Runs dataset gen + LOOO + analysis for one scenario
- Writes all outputs to GCS
- Syncs GCS checkpoints before starting (spot-preemption recovery)

### 6.4 exp-research job spec

See `infra/exp-research/mpc-scale-isolation-study.json` (created in this branch).

---

## 7. Checkpointing Strategy

### Level 1: Per-object (already implemented)

`pipeline.py` writes one parquet file per object to `checkpoints/`. On restart, completed objects are skipped. This works on a VM and survives SIGTERM.

### Level 2: Per-scenario

The main script checks for `looo_results.parquet` existence before running LOOO. This means a fully completed scenario is always skipped on re-run. On a VM, this is reliable.

### Level 3: GCS sync for spot preemption (Phase 2 / cloud)

For exp-research pods:
1. On pod start: `gsutil -m rsync gs://$OUTPUT_PREFIX/checkpoints/$SCENARIO/ /tmp/checkpoints/`
2. After each object: write checkpoint locally + async GCS upload
3. On SIGTERM (spot preemption): flush all pending uploads, GKE restarts the pod
4. `backoff_limit_per_index: 2` gives 2 restarts per shard

With 19s/object and 10K objects, a preemption that kills 50% of work costs at most ~25 min (re-running the half that wasn't checkpointed). Acceptable.

---

## 8. Cost Estimates

### Phase 1 — n2-highmem-96 spot VM

| Component | Rate | Duration | Cost |
|-----------|------|----------|------|
| n2-highmem-96 spot | ~$4.80/hr | 25 hours | ~$120 |
| Egress (checkpoints to GCS) | ~$0.01/GB | ~5 GB | ~$0.05 |
| **Total** | | | **~$120** |

### Phase 2 — exp-research, 10K objects

| Component | Rate | Duration | Cost |
|-----------|------|----------|------|
| 81 pods × 16 CPU spot | ~$0.04/core-hr | 4 hr | ~$210 |
| 81 pods × 32 GiB RAM | ~$0.005/GB-hr | 4 hr | ~$52 |
| GCS storage + ops | | | ~$10 |
| **Total** | | | **~$270** |

### Phase 2 — exp-research, 50K objects

| Component | Rate | Duration | Cost |
|-----------|------|----------|------|
| 81 pods × 32 CPU spot | ~$0.04/core-hr | 9 hr | ~$940 |
| Memory + storage | | | ~$100 |
| **Total** | | | **~$1,040** |

---

## 9. exp-research Job Spec (Phase 2)

See `infra/exp-research/mpc-scale-isolation-study.json`.

Key design decisions:
- **81 completions** = one shard per scenario (station × bias)
- **64 parallelism** = all 81 shards run near-simultaneously (GKE autoscaler scales up nodes)
- **16 CPU, 32 GiB per pod** = supports ASSIST ephemeris cache + 14 parallel object workers
- **`backoff_limit_per_index: 2`** = survives spot preemption for each shard
- **`max_failed_indexes: 8`** = tolerates up to 8/81 scenarios failing (manually re-run)
- Input data staged to GCS before job submission

---

## 10. Execution Plan

### Phase 1 (3,500 objects — now, ~2 weeks)

- [ ] Add `--scenario-parallelism` flag to `10_run_isolation_study.py`
- [ ] Verify `data/looo_sample_3500/` has complete obs + orbits parquet
- [ ] Provision n2-highmem-96 spot VM, clone repo, install deps
- [ ] Run with `--input-dir data/looo_sample_3500 --propagator assist --scenario-parallelism 16 --max-processes 6`
- [ ] Monitor via GCS log tailing or VM SSH session
- [ ] Pull results when done; run `scripts/11_atct_analysis.py` on merged output

### Phase 2 (10K–50K objects — future sprint)

- [ ] Implement `GCSCheckpointStore` in `looo/gcs_checkpoint.py`
- [ ] Write `scripts/12_run_looo_cloud_shard.py` with `JOB_COMPLETION_INDEX` support
- [ ] Create scenario manifest JSON and upload to GCS
- [ ] Finalize `infra/exp-research/mpc-scale-isolation-study.json` spec
- [ ] Build Docker image and push to `us-west1-docker.pkg.dev/moeyens-thor-dev/ai/`
- [ ] Submit with `expctl submit -f infra/exp-research/mpc-scale-isolation-study.json`
- [ ] Watch with `expctl pipeline watch <run_id>`

---

## 11. Open Questions

1. **Two-body vs ASSIST for the scale run?** Two-body is ~10–20× faster, which turns 25 hours into 2 hours on the VM. The question is whether the residual floor difference (two-body vs N-body) matters for bias detection SNR. Recommend a 1K-object pilot with both propagators to check.

2. **Expand beyond 9 Phase 1 stations?** The full MPC has ~2,000 coded stations. Adding more stations doesn't change the per-object LOOO cost much (LOOO iterates over held-out stations, not objects), but it multiplies the scenario count. 100 stations × 9 biases = 900 scenarios — still feasible with exp-research.

3. **Real MPC observations vs synthetic?** The current pipeline uses real MPC obs as a timing template but injects synthetic biases. For the production bias atlas, we'd run LOOO directly on real observations (no bias injection) — simpler and removes the "clean scenario" baseline issue. This eliminates the dataset generation step and halves the pipeline.

4. **GCS input data staging.** For Phase 2, the 50K-object parquet needs to be staged to GCS before job submission. At ~75 MB per 3,500 objects, 50K objects ≈ 1 GB — small, not an issue.

5. **Dagster monitoring for Phase 3.** If we productionize a monthly bias atlas, wrapping this in Dagster assets (with `dagster-k8s` and `adam-dagster-shared` K8s configs) would give us scheduling, retries, and visibility. The `k8s_defaults.py` pattern from `adam_etl` translates directly.
