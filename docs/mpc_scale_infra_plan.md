# MPC-Scale Isolation Study: Infrastructure Plan

**Branch:** `kk/mpc-scale-infra-plan`  
**Status:** Draft — pending team review

---

## 1. Scope: What is "MPC-Scale"?

The current isolation study runs on **100 synthetic asteroids** across **81 scenarios** (9 stations × 9 bias types), using real MPC observations as a template.

"MPC-scale" means increasing the asteroid sample to get statistically meaningful detection power per observatory:

| Scale | Objects | Purpose |
|-------|---------|---------|
| Current | 100 | Prototype / feasibility |
| 1K | 1,000 | Pilot at realistic scale for major stations |
| 10K | 10,000 | Full statistical study for all Phase 1 stations |
| 50K+ | 50,000+ | Production bias atlas, matches published MPC analyses |

The `data/looo_sample_3500/` directory already contains 3,500 objects — that is the near-term target.

---

## 2. Infrastructure Options

### Option A: Single Large Cloud VM

**Description:** n2-highmem-96 (96 vCPU, 768 GiB RAM) on GCP, run the existing script with `--max-processes 90`.

| Property | Assessment |
|----------|-----------|
| Setup time | Minutes |
| Parallelism | Up to 96 workers across sequential scenarios |
| Time for 3,500 objects | ~10–12 hours two-body; ~17–20 hours ASSIST |
| Time for 50K objects | ~6 days (ASSIST) — not feasible |
| Fault tolerance | None — spot preemption kills the run |
| Checkpointing | Per-object checkpoints already built-in; survives restart |
| Cost | ~$5–15/hr spot; ~$75–150 total for 3.5K |
| Ops overhead | Low |

**Verdict:** Best choice for the 3,500-object run.

---

### Option B: exp-research (GKE + Kueue + IndexedJob)

**Description:** Use the existing `exp-research` cluster (`moeyens-thor-dev`, `us-west1-a`) with each of the 81 scenarios as a parallel indexed job shard.

| Property | Assessment |
|----------|-----------|
| Setup time | 1–2 days of code changes + cluster config |
| Parallelism | 81 shards simultaneously; N cores per shard |
| Time for 3,500 objects | ~2–3 hours ASSIST (all scenarios parallel) |
| Time for 50K objects | ~6–9 hours ASSIST |
| Fault tolerance | Per-index retry; GCS checkpoints survive pod kill |
| Checkpointing | Needs GCS-backed checkpoint adapter (new code) |
| Cost | ~$20–80 for 3.5K; ~$300–700 for 50K (spot) |
| Ops overhead | Medium — requires job spec, GCS adapter, cloud runner script |

**Verdict:** Necessary for 50K+ objects. Overkill for 3,500 unless iteration speed matters.

---

### Option C: Dagster on K8s (adam_etl / adam-jobs stack)

**When it makes sense:** Only if this becomes a recurring production pipeline (e.g., monthly MPC bias atlas). Heavy orchestration overhead for a one-time research batch job.

**Verdict:** Skip for now.

---

## 3. Recommendation by Phase

| Phase | Objects | Recommended Infra |
|-------|---------|-------------------|
| **Phase 1 (now)** | 3,500 | Single large VM (n2-highmem-96 spot) |
| **Phase 2** | 10,000–50,000 | exp-research indexed job |
| **Phase 3 (production)** | 50K+, recurring | Dagster on K8s |

---

## 4. Time Estimates

### Calibration (measured from actual run logs)

100 objects, 6 workers, 81 scenarios sequential:

| Run | Log file | Wall-clock | Per-scenario |
|-----|----------|-----------|-------------|
| ASSIST | `10_isolation_assist.log` | 7h 41min | ~5.7 min |
| Two-body | `isolation_study_twobody_atfix_20260326.log` | 4h 26min | ~3.3 min |

**Two-body speedup over ASSIST: ~1.7×** — much less than the analytic vs. numerical
integration difference would suggest. Python/process overhead and DC solver
convergence iterations dominate over per-propagation cost at these arc lengths.

Core-seconds per object (6 workers):
- ASSIST: ~20 core-sec/object
- Two-body: ~12 core-sec/object

### Phase 1 — 3,500 objects, single VM (n2-highmem-96)

The existing script runs 81 scenarios sequentially. With `--max-processes 90` (one
object-worker pool, all cores to LOOO):

```
ASSIST:    (3,500 / 100) × 5.7 min  ≈  200 min/scenario  ×  81  ≈  270 hours  ← too slow
Two-body:  (3,500 / 100) × 3.3 min  ≈  115 min/scenario  ×  81  ≈  156 hours  ← too slow
```

Scenarios must also be parallelised. With `--scenario-parallelism 16 --max-processes 6`
(16 × 6 = 96 cores total, 6 object-workers per scenario):

```
ASSIST:    ceil(81/16) batches × (3,500 × 20s / 6 workers)  ≈  6 × 11,667s  ≈  19 hr
Two-body:  ceil(81/16) batches × (3,500 × 12s / 6 workers)  ≈  6 × 7,000s   ≈  12 hr
```

**Practical estimates (startup + variance):**
- ASSIST: **~20–24 hours**
- Two-body: **~13–16 hours**

### Phase 2 — exp-research (all 81 scenarios in parallel)

Each pod handles one scenario; wall-clock = time for the slowest pod.

| Scale | Propagator | Cores/pod | Wall-clock |
|-------|-----------|-----------|-----------|
| 3,500 | ASSIST | 16 | ~1.5–2 hr |
| 10,000 | ASSIST | 16 | ~4–5 hr |
| 50,000 | ASSIST | 32 | ~9–11 hr |

Derivation for 10K, 16 cores/pod:
```
10,000 × 20s / 16 workers = 12,500s ≈ 3.5 hr + 30 min overhead = ~4 hr
```

---

## 5. Code Changes Required

### 5.1 Scenario-level parallelism in `10_run_isolation_study.py`

Add `--scenario-parallelism` (int, default 1) wrapping the scenario loop in a
`ThreadPoolExecutor`. Each scenario thread dispatches its own `ProcessPoolExecutor`
for object-level workers.

**Note:** Use `ThreadPoolExecutor` at the scenario level (not Process) — threads share
memory without pickling overhead, and the CPU-bound work happens in the inner
subprocess pool.

```
--scenario-parallelism 16 --max-processes 6  →  16 × 6 = 96 cores
```

### 5.2 GCS-backed checkpoints (Phase 2 / exp-research)

Current checkpoints write to local `checkpoints/`. Cloud pods need GCS:

- Before starting: `gsutil -m rsync gs://$PREFIX/checkpoints/$SCENARIO/ /tmp/checkpoints/`
- After each object: write local + async upload
- On SIGTERM: flush uploads, GKE restarts pod; `backoff_limit_per_index: 2` gives 2 retries

### 5.3 Cloud runner script (Phase 2)

`scripts/12_run_looo_cloud_shard.py`:
- Reads `JOB_COMPLETION_INDEX` → selects scenario from manifest JSON in GCS
- Runs dataset gen + LOOO + analysis for that scenario
- Writes outputs to GCS

---

## 6. exp-research Job Spec (Phase 2)

See `infra/exp-research/mpc-scale-isolation-study.json`.

Key parameters:
- **81 completions** — one shard per scenario
- **64 parallelism** — all 81 shards run near-simultaneously
- **16 CPU / 32 GiB per pod** — supports ASSIST cache + 14 parallel object workers
- **`backoff_limit_per_index: 2`** — survives spot preemption per shard
- **`max_failed_indexes: 8`** — tolerates up to 8/81 scenario failures

---

## 7. Checkpointing Summary

| Level | Mechanism | Implemented |
|-------|-----------|-------------|
| Per-object | `checkpoints/<object_id>.parquet` written after each object | Yes |
| Per-scenario | `looo_results.parquet` presence check on re-run | Yes |
| GCS sync (cloud) | `gsutil rsync` on pod start + async upload | Phase 2 |

---

## 8. Cost Estimates

### Phase 1 — n2-highmem-96 spot VM (3,500 objects)

| Component | Rate | Duration | Cost |
|-----------|------|----------|------|
| n2-highmem-96 spot | ~$4.80/hr | 20 hours | ~$96 |
| Storage / egress | | | ~$5 |
| **Total** | | | **~$100** |

### Phase 2 — exp-research, 10K objects (ASSIST)

| Component | Rate | Duration | Cost |
|-----------|------|----------|------|
| 81 pods × 16 CPU spot | ~$0.04/core-hr | 5 hr | ~$260 |
| Memory + storage | | | ~$30 |
| **Total** | | | **~$290** |

### Phase 2 — exp-research, 50K objects (ASSIST, 32 cores/pod)

| Component | Rate | Duration | Cost |
|-----------|------|----------|------|
| 81 pods × 32 CPU spot | ~$0.04/core-hr | 10 hr | ~$1,040 |
| Memory + storage | | | ~$80 |
| **Total** | | | **~$1,120** |

---

## 9. Execution Plan

### Phase 1 (3,500 objects — now)

- [ ] Add `--scenario-parallelism` flag to `10_run_isolation_study.py`
- [ ] Verify `data/looo_sample_3500/` has complete obs + orbits parquet
- [ ] Provision n2-highmem-96 spot VM, clone repo, install deps
- [ ] Run: `python scripts/10_run_isolation_study.py --input-dir data/looo_sample_3500 --propagator assist --scenario-parallelism 16 --max-processes 6`
- [ ] Pull results; run `scripts/11_atct_analysis.py`

### Phase 2 (10K–50K objects — future sprint)

- [ ] Implement `GCSCheckpointStore` in `looo/gcs_checkpoint.py`
- [ ] Write `scripts/12_run_looo_cloud_shard.py` with `JOB_COMPLETION_INDEX` support
- [ ] Create scenario manifest JSON, upload to GCS
- [ ] Finalize `infra/exp-research/mpc-scale-isolation-study.json`
- [ ] Build + push Docker image
- [ ] Submit: `expctl submit -f infra/exp-research/mpc-scale-isolation-study.json`
- [ ] Monitor: `expctl pipeline watch <run_id>`

---

## 10. Open Questions

1. **Two-body vs ASSIST for scale run?** Two-body is ~1.7× faster (not 10–20× as one
   might expect). For a 3,500-object run, that saves ~6 hours. Worth doing a 500-object
   pilot with both to confirm detection rates are comparable before committing to a full run.

2. **Expand beyond 9 Phase 1 stations?** 100 stations × 9 biases = 900 scenarios — still
   feasible with exp-research (just set `completions: 900`).

3. **Real MPC observations vs synthetic?** For the production bias atlas, running LOOO
   directly on real MPC observations (no bias injection) simplifies the pipeline and removes
   the clean-baseline comparison issue under investigation.

4. **Dagster for Phase 3.** The `k8s_defaults.py` pattern from `adam_etl` and
   `adam-dagster-shared` translates directly if a recurring monthly atlas is needed.
