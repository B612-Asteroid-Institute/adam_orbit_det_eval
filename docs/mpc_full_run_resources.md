# Full-MPC LOOO Run — Resource & Infrastructure Estimate

*Based on empirical timing from the 100-object pilot run (2026-03-13).*

---

## Pilot Run Baseline

| Metric | Value |
|--------|-------|
| Objects processed | 100 |
| Held-out observations | 53,864 |
| Wall-clock time (6 cores) | ~30 minutes |
| Workers | 6 (`ProcessPoolExecutor`) |
| Core-minutes consumed | ~180 |
| **Per-object average** | **~1.8 core-minutes** |
| Per-object wall clock (6 cores) | ~18 seconds |
| Propagator | ASSIST (N-body) |

Note: the pilot sample was biased toward objects with ≥50 observations (median 194 obs/object,
max 6,402). Many MPC objects have fewer observations and will process faster.

---

## MPC Scale

| Population | Count | Eligibility estimate |
|-----------|-------|---------------------|
| Numbered asteroids | ~1,300,000 | ~900,000 pass (≥6 obs, ≥7-day arc) |
| Unnumbered provisionals | ~700,000 | ~200,000 pass |
| Comets | ~5,000 | ~3,000 pass |
| **Total eligible** | | **~1,100,000** |

### Compute estimate

Using 1.8 core-minutes/object as a conservative upper bound (most objects are
less well-observed than the pilot sample):

| Scenario | Core-hours | Wall clock @ N cores | GCP n2 cost (spot ~$0.015/core-hr) |
|----------|-----------|----------------------|-------------------------------------|
| Conservative (1.8 core-min/obj) | 33,000 | ~33 hrs @ 1,000 cores | ~$495 |
| Optimistic (0.8 core-min/obj) | 14,700 | ~15 hrs @ 1,000 cores | ~$220 |
| Very conservative (3.0 core-min/obj) | 55,000 | ~55 hrs @ 1,000 cores | ~$825 |

**Bottom line: a full-MPC run needs ~1,000 cores for ~1–2 days and costs ~$300–900
on GCP spot instances.**

---

## Recommended Cloud Architecture

### Option A: GCP + Ray (preferred — integrates with existing adam_core Ray usage)

```
GCP Project
├── Cloud Storage bucket
│   ├── gs://adam-looo/input/mpc_observations.parquet   (sharded by object)
│   ├── gs://adam-looo/input/mpc_orbits.parquet
│   └── gs://adam-looo/runs/<run_id>/checkpoints/       (one file per object)
│
├── GKE cluster (or Ray on GCE)
│   ├── Head node: 1× n2-standard-8 (schedules work, merges outputs)
│   └── Worker pool: 200× n2-standard-4 spot (800 cores total)
│       each worker runs _worker() for one object at a time
│
└── BigQuery (existing)
    └── mpc_sbn_aurora  (source data)
```

**Key changes needed to the current pipeline:**
1. Replace `ProcessPoolExecutor` with `ray.remote` tasks (already used in `iod.py`)
2. Read/write checkpoints to GCS instead of local disk (`gcsfs` or `google-cloud-storage`)
3. Shard the observations parquet into per-object files in GCS (avoids each worker
   loading the full ~TB table)
4. Add a Ray actor for progress tracking / fault tolerance

### Option B: GCP Batch (simpler, no Ray dependency)

Each object becomes one GCP Batch job task. GCP Batch handles retries, spot
preemption recovery, and parallelism automatically.

```
GCP Batch job array
├── 1,100,000 tasks, each running:
│   python scripts/02_run_looo.py --object-ids <provid> \
│       --input-dir gs://... --output-dir gs://...
└── Parallelism: 1,000 concurrent tasks
```

Simpler to set up, slightly less efficient (per-task startup overhead ~10s).

### Option C: Spark / Dataproc (if data team prefers)

Use PySpark on Cloud Dataproc with the LOOO logic as a UDF over a partitioned
observations table. More complex to implement but integrates naturally with
BigQuery via the BigQuery Storage API.

---

## Data Volume

| File | Estimated size |
|------|---------------|
| Input observations (full MPC) | ~500 GB (parquet) |
| Input orbits | ~2 GB |
| Per-object checkpoints | ~50 bytes/row × 600M held-out obs = ~30 GB |
| Final merged result | ~30 GB |
| Observatory/catalog stats | <10 MB |

---

## Recommended Phased Approach

### Phase 1 — Numbered asteroids only (highest scientific value)
- ~900K objects, better-constrained orbits, more observations per object
- Estimated cost: ~$400–700

### Phase 2 — Unnumbered provisionals
- ~200K eligible objects, shorter arcs, more eligibility failures
- Add after Phase 1 to validate short-arc behaviour

### Phase 3 — Re-run with improved sigma model
- Once Veres 2017 sigmas are validated on the pilot, re-run Phase 1 with
  observatory-specific calibration corrections applied iteratively

---

## Infrastructure Prerequisites

1. **BigQuery access**: existing (`mpc_sbn_aurora`, `moeyens-thor-dev` project) ✓
2. **GCS bucket**: create `gs://adam-looo/` with lifecycle rules
3. **SPICE kernels**: bundle into worker Docker image (currently loaded from
   `naif-de440` package — works in containers)
4. **ASSIST ephemeris files**: bundle `jpl-small-bodies-de441-n16` into image ✓
5. **Worker Docker image**: extend `adam-assist` base image with
   `adam_orbit_det_eval` installed
6. **IAM**: workers need read access to BigQuery + GCS read/write on the output bucket
