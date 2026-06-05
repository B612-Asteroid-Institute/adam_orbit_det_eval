# MPC-Scale LOOO Cloud Run — Operational Runbook

Operational procedure for running the full MPC observatory-bias LOOO study on
the exp-research GKE batch cluster. Distilled from the v1 (pilot-v12) full run
and the ~11 hours of Pending / cohort-rejection churn that preceded it. Follow
this top to bottom; it captures the two facts that cost the most time —
**the 8-vCPU node ceiling** and **the Kueue all-or-nothing admission model**.

- **Reference run:** job `mpc-real-data-looo-20260510060359`, image
  `pilot-v12-20260507`, queue `cpu-ondemand`, namespace `research-kat`.
  **19/19 SUCCESS in 5h18m.** Filter loss 2.65%, 13/13 anchors present,
  max `|residual_ra|` = 58.7″ after ITF cleanup.
- **Source incident:** bead `a6w`. Pilot history: beads `dez` (v12 build),
  `hi7` (sanity pilot, skipped), `ojo` (cloud-path validation), `rwj` (kubectl auth).
- Operational only. Scientific interpretation: `docs/mpc-bias-catalog-interpretation.md`.
  Original infra plan: `infra/exp-research/README.md` — its 16-CPU resource table
  is **obsolete**; this runbook supersedes it.

---

## 1. Cluster sizing reality (read first)

All cpu nodepools on exp-research (`cpu-spot`, `cpu-ondemand`, `cpu-highmem-spot`)
are **e2-standard-8 / e2-highmem-8 — 8 vCPU per node**.

- **Per-pod CPU request MUST be ≤ 7.** Leave 1 vCPU for kubelet/system, or the
  pod never schedules and sits `Pending`.
- **Per-pod memory ≤ ~28 GiB** (rough node headroom; e2-standard-8 = 32 GiB total).
- The **legacy 16-CPU spec** from the 3,500-object era (still printed in
  `infra/exp-research/README.md`) **cannot schedule on this cluster.** Do not copy it.

### Fairshare quotas and cohort caps (`research-kat` tenant)

| Queue | Tenant quota | Cohort cap |
|-------|--------------|------------|
| `cpu-spot` | 100 CPU / 400 GiB | `research-cpu-spot-fairshare` 400 CPU |
| `cpu-ondemand` | 25 CPU / 100 GiB | `research-cpu-ondemand-fairshare` 100 CPU |

Borrowing above the tenant quota (up to the cohort cap) is allowed **when
uncontested**. Definitions live in
`exp-research/manifests/kueue/team-fairshare-queues.yaml` and `queues.yaml`.

### Kueue admission model (the silent blocker)

Kueue reserves `min(parallelism, completions) × per-pod resources` **upfront,
all-or-nothing**. The whole job is admitted or none of it is.

```
total CPU at admission    = parallelism × per-pod CPU
total memory at admission = parallelism × per-pod memory
```

Both totals must fit under the tenant quota + cohort cap, *or the job hangs
`Pending` indefinitely with no obvious error*. This is what produced the ~11h
Pending state pre-v1.

---

## 2. The v1 working spec (validated, `infra/exp-research/mpc-scale-real-data-looo.json`)

```jsonc
{
  "name": "mpc-real-data-looo",
  "namespace": "research-kat",      // pairs with queue cpu-ondemand
  "queue": "cpu-ondemand",
  "mode": "indexed",
  "image": "us-west1-docker.pkg.dev/moeyens-thor-dev/ai/mpc-real-data-looo:pilot-v12-20260507",
  "resources": { "cpu": "7", "memory": "16Gi", "gpu": 0 },
  "job": { "completions": 19, "parallelism": 14, "ttl_seconds_after_finished": 172800 },
  "retry": { "backoff_limit_per_index": 6, "max_failed_indexes": 4 },
  "outputs": { "gcs_prefix": "gs://exp-research/mpc-real-data-looo-20260509" },
  "command": [
    "python", "/app/adam_orbit_det_eval/scripts/12_run_looo_cloud_shard.py",
    "--gcs-input-prefix",  "gs://exp-research/mpc-real-data-looo/input",
    "--gcs-output-prefix", "gs://exp-research/mpc-real-data-looo-20260509/output",
    "--propagator", "assist", "--orbit-fitter", "findorb",
    "--strict-fitter", "--max-processes", "5"
  ]
}
```

Why these numbers:

- `cpu: "7"` — node ceiling (§1).
- `--max-processes 5` — the per-pod parallel worker pool. Kept under the 7-CPU
  budget so the pod doesn't oversubscribe.
- `parallelism: 14` — admission footprint 14×7 = 98 CPU (just under the
  `cpu-ondemand` cohort cap of 100) and 14×16 = 224 GiB. **This exceeds the
  ondemand *tenant* quota (25 CPU / 100 GiB) and admits only via cohort
  borrowing — confirm the cohort is uncontested before submit** (see §7 gap note).
- `max_failed_indexes: 4` — ~20% of 19 completions. **Must be ≤ completions** or
  submit fails validation. Scale to ~10–20% of your shard count.
- `gcs_prefix` — **dated, never reused.** A reused prefix surfaces as a 403 on
  upload (stale checkpoints, see §6).

---

## 3. Image build → audit → submit (canonical overlay strategy)

Two Dockerfiles, both in `infra/exp-research/`:

- **`Dockerfile.production`** — full from-scratch build.
- **`Dockerfile.pilot-overlay`** — the canonical fast path. Layers workspace
  `adam_core` + `adam_orbit_det_eval` onto the `mpc-isolation-study` base, then
  builds Find_Orb (binary + ephemerides) in a Bookworm GCC-12 stage (Find_Orb's
  pinned commits don't compile under the base's GCC 14, bead `mxw`).

The build embeds the **non-negotiable gates** from CLAUDE.md → *Cloud Image
Validation Rules*. Pilots v6–v9 burned four full ~30-min cycles on bugs these
would have caught:

1. **Build-time end-to-end smoke `RUN` step** against a 1–3 object fixture —
   the hard gate. Import-only smokes are insufficient (they missed the DE440t
   ephemeris gap that silently produced zero-row output across four pilots).
2. **DE440t + bc405 verification `RUN` step.** **DE440t (~800 MB) must ship
   inside the image.**
3. **Interactive audit before promoting to a real job image:** `docker run
   --rm -it --entrypoint /bin/bash <tag>` (or a `kubectl run` debug pod when
   local Docker is unavailable), run the entrypoint against a tiny fixture,
   verify pins resolve and `HOME`/`PATH`/data paths.

```bash
# Build via Cloud Build (always a fresh UTC-dated tag — never reuse a tag)
gcloud builds submit \
  --tag us-west1-docker.pkg.dev/moeyens-thor-dev/ai/mpc-real-data-looo:pilot-vN-YYYYMMDD \
  --project moeyens-thor-dev \
  --machine-type e2-highcpu-8 --timeout 2400s \
  /path/to/workspace_root/
```

If the smoke `RUN` step fails, the build fails and no image is pushed — that is
the intended behavior.

---

## 4. Fetch, upload, submit, monitor

```bash
cd adam_orbit_det_eval

# 4a. Fetch + shard input from BigQuery (~1000 obj/shard). Excludes comets,
#     requires ≥3 distinct stations and ≥20 obs per object.
python scripts/15_fetch_mpc_scale.py --output-dir data/mpc_scale_full --shard-size 1000
#     → set job.completions = the shard count in data/mpc_scale_full/fetch_metadata.json

# 4b. Upload shards to GCS
gsutil -m cp -r data/mpc_scale_full/shard_* gs://exp-research/mpc-real-data-looo/input/

# 4c. Submit (from the exp-research repo checkout)
cd ../exp-research
./expctl submit -f ../adam_orbit_det_eval/infra/exp-research/mpc-scale-real-data-looo.json
```

**Always launch the monitor in a second terminal at submit time.** Its primary
catch is the deadlock/stall signature: pods `Running` at near-zero CPU with no
new GCS SUCCESS markers.

```bash
cd adam_orbit_det_eval
./scripts/monitor_cloud_job.sh \
  --job-name   mpc-real-data-looo \
  --namespace  research-kat \
  --context    gke_moeyens-thor-dev_us-west1-a_exp-research \
  --output-prefix gs://exp-research/mpc-real-data-looo-20260509/output \
  --expected-shard-minutes 420 --interval-seconds 300
```

The monitor uses read-only `kubectl`/`gsutil` verbs only; threshold trips ring
the bell, fire a macOS notification, and dump `describe pod` + last 200 log lines
to `logs/forensics_*.log`. `kubectl` works directly now — `gke-gcloud-auth-plugin`
is installed (bead `rwj`); the old bearer-token workaround is retired.

Ad-hoc status:

```bash
CTX=gke_moeyens-thor-dev_us-west1-a_exp-research
kubectl --context $CTX get pods -n research-kat -l job-name=mpc-real-data-looo -w
kubectl --context $CTX logs  -n research-kat mpc-real-data-looo-<index>
kubectl --context $CTX get job mpc-real-data-looo -n research-kat
```

---

## 5. Pre-submit checklist

- [ ] Image built with smoke gate green; interactively audited (§3).
- [ ] `parallelism × memory` ≤ cohort cap memory **and** `parallelism × cpu` ≤
      cohort cap CPU (§1).
- [ ] Per-pod `cpu ≤ 7`, `memory ≤ 28 GiB`.
- [ ] `max_failed_indexes ≤ completions`, set to ~10–20% of completions.
- [ ] GCS output prefix is **new and dated** (never reused).
- [ ] `queue` and `namespace` paired correctly (`cpu-ondemand` ↔ `research-kat`).

---

## 6. Failure modes and recovery

| Symptom | Cause | Fix |
|---------|-------|-----|
| Submit validation error | `max_failed_indexes > completions` | Lower `max_failed_indexes`. |
| "spec asks for more memory than the cohort cap allows" | `parallelism × memory` > cohort cap | Shrink per-pod memory or `parallelism`; or split into smaller jobs. |
| Pods stuck `Pending` | per-pod resource > node capacity (CPU > 7), or admission footprint exceeds quota+cohort | `kubectl describe pod` → look for `Unschedulable` / node-selector. Drop CPU to ≤7 (§1) or shrink admission footprint (§1 Kueue). |
| 403 on GCS upload | stale checkpoints in a **reused** output prefix | Use a new dated prefix, or grant `roles/storage.objectAdmin` on the bucket and clean the stale prefix. |
| Pods `Running`, ~0% CPU, no new SUCCESS | deadlock/stall | The monitor pages on this; pull `logs/forensics_*.log` and restart the affected index. |

Two **cohort-cap rejections** and the ~11h Pending state in the pre-v1 path were
all the top two rows of this table. See bead `a6w` for the incident timeline.

---

## 7. Post-run: collect, hygiene, validate

```bash
cd adam_orbit_det_eval

# 7a. Scan SUCCESS/FAILED markers, download + merge completed shards
python scripts/13_collect_cloud_results.py \
  --gcs-output-prefix gs://exp-research/mpc-real-data-looo-20260509/output \
  --output-dir data/mpc_scale_results_YYYYMMDD \
  --min-shards-pct 90
#   → merged_looo_results.parquet (+ observatory/program stats, collection_report.json)

# 7b. Publication hygiene → the published catalog (drops OCC + unknown-stn rows,
#     re-aggregates with the small-sample cutoff n_obs≥100 AND n_objects≥20)
python scripts/18_apply_publication_hygiene.py \
  --results-dir data/mpc_scale_results_YYYYMMDD \
  --source-obs-dir data/mpc_scale_full
#   → merged_looo_results_published.parquet + observatory_stats_published.parquet
#     + publication_hygiene_audit.json

# 7c. v2 final catalog (CANONICAL entrypoint). Chains AT/CT decomposition
#     (script 16) and bias-table generation (script 17) into one atomic,
#     idempotent step. Hard-aborts BEFORE writing any bias table if AT/CT
#     came out empty — the d5b anti-repeat guard. Re-run is overwrite-gated.
python scripts/19_publish_catalog.py \
  --residuals    data/mpc_scale_results_YYYYMMDD/merged_looo_results_published.parquet \
  --observations data/mpc_scale_full/mpc_observations.parquet \
  --orbits       data/mpc_scale_full/mpc_orbits.parquet \
  --output-dir   data/bias_catalog/v2_YYYYMMDD
#   → v2_YYYYMMDD/looo_results_atct.parquet   (raw: AT/CT-augmented residuals)
#   → v2_YYYYMMDD/bias_catalog/               (corrected: bias_table.parquet + .csv + config)
#   (sharded full run: --observations/--orbits are the *merged* source obs +
#    catalog orbits; for a single-shard or 3500-obj run point them at that
#    shard's mpc_observations.parquet / mpc_orbits.parquet. Add --validate-anchors
#    to also emit the anchor validation report.)
```

**v2 final catalog — use script 19, not 16 + 17 by hand.**
`scripts/19_publish_catalog.py` is the canonical "produce the final catalog"
entrypoint for v2. It runs AT/CT decomposition and then the bias table as a
single command, emitting **both** the *raw* AT/CT-augmented residuals parquet
and the *corrected* `bias_catalog/` directory under `--output-dir`. It exists
to prevent a repeat of the v1/d5b incident — where the catalog shipped with
empty AT/CT columns because scripts 16 and 17 were separate manual steps:
script 17 only ever sees the augmented parquet, and the wrapper refuses to
generate the bias table if AT/CT is all-null, so a catalog with empty AT/CT
can no longer ship. Re-running is idempotent and overwrite-gated (pass
`--overwrite` to regenerate in place).

**First-accuracy pass:** compare the published catalog's 13 anchor stations to
`data/bias_catalog/3500obj/`. v1 baseline: 13/13 anchors present, filter loss
2.65%, max `|residual_ra|` = 58.7″ after ITF mitigation.

**ITF guard:** the `status='I'` (ITF) row-drop is upstream in
`scripts/15_fetch_mpc_scale.py` (commit `6b78d9f`). If a future run shows extreme
residuals (> 1000″), check whether ITF rows leaked in by re-running the BQ status
audit (see bd memory `mpc-itf-status-filter`). v11's catastrophic tail
(max 261,228″) was the warm-start regression, fixed in v12 (bead `dez`).

---

## 8. Open operational gaps (verify before v2)

- **`cpu-ondemand` quota vs. v1 footprint.** The v1 spec's 98 CPU / 224 GiB
  footprint exceeds the documented `cpu-ondemand` *tenant* quota (25 CPU /
  100 GiB) and admitted only via cohort borrowing. Before a v2 submit, confirm
  current quotas with `kubectl describe clusterqueue` / `kubectl get
  localqueue -n research-kat` — borrowing succeeds only when the cohort is
  uncontested, so a contended cluster could re-trigger the Pending state.
- **No JOURNAL.md** exists in this repo; the pre-v1 submit-by-submit timeline was
  reconstructed from beads `a6w`/`dez`/`hi7`/`ojo`, not a primary log.
