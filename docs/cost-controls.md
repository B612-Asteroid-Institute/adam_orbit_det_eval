# MPC LOOO — Cloud Cost Controls

Tracks: bead `beads_agent_setup-cue` (P2). Audit + recommendations for the
full-MPC observatory-bias run, after pilot v10 (2026-04-24, 4 shards × 6 CPU).
Production target: 200 shards × 16 CPU × 4–8 h each.

## Audit summary (2026-04-27)

| Area | Current state | Verdict |
|---|---|---|
| Bucket region | `gs://exp-research` is `us-west1` STANDARD, 3.9 TiB total | OK — matches cluster |
| Artifact Registry region | `us-west1-docker.pkg.dev/moeyens-thor-dev/ai`, repo total 2.5 TiB | OK — region; size is shared with non-MPC packages |
| GCS lifecycle | None on `gs://exp-research` | Gap — pilot/test outputs accumulate forever |
| Cloud Logging _Default sink | 2 exclusions, both `disabled: true` | Gap — verbose pod stdout shipped at full volume |
| Cloud Build region | Submits go to `locations/global`; source bucket `moeyens-thor-dev_cloudbuild` is **multi-region US** | Gap — cross-region pull from US into us-west1 worker on every submit |
| AR cleanup policy | `cleanupPolicyDryRun: true`, but no policies attached | Gap — every pilot tag persists indefinitely |
| Pilot image footprint (`mpc-real-data-looo`) | 7 tagged pilot images × ~6.6 GB ≈ 46 GB | $4–5/mo at AR list price |

## Pilot v10 measurements

Logging volume (per-pod, per-hour) sampled from
`gcloud logging read resource.labels.pod_name=mpc-pilot-real-data-looo-...`:

| Pod | Window | Entries/hr | JSON envelope |
|---|---|---|---|
| shard 0 (`-0-5nngz`) | 1 h mid-run | ~1,700 | ~2.5 MB |
| shard 3 (`-3-qcjxd`) | 1 h end-of-run | ~3,087 | ~4.5 MB |

So pilot pods emit ~5 MB/pod-hour (envelope; chargeable ingestion is ~70% of
that). Well **under** the bead's 100 MB/pod-hour threshold for "must reduce
verbosity", but still worth filtering at production scale.

Pilot output footprint (`gs://exp-research/mpc-real-data-looo-pilot/`):

| Prefix | Size |
|---|---|
| `input/` | 76.78 MiB |
| `output/shard_000/` | 4.32 MiB |
| `output/shard_001/` | 6.20 MiB |
| `output/shard_002/` | 8.63 MiB |
| `output/shard_003/` | 4.59 MiB |
| `runs/` | 2.55 KiB |
| **Total** | **100.5 MiB** |

A unique side observation: every pod log line lands at `severity=ERROR` in
Cloud Logging because Python's default handler writes to stderr and GKE
auto-tags stderr as ERROR. The Python level (`INFO`/`WARNING`/`ERROR`) lives
inside `textPayload`. Any exclusion filter using `severity<WARNING` will drop
**nothing** — must filter on `textPayload` instead.

## Production-run cost projection

Extrapolating pilot v10 to 200 shards × 6 h average (mid-point of 4–8 h):

| Item | Volume | Unit | Monthly $ |
|---|---|---|---|
| Cloud Logging ingestion | 200 × 6 × ~5 MB ≈ **6 GB / run** | $0.50 / GB | ~$3 / run |
| GCS output (Standard) | 200 × ~6 MB ≈ 1.2 GB; + 3.8 GB inputs | $0.020 / GB-mo | $0.10 / mo |
| AR storage (new prod tags) | ~6.6 GB / image | $0.10 / GB-mo | $0.66 / mo / tag |
| AR storage (existing 7 mpc-real-data-looo pilot tags) | ~46 GB | $0.10 / GB-mo | $4.60 / mo |
| Cloud Build cross-region pull | source bucket is US multi-region; egress to global build worker | $0.01–0.02 / GB | low (per submit) |

**Bottom line**: at the current cadence the hard dollar cost from "stuff we
control via this bead" is dominated by AR image storage. Cloud Logging is in
the noise unless verbosity blows up. GCS output is negligible. The cross-region
build path costs almost nothing per submit but is the single thing that's
*architecturally wrong* — it'll keep biting any future scale-up too.

## Recommendations

### Logging

| Change | Risk | Status | Est. savings |
|---|---|---|---|
| Add `_Default` sink exclusion for MPC pod INFO/DEBUG `textPayload` (keep WARN/ERROR) | Reversible | **Applied** | ~70% of MPC log GB → ~$2/run |
| Drop per-object `Downloaded …` / `Uploaded …` info lines in `12_run_looo_cloud_shard.py` | Reversible (code change) | **Deferred** — needs code review | Smaller than the exclusion |
| Disable cluster `WORKLOADS` logging | Cluster-wide, **destructive** | **Deferred** — bead non-goal | n/a |

The exclusion filter (Python `INFO` and `DEBUG` levels embedded in
`textPayload`, scoped to MPC pods only):

```
resource.type="k8s_container"
AND resource.labels.cluster_name="exp-research"
AND resource.labels.namespace_name=("research" OR "research-kat")
AND resource.labels.pod_name=~"^mpc-(pilot-)?real-data-looo-.*"
AND textPayload=~" (INFO|DEBUG) "
```

This preserves WARNING / ERROR / CRITICAL from `adam_orbit_det_eval.looo`,
`adam_fo`, and the runner. Reverse with
`gcloud logging sinks update _Default --remove-exclusion=mpc-pod-info`.

### Storage Lifecycle

| Change | Risk | Status |
|---|---|---|
| Lifecycle rules on `gs://exp-research` scoped to pilot/test prefixes only (Nearline @ 30 d, Delete @ 90 d) | Reversible | **Applied** |
| Lifecycle rules on `gs://exp-research/mpc-real-data-looo/` (production) | **Destructive** | **Deferred** — never auto-delete production results without operator sign-off |
| Convert `gs://precovery_experiments` (multi-region US, same project) to regional | Out of scope | Flag for Kathleen — not MPC-owned |

Applied prefixes (explicit, never matches the `mpc-real-data-looo/` production
prefix):

- `mpc-real-data-looo-pilot/`
- `mpc-isolation-study/`

### Image Cleanup

| Change | Risk | Status | Est. savings |
|---|---|---|---|
| AR cleanup policy on `ai` repo: keep most recent 10 versions per package, **dry-run only**, scoped to `mpc-` package prefixes | Dry-run = safe | **Applied** | After review + apply: ~$4/mo on existing 6 stale pilot tags |
| Delete the 6 superseded mpc-real-data-looo pilot images (`pilot-v6` through `pilot-v9`, `pilot-nodeps-20260416`, `pilot-20260416114936`) | **Destructive** | **Deferred** — needs explicit sign-off | ~$4/mo |
| Set repository-level cleanup-on-untag for orphan layers | Low risk but project-wide | **Deferred** | low |

After running for ~7 days the dry-run policy surfaces what would be deleted
in the AR console; flip `cleanupPolicyDryRun` to `false` only after operator
review.

### Region Verification

| Item | Current | Action |
|---|---|---|
| Cluster | `us-west1-a` | OK |
| Bucket `gs://exp-research` | `us-west1` STANDARD | OK |
| Artifact Registry `ai` | `us-west1` | OK |
| Cloud Build submit region | `global` (default) | **Deferred** — needs `--region=us-west1` added to the `gcloud builds submit` invocation in `exp-research/expctl` (reference repo, out of scope here) |
| Cloud Build source bucket `moeyens-thor-dev_cloudbuild` | Multi-region US | **Deferred** — recreating this bucket as us-west1 is project-wide and affects every team using Cloud Build |

The Cloud Build cross-region issue is small per submit but architecturally
wrong: every `expctl submit` uploads source to a US-multi-region bucket, then
a global Cloud Build worker pulls it before pushing to the us-west1 AR. The
fix lives in `exp-research/expctl` (line ~717: add `"--region=us-west1"` to
the `gcloud builds submit` call).

## Operator checklist (after each wave)

1. `expctl cleanup --dry-run` — review which run prefixes / images would be
   deleted; rerun with `--apply` for the targeted set.
2. `gsutil ls gs://exp-research/mpc-real-data-looo-pilot/output/` — confirm
   pilot-only outputs are aging into Nearline (visible via
   `gsutil stat` → `Storage class: NEARLINE`).
3. `gcloud logging read 'jsonPayload.message:"…" OR textPayload:"…"' …` —
   spot-check that errors/warnings still surface despite the exclusion.
4. `gcloud artifacts repositories describe ai --location=us-west1 \
   --format='value(cleanupPolicies)'` — confirm dry-run policy is logging
   intended deletions before flipping to live mode.
