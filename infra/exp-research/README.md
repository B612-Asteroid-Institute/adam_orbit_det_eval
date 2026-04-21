# MPC-Scale Real-Data LOOO Cloud Run

Run the full MPC observatory bias evaluation at scale using the exp-research
GKE batch framework. Each pod processes one shard of ~1000 objects through
the Leave-One-Observatory-Out pipeline with ASSIST propagation and FindOrb
orbit fitting.

## Prerequisites

- `gcloud` CLI authenticated with `moeyens-thor-dev` project
- `expctl` CLI (from the `exp-research` repo)
- Kubeconfig for the `exp-research` GKE cluster:
  ```bash
  gcloud container clusters get-credentials exp-research \
      --region us-west1-a --project moeyens-thor-dev
  ```

## Step 1: Fetch and shard input data

Run the MPC-scale fetch script to download eligible objects from BigQuery
and shard them into groups of ~1000:

```bash
cd adam_orbit_det_eval
python scripts/15_fetch_mpc_scale.py \
    --output-dir data/mpc_scale_full \
    --shard-size 1000
```

This produces `data/mpc_scale_full/shard_NNN/` directories with
`mpc_observations.parquet` and `mpc_orbits.parquet` in each.

Check `data/mpc_scale_full/fetch_metadata.json` for the total shard count.
Update `job.completions` in the job spec to match.

## Step 2: Upload input shards to GCS

```bash
gsutil -m cp -r data/mpc_scale_full/shard_* \
    gs://exp-research/mpc-real-data-looo/input/
```

## Step 3: Build the Docker image and submit the job

Using the exp-research spec (builds the image automatically via Cloud Build):

```bash
cd exp-research
./expctl submit -f ../adam_orbit_det_eval/infra/exp-research/mpc-scale-real-data-looo.json
```

Or build the image locally first (for testing):

```bash
cd adam_orbit_det_eval
docker build -t mpc-real-data-looo:local -f infra/exp-research/Dockerfile ..
```

## Step 4: Monitor the job

```bash
# Watch pod status
kubectl --context gke_moeyens-thor-dev_us-west1-a_exp-research \
    get pods -n research -l job-name=mpc-real-data-looo -w

# Check a specific pod's logs
kubectl --context gke_moeyens-thor-dev_us-west1-a_exp-research \
    logs -n research mpc-real-data-looo-<index>

# Check job completion status
kubectl --context gke_moeyens-thor-dev_us-west1-a_exp-research \
    get job mpc-real-data-looo -n research
```

## Step 5: Collect and merge results

Once enough shards have completed (check for SUCCESS markers):

```bash
cd adam_orbit_det_eval
python scripts/13_collect_cloud_results.py \
    --gcs-output-prefix gs://exp-research/mpc-real-data-looo/output \
    --output-dir data/mpc_scale_results \
    --min-shards-pct 90
```

This downloads all completed shard results, merges them into unified parquet
files, re-computes observatory and program code statistics on the full
dataset, and writes a `collection_report.json` summary.

## Resource configuration

| Setting | Value | Rationale |
|---------|-------|-----------|
| CPU per pod | 16 | LOOO pipeline is CPU-bound (propagation + orbit fitting) |
| Memory per pod | 32 GiB | ~1000 objects × ~200 obs each; FindOrb needs working memory |
| Parallelism | 64 | Max concurrent pods (cpu-spot queue limit: 200 CPUs) |
| Completions | 200 | ~200K eligible objects / 1000 per shard (adjust to actual count) |
| Spot retry | 2 per index | Handles preemption; GCS checkpoints enable resume |
| Max failed | 20 | ~10% failure budget; collector handles partial results |

## Expected cost and time

Based on the 3,500-object isolation study run:

| Metric | Estimate |
|--------|----------|
| Wall time per pod | 4-8 hours (depends on object complexity) |
| Total pod-hours | 800-1600 |
| Spot cost | ~$100-250 (16 CPU spot @ ~$0.15/CPU-hr) |
| Output size | ~10-20 GB across all shards |
| End-to-end wall time | ~12-24 hours with 64-way parallelism |

## Monitoring

The `scripts/monitor_cloud_job.sh` watcher catches the deadlock/stall signature
we saw on the ojo pilot (pods `Running` but hung at near-zero CPU with no new
GCS SUCCESS markers). Launch it alongside every submit in a second terminal:

```bash
cd adam_orbit_det_eval
./scripts/monitor_cloud_job.sh \
    --job-name mpc-real-data-looo \
    --namespace research \
    --output-prefix gs://exp-research/mpc-real-data-looo/output \
    --expected-shard-minutes 420 \
    --interval-seconds 300
```

Each cycle writes a SUMMARY line to stdout and to a rotating log under
`logs/cloud_monitor_<job>_<UTC>.log`. Threshold trips raise the terminal bell
and a macOS notification, and dump `describe pod` + the last 200 log lines
to `logs/forensics_*.log` for post-mortem. The script only calls read-only
`kubectl` / `gsutil` verbs (no apply/delete). Run `--help` for the full flag
list (warmup grace, CPU %, FAILED budget, etc.). Stop with `Ctrl-C` or let
`--cycles N` bound the run (e.g. `--cycles 1` for a one-shot health check).

## Customization

The GCS paths in `mpc-scale-real-data-looo.json` can be edited directly.
Key fields to adjust:

- `job.completions`: match the number of shards from the fetch step
- `command` args: `--gcs-input-prefix` and `--gcs-output-prefix`
- `resources`: increase memory if OOM errors occur
- `retry.max_failed_indexes`: increase tolerance for transient failures
