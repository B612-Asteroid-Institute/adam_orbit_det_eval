#!/usr/bin/env bash
# monitor_cloud_job.sh — periodic health check for GKE batch jobs.
#
# Flags deadlock/stall signatures for an exp-research batch job and pages via
# terminal bell + macOS notification. Read-only kubectl + gsutil only.
#
# Checks per cycle:
#   1. kubectl top pods -l job-name=<job>: per-pod CPU vs its request.
#      Running pods past --warmup-minutes with CPU below --cpu-threshold-pct
#      of their request are flagged (deadlock signature from the ojo incident).
#   2. gsutil ls <output-prefix>/*/SUCCESS and .../FAILED: SUCCESS count
#      that has not advanced in 2× --expected-shard-minutes while pods are
#      still running trips a stall alert. FAILED count >= --max-failed trips
#      a failure-budget alert.
#   3. kubectl get pods: restart count >= --backoff-limit-per-index trips.
# Flagged pods get a describe + tail-N log dump written to logs/forensics_*.log.
#
# Dry-run output (ojo resubmit, 2026-04-21, --cycles 1):
# ------------------------------------------------------------------------------
#   ./scripts/monitor_cloud_job.sh \
#       --job-name mpc-pilot-real-data-looo-20260420213109 \
#       --namespace research \
#       --output-prefix gs://exp-research/mpc-real-data-looo-pilot/output \
#       --cycles 1
#
#   [2026-04-21T06:58:02Z] START monitor job=mpc-pilot-real-data-looo-20260420213109
#       ns=research ctx=gke_moeyens-thor-dev_us-west1-a_exp-research
#       output=gs://exp-research/mpc-real-data-looo-pilot/output
#       interval=300s warmup=10m expected_shard=420m max_failed=20
#       cpu_threshold_pct=5 backoff_limit=2
#   [2026-04-21T06:58:05Z] SUMMARY pods_total=0 running=0 SUCCESS=0 FAILED=3
#       stalled=0 backoff=0 stale_min=0 anomalies=0
#   [2026-04-21T06:58:05Z] STOP monitor (cycles=1 reached)
# ------------------------------------------------------------------------------
# The ojo pilot job had already terminated Failed (0/4 completions, all pods
# reaped) by the time the dry-run ran; so no pods to sample CPU from, and
# SUCCESS/FAILED marker counts read correctly from GCS (3 FAILED, below the
# --max-failed=20 budget, no alert).

set -o pipefail

# --- defaults ---------------------------------------------------------------
NAMESPACE="research-kat"
JOB_NAME=""
OUTPUT_PREFIX=""
EXPECTED_SHARD_MINUTES=420
INTERVAL_SECONDS=300
WARMUP_MINUTES=10
MAX_FAILED=20
CPU_THRESHOLD_PCT=5
BACKOFF_LIMIT_PER_INDEX=2
FORENSIC_TAIL_LINES=200
CONTEXT=""
CYCLES=0  # 0 = run forever

usage() {
    cat <<EOF
Usage: $(basename "$0") --job-name NAME --output-prefix GS_URL [options]

Required:
  --job-name NAME                Kubernetes Job name (label selector).
  --output-prefix GS_URL         GCS prefix, e.g. gs://exp-research/foo/output

Optional:
  --namespace NAME               Kubernetes namespace (default: research-kat).
  --context CTX                  kubectl context (default: current-context).
  --expected-shard-minutes N     Expected wall-time per shard (default: 420).
  --interval-seconds N           Cycle interval (default: 300).
  --warmup-minutes N             Grace period after pod start before CPU check
                                 (default: 10).
  --max-failed N                 FAILED marker count that trips alert
                                 (default: 20).
  --cpu-threshold-pct N          Per-pod CPU % of request below which Running
                                 pods are flagged stalled (default: 5).
  --backoff-limit-per-index N    Restart count that trips alert (default: 2).
  --forensic-tail-lines N        Log lines captured per flagged pod (default: 200).
  --cycles N                     Run N cycles then exit; 0 = forever (default: 0).
  -h, --help                     Show this help.
EOF
}

# --- arg parse --------------------------------------------------------------
while [[ $# -gt 0 ]]; do
    case "$1" in
        --namespace) NAMESPACE="$2"; shift 2 ;;
        --job-name) JOB_NAME="$2"; shift 2 ;;
        --output-prefix) OUTPUT_PREFIX="$2"; shift 2 ;;
        --expected-shard-minutes) EXPECTED_SHARD_MINUTES="$2"; shift 2 ;;
        --interval-seconds) INTERVAL_SECONDS="$2"; shift 2 ;;
        --warmup-minutes) WARMUP_MINUTES="$2"; shift 2 ;;
        --max-failed) MAX_FAILED="$2"; shift 2 ;;
        --cpu-threshold-pct) CPU_THRESHOLD_PCT="$2"; shift 2 ;;
        --backoff-limit-per-index) BACKOFF_LIMIT_PER_INDEX="$2"; shift 2 ;;
        --forensic-tail-lines) FORENSIC_TAIL_LINES="$2"; shift 2 ;;
        --context) CONTEXT="$2"; shift 2 ;;
        --cycles) CYCLES="$2"; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        *) echo "unknown arg: $1" >&2; usage >&2; exit 2 ;;
    esac
done

if [[ -z "$JOB_NAME" ]]; then
    echo "ERROR: --job-name required" >&2; usage >&2; exit 2
fi
if [[ -z "$OUTPUT_PREFIX" ]]; then
    echo "ERROR: --output-prefix required" >&2; usage >&2; exit 2
fi
if [[ -z "$CONTEXT" ]]; then
    CONTEXT=$(kubectl config current-context 2>/dev/null || true)
fi

mkdir -p logs

STAMP=$(date -u +%Y%m%dT%H%M%SZ)
LOG_FILE="logs/cloud_monitor_${JOB_NAME}_${STAMP}.log"
PROGRESS_FILE="logs/.monitor_progress_${JOB_NAME}.log"

log() {
    local line
    line="[$(date -u +%Y-%m-%dT%H:%M:%SZ)] $*"
    echo "$line"
    echo "$line" >>"$LOG_FILE"
}

notify() {
    local msg="$1"
    printf '\a'
    if command -v osascript >/dev/null 2>&1; then
        osascript -e "display notification \"${msg//\"/\\\"}\" with title \"cloud-job monitor: $JOB_NAME\"" \
            >/dev/null 2>&1 || true
    fi
}

dump_forensics() {
    local pod="$1"
    local reason="$2"
    local out="logs/forensics_${JOB_NAME}_${pod}_$(date -u +%Y%m%dT%H%M%SZ).log"
    {
        echo "=== reason: $reason ==="
        echo "=== kubectl describe pod $pod ==="
        kubectl ${CONTEXT:+--context "$CONTEXT"} describe pod -n "$NAMESPACE" "$pod" 2>&1
        echo ""
        echo "=== kubectl logs (tail $FORENSIC_TAIL_LINES) ==="
        kubectl ${CONTEXT:+--context "$CONTEXT"} logs -n "$NAMESPACE" "$pod" \
            --tail="$FORENSIC_TAIL_LINES" 2>&1 || true
    } >"$out"
    log "forensics dumped: $out"
}

count_lines() {
    local raw
    raw=$(cat)
    if [[ -z "$raw" ]]; then echo 0; return; fi
    printf '%s\n' "$raw" | wc -l | tr -d ' '
}

cycle() {
    local now_epoch
    now_epoch=$(date -u +%s)
    local flagged_pods=""

    # 1. pod inventory (name, phase, startTime, restartCount, cpuRequest_mc)
    local pods_json pod_lines
    pods_json=$(kubectl ${CONTEXT:+--context "$CONTEXT"} get pods -n "$NAMESPACE" \
            -l "job-name=$JOB_NAME" -o json 2>/dev/null || echo '{"items":[]}')

    pod_lines=$(printf '%s' "$pods_json" | python3 - 2>/dev/null <<'PY' || true
import json, sys
def parse_cpu(s):
    if not s: return 0
    s = str(s).strip()
    if s.endswith("m"):
        try: return int(float(s[:-1]))
        except Exception: return 0
    try: return int(float(s) * 1000)
    except Exception: return 0
raw = sys.stdin.read()
try:
    d = json.loads(raw)
except Exception:
    sys.exit(0)
for p in d.get("items", []):
    name = p.get("metadata", {}).get("name", "")
    phase = p.get("status", {}).get("phase", "")
    start = p.get("status", {}).get("startTime", "")
    rc = 0
    for cs in p.get("status", {}).get("containerStatuses", []) or []:
        rc = max(rc, cs.get("restartCount", 0))
    cpu_req = 0
    for c in p.get("spec", {}).get("containers", []) or []:
        cpu_req = parse_cpu(c.get("resources", {}).get("requests", {}).get("cpu", ""))
        if cpu_req: break
    print(f"{name}\t{phase}\t{start}\t{rc}\t{cpu_req}")
PY
)

    local total_count=0 running_count=0
    if [[ -n "$pod_lines" ]]; then
        total_count=$(printf '%s\n' "$pod_lines" | wc -l | tr -d ' ')
        running_count=$(printf '%s\n' "$pod_lines" | awk -F'\t' '$2=="Running"' | wc -l | tr -d ' ')
    fi

    # 2. kubectl top (millicores)
    local top_out
    top_out=$(kubectl ${CONTEXT:+--context "$CONTEXT"} top pods -n "$NAMESPACE" \
            -l "job-name=$JOB_NAME" --no-headers 2>/dev/null || true)

    # 3. per-pod checks
    local stalled=0 backoff=0
    if [[ -n "$pod_lines" ]]; then
        while IFS=$'\t' read -r name phase start rc cpu_req; do
            [[ -z "$name" ]] && continue

            if [[ "$rc" =~ ^[0-9]+$ && "$rc" -ge "$BACKOFF_LIMIT_PER_INDEX" ]]; then
                backoff=$((backoff+1))
                flagged_pods="$flagged_pods $name"
                log "ALERT pod=$name restart_count=$rc >= backoff_limit_per_index=$BACKOFF_LIMIT_PER_INDEX"
            fi

            if [[ "$phase" == "Running" && -n "$start" && "$cpu_req" -gt 0 ]]; then
                local start_epoch age_min
                start_epoch=$(python3 -c "import datetime,sys; print(int(datetime.datetime.fromisoformat(sys.argv[1].replace('Z','+00:00')).timestamp()))" "$start" 2>/dev/null || echo 0)
                age_min=$(( (now_epoch - start_epoch) / 60 ))
                if [[ "$age_min" -ge "$WARMUP_MINUTES" ]]; then
                    local cpu_mc cpu_num pct
                    cpu_mc=$(printf '%s\n' "$top_out" | awk -v n="$name" '$1==n {print $2}')
                    if [[ -n "$cpu_mc" ]]; then
                        cpu_num=${cpu_mc%m}
                        if [[ "$cpu_num" =~ ^[0-9]+$ ]]; then
                            pct=$(( cpu_num * 100 / cpu_req ))
                            if [[ "$pct" -lt "$CPU_THRESHOLD_PCT" ]]; then
                                stalled=$((stalled+1))
                                flagged_pods="$flagged_pods $name"
                                log "ALERT pod=$name phase=Running age_min=$age_min cpu=${cpu_mc} pct_of_request=${pct}% < ${CPU_THRESHOLD_PCT}% (request=${cpu_req}m)"
                            fi
                        fi
                    fi
                fi
            fi
        done <<<"$pod_lines"
    fi

    # 4. GCS marker counts
    local success_count failed_count
    success_count=$(gsutil ls "${OUTPUT_PREFIX%/}/*/SUCCESS" 2>/dev/null | count_lines)
    failed_count=$(gsutil ls "${OUTPUT_PREFIX%/}/*/FAILED" 2>/dev/null | count_lines)

    # 5. progress stagnation
    local last_success=0 last_progress_epoch="$now_epoch"
    if [[ -f "$PROGRESS_FILE" ]]; then
        last_success=$(awk -F'\t' 'NR==1{print $1}' "$PROGRESS_FILE" 2>/dev/null || echo 0)
        last_progress_epoch=$(awk -F'\t' 'NR==1{print $2}' "$PROGRESS_FILE" 2>/dev/null || echo "$now_epoch")
        [[ -z "$last_success" ]] && last_success=0
        [[ -z "$last_progress_epoch" ]] && last_progress_epoch="$now_epoch"
    fi
    if [[ "$success_count" -gt "$last_success" ]]; then
        last_progress_epoch="$now_epoch"
    fi
    printf '%s\t%s\n' "$success_count" "$last_progress_epoch" >"$PROGRESS_FILE"

    local stale_min=$(( (now_epoch - last_progress_epoch) / 60 ))
    local stall_budget=$(( 2 * EXPECTED_SHARD_MINUTES ))
    local marker_stalled=0
    if [[ "$stale_min" -ge "$stall_budget" && "$running_count" -gt 0 ]]; then
        marker_stalled=1
        log "ALERT GCS SUCCESS stagnant: count=$success_count stale_min=$stale_min >= budget=${stall_budget}m while $running_count pods Running"
    fi
    local failed_tripped=0
    if [[ "$failed_count" -ge "$MAX_FAILED" ]]; then
        failed_tripped=1
        log "ALERT FAILED marker count $failed_count >= --max-failed=$MAX_FAILED"
    fi

    local anomalies=$(( stalled + backoff + marker_stalled + failed_tripped ))

    log "SUMMARY pods_total=$total_count running=$running_count SUCCESS=$success_count FAILED=$failed_count stalled=$stalled backoff=$backoff stale_min=$stale_min anomalies=$anomalies"

    if [[ "$anomalies" -gt 0 ]]; then
        notify "anomalies=$anomalies stalled=$stalled backoff=$backoff failed=$failed_count"
        # dedupe forensic dumps per cycle
        local seen=""
        for p in $flagged_pods; do
            case " $seen " in
                *" $p "*) continue ;;
            esac
            seen="$seen $p"
            dump_forensics "$p" "flagged_cycle"
        done
    fi
}

log "START monitor job=$JOB_NAME ns=$NAMESPACE ctx=$CONTEXT output=$OUTPUT_PREFIX interval=${INTERVAL_SECONDS}s warmup=${WARMUP_MINUTES}m expected_shard=${EXPECTED_SHARD_MINUTES}m max_failed=$MAX_FAILED cpu_threshold_pct=$CPU_THRESHOLD_PCT backoff_limit=$BACKOFF_LIMIT_PER_INDEX"

trap 'log "STOP monitor (signal)"; exit 0' INT TERM

run=0
while true; do
    cycle
    run=$((run+1))
    if [[ "$CYCLES" -gt 0 && "$run" -ge "$CYCLES" ]]; then
        log "STOP monitor (cycles=$CYCLES reached)"
        exit 0
    fi
    sleep "$INTERVAL_SECONDS"
done
