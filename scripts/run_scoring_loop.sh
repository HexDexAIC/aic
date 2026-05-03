#!/usr/bin/env bash
# Run an AIC policy N times against the eval stack and collect scoring.yaml from each.
#
# Usage:
#   scripts/run_scoring_loop.sh POLICY N [options]
#
# POLICY is the short class name (e.g. CheatCode, RunACT, CheatCodeMJ).
# It will be prefixed with `aic_example_policies.ros.` when passed to aic_model.
#
# Options:
#   --ground-truth          Pass ground_truth:=true to the launch (needed for CheatCode).
#   --output-dir DIR        Base output dir. Default: ~/aic_results/sweep_<ts>_<policy>.
#   --timeout SEC           Wall-clock timeout per run. Default: 600.
#   --ready-wait SEC        How long to wait for engine to advertise before starting policy. Default: 90.
#   --no-pkill              Skip post-run pkill cleanup (leaves stragglers for debugging).
#   --no-bag                Delete per-trial bag_trial_* dirs after scoring.yaml is written (disk saver).
#
# Layout produced:
#   <OUTPUT_DIR>/
#     run_001/
#       scoring.yaml
#       bag_trial_*           (per-trial rosbags written by the engine)
#       terminal1_eval.log
#       terminal2_policy.log
#     run_002/
#     ...
#     summary.csv             (one row per run)
#     summary.md              (aggregate)

set -euo pipefail

POLICY="${1:-}"
N="${2:-}"
if [[ -z "$POLICY" || -z "$N" ]]; then
    sed -n '2,20p' "$0" >&2
    exit 2
fi
shift 2

GROUND_TRUTH=false
OUTPUT_BASE=""
RUN_TIMEOUT=600
READY_WAIT=90
DO_PKILL=true
GAZEBO_GUI=true
LAUNCH_RVIZ=true
NO_BAG=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --ground-truth) GROUND_TRUTH=true; shift ;;
        --output-dir)   OUTPUT_BASE="$2"; shift 2 ;;
        --timeout)      RUN_TIMEOUT="$2"; shift 2 ;;
        --ready-wait)   READY_WAIT="$2"; shift 2 ;;
        --no-pkill)     DO_PKILL=false; shift ;;
        --no-bag)       NO_BAG=true; shift ;;
        --no-gui)       GAZEBO_GUI=false; shift ;;
        --no-rviz)      LAUNCH_RVIZ=false; shift ;;
        --headless)     GAZEBO_GUI=false; LAUNCH_RVIZ=false; shift ;;
        *) echo "unknown arg: $1" >&2; exit 2 ;;
    esac
done

# Derive paths from the script's own location so it works regardless of
# where the tree is checked out / moved to.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC="$(cd "$SCRIPT_DIR/.." && pwd)"        # .../src/aic
WS="$(cd "$SRC/../.." && pwd)"              # .../ws_aic (src/aic's grandparent)
TS="$(date +%Y-%m-%d_%H%M%S)"

# distrobox on this machine must use docker as the container backend;
# without this it defaults to podman, fails to find aic_eval, and prompts
# to create a new Fedora toolbox (requiring sudo) — which silently breaks
# the whole orchestration.
export DBX_CONTAINER_MANAGER=docker
: "${OUTPUT_BASE:=$WS/aic_results/sweep_${TS}_${POLICY}}"
mkdir -p "$OUTPUT_BASE"

echo "sweep:        $POLICY × $N iterations"
echo "output:       $OUTPUT_BASE"
echo "ground_truth: $GROUND_TRUTH   gazebo_gui: $GAZEBO_GUI   launch_rviz: $LAUNCH_RVIZ"
echo "timeout/run:  ${RUN_TIMEOUT}s"
echo

# CheatCode-family policies read ground-truth TF frames (task_board/.../port_link,
# <cable>/<plug>_link) and will hang indefinitely without them. Refuse to start
# without --ground-truth instead of wasting a whole run.
if [[ "$POLICY" == CheatCode* ]] && [[ "$GROUND_TRUTH" != "true" ]]; then
    echo "ERROR: policy '$POLICY' requires --ground-truth (it reads port/plug TF frames" >&2
    echo "       that are only published when ground_truth:=true)." >&2
    echo "       Re-run with: scripts/run_scoring_loop.sh $POLICY $N --ground-truth ..." >&2
    exit 2
fi

# Preflight: aic_eval container must be running.
if ! docker inspect aic_eval --format '{{.State.Running}}' 2>/dev/null | grep -q '^true$'; then
    echo "ERROR: aic_eval container is not running. Start it first." >&2
    echo "  distrobox enter -r aic_eval -- true   # or equivalent" >&2
    exit 1
fi

# Preflight: distrobox must see the container under the configured backend.
# Catches the "default-to-podman → prompt for sudo fedora pull" trap.
if ! distrobox list 2>/dev/null | grep -q "^[[:alnum:]]\+[[:space:]]*|[[:space:]]*aic_eval[[:space:]]"; then
    echo "ERROR: distrobox (backend=$DBX_CONTAINER_MANAGER) doesn't see a container named aic_eval." >&2
    echo "Check: DBX_CONTAINER_MANAGER=docker distrobox list" >&2
    exit 1
fi

# Pattern that matches any stale policy process on the host.
# Covers: `ros2 run aic_model aic_model ...`, the pixi-installed binary
# `/.pixi/envs/default/lib/aic_model/aic_model`, and any cmdline with
# aic_example_policies in it (the `policy:=...` parameter).
STALE_PATTERN='aic_example_policies|/aic_model/aic_model|ros2 run aic_model'

# Force-kill matching processes. SIGTERM first; verify; escalate to SIGKILL.
kill_policy_on_host() {
    if ! pgrep -f "$STALE_PATTERN" > /dev/null 2>&1; then
        return 0
    fi
    pkill -TERM -f "$STALE_PATTERN" 2>/dev/null || true
    # Wait up to 3s for clean exit.
    for _ in 1 2 3; do
        sleep 1
        if ! pgrep -f "$STALE_PATTERN" > /dev/null 2>&1; then
            return 0
        fi
    done
    # Still alive — force-kill.
    pkill -KILL -f "$STALE_PATTERN" 2>/dev/null || true
    sleep 1
    if pgrep -f "$STALE_PATTERN" > /dev/null 2>&1; then
        echo "WARN: could not kill stale policy process(es):" >&2
        pgrep -af "$STALE_PATTERN" | sed 's/^/  /' >&2
        return 1
    fi
    return 0
}

# Gotcha: if cleanup_container's docker exec carries the kill patterns as
# literal arguments (e.g. `bash -c 'for pat in "aic_adapter" ...'`), then
# `pkill -f "aic_adapter"` inside the container matches OUR OWN docker-exec
# wrapper and SIGKILLs the driver mid-cleanup — leaving stragglers alive.
# Mitigation: write the kill logic to a temp script and invoke it by path
# only. The docker exec cmdline then contains no pattern strings.
CLEANUP_SCRIPT="$(mktemp /tmp/aic_cleanup_XXXXXX.sh)"
cat > "$CLEANUP_SCRIPT" <<'CLEANUP_EOF'
#!/bin/bash
# Kill every AIC-related node inside the aic_eval container. Empirically,
# when an `ros2 launch` tree exits, not all child processes receive the
# shutdown — aic_adapter, rviz2, component_container (ros_gz bridge),
# ground_truth publishers, and gz clients tend to linger. Stale subscribers
# from one run cross-talk with the next run's sim, producing spurious
# motion commands before the new trial has even started.
for pat in \
    "ros2 launch aic_bringup" \
    "aic_engine" \
    "aic_adapter" \
    "ros_gz_container" \
    "ros_gz_bridge" \
    "component_container" \
    "controller_manager" \
    "robot_state_publisher" \
    "joint_state_broadcaster" \
    "fts_broadcaster" \
    "aic_controller" \
    "ground_truth_static_tf_publisher" \
    "ground_truth_tf_relay" \
    "static_transform_publisher" \
    "rviz2" \
    "gz sim" \
    "gzserver" \
    "gzclient" \
    "rmw_zenohd" ; do
    pkill -9 -f "$pat" 2>/dev/null
done
true
CLEANUP_EOF
chmod +x "$CLEANUP_SCRIPT"
# Ensure temp cleanup script is removed on exit (distrobox bind-mounts /tmp).
trap "rm -f '$CLEANUP_SCRIPT' 2>/dev/null || true" EXIT

cleanup_container() {
    # docker exec carries only the script path — no pattern strings in argv,
    # so pkill-f inside cannot match our wrapper.
    docker exec aic_eval "$CLEANUP_SCRIPT" 2>/dev/null || true
    # Host-side stragglers. pgrep/pkill self-exclude by default.
    pkill -9 -f "aic_example_policies|/aic_model/aic_model|ros2 run aic_model|gz sim -g" 2>/dev/null || true
}

# Preflight: no stray policy processes from a previous attempt.
if pgrep -f "$STALE_PATTERN" > /dev/null 2>&1; then
    echo "WARN: existing aic_model process detected. Killing before starting sweep:" >&2
    pgrep -af "$STALE_PATTERN" | sed 's/^/  /' >&2
    if ! kill_policy_on_host; then
        echo "ERROR: could not clear stale policy processes — aborting to avoid" >&2
        echo "       the new run colliding with the old one." >&2
        exit 1
    fi
fi

# Preflight: also purge container-side stragglers (aic_adapter, rviz2,
# component_container, gz clients, etc). These accumulate across failed
# runs and cross-talk with the new run's sim — that's the root cause of
# "robot moves randomly before the env spawns".
echo "Purging any container-side stragglers from prior runs..."
cleanup_container
sleep 2

# INT/TERM handler. Order matters:
#  1. Kill the local engine handle (T1_PID) so docker-exec/distrobox can release.
#  2. Kill host-side policy procs.
#  3. Cleanup container-side stragglers (engine/sim/etc).
# Without (1) the local `docker exec` parent stays alive holding the TTY/pipe,
# and the inner engine process keeps running because docker exec doesn't
# forward SIGINT to grandchildren by default.
on_interrupt() {
    echo
    echo "Interrupted. Cleaning up..."
    if [[ -n "${T1_PID:-}" ]] && kill -0 "$T1_PID" 2>/dev/null; then
        kill -INT  "$T1_PID" 2>/dev/null || true; sleep 1
        kill -TERM "$T1_PID" 2>/dev/null || true; sleep 1
        kill -KILL "$T1_PID" 2>/dev/null || true
    fi
    kill_policy_on_host
    cleanup_container
    exit 130
}
trap on_interrupt INT TERM

for i in $(seq 1 "$N"); do
    RUN_ID="$(printf "run_%03d" "$i")"
    RUN_DIR="$OUTPUT_BASE/$RUN_ID"
    mkdir -p "$RUN_DIR"
    T1_LOG="$RUN_DIR/terminal1_eval.log"
    T2_LOG="$RUN_DIR/terminal2_policy.log"

    echo "┌─ $RUN_ID ($i/$N)  $(date +%H:%M:%S)"
    START="$(date +%s)"

    # Terminal 1 — engine + sim in the container.
    # distrobox enter doesn't reliably forward env vars, so set AIC_RESULTS_DIR
    # explicitly inside the subshell it spawns.
    #
    # AIC_USE_DOCKER_EXEC=1 swaps `distrobox enter -r` for `docker exec`. The
    # distrobox path needs sudo (rootful), which fails from non-TTY shells; the
    # docker-exec path works headlessly. Same container either way.
    ENGINE_CMD_SL="export AIC_RESULTS_DIR='$RUN_DIR' && exec /entrypoint.sh ground_truth:=$GROUND_TRUTH start_aic_engine:=true shutdown_on_aic_engine_exit:=true gazebo_gui:=$GAZEBO_GUI launch_rviz:=$LAUNCH_RVIZ"
    (
        # shutdown_on_aic_engine_exit defaults to false upstream — without it
        # the launch keeps gzserver/aic_adapter/rviz spinning long after the
        # engine has exited cleanly and scoring.yaml is written. Always true
        # for batch sweeps so the run terminates promptly.
        if [[ "${AIC_USE_DOCKER_EXEC:-0}" == "1" ]]; then
            exec docker exec -i aic_eval bash -c "$ENGINE_CMD_SL"
        else
            exec distrobox enter -r aic_eval -- bash -c "$ENGINE_CMD_SL"
        fi
    ) > "$T1_LOG" 2>&1 &
    T1_PID=$!

    # Wait for engine to be ready for the policy.
    # Signal we grep for: "No node with name 'aic_model' found. Retrying..."
    # (engine emits this once per second until the policy shows up)
    ready=0
    t1_died_early=0
    secs=0
    while (( secs < READY_WAIT )); do
        if grep -q "No node with name 'aic_model'" "$T1_LOG" 2>/dev/null; then
            ready=1
            break
        fi
        if ! kill -0 "$T1_PID" 2>/dev/null; then
            t1_died_early=1
            break
        fi
        sleep 1
        secs=$((secs+1))
    done
    if (( t1_died_early == 1 )); then
        echo "│  Terminal 1 exited after ${secs}s before ready — aborting this run"
        echo "│  Last T1 log lines:"
        tail -n 10 "$T1_LOG" | sed 's/^/│    /'
        # No T2 launched yet, nothing to kill; just reap and settle.
        wait "$T1_PID" 2>/dev/null || true
        cleanup_container
        sleep 2
        echo "└─ $RUN_ID FAILED in $(( $(date +%s) - START ))s — Terminal 1 error"
        echo
        continue
    fi
    if (( ready == 0 )); then
        echo "│  engine did not reach ready state within ${READY_WAIT}s (continuing anyway)"
    fi

    # Terminal 2 — policy in pixi.
    (
        cd "$SRC"
        exec pixi run ros2 run aic_model aic_model --ros-args \
            -p use_sim_time:=true \
            -p policy:="aic_example_policies.ros.$POLICY"
    ) > "$T2_LOG" 2>&1 &
    T2_PID=$!

    # Wait for Terminal 1 to finish. Two signals we can detect:
    #   (a) T1 exits naturally — ideal, means launch tore down cleanly.
    #   (b) engine prints "process has finished cleanly" AND scoring.yaml
    #       exists, but the launch tree hasn't exited yet. Observed empirically:
    #       rviz2 / aic_adapter / component_container sometimes ignore the
    #       Shutdown event, blocking the launch. No point waiting out the full
    #       RUN_TIMEOUT — give a short grace then force-kill.
    # Falls back to RUN_TIMEOUT only if neither signal appears (real hang).
    POST_DONE_GRACE=15
    secs=0
    engine_done_at=0
    while kill -0 "$T1_PID" 2>/dev/null && (( secs < RUN_TIMEOUT )); do
        if (( engine_done_at == 0 )); then
            if [[ -f "$RUN_DIR/scoring.yaml" ]] && \
               grep -q "aic_engine.*process has finished cleanly" "$T1_LOG" 2>/dev/null; then
                engine_done_at=$secs
                echo "│  engine finished cleanly — giving launch ${POST_DONE_GRACE}s to tear down"
            fi
        else
            if (( secs - engine_done_at >= POST_DONE_GRACE )); then
                echo "│  launch did not exit ${POST_DONE_GRACE}s after engine — force-shutting down"
                break
            fi
        fi
        sleep 2
        secs=$((secs+2))
    done
    if kill -0 "$T1_PID" 2>/dev/null; then
        if (( engine_done_at == 0 )); then
            echo "│  TIMEOUT (${RUN_TIMEOUT}s) with no clean-engine signal — forcing shutdown"
        fi
        # SIGINT → graceful; escalate if needed.
        kill -INT "$T1_PID" 2>/dev/null || true
        sleep 3
        kill -TERM "$T1_PID" 2>/dev/null || true
        sleep 2
        kill -KILL "$T1_PID" 2>/dev/null || true
    fi
    wait "$T1_PID" 2>/dev/null || true
    T1_RC=$?

    # Give the policy a couple seconds to exit via its own lifecycle shutdown.
    sleep 3

    # Killing T2_PID only hits the outer bash/pixi wrapper — the actual
    # python aic_model process is several layers deep and often survives.
    # kill_policy_on_host() pattern-matches the cmdline and escalates to
    # SIGKILL, which is the only reliable way to clear the child.
    if $DO_PKILL; then
        kill_policy_on_host || true
        cleanup_container
    fi
    wait "$T2_PID" 2>/dev/null || true
    # Short settle to let container release port 7447 before the next run.
    sleep 3

    DURATION=$(( $(date +%s) - START ))

    if [[ "$NO_BAG" == "true" ]]; then
        # --no-bag: scoring.yaml (if written) already summarized this run;
        # the per-trial bags are just disk weight. Prune them either way
        # so even crashed runs don't accumulate bag dirs.
        mapfile -t BAG_DIRS < <(find "$RUN_DIR" -maxdepth 1 -type d -name 'bag_trial_*')
        if (( ${#BAG_DIRS[@]} > 0 )); then
            BAG_SIZE=$(du -sh "${BAG_DIRS[@]}" 2>/dev/null | tail -1 | awk '{print $1}')
            echo "│  --no-bag: pruning ${#BAG_DIRS[@]} bag_trial_* dir(s) (${BAG_SIZE:-?})"
            rm -rf "${BAG_DIRS[@]}"
        fi
    fi

    if [[ -f "$RUN_DIR/scoring.yaml" ]]; then
        # Quick inline extraction of the total so the progress line is useful.
        TOTAL=$(python3 - "$RUN_DIR/scoring.yaml" <<'PY' 2>/dev/null || echo "?"
import sys, yaml
with open(sys.argv[1]) as f:
    data = yaml.safe_load(f)
# Walk any structure looking for a 'total' key at the top or under 'score'.
def find_total(d):
    if isinstance(d, dict):
        for k in ("total_score", "total", "overall_score"):
            if k in d and isinstance(d[k], (int, float)):
                return d[k]
        for v in d.values():
            r = find_total(v)
            if r is not None: return r
    return None
t = find_total(data)
print(f"{t:.2f}" if t is not None else "?")
PY
        )
        echo "└─ $RUN_ID done in ${DURATION}s — total=$TOTAL"
    else
        echo "└─ $RUN_ID FAILED in ${DURATION}s — no scoring.yaml written"
    fi
    echo
done

# Final aggregation.
if command -v python3 >/dev/null 2>&1; then
    python3 "$SCRIPT_DIR/summarize_sweep.py" "$OUTPUT_BASE" | tee "$OUTPUT_BASE/summary.md"
else
    echo "python3 not available on host — skipping summary. Runs are under $OUTPUT_BASE"
fi

echo
echo "Sweep complete: $OUTPUT_BASE"
