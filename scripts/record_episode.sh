#!/usr/bin/env bash
# Record a single deterministic episode of the AIC eval into a LeRobot dataset.
#
# Launches the eval stack with one of the single-trial configs (sfp/sc) and
# runs the TeleopAssist policy, which optionally adds keyboard teleop on top
# of an inner policy and writes a LeRobotDataset to disk.
#
# Usage:
#   src/aic/scripts/record_episode.sh PORT [options]
#
# PORT is "sfp" or "sc"; selects single_trial_<PORT>.yaml.
#
# Options:
#   --inner POLICY        Inner policy class name (e.g. CheatCodeMJ, WaveArm,
#                         RunACT) or "none" for pure-teleop hold mode.
#                         Default: WaveArm.
#   --no-teleop           Disable keyboard teleop (run inner policy unchanged,
#                         still record). Default: teleop on.
#   --no-record           Disable dataset recording (just run the policy).
#   --dataset-root DIR    Base dir under which to create the dataset.
#                         Default: ~/ws_aic/aic_data.
#   --task PROMPT         Task prompt string written into each frame.
#                         Default: derived from PORT.
#   --gui                 Launch Gazebo GUI client (default: OFF — laptop-friendly).
#   --no-rviz             Skip RViz                (default: ON  — lightweight viz).
#   --headless            --no-gui + --no-rviz (no viz at all).
#   --no-gui              Explicit form of the default.
#   --timeout SEC         Wall-clock timeout. Default: 300.
#   --ready-wait SEC      Engine-ready wait. Default: 90.
#   --ground-truth        Pass ground_truth:=true (needed if inner is CheatCode*).
#
# Determinism: per [[spawn-determinism-from-yaml]], the engine spawn from a
# fixed YAML is bit-deterministic for static scene and ≤0.05 mm for cable on
# trial 1. Multi-trial runs accumulate arm drift — record one trial per
# invocation to stay deterministic.

set -euo pipefail

PORT="${1:-}"
if [[ -z "$PORT" || ( "$PORT" != "sfp" && "$PORT" != "sc" ) ]]; then
    sed -n '2,30p' "$0" >&2
    exit 2
fi
shift

INNER="WaveArm"
ENABLE_TELEOP=1
ENABLE_RECORD=1
DATASET_ROOT=""
TASK_PROMPT=""
GROUND_TRUTH=false
RUN_TIMEOUT=300
READY_WAIT=90
GAZEBO_GUI=false
LAUNCH_RVIZ=true

while [[ $# -gt 0 ]]; do
    case "$1" in
        --inner)         INNER="$2"; shift 2 ;;
        --no-teleop)     ENABLE_TELEOP=0; shift ;;
        --no-record)     ENABLE_RECORD=0; shift ;;
        --dataset-root)  DATASET_ROOT="$2"; shift 2 ;;
        --task)          TASK_PROMPT="$2"; shift 2 ;;
        --ground-truth)  GROUND_TRUTH=true; shift ;;
        --timeout)       RUN_TIMEOUT="$2"; shift 2 ;;
        --ready-wait)    READY_WAIT="$2"; shift 2 ;;
        --gui)           GAZEBO_GUI=true; shift ;;
        --no-gui)        GAZEBO_GUI=false; shift ;;
        --no-rviz)       LAUNCH_RVIZ=false; shift ;;
        --headless)      GAZEBO_GUI=false; LAUNCH_RVIZ=false; shift ;;
        *) echo "unknown arg: $1" >&2; exit 2 ;;
    esac
done

# Derive paths from the script's location.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC="$(cd "$SCRIPT_DIR/.." && pwd)"             # .../src/aic
WS="$(cd "$SRC/../.." && pwd)"                   # .../ws_aic
TS="$(date +%Y-%m-%d_%H%M%S)"

CONFIG="$SRC/aic_engine/config/single_trial_${PORT}.yaml"
if [[ ! -f "$CONFIG" ]]; then
    echo "ERROR: config not found: $CONFIG" >&2
    exit 1
fi

: "${DATASET_ROOT:=$WS/aic_data}"
mkdir -p "$DATASET_ROOT"

OUTPUT_DIR="$WS/aic_results/recording_${TS}_${PORT}"
mkdir -p "$OUTPUT_DIR"
T1_LOG="$OUTPUT_DIR/terminal1_eval.log"
T2_LOG="$OUTPUT_DIR/terminal2_policy.log"

if [[ -z "$TASK_PROMPT" ]]; then
    TASK_PROMPT="insert ${PORT} cable"
fi

# CheatCode-family policies need ground-truth.
if [[ "$INNER" == CheatCode* ]] && [[ "$GROUND_TRUTH" != "true" ]]; then
    echo "INFO: inner=$INNER requires ground-truth — auto-enabling --ground-truth"
    GROUND_TRUTH=true
fi

export DBX_CONTAINER_MANAGER=docker

echo "config:        $CONFIG"
echo "inner policy:  $INNER"
echo "teleop:        $([[ $ENABLE_TELEOP == 1 ]] && echo on || echo off)"
echo "record:        $([[ $ENABLE_RECORD == 1 ]] && echo "on  → $DATASET_ROOT" || echo off)"
echo "ground_truth:  $GROUND_TRUTH"
echo "task prompt:   '$TASK_PROMPT'"
echo "output:        $OUTPUT_DIR"
echo

# Preflight: aic_eval container running.
if ! docker inspect aic_eval --format '{{.State.Running}}' 2>/dev/null | grep -q '^true$'; then
    echo "ERROR: aic_eval container is not running. Start it first." >&2
    exit 1
fi

# Stale process cleanup (mirrors run_scoring_loop.sh's logic).
STALE_PATTERN='aic_example_policies|/aic_model/aic_model|ros2 run aic_model'

kill_policy_on_host() {
    if ! pgrep -f "$STALE_PATTERN" > /dev/null 2>&1; then return 0; fi
    pkill -TERM -f "$STALE_PATTERN" 2>/dev/null || true
    for _ in 1 2 3; do
        sleep 1
        pgrep -f "$STALE_PATTERN" > /dev/null 2>&1 || return 0
    done
    pkill -KILL -f "$STALE_PATTERN" 2>/dev/null || true
    sleep 1
}

CLEANUP_SCRIPT="$(mktemp /tmp/aic_cleanup_XXXXXX.sh)"
cat > "$CLEANUP_SCRIPT" <<'CLEANUP_EOF'
#!/bin/bash
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
trap "rm -f '$CLEANUP_SCRIPT' 2>/dev/null || true" EXIT

cleanup_container() {
    docker exec aic_eval "$CLEANUP_SCRIPT" </dev/null >/dev/null 2>&1 || true
    pkill -9 -f "aic_example_policies|/aic_model/aic_model|ros2 run aic_model|gz sim -g" 2>/dev/null || true
}

if pgrep -f "$STALE_PATTERN" > /dev/null 2>&1; then
    echo "Killing stale policy processes from a previous run..."
    kill_policy_on_host || true
fi
echo "Purging container-side stragglers..."
cleanup_container
sleep 2

START="$(date +%s)"

# Terminal 1 — engine + sim.
# Single-line bash -c with && chain matches the working pattern in
# run_scoring_loop.sh; multi-line forms get flattened by distrobox's
# wrapping and exec ends up parsed as another export arg.
(
    distrobox enter -r aic_eval -- \
        bash -c "export AIC_RESULTS_DIR='$OUTPUT_DIR' && exec /entrypoint.sh ground_truth:=$GROUND_TRUTH start_aic_engine:=true shutdown_on_aic_engine_exit:=true gazebo_gui:=$GAZEBO_GUI launch_rviz:=$LAUNCH_RVIZ aic_engine_config_file:='$CONFIG'"
) > "$T1_LOG" 2>&1 &
T1_PID=$!

# Wait for engine ready.
ready=0
secs=0
while (( secs < READY_WAIT )); do
    if grep -q "No node with name 'aic_model'" "$T1_LOG" 2>/dev/null; then
        ready=1
        break
    fi
    if ! kill -0 "$T1_PID" 2>/dev/null; then
        echo "ERROR: terminal 1 died before ready. Last lines:" >&2
        tail -n 20 "$T1_LOG" | sed 's/^/  /' >&2
        exit 1
    fi
    sleep 1
    secs=$((secs+1))
done
if (( ready == 0 )); then
    echo "WARN: engine not ready after ${READY_WAIT}s — proceeding anyway"
fi

# Terminal 2 — policy in pixi with teleop/record env vars.
(
    cd "$SRC"
    export INNER_POLICY="$INNER"
    export ENABLE_TELEOP="$ENABLE_TELEOP"
    if [[ "$ENABLE_RECORD" == "1" ]]; then
        export RECORD_DATASET_PATH="$DATASET_ROOT"
    fi
    export RECORD_TASK_PROMPT="$TASK_PROMPT"
    exec pixi run ros2 run aic_model aic_model --ros-args \
        -p use_sim_time:=true \
        -p policy:="aic_example_policies.ros.TeleopAssist"
) > "$T2_LOG" 2>&1 &
T2_PID=$!

echo "Recording in progress. Ctrl-C in this terminal to abort."
echo "(Once Gazebo + the policy node are up and 'TeleopAssist.insert_cable enter'"
echo " appears in the policy log, ESC inside the launch terminal will also stop"
echo " the policy via the keyboard listener.)"

# Wait for run to finish (T1 exit or scoring.yaml + grace).
POST_DONE_GRACE=15
secs=0
engine_done_at=0
while kill -0 "$T1_PID" 2>/dev/null && (( secs < RUN_TIMEOUT )); do
    if (( engine_done_at == 0 )); then
        if [[ -f "$OUTPUT_DIR/scoring.yaml" ]] && \
           grep -q "aic_engine.*process has finished cleanly" "$T1_LOG" 2>/dev/null; then
            engine_done_at=$secs
            echo "│  engine finished — giving launch ${POST_DONE_GRACE}s to tear down"
        fi
    elif (( secs - engine_done_at >= POST_DONE_GRACE )); then
        echo "│  forcing teardown"
        break
    fi
    sleep 2
    secs=$((secs+2))
done
if kill -0 "$T1_PID" 2>/dev/null; then
    if (( engine_done_at == 0 )); then
        echo "│  TIMEOUT (${RUN_TIMEOUT}s) — forcing shutdown"
    fi
    kill -INT "$T1_PID" 2>/dev/null || true; sleep 3
    kill -TERM "$T1_PID" 2>/dev/null || true; sleep 2
    kill -KILL "$T1_PID" 2>/dev/null || true
fi
wait "$T1_PID" 2>/dev/null || true

sleep 3
kill_policy_on_host || true
cleanup_container
wait "$T2_PID" 2>/dev/null || true

DURATION=$(( $(date +%s) - START ))
echo
echo "Episode done in ${DURATION}s"
echo "Logs:    $OUTPUT_DIR/"
if [[ "$ENABLE_RECORD" == "1" ]]; then
    echo "Dataset: under $DATASET_ROOT/"
    ls -dt "$DATASET_ROOT"/aic_recording_* 2>/dev/null | head -3 | sed 's/^/  /' || true
fi
if [[ -f "$OUTPUT_DIR/scoring.yaml" ]]; then
    echo "Scoring: $OUTPUT_DIR/scoring.yaml"
fi
