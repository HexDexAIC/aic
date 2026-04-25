#!/usr/bin/env bash
# Run one deterministic AIC trial and (optionally) record it to a LeRobot dataset.
#
# Spawns the engine + sim with a single-trial config, runs the chosen policy,
# and runs `record_lerobot.py` in its own process to write the dataset.
# Recording is bookended by /insert_cable/_action/status — the recorder
# auto-starts on STATUS_EXECUTING (i.e. after the engine's arm-stabilization
# wait, the moment the policy actually begins) and saves on terminal status.
#
# Usage:
#   src/aic/scripts/record_episode.sh PORT [options]
#
# PORT is "sfp" or "sc"; selects single_trial_<PORT>.yaml.
#
# Options:
#   --policy NAME         Policy class name from aic_example_policies.ros.
#                         Default: CheatCodeMJ. CheatCode-family auto-enables
#                         --ground-truth.
#   --no-record           Skip the LeRobot recorder; just run the policy.
#   --dataset-root DIR    Where to write the dataset.
#                         Default: ~/ws_aic/aic_data.
#   --task PROMPT         Task prompt string written into each frame.
#                         Default: derived from PORT.
#   --vcodec CODEC        Video codec for the dataset (default: h264).
#                         Use libsvtav1 for smaller files (slower encode).
#   --no-videos           Store images as PNG-per-frame instead of MP4
#                         (skips encoding entirely; bigger disk).
#   --insertion-threshold M
#                         Plug-port distance threshold (m) below which
#                         CheatCodeMJ considers an insertion successful.
#                         Default: 0.005.
#   --max-retries N       Max retry attempts for CheatCodeMJ if the first
#                         descent doesn't seat. Default: 1.
#   --bad-port-offset-x M Inject a deliberate XY offset into CheatCodeMJ's
#                         port_tf so the descent aims off-target. Useful
#                         for testing the retry loop. Default: 0.
#   --bad-port-offset-y M  Same for Y. Default: 0.
#   --stuck-min-fraction F Don't check stuck-detection until past this
#                         fraction of the descent. Default: 0.3.
#   --stuck-window-s S    Look-back window (s) for stuck progress check.
#                         Default: 1.5.
#   --stuck-progress-m M  Min net distance reduction over the window to
#                         keep going. Default: 0.002.
#   --gui                 Launch Gazebo GUI client (default OFF — laptop-friendly).
#   --no-rviz             Skip RViz (default ON — lightweight viz).
#   --headless            --no-gui + --no-rviz (no viz at all).
#   --no-gui              Explicit form of the default.
#   --timeout SEC         Wall-clock timeout. Default: 300.
#   --ready-wait SEC      Engine-ready wait. Default: 90.
#   --save-grace SEC      Time after engine exits to let the recorder finish
#                         encoding/saving. Default: 90.
#   --ground-truth        Pass ground_truth:=true (auto-on for CheatCode*).
#
# Determinism: per [[spawn-determinism-from-yaml]], the engine spawn from a
# fixed YAML is bit-deterministic for static scene and ≤0.05 mm for cable on
# trial 1. One trial per invocation keeps the arm-state slate clean.

set -euo pipefail

PORT="${1:-}"
if [[ -z "$PORT" || ( "$PORT" != "sfp" && "$PORT" != "sc" ) ]]; then
    sed -n '2,33p' "$0" >&2
    exit 2
fi
shift

POLICY="CheatCodeMJ"
ENABLE_RECORD=1
DATASET_ROOT=""
TASK_PROMPT=""
VCODEC="h264"
USE_VIDEOS=1
INSERTION_THRESHOLD=""    # empty = use the policy's default
MAX_RETRIES=""            # empty = use the policy's default
BAD_PORT_OFFSET_X=""      # empty = no offset (default)
BAD_PORT_OFFSET_Y=""      # empty = no offset (default)
STUCK_MIN_FRACTION=""     # empty = use the policy's default (0.3)
STUCK_WINDOW_S=""         # empty = use the policy's default (1.5)
STUCK_PROGRESS_M=""       # empty = use the policy's default (0.002)
GROUND_TRUTH=false
RUN_TIMEOUT=300
READY_WAIT=90
SAVE_GRACE=90
GAZEBO_GUI=false
LAUNCH_RVIZ=true

while [[ $# -gt 0 ]]; do
    case "$1" in
        --policy)        POLICY="$2"; shift 2 ;;
        --no-record)     ENABLE_RECORD=0; shift ;;
        --dataset-root)  DATASET_ROOT="$2"; shift 2 ;;
        --task)          TASK_PROMPT="$2"; shift 2 ;;
        --vcodec)        VCODEC="$2"; shift 2 ;;
        --no-videos)     USE_VIDEOS=0; shift ;;
        --insertion-threshold) INSERTION_THRESHOLD="$2"; shift 2 ;;
        --max-retries)   MAX_RETRIES="$2"; shift 2 ;;
        --bad-port-offset-x) BAD_PORT_OFFSET_X="$2"; shift 2 ;;
        --bad-port-offset-y) BAD_PORT_OFFSET_Y="$2"; shift 2 ;;
        --stuck-min-fraction) STUCK_MIN_FRACTION="$2"; shift 2 ;;
        --stuck-window-s)    STUCK_WINDOW_S="$2"; shift 2 ;;
        --stuck-progress-m)  STUCK_PROGRESS_M="$2"; shift 2 ;;
        --ground-truth)  GROUND_TRUTH=true; shift ;;
        --timeout)       RUN_TIMEOUT="$2"; shift 2 ;;
        --ready-wait)    READY_WAIT="$2"; shift 2 ;;
        --save-grace)    SAVE_GRACE="$2"; shift 2 ;;
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
T3_LOG="$OUTPUT_DIR/terminal3_recorder.log"

if [[ -z "$TASK_PROMPT" ]]; then
    TASK_PROMPT="insert ${PORT} cable"
fi

# CheatCode-family policies need ground-truth.
if [[ "$POLICY" == CheatCode* ]] && [[ "$GROUND_TRUTH" != "true" ]]; then
    echo "INFO: policy=$POLICY requires ground-truth — auto-enabling --ground-truth"
    GROUND_TRUTH=true
fi

export DBX_CONTAINER_MANAGER=docker

echo "config:        $CONFIG"
echo "policy:        $POLICY"
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
STALE_PATTERN='aic_example_policies|/aic_model/aic_model|ros2 run aic_model|record_lerobot'

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
    echo "Killing stale processes from a previous run..."
    kill_policy_on_host || true
fi
echo "Purging container-side stragglers..."
cleanup_container
sleep 2

START="$(date +%s)"

# ── Terminal 1 — engine + sim ────────────────────────────────────────────
# Single-line bash -c with && chain matches run_scoring_loop.sh's pattern;
# multi-line forms get flattened by distrobox's wrapping.
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

# ── Terminal 2 — policy ───────────────────────────────────────────────────
# Build the ROS-args parameter list, including any policy-specific overrides
# (only emit -p flags when the user explicitly set them, so the policy's
# declared defaults are otherwise used).
POLICY_ARGS=( -p use_sim_time:=true -p policy:="aic_example_policies.ros.$POLICY" )
if [[ -n "$INSERTION_THRESHOLD" ]]; then
    POLICY_ARGS+=( -p insertion_threshold_m:=$INSERTION_THRESHOLD )
fi
if [[ -n "$MAX_RETRIES" ]]; then
    POLICY_ARGS+=( -p max_insertion_retries:=$MAX_RETRIES )
fi
if [[ -n "$BAD_PORT_OFFSET_X" ]]; then
    POLICY_ARGS+=( -p bad_port_offset_x:=$BAD_PORT_OFFSET_X )
fi
if [[ -n "$BAD_PORT_OFFSET_Y" ]]; then
    POLICY_ARGS+=( -p bad_port_offset_y:=$BAD_PORT_OFFSET_Y )
fi
if [[ -n "$STUCK_MIN_FRACTION" ]]; then
    POLICY_ARGS+=( -p stuck_min_fraction:=$STUCK_MIN_FRACTION )
fi
if [[ -n "$STUCK_WINDOW_S" ]]; then
    POLICY_ARGS+=( -p stuck_window_s:=$STUCK_WINDOW_S )
fi
if [[ -n "$STUCK_PROGRESS_M" ]]; then
    POLICY_ARGS+=( -p stuck_progress_m:=$STUCK_PROGRESS_M )
fi
(
    cd "$SRC"
    exec pixi run ros2 run aic_model aic_model --ros-args "${POLICY_ARGS[@]}"
) > "$T2_LOG" 2>&1 &
T2_PID=$!

# ── Terminal 3 — LeRobot recorder ─────────────────────────────────────────
T3_PID=""
if [[ "$ENABLE_RECORD" == "1" ]]; then
    RECORDER_ARGS=( --root "$DATASET_ROOT" --task "$TASK_PROMPT" --vcodec "$VCODEC" )
    if [[ "$USE_VIDEOS" == "0" ]]; then
        RECORDER_ARGS+=( --no-videos )
    fi
    (
        cd "$SRC"
        exec pixi run python "$SRC/scripts/record_lerobot.py" "${RECORDER_ARGS[@]}"
    ) > "$T3_LOG" 2>&1 &
    T3_PID=$!
    echo "recorder:      pid $T3_PID, log $T3_LOG"
fi

echo "Run in progress. Ctrl-C in this terminal to abort."

# ── Wait for engine to complete ───────────────────────────────────────────
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
        echo "│  forcing engine teardown"
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

# Stop the policy promptly — the trial is over either way.
sleep 3
kill_policy_on_host || true
wait "$T2_PID" 2>/dev/null || true

# ── Wait for the recorder to finish encoding/saving ──────────────────────
if [[ -n "$T3_PID" ]]; then
    if kill -0 "$T3_PID" 2>/dev/null; then
        echo "│  waiting up to ${SAVE_GRACE}s for recorder to encode + save..."
        secs=0
        while kill -0 "$T3_PID" 2>/dev/null && (( secs < SAVE_GRACE )); do
            sleep 2
            secs=$((secs+2))
        done
        if kill -0 "$T3_PID" 2>/dev/null; then
            echo "│  recorder still running after ${SAVE_GRACE}s — sending SIGINT"
            kill -INT "$T3_PID" 2>/dev/null || true; sleep 5
            kill -KILL "$T3_PID" 2>/dev/null || true
        fi
    fi
    wait "$T3_PID" 2>/dev/null || true
fi

cleanup_container

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
