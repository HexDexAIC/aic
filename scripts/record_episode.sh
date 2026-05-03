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
#   --config PATH         Override the engine YAML path. Default:
#                         single_trial_<PORT>.yaml in the engine config dir.
#                         Used by the spawn-sweep tool to run templated
#                         per-episode configs.
#   --policy NAME         Policy class name from aic_example_policies.ros.
#                         Default: CheatCodeMJ. CheatCode-family auto-enables
#                         --ground-truth.
#   --no-record           Skip the LeRobot recorder; just run the policy.
#   --output-dir DIR      Override the auto-stamped run dir. Default:
#                         aic_results/recording_<TS>_<PORT>/. Used by
#                         spawn_sweep_sfp.py to land each seed under
#                         <sweep_dir>/seeds/seed_NN/.
#   --policy-config PATH  Path to a ROS2 params YAML for the policy. Default:
#                         aic_example_policies/config/<policy>.yaml (lowercased).
#                         CLI -p flags below still override individual values.
#   --dataset-root DIR    Where to write the dataset. By default the dataset
#                         lands under the run dir as <run>/dataset/ (single
#                         tree per trial). Passing this overrides that and
#                         restores the legacy <root>/aic_recording_<TS>/
#                         layout.
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
#                         descent doesn't seat. Default: 2.
#   --bad-port-offset-x M Inject a deliberate XY offset into CheatCodeMJ's
#                         port_tf so the descent aims off-target. Default 0.
#                         Used to test the retry path.
#   --bad-port-offset-y M  Same for Y. Default: 0.
#   --bad-offset-decay D  Multiplier applied to the bad offset on each
#                         retry. Default: 1.0 (no decay — all attempts see
#                         the same offset). Set <1 (e.g. 0.5) to exercise
#                         "retry recovers" path. Only meaningful when
#                         --bad-port-offset-{x,y} is non-zero.
#   --stuck-min-fraction F Don't run stuck-detection until past this
#                         fraction of the descent. Default: 0.2. Required
#                         to avoid false-triggering on the natural min-jerk
#                         ramp-up.
#   --stuck-window-s S    Look-back window (s) for stuck progress check.
#                         Default: 1.5.
#   --stuck-progress-m M  Min net distance reduction over the window to
#                         keep going. Default: 0.002.
#   --lift-time-frac F    How long the lift-to-hover takes, as a fraction
#                         of descent_time. The dominant wait between a
#                         failed attempt and the next descent. Default: 0.5.
#   --hover-hold-s S      Steady-state hold at hover between attempts.
#                         Default: 0.5.
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

CONFIG_OVERRIDE=""
OUTPUT_DIR_OVERRIDE=""
POLICY_CONFIG_OVERRIDE=""
POLICY="CheatCodeMJ"
ENABLE_RECORD=1
DATASET_ROOT=""
DATASET_ROOT_USER_SET=0
DATASET_NAME_OVERRIDE=""
TASK_PROMPT=""
VCODEC="h264"
USE_VIDEOS=1
INSERTION_THRESHOLD=""    # empty = use the policy's default
MAX_RETRIES=""            # empty = use the policy's default
BAD_PORT_OFFSET_X=""      # empty = use the policy's default (0)
BAD_PORT_OFFSET_Y=""      # empty = use the policy's default (0)
BAD_OFFSET_DECAY=""       # empty = use the policy's default (1.0)
STUCK_MIN_FRACTION=""     # empty = use the policy's default (0.2)
STUCK_WINDOW_S=""         # empty = use the policy's default (1.0)
STUCK_PROGRESS_M=""       # empty = use the policy's default (0.002)
LIFT_TIME_FRAC=""         # empty = use the policy's default (0.5)
HOVER_HOLD_S=""           # empty = use the policy's default (0.5)
GROUND_TRUTH=false
RUN_TIMEOUT=300
READY_WAIT=90
SAVE_GRACE=90
GAZEBO_GUI=false
LAUNCH_RVIZ=true

while [[ $# -gt 0 ]]; do
    case "$1" in
        --config)        CONFIG_OVERRIDE="$2"; shift 2 ;;
        --output-dir)    OUTPUT_DIR_OVERRIDE="$2"; shift 2 ;;
        --policy-config) POLICY_CONFIG_OVERRIDE="$2"; shift 2 ;;
        --policy)        POLICY="$2"; shift 2 ;;
        --no-record)     ENABLE_RECORD=0; shift ;;
        --dataset-root)  DATASET_ROOT="$2"; DATASET_ROOT_USER_SET=1; shift 2 ;;
        --dataset-name)  DATASET_NAME_OVERRIDE="$2"; shift 2 ;;
        --task)          TASK_PROMPT="$2"; shift 2 ;;
        --vcodec)        VCODEC="$2"; shift 2 ;;
        --no-videos)     USE_VIDEOS=0; shift ;;
        --insertion-threshold) INSERTION_THRESHOLD="$2"; shift 2 ;;
        --max-retries)   MAX_RETRIES="$2"; shift 2 ;;
        --bad-port-offset-x) BAD_PORT_OFFSET_X="$2"; shift 2 ;;
        --bad-port-offset-y) BAD_PORT_OFFSET_Y="$2"; shift 2 ;;
        --bad-offset-decay)  BAD_OFFSET_DECAY="$2"; shift 2 ;;
        --stuck-min-fraction) STUCK_MIN_FRACTION="$2"; shift 2 ;;
        --stuck-window-s)    STUCK_WINDOW_S="$2"; shift 2 ;;
        --stuck-progress-m)  STUCK_PROGRESS_M="$2"; shift 2 ;;
        --lift-time-frac)    LIFT_TIME_FRAC="$2"; shift 2 ;;
        --hover-hold-s)      HOVER_HOLD_S="$2"; shift 2 ;;
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

if [[ -n "$CONFIG_OVERRIDE" ]]; then
    CONFIG="$CONFIG_OVERRIDE"
else
    CONFIG="$SRC/aic_engine/config/single_trial_${PORT}.yaml"
fi
if [[ ! -f "$CONFIG" ]]; then
    echo "ERROR: config not found: $CONFIG" >&2
    exit 1
fi

if [[ -n "$OUTPUT_DIR_OVERRIDE" ]]; then
    OUTPUT_DIR="$OUTPUT_DIR_OVERRIDE"
else
    OUTPUT_DIR="$WS/aic_results/recording_${TS}_${PORT}"
fi
mkdir -p "$OUTPUT_DIR"

# Single-tree layout by default: dataset goes under the run dir as "dataset/".
# spawn_sweep_sfp.py passes --dataset-root <sweep> --dataset-name dataset
# so all seeds share <sweep>/dataset/ and the recorder appends episodes
# via LeRobotDataset.resume — ready to push to HF without a consolidate step.
if [[ -n "$DATASET_NAME_OVERRIDE" ]]; then
    DATASET_NAME="$DATASET_NAME_OVERRIDE"
elif [[ "$DATASET_ROOT_USER_SET" == "1" ]]; then
    DATASET_NAME=""
else
    DATASET_ROOT="$OUTPUT_DIR"
    DATASET_NAME="dataset"
fi
mkdir -p "$DATASET_ROOT"
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

# Stale process cleanup. Two patterns:
#   STALE_PATTERN_STARTUP — wider; used at the start of a run to kill
#     any leftover policy/recorder from a previous invocation.
#   STALE_PATTERN_POLICY  — used post-trial to kill the policy only.
#     The recorder must NOT be in this list — it needs the full
#     SAVE_GRACE window to finish encoding + saving the episode. The
#     recorder exits on its own when STATUS_SUCCEEDED/ABORTED fires.
STALE_PATTERN_STARTUP='aic_example_policies|/aic_model/aic_model|ros2 run aic_model|record_lerobot'
STALE_PATTERN_POLICY='aic_example_policies|/aic_model/aic_model|ros2 run aic_model'

kill_policy_on_host() {
    if ! pgrep -f "$STALE_PATTERN_POLICY" > /dev/null 2>&1; then return 0; fi
    pkill -TERM -f "$STALE_PATTERN_POLICY" 2>/dev/null || true
    for _ in 1 2 3; do
        sleep 1
        pgrep -f "$STALE_PATTERN_POLICY" > /dev/null 2>&1 || return 0
    done
    pkill -KILL -f "$STALE_PATTERN_POLICY" 2>/dev/null || true
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

# INT/TERM handler. docker exec / distrobox enter don't forward SIGINT to
# grandchildren reliably, so on Ctrl-C we have to kill T1_PID (engine
# launcher) and T2_PID (policy launcher) explicitly, then sweep the
# container for stragglers. Without this the engine + sim keep running
# after the user pressed Ctrl-C.
on_interrupt() {
    echo
    echo "Interrupted. Cleaning up..."
    if [[ -n "${T2_PID:-}" ]] && kill -0 "$T2_PID" 2>/dev/null; then
        kill -INT  "$T2_PID" 2>/dev/null || true; sleep 1
        kill -KILL "$T2_PID" 2>/dev/null || true
    fi
    if [[ -n "${T1_PID:-}" ]] && kill -0 "$T1_PID" 2>/dev/null; then
        kill -INT  "$T1_PID" 2>/dev/null || true; sleep 1
        kill -TERM "$T1_PID" 2>/dev/null || true; sleep 1
        kill -KILL "$T1_PID" 2>/dev/null || true
    fi
    cleanup_container
    exit 130
}
trap on_interrupt INT TERM

if pgrep -f "$STALE_PATTERN_STARTUP" > /dev/null 2>&1; then
    echo "Killing stale processes from a previous run..."
    pkill -TERM -f "$STALE_PATTERN_STARTUP" 2>/dev/null || true
    sleep 2
    pkill -KILL -f "$STALE_PATTERN_STARTUP" 2>/dev/null || true
    sleep 1
fi
echo "Purging container-side stragglers..."
cleanup_container
sleep 2

START="$(date +%s)"

# ── Terminal 1 — engine + sim ────────────────────────────────────────────
# Single-line bash -c with && chain matches run_scoring_loop.sh's pattern;
# multi-line forms get flattened by distrobox's wrapping.
#
# AIC_USE_DOCKER_EXEC=1 swaps `distrobox enter -r` for `docker exec`. The
# distrobox path needs sudo (rootful), which fails from non-TTY shells; the
# docker-exec path works headlessly. Same container either way.
ENGINE_CMD="export AIC_RESULTS_DIR='$OUTPUT_DIR' && exec /entrypoint.sh ground_truth:=$GROUND_TRUTH start_aic_engine:=true shutdown_on_aic_engine_exit:=true gazebo_gui:=$GAZEBO_GUI launch_rviz:=$LAUNCH_RVIZ aic_engine_config_file:='$CONFIG'"
(
    if [[ "${AIC_USE_DOCKER_EXEC:-0}" == "1" ]]; then
        exec docker exec -i aic_eval bash -c "$ENGINE_CMD"
    else
        exec distrobox enter -r aic_eval -- bash -c "$ENGINE_CMD"
    fi
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

# ── Terminal 3 — LeRobot recorder (launched BEFORE policy) ───────────────
# We launch the recorder first and wait for its WRITER_READY sentinel
# before starting the policy. The LeRobotDataset.create call inside the
# recorder blocks the executor for ~2-3 s wall on first observation; if
# that block lands during the policy's slerp it eats the slerp frames.
# Pre-creating the writer in the engine's idle window avoids that.
T3_PID=""
if [[ "$ENABLE_RECORD" == "1" ]]; then
    RECORDER_ARGS=( --root "$DATASET_ROOT" --task "$TASK_PROMPT" --vcodec "$VCODEC" )
    if [[ "$USE_VIDEOS" == "0" ]]; then
        RECORDER_ARGS+=( --no-videos )
    fi
    if [[ -n "$DATASET_NAME" ]]; then
        RECORDER_ARGS+=( --name "$DATASET_NAME" )
    fi
    (
        cd "$SRC"
        exec pixi run python "$SRC/scripts/record_lerobot.py" "${RECORDER_ARGS[@]}"
    ) > "$T3_LOG" 2>&1 &
    T3_PID=$!
    echo "recorder:      pid $T3_PID, log $T3_LOG"

    # Wait for WRITER_READY before launching the policy.
    WRITER_WAIT=30
    secs=0
    while (( secs < WRITER_WAIT )); do
        if grep -q "^WRITER_READY$" "$T3_LOG" 2>/dev/null; then
            echo "│  recorder writer ready after ${secs}s"
            break
        fi
        if ! kill -0 "$T3_PID" 2>/dev/null; then
            echo "ERROR: recorder died before WRITER_READY. Last lines:" >&2
            tail -n 20 "$T3_LOG" | sed 's/^/  /' >&2
            exit 1
        fi
        sleep 1
        secs=$((secs+1))
    done
    if ! grep -q "^WRITER_READY$" "$T3_LOG" 2>/dev/null; then
        echo "WARN: recorder did not emit WRITER_READY after ${WRITER_WAIT}s — proceeding anyway"
    fi
fi

# ── Terminal 2 — policy ───────────────────────────────────────────────────
# Build the ROS-args parameter list, including any policy-specific overrides
# (only emit -p flags when the user explicitly set them, so the policy's
# declared defaults are otherwise used).
# Bundled per-policy YAML defaults. ROS2 reads --params-file FIRST, then
# applies any -p name:=value overrides on top. So a user can drop in a
# tweaked YAML via --policy-config or override individual knobs via the
# existing --insertion-threshold / etc. flags.
POLICY_CONFIG="$POLICY_CONFIG_OVERRIDE"
if [[ -z "$POLICY_CONFIG" ]]; then
    candidate="$SRC/aic_example_policies/config/${POLICY,,}.yaml"
    [[ -f "$candidate" ]] && POLICY_CONFIG="$candidate"
fi
POLICY_ARGS=( -p use_sim_time:=true -p policy:="aic_example_policies.ros.$POLICY" )
if [[ -n "$POLICY_CONFIG" ]]; then
    echo "policy params:  $POLICY_CONFIG"
    POLICY_ARGS=( --params-file "$POLICY_CONFIG" "${POLICY_ARGS[@]}" )
fi
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
if [[ -n "$BAD_OFFSET_DECAY" ]]; then
    POLICY_ARGS+=( -p bad_offset_decay_per_retry:=$BAD_OFFSET_DECAY )
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
if [[ -n "$LIFT_TIME_FRAC" ]]; then
    POLICY_ARGS+=( -p lift_time_frac:=$LIFT_TIME_FRAC )
fi
if [[ -n "$HOVER_HOLD_S" ]]; then
    POLICY_ARGS+=( -p hover_hold_s:=$HOVER_HOLD_S )
fi
(
    cd "$SRC"
    export AIC_RESULTS_DIR="$OUTPUT_DIR"
    exec pixi run ros2 run aic_model aic_model --ros-args "${POLICY_ARGS[@]}"
) > "$T2_LOG" 2>&1 &
T2_PID=$!

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
echo "Run dir: $OUTPUT_DIR/"
if [[ "$ENABLE_RECORD" == "1" ]]; then
    if [[ -n "$DATASET_NAME" ]]; then
        echo "Dataset: $DATASET_ROOT/$DATASET_NAME/"
    else
        echo "Dataset: under $DATASET_ROOT/"
        ls -dt "$DATASET_ROOT"/aic_recording_* 2>/dev/null | head -3 | sed 's/^/  /' || true
    fi
fi
if [[ -f "$OUTPUT_DIR/scoring.yaml" ]]; then
    echo "Scoring: $OUTPUT_DIR/scoring.yaml"
fi
