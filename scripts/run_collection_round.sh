#!/usr/bin/env bash
# One round of ground-truth data collection.
#
# Starts a fresh eval container (with ground_truth:=true) plus the
# LoggingCheatCode policy. Waits for all 3 trials to complete or
# fail. Stops the container, saves artifacts.
#
# Usage:
#   ./run_collection_round.sh [<TAG>]
#
# Designed to be called repeatedly to accumulate data across runs.

set -e
TAG=${1:-$(date +%Y%m%d_%H%M%S)}
SUDO_PW=${SUDO_PW:-keerti}
IMAGE=${IMAGE:-aic_eval:fixed18_patched}
CONTAINER_NAME="aic_eval_round_$TAG"
RESULTS_DIR="$HOME/aic_results_$TAG"

log() { echo -e "\e[1;34m[round $TAG]\e[0m $*"; }

log "Starting round at $(date)"
mkdir -p "$RESULTS_DIR"

log "Mount rshared (sudo)..."
echo "$SUDO_PW" | sudo -S mount --make-rshared / 2>/dev/null || true

log "Removing any stale container..."
docker rm -f "$CONTAINER_NAME" 2>/dev/null || true

log "Starting eval container ($IMAGE)..."
docker run -d --rm \
    --name "$CONTAINER_NAME" \
    --gpus all \
    -e DISPLAY='' -e QT_QPA_PLATFORM=offscreen \
    --network bridge -p 7447:7447 \
    --privileged \
    "$IMAGE" \
    'gazebo_gui:=false' 'launch_rviz:=false' \
    'start_aic_engine:=true' 'ground_truth:=true'

log "Waiting for engine to be ready..."
until docker logs "$CONTAINER_NAME" 2>&1 | grep -q "No node with name 'aic_model' found"; do
    sleep 5
done
log "Engine ready. Starting policy."

# Run policy in background.
export RMW_IMPLEMENTATION=rmw_zenoh_cpp
export ZENOH_ROUTER_CHECK_ATTEMPTS=-1
export ZENOH_CONFIG_OVERRIDE='mode="client";connect/endpoints=["tcp/localhost:7447"];transport/shared_memory/enabled=false;scouting/multicast/enabled=false;scouting/gossip/enabled=false'

cd "$HOME/ws_aic/src/aic"
pixi run ros2 run aic_model aic_model \
    --ros-args -p use_sim_time:=true \
    -p policy:=aic_example_policies.ros.LoggingCheatCode &
POLICY_PID=$!

log "Policy PID=$POLICY_PID. Waiting for all trials to complete..."

until docker logs "$CONTAINER_NAME" 2>&1 | grep -qE "TrialState: ?Complete|All trials complete|Eval Summary" || \
      ! kill -0 $POLICY_PID 2>/dev/null; do
    if ! docker ps --filter "name=$CONTAINER_NAME" --format "{{.Names}}" | grep -q "$CONTAINER_NAME"; then
        log "Container died unexpectedly."
        break
    fi
    sleep 60
done

log "Trials done or policy exited. Cleaning up."
kill $POLICY_PID 2>/dev/null || true
sleep 2

log "Extracting scoring..."
docker cp "$CONTAINER_NAME":/root/aic_results "$RESULTS_DIR/" 2>&1 || true
docker stop "$CONTAINER_NAME" 2>/dev/null || true

log "Round $TAG complete. Logs in ~/aic_logs/, scoring in $RESULTS_DIR/"
