#!/usr/bin/env bash
# Run VisionInsert_v2 against a fresh eval container with ground_truth=false.
# Same shape as run_vision_insert.sh but uses the v2 perception
# (PnP-IPPE-sol-0 + SE3Tracker, port_pose_v2 module).
#
# Usage:
#   ./run_vision_insert_v2.sh
#
# Results land in $HOME/aic_results_visioninsert_v2/
set -e
export PATH=$HOME/.pixi/bin:$PATH
SUDO_PW=${SUDO_PW:-keerti}
IMAGE=${IMAGE:-aic_eval:fixed18_patched}
CONTAINER="aic_eval_vi_v2"
RESULTS="$HOME/aic_results_visioninsert_v2"
WEIGHTS=${AIC_V1_WEIGHTS:-$HOME/aic_runs/v1_h100_results/best.pt}

log() { echo -e "\e[1;34m[VisionInsert_v2]\e[0m $*"; }

log "Step 1: cleanup any prior container/results"
docker rm -f $CONTAINER 2>/dev/null || true
mkdir -p "$RESULTS"

log "Step 2: sudo mount rshared"
echo "$SUDO_PW" | sudo -S mount --make-rshared / 2>/dev/null || true

log "Step 3: verify weights exist at $WEIGHTS"
if [[ ! -f "$WEIGHTS" ]]; then
    log "ERROR: weights file not found at $WEIGHTS"
    exit 1
fi

log "Step 4: start container (ground_truth=false, start_aic_engine=true)"
docker run -d --rm --name $CONTAINER --gpus all \
    -e DISPLAY='' -e QT_QPA_PLATFORM=offscreen \
    --network bridge -p 7447:7447 \
    --privileged \
    "$IMAGE" \
    'gazebo_gui:=false' 'launch_rviz:=false' \
    'start_aic_engine:=true' 'ground_truth:=false'

log "Step 5: wait for engine ready"
until docker logs $CONTAINER 2>&1 | grep -q "No node with name 'aic_model' found"; do
    sleep 5
done
log "Engine ready."

log "Step 6: launch VisionInsert_v2 policy"
export RMW_IMPLEMENTATION=rmw_zenoh_cpp
export ZENOH_ROUTER_CHECK_ATTEMPTS=-1
export ZENOH_CONFIG_OVERRIDE='mode="client";connect/endpoints=["tcp/localhost:7447"];transport/shared_memory/enabled=false;scouting/multicast/enabled=false;scouting/gossip/enabled=false'
export AIC_V1_WEIGHTS="$WEIGHTS"

cd $HOME/ws_aic/src/aic
pixi run env AIC_V1_WEIGHTS="$WEIGHTS" \
    ros2 run aic_model aic_model \
    --ros-args -p use_sim_time:=true \
    -p policy:=aic_example_policies.ros.VisionInsert_v2 \
    > "$RESULTS/policy.log" 2>&1 &
POLICY_PID=$!

log "Policy PID=$POLICY_PID. Waiting for trials to finish..."
TIMEOUT=$((30 * 60))   # 30 minutes max
START=$(date +%s)
while true; do
    if ! docker ps --filter "name=$CONTAINER" --format "{{.Names}}" | grep -q $CONTAINER; then
        log "Container died."
        break
    fi
    if docker logs $CONTAINER 2>&1 | grep -qE "All trials complete|Eval Summary|Final Score"; then
        log "Trials complete."
        break
    fi
    if (( $(date +%s) - START > TIMEOUT )); then
        log "TIMEOUT after $TIMEOUT seconds."
        break
    fi
    sleep 30
done

log "Step 7: collect results"
docker logs $CONTAINER > "$RESULTS/container.log" 2>&1 || true
docker exec $CONTAINER cat /tmp/scoring.yaml > "$RESULTS/scoring.yaml" 2>/dev/null || \
    log "scoring.yaml not found in /tmp; try /aic_logs"

log "Step 8: stop policy and container"
kill $POLICY_PID 2>/dev/null || true
docker stop $CONTAINER 2>/dev/null || true

log "Done. Results in $RESULTS"
log "  - policy.log     ($(wc -l < $RESULTS/policy.log 2>/dev/null) lines)"
log "  - container.log  ($(wc -l < $RESULTS/container.log 2>/dev/null) lines)"
log "  - scoring.yaml   ($(stat -c%s $RESULTS/scoring.yaml 2>/dev/null || echo 'missing'))"
