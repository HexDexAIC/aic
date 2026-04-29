#!/usr/bin/env bash
# Run VisionInsert against a fresh eval container with ground_truth=false.
# This simulates the actual submission scoring environment.
#
# Usage:
#   ./run_vision_insert.sh
#
# After completion, scoring.yaml is at $HOME/aic_results_visioninsert/

set -e
export PATH=$HOME/.pixi/bin:$PATH
SUDO_PW=${SUDO_PW:-keerti}
IMAGE=${IMAGE:-aic_eval:fixed18_patched}
CONTAINER="aic_eval_vi"
RESULTS="$HOME/aic_results_visioninsert"

log() { echo -e "\e[1;34m[VisionInsert]\e[0m $*"; }

log "Step 1: cleanup any prior container/results"
docker rm -f $CONTAINER 2>/dev/null || true
mkdir -p "$RESULTS"

log "Step 2: sudo mount rshared"
echo "$SUDO_PW" | sudo -S mount --make-rshared / 2>/dev/null || true

log "Step 3: start container (ground_truth=false, start_aic_engine=true)"
docker run -d --rm --name $CONTAINER --gpus all \
    -e DISPLAY='' -e QT_QPA_PLATFORM=offscreen \
    --network bridge -p 7447:7447 \
    --privileged \
    "$IMAGE" \
    'gazebo_gui:=false' 'launch_rviz:=false' \
    'start_aic_engine:=true' 'ground_truth:=false'

log "Step 4: wait for engine ready"
until docker logs $CONTAINER 2>&1 | grep -q "No node with name 'aic_model' found"; do
    sleep 5
done
log "Engine ready."

log "Step 5: launch VisionInsert policy"
export RMW_IMPLEMENTATION=rmw_zenoh_cpp
export ZENOH_ROUTER_CHECK_ATTEMPTS=-1
export ZENOH_CONFIG_OVERRIDE='mode="client";connect/endpoints=["tcp/localhost:7447"];transport/shared_memory/enabled=false;scouting/multicast/enabled=false;scouting/gossip/enabled=false'

cd $HOME/ws_aic/src/aic
pixi run ros2 run aic_model aic_model \
    --ros-args -p use_sim_time:=true \
    -p policy:=aic_example_policies.ros.VisionInsert &
POLICY_PID=$!

log "Policy PID=$POLICY_PID. Waiting for trials to finish..."
until docker logs $CONTAINER 2>&1 | grep -qE "All trials complete|Eval Summary|Final Score"; do
    if ! docker ps --filter "name=$CONTAINER" --format "{{.Names}}" | grep -q $CONTAINER; then
        log "Container died."
        break
    fi
    sleep 60
done

log "Stopping policy + container."
kill $POLICY_PID 2>/dev/null || true
sleep 3

log "Extracting scoring."
docker cp $CONTAINER:/root/aic_results "$RESULTS"_cp 2>&1 || true
mv "${RESULTS}_cp"/aic_results/* "$RESULTS"/ 2>/dev/null || true
rm -rf "${RESULTS}_cp"
docker stop $CONTAINER 2>/dev/null || true

log "VisionInsert run complete."
if [ -f "$RESULTS/scoring.yaml" ]; then
    log "Scoring:"
    cat "$RESULTS/scoring.yaml"
fi
