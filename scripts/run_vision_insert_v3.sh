#!/usr/bin/env bash
# Run VisionInsert_v3 against a fresh eval container with ground_truth=false.
# Uses the port_pose_v2 perception (PnP-IPPE-sol-0 + SE3Tracker) and the v3
# motion loops (coarsened approach + descent + 5s settle).
#
# Usage:
#   ./run_vision_insert_v3.sh
#
# Required env (set or override at invocation):
#   AIC_WORKSPACE     Path to the workspace containing this repo, defaults to
#                     ~/ws_aic/src/aic (so AIC_WORKSPACE/aic_example_policies/...).
#                     The script `cd`s here so `pixi run ros2 run aic_model ...`
#                     resolves the policy module.
#   AIC_V1_WEIGHTS    Path to the trained YOLO best.pt. Default ~/aic_runs/
#                     v1_h100_results/best.pt. Download from
#                     https://huggingface.co/HexDexAIC/aic-port-yolo-v1
#   IMAGE             Eval Docker image tag. Default aic_eval:fixed18 (the
#                     public AIC eval image). Override via `IMAGE=mytag ./run_...`
#                     if you have a locally-patched build.
#   SUDO_PW           Sudo password for `mount --make-rshared /` step.
#                     Required only if you haven't already run that mount on
#                     this host. Override via `SUDO_PW=xxx ./run_...`.
#
# Results land in $HOME/aic_results_visioninsert_v3/.
set -e
export PATH=$HOME/.pixi/bin:$PATH
SUDO_PW=${SUDO_PW:-}
IMAGE=${IMAGE:-aic_eval:fixed18}
CONTAINER="aic_eval_vi_v3"
RESULTS="$HOME/aic_results_visioninsert_v3"
WEIGHTS=${AIC_V1_WEIGHTS:-$HOME/aic_runs/v1_h100_results/best.pt}
WORKSPACE=${AIC_WORKSPACE:-$HOME/ws_aic/src/aic}

log() { echo -e "\e[1;34m[VisionInsert_v3]\e[0m $*"; }

log "Step 1: cleanup any prior container/results + stale host-side aic_model procs"
docker rm -f $CONTAINER 2>/dev/null || true
# Kill stale aic_model procs from prior runs — single SIGTERM on the script's
# wrapper PID does not kill the python children, and they will register as a
# duplicate aic_model in the next container's Zenoh router and trip
# "Lifecycle node 'aic_model' is not in 'unconfigured' state. Current state:
# finalized" → all trials score 0 with "Model validation failed".
pkill -KILL -f "aic_example_policies|/aic_model/aic_model|ros2 run aic_model" 2>/dev/null || true
mkdir -p "$RESULTS"

log "Step 2: sudo mount rshared (skipped if SUDO_PW unset; assumes already done)"
if [[ -n "$SUDO_PW" ]]; then
    echo "$SUDO_PW" | sudo -S mount --make-rshared / 2>/dev/null || true
fi

log "Step 3: verify weights exist at $WEIGHTS"
if [[ ! -f "$WEIGHTS" ]]; then
    log "ERROR: weights file not found at $WEIGHTS"
    log "Download from https://huggingface.co/HexDexAIC/aic-port-yolo-v1"
    log "  pip install huggingface_hub"
    log "  python -c 'from huggingface_hub import hf_hub_download; print(hf_hub_download(\"HexDexAIC/aic-port-yolo-v1\", \"best.pt\"))'"
    log "Then export AIC_V1_WEIGHTS=<that path> and re-run."
    exit 1
fi

log "Step 4: verify workspace at $WORKSPACE"
if [[ ! -d "$WORKSPACE" ]]; then
    log "ERROR: workspace not found at $WORKSPACE"
    log "Set AIC_WORKSPACE to the directory containing aic_example_policies/."
    exit 1
fi

log "Step 5: start container (image=$IMAGE, ground_truth=false, start_aic_engine=true)"
docker run -d --rm --name $CONTAINER --gpus all \
    -e DISPLAY='' -e QT_QPA_PLATFORM=offscreen \
    --network bridge -p 7447:7447 \
    --privileged \
    "$IMAGE" \
    'gazebo_gui:=false' 'launch_rviz:=false' \
    'start_aic_engine:=true' 'ground_truth:=false'

log "Step 6: wait for engine ready"
until docker logs $CONTAINER 2>&1 | grep -q "No node with name 'aic_model' found"; do
    sleep 5
done
log "Engine ready."

log "Step 7: launch VisionInsert_v3 policy"
export RMW_IMPLEMENTATION=rmw_zenoh_cpp
export ZENOH_ROUTER_CHECK_ATTEMPTS=-1
export ZENOH_CONFIG_OVERRIDE='mode="client";connect/endpoints=["tcp/localhost:7447"];transport/shared_memory/enabled=false;scouting/multicast/enabled=false;scouting/gossip/enabled=false'
export AIC_V1_WEIGHTS="$WEIGHTS"

cd "$WORKSPACE"
pixi run env AIC_V1_WEIGHTS="$WEIGHTS" \
    ros2 run aic_model aic_model \
    --ros-args -p use_sim_time:=true \
    -p policy:=aic_example_policies.ros.VisionInsert_v3 \
    > "$RESULTS/policy.log" 2>&1 &
POLICY_PID=$!

log "Policy PID=$POLICY_PID. Waiting for trials to finish..."
# NOTE: the engine doesn't reliably emit any of the strings below — instead
# it prints "process has finished cleanly [pid 51]" for the aic_engine after
# the YAML scoring summary block. That's the canonical end signal. The other
# strings are kept for forward-compat in case the engine adds them later.
TIMEOUT=$((180 * 60))   # 180 minutes max
START=$(date +%s)
while true; do
    if ! docker ps --filter "name=$CONTAINER" --format "{{.Names}}" | grep -q $CONTAINER; then
        log "Container exited."
        break
    fi
    if docker logs $CONTAINER 2>&1 | grep -qE "aic_engine.*process has finished cleanly|All trials complete|Eval Summary|Final Score"; then
        log "Trials complete."
        break
    fi
    if (( $(date +%s) - START > TIMEOUT )); then
        log "TIMEOUT after $TIMEOUT seconds."
        break
    fi
    sleep 30
done

log "Step 8: collect results"
docker logs $CONTAINER > "$RESULTS/container.log" 2>&1 || true
# scoring.yaml lives at /root/aic_results/scoring.yaml inside the container
# (NOT /tmp/ as some older docs claim).
docker cp $CONTAINER:/root/aic_results/scoring.yaml "$RESULTS/scoring.yaml" 2>/dev/null || \
    log "scoring.yaml not found in /root/aic_results/"

log "Step 9: stop policy and container"
kill $POLICY_PID 2>/dev/null || true
sleep 2
pkill -KILL -f "aic_example_policies|/aic_model/aic_model|ros2 run aic_model" 2>/dev/null || true
docker stop $CONTAINER 2>/dev/null || true

log "Done. Results in $RESULTS"
log "  - policy.log     ($(wc -l < $RESULTS/policy.log 2>/dev/null || echo '?') lines)"
log "  - container.log  ($(wc -l < $RESULTS/container.log 2>/dev/null || echo '?') lines)"
log "  - scoring.yaml   ($(stat -c%s $RESULTS/scoring.yaml 2>/dev/null || echo 'missing') bytes)"
