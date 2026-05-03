#!/usr/bin/env bash
# Generic policy launcher. Usage: bash start_policy.sh <PolicyName>
# Waits for aic_eval_vi engine, then launches the given policy.

POLICY=${1:-aic_example_policies.ros.WaveArm}
LOG=${LOG:-$HOME/aic_policy_run.log}

exec >> "$LOG" 2>&1
export PATH=$HOME/.pixi/bin:/usr/local/bin:/usr/bin:/bin

log() { echo "[$(date '+%H:%M:%S')] $*"; }
log "Starting policy: $POLICY"

while true; do
    if ! docker ps --filter name=aic_eval_vi --format '{{.Names}}' | grep -q aic_eval_vi; then
        log "container not running"; exit 1
    fi
    if docker logs aic_eval_vi 2>&1 | grep -q "No node with name 'aic_model' found"; then
        log "Engine ready."; break
    fi
    sleep 10
done

export RMW_IMPLEMENTATION=rmw_zenoh_cpp
export ZENOH_ROUTER_CHECK_ATTEMPTS=-1
export ZENOH_CONFIG_OVERRIDE='mode="client";connect/endpoints=["tcp/localhost:7447"];transport/shared_memory/enabled=false;scouting/multicast/enabled=false;scouting/gossip/enabled=false'

cd $HOME/ws_aic/src/aic
log "Launching $POLICY..."
pixi run ros2 run aic_model aic_model --ros-args -p use_sim_time:=true -p "policy:=$POLICY"
log "Policy exited with code $?"
