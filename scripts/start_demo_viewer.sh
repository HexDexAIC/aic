#!/usr/bin/env bash
# Launch the live port-detection demo viewer.
# Requires a running aic_eval_vi container with cameras streaming.
#
# Usage:
#   bash start_demo_viewer.sh [sfp|sc] [port]
#
# Then open http://localhost:8765/ (or your custom port) in a browser.

PORT_TYPE=${1:-sfp}
PORT=${2:-8765}

export PATH=$HOME/.pixi/bin:$PATH
export RMW_IMPLEMENTATION=rmw_zenoh_cpp
export ZENOH_ROUTER_CHECK_ATTEMPTS=-1
export ZENOH_CONFIG_OVERRIDE='mode="client";connect/endpoints=["tcp/localhost:7447"];transport/shared_memory/enabled=false;scouting/multicast/enabled=false;scouting/gossip/enabled=false'

cd $HOME/ws_aic/src/aic
echo "Demo viewer at http://localhost:$PORT/  (port_type=$PORT_TYPE)"
pixi run python /mnt/c/Users/Dell/aic/scripts/demo_viewer.py --port "$PORT" --type "$PORT_TYPE"
