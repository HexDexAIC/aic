#!/usr/bin/env bash
# Persistent launcher for the LeRobot audit viewer.
exec >> $HOME/audit_server.log 2>&1
export PATH=$HOME/.pixi/bin:/usr/local/bin:/usr/bin:/bin
cd $HOME/ws_aic/src/aic
exec pixi run python /home/dell/aic_scripts/audit_lerobot_server.py --port 8001
