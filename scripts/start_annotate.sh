#!/usr/bin/env bash
# Persistent annotation server launcher (avoids pkill-self-targeting issues).
exec >> $HOME/annotate_server.log 2>&1
export PATH=$HOME/.pixi/bin:/usr/local/bin:/usr/bin:/bin
cd $HOME/ws_aic/src/aic
exec pixi run python /home/dell/aic_scripts/annotate_server.py --port 8000
