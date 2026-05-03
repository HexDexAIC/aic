#!/usr/bin/env bash
# Runner used by Claude to execute commands inside the WSL2 pixi env.
# Usage: cat _run_in_wsl.sh | wsl bash -s [SHELL_COMMAND...]
# All args after `-s` are joined and run as a bash command line.
set -e
export PATH="$HOME/.pixi/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
cd /home/dell/ws_aic/src/aic
bash -c "$*"
