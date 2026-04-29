#!/usr/bin/env bash
# Helper: run py-spy as root for a given PID
PID=$1
PYSPY=$HOME/.pixi/envs/default/bin/py-spy
echo keerti | sudo -S "$PYSPY" dump --pid "$PID"
