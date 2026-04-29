#!/usr/bin/env bash
# Autopilot — survives independent of any parent shell.
# Waits for Trial 3 of CheatCode to finish in aic_eval_running, then runs
# the full pipeline (auto_label, train YOLO, run VisionInsert).
#
# Logs to $HOME/aic_autopilot.log. Persists across shell session ends
# when invoked with nohup.

set +e
exec >> $HOME/aic_autopilot.log 2>&1
export PATH=$HOME/.pixi/bin:/usr/local/bin:/usr/bin:/bin
export SUDO_PW=keerti

log() { echo "[$(date '+%H:%M:%S')] $*"; }

log "Autopilot starting (PID $$)"
log "Waiting for trial 3 completion in aic_eval_running..."

# Phase 1: wait for trial 3 done
attempts=0
while true; do
    if ! docker ps --filter name=aic_eval_running --format "{{.Names}}" 2>/dev/null | grep -q aic_eval_running; then
        log "Container aic_eval_running not running. Continuing pipeline anyway."
        break
    fi
    if docker logs aic_eval_running 2>&1 | grep -q "Trial .trial_3. completed successfully"; then
        log "Trial 3 completed."
        break
    fi
    if docker logs aic_eval_running 2>&1 | grep -qE "Trial .trial_3. failed|Trial .trial_3. timed out"; then
        log "Trial 3 failed/timed out, continuing anyway."
        break
    fi
    attempts=$((attempts+1))
    if [ $((attempts % 20)) -eq 0 ]; then
        log "Heartbeat: still waiting (attempt $attempts)"
    fi
    sleep 60
done

# Phase 2: run pipeline
log "Running full pipeline..."
bash $HOME/aic_scripts/run_full_pipeline.sh
rc=$?
log "Pipeline finished with exit code $rc"

log "Autopilot done."
