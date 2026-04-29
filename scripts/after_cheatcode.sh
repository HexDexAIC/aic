#!/usr/bin/env bash
# Run after CheatCode 3-trial run completes.
#  1. Extracts scoring.yaml from container
#  2. Re-runs auto_label on full data
#  3. Updates SC tcp_to_plug offset from trial 3 data
#  4. Trains YOLOv8n-pose on full dataset (50 epochs)
#  5. Stops the eval container
#
# After this, run scripts/run_vision_insert.sh to test VisionInsert.

set -e
export PATH=$HOME/.pixi/bin:$PATH

CONTAINER=${1:-aic_eval_running}
DATASET=${DATASET:-$HOME/aic_dataset_full}
RUN_NAME=${RUN_NAME:-yolopose_full}
EPOCHS=${EPOCHS:-50}

log() { echo -e "\e[1;34m[after]\e[0m $*"; }

log "Step 1: Extract scoring.yaml from container..."
mkdir -p $HOME/aic_results_cheatcode
docker cp $CONTAINER:/root/aic_results $HOME/aic_results_cheatcode/_cp 2>&1 || \
    log "WARN: docker cp failed; container may already be stopped."
mv $HOME/aic_results_cheatcode/_cp/aic_results/* $HOME/aic_results_cheatcode/ 2>/dev/null || true
rm -rf $HOME/aic_results_cheatcode/_cp
ls $HOME/aic_results_cheatcode/
if [ -f "$HOME/aic_results_cheatcode/scoring.yaml" ]; then
    log "Cheatcode scoring:"
    cat $HOME/aic_results_cheatcode/scoring.yaml
fi

log "Step 2: Auto-label full dataset..."
cd $HOME/ws_aic/src/aic
pixi run python /mnt/c/Users/Dell/aic/scripts/auto_label.py --out "$DATASET" 2>&1 | tail -5

log "Step 3: Update tcp_to_plug offsets from full data..."
pixi run python /mnt/c/Users/Dell/aic/scripts/analyze_logs.py 2>&1 | head -40 > $HOME/_offset_full.txt
cat $HOME/_offset_full.txt
log "(SC offset to update by hand if trial 3 ran)"

log "Step 4: Train YOLOv8n-pose ($EPOCHS epochs)..."
pixi run python /mnt/c/Users/Dell/aic/scripts/train_yolopose.py \
    --data "$DATASET/dataset.yaml" --epochs $EPOCHS --imgsz 640 --batch 16 \
    --name "$RUN_NAME" 2>&1 | tail -10

log "Step 5: Stop eval container..."
docker stop $CONTAINER 2>/dev/null || true

log "Done. Best ONNX at: $HOME/aic_runs/$RUN_NAME/weights/best.onnx"
log "Next: kick off VisionInsert run via scripts/run_vision_insert.sh"
