#!/usr/bin/env bash
# Master orchestration: run everything from "CheatCode just finished" to
# "VisionInsert scored, results extracted".
#
# Should be invoked AFTER trial 3 completes in the running aic_eval_running.

set -e
export PATH=$HOME/.pixi/bin:$PATH
SUDO_PW=${SUDO_PW:-keerti}

log() { echo -e "\e[1;35m[full]\e[0m $*"; }

log "Step 1: Extract CheatCode scoring..."
mkdir -p $HOME/aic_results_cheatcode
docker cp aic_eval_running:/root/aic_results $HOME/aic_results_cheatcode/_cp 2>&1 || \
    log "WARN: docker cp failed"
mv $HOME/aic_results_cheatcode/_cp/aic_results/* $HOME/aic_results_cheatcode/ 2>/dev/null || true
rm -rf $HOME/aic_results_cheatcode/_cp || true
ls $HOME/aic_results_cheatcode/ || true
if [ -f "$HOME/aic_results_cheatcode/scoring.yaml" ]; then
    log "CheatCode scoring.yaml:"
    cat $HOME/aic_results_cheatcode/scoring.yaml | head -50
fi
docker logs aic_eval_running 2>&1 | grep -E "Trial .* completed|total score|Score:" \
    | sed -E "s/\x1b\[[0-9;]*m//g" > $HOME/aic_results_cheatcode/log_scores.txt
cat $HOME/aic_results_cheatcode/log_scores.txt

log "Step 2: Stop CheatCode container..."
docker stop aic_eval_running 2>/dev/null || true
sleep 5

log "Step 3: Auto-label full dataset..."
cd $HOME/ws_aic/src/aic
pixi run python $HOME/aic_scripts/auto_label.py --out $HOME/aic_dataset_full 2>&1 | tail -5

log "Step 4: Compute final tcp_to_plug offsets per port type..."
pixi run python $HOME/aic_scripts/analyze_logs.py 2>&1 | head -60

log "Step 5: Train YOLOv8n-pose, 50 epochs (~30-45 min)..."
pixi run python $HOME/aic_scripts/train_yolopose.py \
    --data $HOME/aic_dataset_full/dataset.yaml \
    --epochs 50 --imgsz 640 --batch 16 --name yolopose_full \
    2>&1 | tail -15

log "Step 6: Verify ONNX export..."
ls -la $HOME/aic_runs/yolopose_full/weights/
test -f $HOME/aic_runs/yolopose_full/weights/best.onnx || {
    log "ERROR: ONNX export missing"
    exit 1
}

log "Step 7: Sanity-check trained model on saved frames..."
pixi run python $HOME/aic_scripts/test_yolo_onnx.py \
    $HOME/aic_runs/yolopose_full/weights/best.onnx 2>&1 | head -30

log "Step 8: Run VisionInsert (ground_truth=false)..."
bash $HOME/aic_scripts/run_vision_insert.sh

log "Step 9: Extract VisionInsert scoring..."
docker logs aic_eval_vi 2>&1 | grep -E "Trial .* completed|total score|Score:" \
    | sed -E "s/\x1b\[[0-9;]*m//g" > $HOME/aic_results_visioninsert/log_scores.txt 2>/dev/null || true
cat $HOME/aic_results_visioninsert/log_scores.txt 2>/dev/null

log "DONE. Compare:"
echo "CheatCode (ground_truth=true):"
cat $HOME/aic_results_cheatcode/log_scores.txt 2>/dev/null
echo
echo "VisionInsert (ground_truth=false, what counts for submission):"
cat $HOME/aic_results_visioninsert/log_scores.txt 2>/dev/null
