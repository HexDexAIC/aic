# Autonomous AIC Run — Status Snapshot

This file summarizes the current state of the autonomous run so you can pick
up where it left off when you check back. Updated by Claude during the run.

## What's running right now

A long-running autopilot (`bawhjf5ro` background task) that is:

1. Waiting for the running `aic_eval_running` container's Trial 3 to complete
2. As soon as Trial 3 finishes, automatically running `~/aic_scripts/run_full_pipeline.sh` which:
   - Extracts CheatCode scoring artifacts to `~/aic_results_cheatcode/`
   - Stops the CheatCode container
   - Runs `auto_label.py` on the full ground-truth log dump
   - Trains YOLOv8n-pose for 50 epochs (model at `~/aic_runs/yolopose_full/`)
   - Tests the trained ONNX on saved frames
   - Starts a **fresh** eval container with `ground_truth:=false`
   - Runs `VisionInsert` policy against it
   - Extracts the VisionInsert scoring artifacts to `~/aic_results_visioninsert/`

3. Logs everything to `~/aic_pipeline.log`

## Trial scores so far

- **Trial 1 (SFP, nic_card_mount_0): 68.379149** (CheatCode + ground_truth — full insertion with some Tier 2 penalty)
- **Trial 2 (SFP, nic_card_mount_1): in progress** (z_offset descending)
- **Trial 3 (SC, sc_port_1): not yet started**

Watch live: `cat ~/aic_pipeline.log`

## Files generated so far

- `~/aic_logs/<run_ts>/trial_01_sfp/` — Trial 1 frames + JSON metadata + ground-truth TFs
- `~/aic_logs/<run_ts>/trial_02_sfp/` — Trial 2 frames (in progress)
- `~/aic_logs/tcp_to_plug_offset.json` — calibrated SFP offset (SC will be added when Trial 3 data lands)
- `~/aic_dataset/`, `~/aic_dataset_v2/` — early intermediate datasets (smoke test)
- `~/aic_runs/yolo_smoke/` — smoke YOLO model (validates pipeline)
- `~/aic_scripts/` — full toolchain copied to WSL filesystem

## Code under `aic_example_policies/aic_example_policies/ros/`

- `LoggingCheatCode.py` — CheatCode + per-frame logger (the policy currently running)
- `port_detector.py` — classical SFP + SC detector (HSV/contour for SFP, HoughCircles for SC)
- `port_detector_yolo.py` — ONNX-based YOLOv8-pose detector wrapper
- `port_pose.py` — known-size, stereo, and PnP pose lifters
- `VisionInsert.py` — submission-targeted policy (uses vision instead of ground-truth TF)

## Disk

- WSL `/`: 904 GB free (plenty)
- Windows C:: 14 GB free (tight; only used for small scripts)

## What I've validated offline so far

- Classical detector on Trial 1 frames: 100% detection, 15.8 mm median PnP error
- Classical detector on Trial 2 frames: 60% detection, 31.7 mm median error (worse — confirms YOLO is needed)
- YOLO smoke train (5 epochs, 333 frames): pipeline works, ONNX export works, but model is overfit on smoke data
- Full YOLO train will run on ~1800 labels covering all 3 trial scenes

## Expected completion

Roughly 4 hours from now (00:44 +4h ≈ 04:44):
- ~30 min more for Trial 2
- ~30 min for Trial 3
- ~30-45 min for YOLO training
- ~30 min for VisionInsert (3 trials, sim ~80% real-time)

## If you want to bail out and inspect manually

```bash
# Stop the autopilot
# (the bg task will be killed when this Claude session ends, but you can also)
docker stop aic_eval_running aic_eval_vi 2>/dev/null
pkill -f "aic_model"
```

## Final scoring will be in

- `~/aic_results_cheatcode/scoring.yaml` — upper bound (with ground truth)
- `~/aic_results_visioninsert/scoring.yaml` — submittable result (vision-only)
