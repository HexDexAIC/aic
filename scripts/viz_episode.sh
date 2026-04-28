#!/usr/bin/env bash
# Launch the LeRobot rerun.io viewer on one recorded episode.
#
# Usage:
#   src/aic/scripts/viz_episode.sh <sweep_dir> <seed>     # spawn local viewer
#   src/aic/scripts/viz_episode.sh <sweep_dir> <seed> --save     # save .rrd to /tmp
#   src/aic/scripts/viz_episode.sh <dataset_dir> --direct        # point at one dataset dir
#
# A "dataset_dir" is the aic_recording_<TS>/ directory directly
# (the one containing data/, videos/, meta/).
set -euo pipefail

EPISODE_INDEX=0
if [[ "${2:-}" == "--direct" ]]; then
    DS="$1"
    SAVE=0
    [[ "${3:-}" == "--save" ]] && SAVE=1
elif [[ $# -lt 2 ]]; then
    sed -n '2,12p' "$0" >&2
    exit 2
else
    SWEEP="$1"
    SEED="$2"
    EPISODE_INDEX="$SEED"     # one episode per seed in the shared dataset
    SAVE=0
    [[ "${3:-}" == "--save" ]] && SAVE=1
    if [[ -d "$SWEEP/dataset" ]]; then
        # New shared-dataset layout — one multi-episode dataset per sweep.
        DS="$SWEEP/dataset"
    elif [[ -d "$SWEEP/seeds/seed_$(printf '%02d' "$SEED")/dataset" ]]; then
        # Legacy per-seed dataset layout.
        DS="$SWEEP/seeds/seed_$(printf '%02d' "$SEED")/dataset"
        EPISODE_INDEX=0
    else
        echo "ERROR: no dataset under $SWEEP/dataset or $SWEEP/seeds/seed_*/dataset/" >&2
        exit 1
    fi
fi

# Resolve to absolute path BEFORE we cd into src/aic for pixi —
# relative paths break after the cd.
DS=$(readlink -f "$DS")

REPO_ID="local/$(basename "$DS")"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC="$(cd "$SCRIPT_DIR/.." && pwd)"

echo "dataset: $DS"
echo "repo_id: $REPO_ID"
echo

cd "$SRC"
# --num-workers 0 keeps the DataLoader single-process. The default
# multi-worker loader exhausts /dev/shm and crashes mid-stream with
# "DataLoader worker killed by signal: Bus error".
if (( SAVE )); then
    OUT=/tmp/$(basename "$DS")_ep${EPISODE_INDEX}.rrd
    pixi run lerobot-dataset-viz \
        --repo-id "$REPO_ID" \
        --root "$DS" \
        --episode-index "$EPISODE_INDEX" \
        --num-workers 0 \
        --save 1 \
        --output-dir /tmp
    echo
    echo "Saved: $OUT (open later with: rerun $OUT)"
else
    pixi run lerobot-dataset-viz \
        --repo-id "$REPO_ID" \
        --root "$DS" \
        --episode-index "$EPISODE_INDEX" \
        --num-workers 0
fi
