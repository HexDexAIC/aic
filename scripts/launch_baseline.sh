#!/usr/bin/env bash
# Launch the strict-clean ACT baseline training job on Velda.
#
# This script is the actual command that gets passed to `vbatch -P
# anycloud-a100-1`. Putting the lerobot-train invocation in a script
# file (rather than inline with `vbatch -- lerobot-train ...`) avoids
# shell-parsing issues with line-continuation backslashes — see gotcha
# §6 in wiki/methodology/training-act-baseline.md.
#
# Pre-conditions (do once on the Velda CPU container):
#   - venv at ~/aic-venv with lerobot installed (per training-act-baseline runbook)
#   - HF auth done with HexDexAIC-scoped fine-grained token
#   - ~/clean_eps.txt populated by `build_clean_eps.py` (488 strict-clean episodes)
#
# Launch:
#   JOB=$(vbatch -P anycloud-a100-1 -- bash ~/launch_baseline.sh)
#   echo "Job ID: $JOB"
#   velda task watch $JOB
#
# Customize via env vars:
#   REPO        — HF dataset repo id (default HexDexAIC/aic-sfp-500)
#   PUSH_REPO   — HF repo to push final checkpoint to (default HexDexAIC/act-aic-sfp-500-v1)
#   STEPS       — total training steps (default 80000)
#   BATCH       — batch size (default 8)
#   OUT         — output dir (default ./outputs/act-baseline-v1)
#   VENV        — venv path (default ~/aic-venv)

set -euo pipefail

REPO="${REPO:-HexDexAIC/aic-sfp-500}"
PUSH_REPO="${PUSH_REPO:-HexDexAIC/act-aic-sfp-500-v1}"
STEPS="${STEPS:-80000}"
BATCH="${BATCH:-8}"
OUT="${OUT:-./outputs/act-baseline-v1}"
VENV="${VENV:-$HOME/aic-venv}"
EPS_FILE="${EPS_FILE:-$HOME/clean_eps.txt}"

if [[ ! -f "$EPS_FILE" ]]; then
    echo "FATAL: $EPS_FILE not found — run build_clean_eps.py first" >&2
    exit 1
fi
EPS=$(cat "$EPS_FILE")

if [[ ! -x "$VENV/bin/lerobot-train" ]]; then
    echo "FATAL: $VENV/bin/lerobot-train not found — install lerobot in the venv" >&2
    exit 1
fi

echo "[$(date -u +%H:%M:%SZ)] launching ACT baseline:"
echo "  dataset      : $REPO"
echo "  push_to_hub  : $PUSH_REPO"
echo "  episodes     : $(tr ',' '\n' <<<"$EPS" | wc -l) (from $EPS_FILE)"
echo "  steps        : $STEPS"
echo "  batch_size   : $BATCH"
echo "  output_dir   : $OUT"

exec "$VENV/bin/lerobot-train" \
    --dataset.repo_id="$REPO" \
    --dataset.episodes="[$EPS]" \
    --policy.type=act \
    --policy.chunk_size=100 \
    --batch_size="$BATCH" \
    --steps="$STEPS" \
    --num_workers=4 \
    --output_dir="$OUT" \
    --policy.push_to_hub=true \
    --policy.repo_id="$PUSH_REPO" \
    --policy.private=true \
    --wandb.enable=false
