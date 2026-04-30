#!/usr/bin/env bash
# Launch ACT-PR training on Velda.
#
# Pre-conditions on Velda:
#   1. ~/aic-venv with lerobot 0.5.1 installed (per training-act-baseline runbook)
#   2. HF token has write access to HexDexAIC/act-pr-aic-sfp-500-v1
#   3. ~/ws_aic/src/aic/scripts/{train_actpr.py,act_pr_policy.py} present
#      (clone the hariharan/runact-local branch onto Velda first if needed)
#   4. ~/clean_eps.txt populated by build_clean_eps.py (488 strict-clean
#      episode ids) — only needed if you want strict-clean filter; default
#      omits the filter since the lenient filter setting hasn't been
#      pre-decided.
#
# Launch:
#   JOB=$(vbatch -P anycloud-a100-1 -- bash ~/launch_actpr.sh)
#   echo $JOB && velda task watch $JOB
#
# Single-line lerobot CLI is avoided here in favour of running our custom
# train_actpr.py directly, since lerobot-train doesn't know about ACT-PR.
#
# Customize via env vars:
#   STEPS                  default 80000
#   BATCH                  default 8
#   PORT_LAMBDA            default 1.0
#   OUT                    default ./outputs/act-pr-v1
#   PUSH_REPO              default HexDexAIC/act-pr-aic-sfp-500-v1
#   USE_CLEAN_FILTER       1 to filter to strict-clean (set EPS_FILE)
#   EPS_FILE               clean_eps.txt path

set -euo pipefail

STEPS="${STEPS:-80000}"
BATCH="${BATCH:-8}"
PORT_LAMBDA="${PORT_LAMBDA:-1.0}"
OUT="${OUT:-./outputs/act-pr-v1}"
PUSH_REPO="${PUSH_REPO:-HexDexAIC/act-pr-aic-sfp-500-v1}"
USE_CLEAN_FILTER="${USE_CLEAN_FILTER:-0}"
EPS_FILE="${EPS_FILE:-$HOME/clean_eps.txt}"
VENV="${VENV:-$HOME/aic-venv}"
SCRIPTS_DIR="${SCRIPTS_DIR:-$HOME/ws_aic/src/aic/scripts}"

if [[ ! -x "$VENV/bin/python" ]]; then
    echo "FATAL: $VENV/bin/python not found" >&2
    exit 1
fi
if [[ ! -f "$SCRIPTS_DIR/train_actpr.py" ]]; then
    echo "FATAL: $SCRIPTS_DIR/train_actpr.py not found — clone the branch on Velda" >&2
    exit 1
fi

ARGS=(
    --output_dir "$OUT"
    --steps "$STEPS"
    --batch_size "$BATCH"
    --num_workers 2
    --port_pose_loss_weight "$PORT_LAMBDA"
    --push_repo "$PUSH_REPO"
)

if [[ "$USE_CLEAN_FILTER" == "1" ]]; then
    if [[ ! -f "$EPS_FILE" ]]; then
        echo "FATAL: USE_CLEAN_FILTER=1 but $EPS_FILE not found" >&2
        exit 1
    fi
    ARGS+=( --episodes_file "$EPS_FILE" )
fi

echo "[$(date -u +%H:%M:%SZ)] launching ACT-PR train"
for a in "${ARGS[@]}"; do echo "  $a"; done

exec "$VENV/bin/python" "$SCRIPTS_DIR/train_actpr.py" "${ARGS[@]}"
