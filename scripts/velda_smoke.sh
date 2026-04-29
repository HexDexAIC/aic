#!/usr/bin/env bash
# Velda smoke test — verify a GPU instance can train ACT on aic-sfp-500.
#
# Run this ONCE on a freshly-launched Velda instance to catch infra issues
# before committing to an 8-hour training run. Cost: ~$0.50–$1 on a small
# GPU (L4/A10/T4); ~$2 on A100. Picks up-to-date HF token from `hf auth login`
# (run that interactively first if needed).
#
# Usage (from inside the Velda VM):
#     curl -fsSL https://raw.githubusercontent.com/HexDexAIC/aic/main/scripts/velda_smoke.sh | bash
# Or copy this file in and run:
#     bash velda_smoke.sh
#
# Each step is a gate. If anything FAILs, stop and debug — don't waste
# training-cluster time on a half-working env.

set -euo pipefail

GREEN='\033[0;32m'; RED='\033[0;31m'; YELLOW='\033[0;33m'; NC='\033[0m'
ok()   { echo -e "${GREEN}[OK]${NC}   $*"; }
warn() { echo -e "${YELLOW}[WARN]${NC} $*"; }
fail() { echo -e "${RED}[FAIL]${NC} $*"; exit 1; }
hdr()  { echo; echo "════ $* ════"; }

REPO="${REPO:-HexDexAIC/aic-sfp-500}"
WORK="${WORK:-$HOME/aic-smoke}"
mkdir -p "$WORK"
cd "$WORK"

hdr "1/6  GPU + CUDA"
command -v nvidia-smi >/dev/null || fail "nvidia-smi not found — no GPU driver?"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || fail "nvidia-smi failed"
ok "GPU visible"

hdr "2/6  Python + pip"
python3 --version || fail "no python3"
python3 -c "import sys; assert sys.version_info >= (3, 10), 'need py >= 3.10'" || fail "python too old"
ok "python OK"

hdr "3/6  Minimal install (lerobot[act] + huggingface_hub)"
pip install --quiet --upgrade pip
pip install --quiet "huggingface_hub>=0.25" || fail "hf install failed"
pip install --quiet "lerobot[act]" || fail "lerobot install failed (try: pip install lerobot or check version)"
python3 -c "
import torch, lerobot
print(' torch:   ', torch.__version__)
print(' cuda:    ', torch.cuda.is_available(), torch.version.cuda)
print(' devices: ', torch.cuda.device_count())
print(' lerobot: ', lerobot.__version__)
" || fail "import smoke failed"
torch_cuda=$(python3 -c "import torch; print(torch.cuda.is_available())")
[ "$torch_cuda" = "True" ] || fail "torch.cuda.is_available() == False — wrong torch wheel?"
ok "torch+cuda+lerobot OK"

hdr "4/6  HF auth (HexDexAIC scope)"
HF_USER=$(hf auth whoami 2>/dev/null | grep -oP 'user=\K\S+' || echo "")
if [ -z "$HF_USER" ]; then
    warn "Not logged in to HF. Run 'hf auth login' interactively, paste your token, then re-run this script."
    fail "no HF auth"
fi
echo " logged in as: $HF_USER"
python3 -c "
from huggingface_hub import HfApi
api = HfApi()
me = api.whoami()
fine = me.get('auth', {}).get('accessToken', {}).get('fineGrained', {})
scopes = {s['entity']['name']: s.get('permissions', []) for s in fine.get('scoped', [])}
if 'HexDexAIC' not in scopes:
    raise SystemExit('FAIL: token has no HexDexAIC scope. Edit at https://huggingface.co/settings/tokens')
if 'repo.content.read' not in scopes['HexDexAIC']:
    raise SystemExit('FAIL: HexDexAIC scope lacks repo.content.read')
print(' HexDexAIC scope:', scopes['HexDexAIC'])
" || fail "HF token not scoped to HexDexAIC"
ok "HF token scoped correctly"

hdr "5/6  Dataset pull (1 episode of $REPO)"
python3 - <<PY || fail "dataset load failed"
from lerobot.datasets.lerobot_dataset import LeRobotDataset
ds = LeRobotDataset(repo_id="$REPO", episodes=[0])
sample = ds[0]
print(' frames in episode 0:', len(ds))
print(' state shape:        ', sample['observation.state'].shape)
print(' action shape:       ', sample['action'].shape)
print(' has num_attempts:   ', 'num_attempts' in sample)
print(' has episode_success:', 'episode_success' in sample)
assert sample['observation.state'].shape[0] == 27, 'state should be 27-D'
assert sample['action'].shape[0] == 9, 'action should be 9-D'
PY
ok "dataset load + schema check OK"

hdr "6/6  10-step lerobot-train smoke (3 episodes, batch=2)"
mkdir -p outputs
# Build clean-eps list (just first 3 for the smoke, no need for all 488)
EPS="0,1,2"
echo " using episodes: $EPS"
echo " kicking off 10-step train run..."
lerobot-train \
    --dataset.repo_id="$REPO" \
    --dataset.episodes="[$EPS]" \
    --policy.type=act \
    --batch_size=2 \
    --steps=10 \
    --num_workers=2 \
    --output_dir="./outputs/smoke" \
    --policy.push_to_hub=false \
    --wandb.enable=false 2>&1 | tee outputs/smoke.log | tail -30 || fail "lerobot-train failed (check outputs/smoke.log)"

# Sanity: did it actually do steps?
grep -qE "step.*10|loss" outputs/smoke.log || warn "no 'step 10' or 'loss' line in log — check outputs/smoke.log manually"
ok "lerobot-train ran 10 steps"

hdr "DONE"
echo
echo " All 6 gates passed. Safe to launch the full 80k-step train run."
echo
echo " Next: see scripts/train_baseline.sh (or paste the lerobot-train command from"
echo " the wiki's adaptive-retry-policy.md page)."
echo
echo " Reminder: STOP THIS INSTANCE from the Velda dashboard when done."
echo " The smoke ran ~\$0.10-\$0.50 of compute; continued idle = wasted credit."
