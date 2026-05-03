#!/usr/bin/env bash
# AIC perception/insertion preflight check.
# Run from WSL2 Ubuntu. Safe to re-run; does not start the eval or modify state.
#
# Usage:
#   bash /mnt/c/Users/Dell/aic/scripts/aic_preflight.sh
# Or copy to your WSL home and run:
#   bash ~/aic_preflight.sh

set -u

PASS="[ OK ]"
FAIL="[FAIL]"
WARN="[WARN]"
INFO="[INFO]"

results=()
fail_count=0
warn_count=0

ok()    { results+=("$PASS $1"); }
ko()    { results+=("$FAIL $1"); fail_count=$((fail_count+1)); }
warn()  { results+=("$WARN $1"); warn_count=$((warn_count+1)); }
info()  { results+=("$INFO $1"); }

section() { echo; echo "=== $1 ==="; }

# ------------------------------------------------------------
# 1. Environment
# ------------------------------------------------------------
section "Environment"
if grep -qi microsoft /proc/version 2>/dev/null; then
    ok "Running inside WSL2"
else
    warn "Not inside WSL — script designed for WSL2 Ubuntu, may still work on native Linux"
fi
echo "    OS: $(. /etc/os-release && echo "$PRETTY_NAME")"
echo "    Kernel: $(uname -r)"

# ------------------------------------------------------------
# 2. GPU
# ------------------------------------------------------------
section "GPU"
if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi >/dev/null 2>&1; then
    ok "nvidia-smi works"
    free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1 | tr -d ' ')
    total=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1 | tr -d ' ')
    name=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
    echo "    GPU: $name"
    echo "    VRAM: $free / $total MiB free"
    if [ "${free:-0}" -ge 8000 ]; then
        ok "VRAM >= 8000 MiB free"
    else
        warn "VRAM only ${free} MiB free — may need to stop other GPU users before training"
    fi
else
    ko "nvidia-smi not found or not functional. Install NVIDIA Container Toolkit and ensure WSL has GPU passthrough."
fi

# ------------------------------------------------------------
# 3. Docker
# ------------------------------------------------------------
section "Docker"
if command -v docker >/dev/null 2>&1; then
    if docker ps >/dev/null 2>&1; then
        ok "Docker daemon accessible"
        docker_version=$(docker --version)
        echo "    $docker_version"
    else
        ko "Docker installed but daemon not accessible. Try: sudo usermod -aG docker \$USER && newgrp docker"
    fi
else
    ko "Docker not installed."
fi

# ------------------------------------------------------------
# 4. AIC eval image / container
# ------------------------------------------------------------
section "AIC Eval Image / Container"
if command -v docker >/dev/null 2>&1 && docker ps >/dev/null 2>&1; then
    aic_images=$(docker images --format '{{.Repository}}:{{.Tag}}' 2>/dev/null | grep -i aic || true)
    if [ -n "$aic_images" ]; then
        ok "AIC docker image(s) present"
        echo "$aic_images" | sed 's/^/    /'
    else
        warn "No AIC docker image. Run: docker pull ghcr.io/intrinsic-dev/aic/aic_eval:latest"
    fi
fi

if command -v distrobox >/dev/null 2>&1; then
    ok "distrobox installed"
    if distrobox list 2>/dev/null | grep -qi aic_eval; then
        ok "aic_eval distrobox container exists"
        distrobox list 2>/dev/null | grep -i aic | sed 's/^/    /'
    else
        warn "aic_eval distrobox container not yet created."
        echo "    Create with: export DBX_CONTAINER_MANAGER=docker && \\"
        echo "                 distrobox create -r --nvidia -i ghcr.io/intrinsic-dev/aic/aic_eval:latest aic_eval"
    fi
else
    warn "distrobox not installed (sudo apt install distrobox). The README expects distrobox; if you've been running raw 'docker run' instead, that's fine but commands differ."
fi

# ------------------------------------------------------------
# 5. Pixi
# ------------------------------------------------------------
section "Pixi"
if command -v pixi >/dev/null 2>&1; then
    ok "pixi installed: $(pixi --version)"
else
    ko "pixi not installed. Install: curl -fsSL https://pixi.sh/install.sh | sh"
fi

# ------------------------------------------------------------
# 6. Locate AIC repo in WSL filesystem
# ------------------------------------------------------------
section "AIC Repository (WSL-side)"
candidate_paths=(
    "$HOME/ws_aic/src/aic"
    "$HOME/aic"
    "$HOME/code/aic"
    "$HOME/work/aic"
    "$HOME/projects/aic"
)
aic_path=""
for p in "${candidate_paths[@]}"; do
    if [ -f "$p/pixi.toml" ] && [ -d "$p/aic_example_policies" ]; then
        aic_path="$p"
        break
    fi
done

if [ -z "$aic_path" ]; then
    found=$(find "$HOME" -maxdepth 6 -name pixi.toml -path "*/aic/*" 2>/dev/null | head -3)
    if [ -n "$found" ]; then
        candidate=$(dirname "$(echo "$found" | head -1)")
        if [ -d "$candidate/aic_example_policies" ]; then
            aic_path="$candidate"
        fi
    fi
fi

if [ -n "$aic_path" ]; then
    ok "AIC repo found: $aic_path"
else
    ko "AIC repo not found in WSL filesystem."
    echo "    Clone it with:"
    echo "      mkdir -p ~/ws_aic/src && cd ~/ws_aic/src"
    echo "      git clone https://github.com/intrinsic-dev/aic"
    echo "    (the Windows-side clone at /mnt/c/Users/Dell/aic is NOT suitable for pixi/ROS — clone in WSL native fs.)"
fi

# ------------------------------------------------------------
# 7. Pixi env state (only if repo found)
# ------------------------------------------------------------
if [ -n "$aic_path" ]; then
    section "Pixi Environment ($aic_path)"
    if [ -d "$aic_path/.pixi" ]; then
        ok ".pixi/ exists (env installed)"
    else
        warn ".pixi/ missing — run: cd $aic_path && pixi install"
    fi

    section "Port Assets"
    for asset in \
        "$aic_path/aic_assets/models/SFP Module/sfp_module_visual.glb" \
        "$aic_path/aic_assets/models/SC Port/sc_port_visual.glb" \
        "$aic_path/aic_assets/models/NIC Card Mount/nic_card_mount_visual.glb" \
        "$aic_path/aic_assets/models/SC Plug/sc_plug_visual.glb"; do
        if [ -f "$asset" ]; then
            ok "$(basename "$asset")"
        else
            ko "missing: $asset"
        fi
    done

    section "Reference Policies"
    for p in CheatCode.py RunACT.py WaveArm.py; do
        f="$aic_path/aic_example_policies/aic_example_policies/ros/$p"
        if [ -f "$f" ]; then
            ok "$p"
        else
            ko "$p missing"
        fi
    done

    # Pixi runtime imports (single python invocation to amortize activation)
    if [ -d "$aic_path/.pixi" ] && command -v pixi >/dev/null 2>&1; then
        section "Pixi Runtime Imports (this may take 10-20s on first run)"
        cd "$aic_path"
        py_out=$(pixi run python -c "
import importlib, sys
required = ['rclpy', 'numpy', 'cv2', 'tf2_ros', 'geometry_msgs.msg']
optional_phase2 = ['ultralytics', 'onnxruntime', 'trimesh']
for m in required:
    try:
        importlib.import_module(m); print('REQ_OK', m)
    except Exception as e:
        print('REQ_MISS', m, type(e).__name__)
for m in optional_phase2:
    try:
        importlib.import_module(m); print('OPT_OK', m)
    except Exception as e:
        print('OPT_MISS', m)
" 2>&1)
        while IFS= read -r line; do
            case "$line" in
                "REQ_OK "*)   ok "  ${line#REQ_OK } (required)" ;;
                "REQ_MISS "*) ko "  ${line#REQ_MISS } (required, but missing or import error)" ;;
                "OPT_OK "*)   ok "  ${line#OPT_OK } (Phase 2)" ;;
                "OPT_MISS "*) info "  ${line#OPT_MISS } (Phase 2, install via 'pixi run pip install ...' when ready)" ;;
            esac
        done <<< "$py_out"

        # Try resolving the aic_model node — this verifies the colcon-built workspace is on PYTHONPATH
        section "ROS Package Visibility"
        if pixi run ros2 pkg list 2>/dev/null | grep -qx aic_model; then
            ok "aic_model package visible to ros2"
        else
            warn "aic_model not in 'ros2 pkg list' — may need to build first or pixi env is incomplete"
        fi
        if pixi run ros2 pkg list 2>/dev/null | grep -qx aic_example_policies; then
            ok "aic_example_policies package visible"
        else
            warn "aic_example_policies not in 'ros2 pkg list'"
        fi

        cd - >/dev/null
    fi
fi

# ------------------------------------------------------------
# 8. Results directory
# ------------------------------------------------------------
section "Results Directory"
results_dir="${AIC_RESULTS_DIR:-$HOME/aic_results}"
if [ -d "$results_dir" ]; then
    ok "Results dir exists: $results_dir"
    latest=$(ls -1t "$results_dir" 2>/dev/null | head -3)
    if [ -n "$latest" ]; then
        echo "    Recent contents:"
        echo "$latest" | sed 's/^/      /'
    fi
    if [ -f "$results_dir/scoring.yaml" ]; then
        info "Previous scoring.yaml found — useful as a sanity baseline"
    fi
else
    info "$results_dir doesn't exist yet (will be created on first eval run)"
fi

# ------------------------------------------------------------
# Summary
# ------------------------------------------------------------
echo
echo "==================================================="
echo "                  PREFLIGHT REPORT"
echo "==================================================="
for r in "${results[@]}"; do
    echo "$r"
done
echo "==================================================="
echo "Fail: $fail_count    Warn: $warn_count"
echo "==================================================="

if [ "$fail_count" -eq 0 ]; then
    echo
    echo "All hard checks passed."
    if [ -n "$aic_path" ]; then
        cat <<EOF

Suggested next commands:

  Terminal 2 (eval, inside distrobox):
    export DBX_CONTAINER_MANAGER=docker
    distrobox enter -r aic_eval
    /entrypoint.sh ground_truth:=true start_aic_engine:=true

  Terminal 3 (policy, in WSL host with pixi):
    cd $aic_path
    pixi run ros2 run aic_model aic_model --ros-args \\
      -p use_sim_time:=true \\
      -p policy:=aic_example_policies.ros.CheatCode

After the run finishes, check: cat \$HOME/aic_results/scoring.yaml
EOF
    fi
    exit 0
else
    echo
    echo "$fail_count hard check(s) failed. Address [FAIL] items above before proceeding."
    exit 1
fi
