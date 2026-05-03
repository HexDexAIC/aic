FROM docker.io/library/ros:kilted-ros-core AS build

# Install pixi
RUN apt update && apt install -y git && curl -fsSL https://pixi.sh/install.sh | bash
ENV PATH="/root/.pixi/bin:$PATH"

# Add base dependencies
COPY aic_example_policies /ws_aic/src/aic/aic_example_policies
COPY aic_model /ws_aic/src/aic/aic_model
COPY aic_interfaces /ws_aic/src/aic/aic_interfaces
COPY aic_utils /ws_aic/src/aic/aic_utils
COPY pixi.toml pixi.lock /ws_aic/src/aic/
COPY pixi_env_setup.sh /ws_aic/src/aic/

SHELL ["/bin/bash", "-c"]
RUN --mount=type=cache,target=/root/.cache/rattler/cache \
    --mount=type=cache,target=/ws_aic/src/aic/.pixi/build \
    cd /ws_aic/src/aic && pixi install --locked

# ── Bake the YOLO weights into the image ───────────────────────────────
# Eval network has no internet at run time, so the weights need to be
# present locally. The `yolo_weights` named build context is wired up
# in docker-compose.yaml's `additional_contexts`. Default points at
# Keerti's HF-cached snapshot; override with YOLO_WEIGHTS_DIR.
#
# We copy into the same path VisionInsert_v3 expects by default
# (~/aic_runs/v1_h100_results/best.pt), then chown to root since the
# container runs as root.
RUN mkdir -p /root/aic_runs/v1_h100_results
COPY --from=yolo_weights / /tmp/yolo_src/
RUN cp -L /tmp/yolo_src/best.pt /root/aic_runs/v1_h100_results/best.pt && \
    rm -rf /tmp/yolo_src && \
    ls -la /root/aic_runs/v1_h100_results/best.pt

# ── Pre-cache the ResNet18 backbone (in case ultralytics needs it) ─────
# YOLOv8 ships its own backbones, but cv2 / torch may try to download
# torchvision models on first import. Pre-warm the cache.
RUN cd /ws_aic/src/aic && pixi run --as-is python -c "\
import torchvision.models as M; \
M.resnet18(weights=M.ResNet18_Weights.IMAGENET1K_V1); \
import os; print('cached:', os.listdir(os.path.expanduser('~/.cache/torch/hub/checkpoints/')))"

# Pre-load the YOLO model once during build to make sure it imports
# cleanly and pull any missing ultralytics deps. Smoke check.
RUN cd /ws_aic/src/aic && pixi run --as-is python -c "\
from ultralytics import YOLO; \
m = YOLO('/root/aic_runs/v1_h100_results/best.pt'); \
print('YOLO loaded ok, names:', m.names)"

WORKDIR /ws_aic/src/aic
COPY --chmod=755 <<"EOF" /entrypoint.sh
#!/bin/bash
set -e

export RMW_IMPLEMENTATION=rmw_zenoh_cpp

if [[ -z "$AIC_ROUTER_ADDR" ]]; then
  echo "AIC_ROUTER_ADDR must be provided"
  exit 1
fi

should_enable_acl() {
  [[ "$AIC_ENABLE_ACL" == "true" || "$AIC_ENABLE_ACL" == "1" ]]
}

if should_enable_acl; then
  if [[ ! (-n "$AIC_MODEL_PASSWD" ) ]]; then
    echo "AIC_MODEL_PASSWD must be provided"
    exit 1
  fi
  echo "model:$AIC_MODEL_PASSWD" >> /credentials.txt
fi

ZENOH_CONFIG_OVERRIDE='connect/endpoints=["tcp/'"$AIC_ROUTER_ADDR"'"]'
ZENOH_CONFIG_OVERRIDE+=';transport/shared_memory/enabled=false'
if should_enable_acl; then
  ZENOH_CONFIG_OVERRIDE+=';transport/auth/usrpwd/user="model"'
  ZENOH_CONFIG_OVERRIDE+=';transport/auth/usrpwd/password="'"$AIC_MODEL_PASSWD"'"'
  ZENOH_CONFIG_OVERRIDE+=';transport/auth/usrpwd/dictionary_file="/credentials.txt"'
fi
export ZENOH_CONFIG_OVERRIDE
echo "ZENOH_CONFIG_OVERRIDE=$ZENOH_CONFIG_OVERRIDE"

# VisionInsert_v3 reads weights from this env var (default
# ~/aic_runs/v1_h100_results/best.pt — already where we baked them).
export AIC_V1_WEIGHTS=/root/aic_runs/v1_h100_results/best.pt

exec pixi run --as-is ros2 run aic_model aic_model "$@"
EOF

ENTRYPOINT ["/entrypoint.sh"]

# VisionInsert_v3 reads its tunables from module-level constants
# (no ROS params), so no --params-file needed.
CMD ["--ros-args", \
     "-p", "policy:=aic_example_policies.ros.VisionInsert_v3", \
     "-p", "use_sim_time:=true"]
