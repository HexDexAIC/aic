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

# ── Bake the ACT checkpoint into the image ──────────────────────────────
# The eval network is `internal: true` — no internet at run time. We pull
# the 80k pretrained_model from the local host filesystem at build time via
# the `act_checkpoint` named build context (defined in docker-compose.yaml's
# `additional_contexts`). This keeps the 200 MB checkpoint OUT of the
# source tree and out of git, while still landing it inside the image.
#
# To rebuild against a different checkpoint, set ACT_CHECKPOINT_DIR before
# `docker compose build`, e.g. ACT_CHECKPOINT_DIR=/path/to/pretrained_model.
COPY --from=act_checkpoint . /opt/act_checkpoint/
RUN echo "checkpoint baked at /opt/act_checkpoint:" && ls /opt/act_checkpoint/

# ── Pre-cache the ResNet18 visual backbone ──────────────────────────────
# ACTPolicy lazily downloads `resnet18-f37072fd.pth` from download.pytorch.org
# on first instantiation. Without internet at eval time this fails with
# "Temporary failure in name resolution" and the policy never configures.
# Trigger the download here (build has internet) so the cache file lands
# at /root/.cache/torch/hub/checkpoints/resnet18-f37072fd.pth.
RUN cd /ws_aic/src/aic && pixi run --as-is python -c "\
import torchvision.models as M; \
M.resnet18(weights=M.ResNet18_Weights.IMAGENET1K_V1); \
import os; \
print('cached:', os.listdir(os.path.expanduser('~/.cache/torch/hub/checkpoints/')))"

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
exec pixi run --as-is ros2 run aic_model aic_model "$@"
EOF

ENTRYPOINT ["/entrypoint.sh"]

# RunACTLocal with the baked-in checkpoint. Stiffness/damping/etc come from
# runactlocal.yaml, but checkpoint_path is overridden to the baked location
# (which doesn't exist on the host). Ignore the YAML's `repo_id` since we
# bake the checkpoint and there's no internet at eval time anyway.
CMD ["--ros-args", \
     "--params-file", "/ws_aic/src/aic/aic_example_policies/config/runactlocal.yaml", \
     "-p", "policy:=aic_example_policies.ros.RunACTLocal", \
     "-p", "checkpoint_path:=/opt/act_checkpoint", \
     "-p", "use_sim_time:=true"]
