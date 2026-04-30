#!/usr/bin/env python3
"""ACT-PR training driver.

Loads HexDexAIC/aic-sfp-500-pr (the derived dataset with port-frame residual
action + observation.port_pose_gt label), builds ACTPRPolicy (ACT + auxiliary
port-pose head), trains with combined loss:

    loss = action_l1 + kl_weight * kl_div + port_pose_loss_weight * port_pose_l1

Saves checkpoints in the lerobot pretrained_model/ format so RunACTPR (the
deployment wrapper) can load them via the same pattern as RunACTLocal.

Usage on Velda (single A100):
    python train_actpr.py \
        --output_dir ~/outputs/act-pr-v1 \
        --steps 80000 \
        --batch_size 8 \
        --num_workers 2 \
        --port_pose_loss_weight 1.0 \
        --push_repo HexDexAIC/act-pr-aic-sfp-500-v1

Smoke (Phase 2 micro-run before full training):
    python train_actpr.py --steps 10000 --eval_freq 0 ...

Strict-clean filter is applied via --episodes_file (clean_eps.txt-style) at
the dataset-load level. Without it, all 500 episodes train including the
12 non-strict.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Make the script's dir importable so we can do `from act_pr_policy import ...`.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from act_pr_policy import ACTPR, ACTPRConfig, ACTPRPolicy, PORT_POSE_KEY  # noqa: E402

from lerobot.utils.constants import ACTION, OBS_IMAGES, OBS_STATE  # noqa: E402
from lerobot.datasets.lerobot_dataset import LeRobotDataset  # noqa: E402
from lerobot.configs.types import FeatureType, NormalizationMode, PolicyFeature  # noqa: E402

os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--repo_id", default="HexDexAIC/aic-sfp-500-pr")
    p.add_argument("--episodes_file", default=None,
                   help="comma-separated ep ids (clean_eps.txt-style). If unset, all 500.")
    p.add_argument("--output_dir", default="./outputs/act-pr-v1")
    p.add_argument("--steps", type=int, default=80000)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--lr_backbone", type=float, default=1e-5)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--save_freq", type=int, default=20000)
    p.add_argument("--log_freq", type=int, default=200)
    p.add_argument("--port_pose_loss_weight", type=float, default=1.0)
    p.add_argument("--chunk_size", type=int, default=100)
    p.add_argument("--push_repo", default="",
                   help="HF repo to push final checkpoint. Empty disables.")
    p.add_argument("--push_private", action="store_true", default=True)
    p.add_argument("--resume", action="store_true",
                   help="resume from output_dir/checkpoints/last")
    return p.parse_args()


def read_episodes(path: str | None):
    if not path:
        return None
    txt = Path(path).read_text().strip()
    return [int(s) for s in txt.split(",") if s.strip()]


def build_features(dataset: LeRobotDataset) -> tuple[dict, dict]:
    """Build input_features / output_features for ACTPRConfig from the
    dataset's info.json."""
    feats = dataset.meta.info["features"]
    input_features = {}
    for k in ("observation.images.left", "observation.images.center",
              "observation.images.right"):
        f = feats[k]
        # lerobot expects (C, H, W) shape in PolicyFeature
        h, w, c = f["shape"]
        input_features[k] = PolicyFeature(type=FeatureType.VISUAL, shape=(c, h, w))
    input_features["observation.state"] = PolicyFeature(
        type=FeatureType.STATE, shape=(feats["observation.state"]["shape"][0],),
    )
    # NEW: aux supervision target lives as an "input feature" for the
    # preprocessor's normalizer to handle. ACTPRPolicy.forward reads it
    # from the batch.
    input_features[PORT_POSE_KEY] = PolicyFeature(
        type=FeatureType.STATE, shape=(feats[PORT_POSE_KEY]["shape"][0],),
    )
    output_features = {
        "action": PolicyFeature(type=FeatureType.ACTION,
                                shape=(feats["action"]["shape"][0],)),
    }
    return input_features, output_features


def build_dataset_stats(dataset: LeRobotDataset) -> dict:
    """Convert dataset.meta.stats into a dict of dicts of float32 tensors."""
    stats = {}
    for key, st in dataset.meta.stats.items():
        if not isinstance(st, dict):
            continue
        stats[key] = {}
        for stat_name in ("mean", "std", "min", "max"):
            if stat_name in st:
                stats[key][stat_name] = torch.tensor(st[stat_name], dtype=torch.float32)
    return stats


class SimpleNormalizer:
    """Minimal MEAN_STD normalizer keyed off dataset stats. Substitutes for
    lerobot's PolicyProcessorPipeline since we bypass the factory.

    Image features are normalized using ImageNet mean/std (matching ACT's
    default preprocessing), since stats.json typically doesn't contain
    image stats and the ResNet18 backbone expects ImageNet-normalized RGB.
    """

    IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    IMAGENET_STD  = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

    def __init__(self, stats: dict, device: torch.device, image_keys: list[str],
                 state_keys: list[str], action_keys: list[str]):
        self.stats = stats
        self.image_keys = image_keys
        self.state_keys = state_keys
        self.action_keys = action_keys
        self._mean = SimpleNormalizer.IMAGENET_MEAN.to(device)
        self._std  = SimpleNormalizer.IMAGENET_STD.to(device)
        self.device = device
        # Pre-cast scalar stats to device tensors
        self._cached: dict[str, dict[str, torch.Tensor]] = {}
        for k, st in stats.items():
            self._cached[k] = {n: v.to(device) for n, v in st.items()}

    def normalize_state(self, batch: dict) -> dict:
        out = dict(batch)
        for k in self.state_keys:
            if k in batch and k in self._cached:
                m, s = self._cached[k].get("mean"), self._cached[k].get("std")
                if m is not None and s is not None:
                    out[k] = (batch[k] - m) / s
        return out

    def normalize_action(self, batch: dict) -> dict:
        out = dict(batch)
        for k in self.action_keys:
            if k in batch and k in self._cached:
                m, s = self._cached[k].get("mean"), self._cached[k].get("std")
                if m is not None and s is not None:
                    # action is (B, T, D) — broadcast (D,) → (1, 1, D)
                    while m.dim() < batch[k].dim():
                        m = m.unsqueeze(0); s = s.unsqueeze(0)
                    out[k] = (batch[k] - m) / s
        return out

    def normalize_image(self, batch: dict) -> dict:
        out = dict(batch)
        for k in self.image_keys:
            if k in batch:
                # Images are uint8 [0, 255] from the dataloader. Convert to
                # float [0, 1] then ImageNet-normalize.
                x = batch[k].to(self.device, non_blocking=True)
                if x.dtype == torch.uint8:
                    x = x.float() / 255.0
                out[k] = (x - self._mean) / self._std
        return out

    def __call__(self, batch: dict) -> dict:
        batch = self.normalize_image(batch)
        batch = self.normalize_state(batch)
        batch = self.normalize_action(batch)
        return batch


def main():
    args = parse_args()
    out_dir = Path(args.output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── dataset ───────────────────────────────────────────────────
    print(f"[{time.strftime('%H:%M:%S')}] loading dataset {args.repo_id}", flush=True)
    eps = read_episodes(args.episodes_file)
    dataset = LeRobotDataset(
        repo_id=args.repo_id,
        episodes=eps,
        delta_timestamps={"action": [i / 20.0 for i in range(args.chunk_size)]},
    )
    print(f"  episodes: {dataset.meta.total_episodes}, frames: {dataset.meta.total_frames}",
          flush=True)

    # ── policy ────────────────────────────────────────────────────
    input_features, output_features = build_features(dataset)
    config = ACTPRConfig(
        input_features=input_features,
        output_features=output_features,
        chunk_size=args.chunk_size,
        n_action_steps=args.chunk_size,
        n_obs_steps=1,
        vision_backbone="resnet18",
        pretrained_backbone_weights="ResNet18_Weights.IMAGENET1K_V1",
        replace_final_stride_with_dilation=False,
        pre_norm=False,
        dim_model=512,
        n_heads=8,
        dim_feedforward=3200,
        feedforward_activation="relu",
        n_encoder_layers=4,
        n_decoder_layers=1,
        use_vae=True,
        latent_dim=32,
        n_vae_encoder_layers=4,
        temporal_ensemble_coeff=None,
        dropout=0.1,
        kl_weight=10.0,
        optimizer_lr=args.lr,
        optimizer_weight_decay=args.weight_decay,
        optimizer_lr_backbone=args.lr_backbone,
        port_pose_loss_weight=args.port_pose_loss_weight,
        normalization_mapping={
            "VISUAL": NormalizationMode.MEAN_STD,
            "STATE":  NormalizationMode.MEAN_STD,
            "ACTION": NormalizationMode.MEAN_STD,
        },
    )

    print(f"[{time.strftime('%H:%M:%S')}] building policy", flush=True)
    stats = build_dataset_stats(dataset)
    policy = ACTPRPolicy(config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy.to(device)
    policy.train()

    # Manual normalizer (we bypass lerobot's preprocessor pipeline).
    normalizer = SimpleNormalizer(
        stats=stats, device=device,
        image_keys=list(config.image_features.keys()),
        state_keys=["observation.state", PORT_POSE_KEY],
        action_keys=["action"],
    )

    n_params = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    print(f"  device: {device}", flush=True)
    print(f"  num_learnable_params: {n_params:,}", flush=True)

    # ── optimizer ─────────────────────────────────────────────────
    backbone_params, other_params = [], []
    for name, p in policy.named_parameters():
        if not p.requires_grad: continue
        (backbone_params if "backbone" in name else other_params).append(p)
    optimizer = torch.optim.AdamW([
        {"params": other_params,    "lr": args.lr},
        {"params": backbone_params, "lr": args.lr_backbone},
    ], weight_decay=args.weight_decay)

    # ── resume? ──────────────────────────────────────────────────
    start_step = 0
    last_ckpt = out_dir / "checkpoints" / "last"
    if args.resume and last_ckpt.is_dir():
        ts_path = last_ckpt / "training_state" / "training_state.pt"
        if ts_path.exists():
            ts = torch.load(ts_path, map_location=device)
            start_step = ts.get("step", 0)
            policy.load_state_dict(ts["policy_state_dict"])
            optimizer.load_state_dict(ts["optimizer_state_dict"])
            print(f"  resumed from step {start_step}", flush=True)

    # ── dataloader ───────────────────────────────────────────────
    dataloader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=(device.type == "cuda"),
        drop_last=True, persistent_workers=(args.num_workers > 0),
    )
    data_iter = iter(dataloader)

    # ── training loop ────────────────────────────────────────────
    print(f"[{time.strftime('%H:%M:%S')}] starting training loop "
          f"(steps {start_step} → {args.steps})", flush=True)
    t0 = time.time()
    last_log_t = t0
    last_log_step = start_step

    for step in range(start_step + 1, args.steps + 1):
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            batch = next(data_iter)

        # Move to device. lerobot's dataset returns a flat dict with
        # tensor values; some are float, some are ints.
        batch = {k: v.to(device, non_blocking=True) if torch.is_tensor(v) else v
                 for k, v in batch.items()}

        # Normalize via our manual pipeline (mirrors lerobot's MEAN_STD).
        batch = normalizer(batch)
        # ACT expects images stacked as a list at OBS_IMAGES.
        if config.image_features:
            batch[OBS_IMAGES] = [batch[k] for k in config.image_features]

        loss, loss_dict = policy.forward(batch)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=10.0)
        optimizer.step()

        if step % args.log_freq == 0:
            now = time.time()
            sps = (step - last_log_step) / max(now - last_log_t, 1e-6)
            last_log_t, last_log_step = now, step
            print(
                f"[{time.strftime('%H:%M:%S')}] step:{step}/{args.steps} "
                f"loss:{loss_dict['loss']:.4f} "
                f"l1:{loss_dict.get('l1_loss', 0):.4f} "
                f"port:{loss_dict.get('port_pose_loss', 0):.4f} "
                f"kld:{loss_dict.get('kld_loss', 0):.4f} "
                f"sps:{sps:.2f}",
                flush=True,
            )

        if step % args.save_freq == 0 or step == args.steps:
            ckpt_dir = out_dir / "checkpoints" / f"{step:06d}"
            (ckpt_dir / "pretrained_model").mkdir(parents=True, exist_ok=True)
            (ckpt_dir / "training_state").mkdir(parents=True, exist_ok=True)
            policy.save_pretrained(ckpt_dir / "pretrained_model")
            torch.save(
                {
                    "step": step,
                    "policy_state_dict": policy.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                },
                ckpt_dir / "training_state" / "training_state.pt",
            )
            symlink = out_dir / "checkpoints" / "last"
            if symlink.is_symlink() or symlink.exists():
                symlink.unlink()
            symlink.symlink_to(f"{step:06d}")
            print(f"  saved checkpoint at step {step}", flush=True)

    # ── push to HF ───────────────────────────────────────────────
    if args.push_repo:
        from huggingface_hub import HfApi
        api = HfApi()
        api.create_repo(args.push_repo, repo_type="model",
                        private=args.push_private, exist_ok=True)
        api.upload_folder(
            folder_path=str(out_dir / "checkpoints" / "last" / "pretrained_model"),
            repo_id=args.push_repo, repo_type="model",
            commit_message=f"ACT-PR {args.steps}-step checkpoint",
        )
        print(f"  pushed to {args.push_repo}", flush=True)

    print(f"[{time.strftime('%H:%M:%S')}] DONE in {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
