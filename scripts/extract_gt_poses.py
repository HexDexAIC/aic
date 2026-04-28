#!/usr/bin/env python3
"""Extract ground-truth port/plug/camera/static transforms for a sweep.

For each seed_NN/ under <sweep_dir>/seeds/, opens the bag's /tf_static
and /scoring/tf at the FIRST timestamp and writes the relevant subset
to <sweep_dir>/gt_poses.json.

Output schema (per episode):
{
  "episode_index": 0,
  "static_tf":   {child_frame: {parent, x, y, z, qx, qy, qz, qw}, ...},
  "scoring_tf":  {child_frame: {...}, ...},
  "frames_of_interest": {
    "port":        "task_board/nic_card_mount_X/sfp_port_0_link",
    "plug":        "cable_0/sfp_tip_link",
    "task_board":  "task_board",
    "tcp":         "gripper/tcp",
    "cam_optical": {"left":  "left_camera/optical",
                    "center":"center_camera/optical",
                    "right": "right_camera/optical"}
  }
}

How a teammate uses it (TCP pose is in the LeRobotDataset's
observation.state[0:9], in 'base_link' frame at every frame):

  # Compose static chain TCP -> camera_optical from the dataset's
  # static_tf (one composition per camera, computed once per episode):
  T_cam_tcp = compose_chain(static_tf, from='gripper/tcp',
                                       to='left_camera/optical')

  # Per video frame:
  T_tcp_world = pose_from_observation_state(state[0:9])  # base_link frame
  T_world_baselink = compose_chain(static_tf, from='aic_world',
                                              to='base_link')
  T_cam_world = T_world_baselink @ T_tcp_world @ T_cam_tcp

  # Port pose in world (constant per episode):
  T_port_world = compose_chain({**static_tf, **scoring_tf},
                               from='aic_world', to=port_frame)

  # Project: port-in-camera = inv(T_cam_world) @ T_port_world
  port_in_cam = numpy.linalg.inv(T_cam_world) @ T_port_world

The teammate can use tf2 / scipy / pin / their favourite library for
the chain composition. All transforms are quaternion+translation,
so it's a few lines of numpy.

Run with the side-venv (mcap + mcap-ros2-support):
  src/aic/scripts/.venv/bin/python src/aic/scripts/extract_gt_poses.py <sweep_dir>
"""

from __future__ import annotations

import json
import signal
import sys
import time
from pathlib import Path

from mcap.reader import make_reader
from mcap_ros2.decoder import DecoderFactory


class _BagTimeout(Exception):
    pass


def _alarm_handler(signum, frame):
    raise _BagTimeout()


def read_first_transforms(bag_dir: Path, topics: list[str],
                          timeout_s: int = 30) -> dict:
    """Same as before, with a per-bag wallclock timeout (default 30 s)
    so a single corrupted/stuck bag can't hang the whole sweep."""
    mcaps = list(bag_dir.glob("*.mcap"))
    if not mcaps:
        return {}
    out: dict[str, dict] = {}
    signal.signal(signal.SIGALRM, _alarm_handler)
    signal.alarm(timeout_s)
    try:
        with open(mcaps[0], "rb") as fh:
            reader = make_reader(fh, decoder_factories=[DecoderFactory()])
            for _, _, _, msg in reader.iter_decoded_messages(topics=topics):
                for t in msg.transforms:
                    child = t.child_frame_id
                    if child in out:
                        continue
                    tx, rt = t.transform.translation, t.transform.rotation
                    out[child] = {
                        "parent": t.header.frame_id,
                        "x": tx.x, "y": tx.y, "z": tx.z,
                        "qx": rt.x, "qy": rt.y, "qz": rt.z, "qw": rt.w,
                    }
    finally:
        signal.alarm(0)
    return out


def find_first(d: dict, *suffixes: str) -> str | None:
    for s in suffixes:
        for k in d:
            if k.endswith(s) or k == s:
                return k
    return None


def extract_seed(seed_dir: Path) -> dict | None:
    bags = list(seed_dir.glob("bag_trial_1_*"))
    if not bags:
        return None
    static_tf  = read_first_transforms(bags[0], topics=["/tf_static"])
    scoring_tf = read_first_transforms(bags[0], topics=["/scoring/tf"])
    if not (static_tf or scoring_tf):
        return None
    foi = {
        "port":       find_first(scoring_tf,
                                 "/sfp_port_0_link", "/sc_port_0_link",
                                 "/sfp_port_link", "/sc_port_link"),
        "plug":       find_first(scoring_tf, "/sfp_tip_link", "/sc_tip_link"),
        "task_board": find_first(scoring_tf, "/task_board", "task_board"),
        "tcp":        find_first(static_tf, "gripper/tcp"),
        "cam_optical": {
            "left":   find_first(static_tf, "left_camera/optical"),
            "center": find_first(static_tf, "center_camera/optical"),
            "right":  find_first(static_tf, "right_camera/optical"),
        },
    }
    return {
        "static_tf":  static_tf,
        "scoring_tf": scoring_tf,
        "frames_of_interest": foi,
    }


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: extract_gt_poses.py <sweep_dir>", file=sys.stderr)
        return 2
    sweep = Path(sys.argv[1])
    seeds = sorted((sweep / "seeds").glob("seed_*"))

    out = []
    skipped = []
    t0 = time.time()
    for i, sd in enumerate(seeds):
        seed_n = int(sd.name.split("_")[1])
        try:
            gt = extract_seed(sd)
        except _BagTimeout:
            print(f"  [{i+1}/{len(seeds)}] seed_{seed_n:03d}: TIMEOUT — skipping",
                  file=sys.stderr, flush=True)
            skipped.append(seed_n)
            continue
        except Exception as exc:
            print(f"  [{i+1}/{len(seeds)}] seed_{seed_n:03d}: ERR {exc!r}",
                  file=sys.stderr, flush=True)
            skipped.append(seed_n)
            continue
        if gt is None:
            print(f"  [{i+1}/{len(seeds)}] seed_{seed_n:03d}: no bag", file=sys.stderr)
            skipped.append(seed_n)
            continue
        out.append({"episode_index": seed_n, **gt})
        if (i + 1) % 25 == 0 or i == 0:
            elapsed = time.time() - t0
            print(f"  [{i+1}/{len(seeds)}] {elapsed:.0f}s elapsed, "
                  f"{len(out)} ok, {len(skipped)} skipped", flush=True)

    target = sweep / "gt_poses.json"
    target.write_text(json.dumps(out, indent=2) + "\n")
    print(f"wrote {target}")
    if out:
        ep0 = out[0]
        foi = ep0["frames_of_interest"]
        print(f"  episodes:    {len(out)}/{len(seeds)}")
        print(f"  static_tf:   {len(ep0['static_tf'])} transforms (same across episodes)")
        print(f"  scoring_tf:  {len(ep0['scoring_tf'])} transforms (per-episode scene state)")
        print(f"  port frame:  {foi['port']}")
        print(f"  plug frame:  {foi['plug']}")
        print(f"  cameras:     {foi['cam_optical']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
