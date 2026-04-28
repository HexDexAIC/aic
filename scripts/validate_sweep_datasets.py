#!/usr/bin/env python3
"""Deep-validate the shared dataset produced by spawn_sweep_sfp.py.

For <sweep_dir>/dataset/ (one multi-episode LeRobotDataset, one episode
per seed) — and per-episode stats sliced by episode_index in the parquet:
  - parse meta/info.json (schema check)
  - read first-chunk parquet (column existence, frame count agreement)
  - check 6-D rotation columns are unit-norm at the first frame
  - check observation.state has 27 dims, action has 9 dims
  - confirm action.z is monotonically non-increasing over the descent

Writes <sweep_dir>/dataset_validation.json.

Run with the .venv python (has pyarrow):
  src/aic/scripts/.venv/bin/python src/aic/scripts/validate_sweep_datasets.py <sweep_dir>
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import pyarrow.parquet as pq


def find_dataset(sweep_dir: Path) -> Path | None:
    """Return <sweep_dir>/dataset/ if it has a valid info.json."""
    ds = sweep_dir / "dataset"
    if ds.is_dir() and (ds / "meta" / "info.json").exists():
        return ds
    return None


def per_episode_stats(ds: Path) -> list[dict]:
    """Slice the dataset parquet(s) by episode_index and report per-episode
    frames + 6-D rotation unit-norm at first frame.

    LeRobot v3.0 writes one parquet per episode under chunk-000/, so read
    them all and concatenate before slicing.
    """
    chunks = sorted((ds / "data").glob("chunk-*/file-*.parquet"))
    if not chunks:
        return []
    eps: list[int] = []
    states: list[list[float]] = []
    actions: list[list[float]] = []
    for c in chunks:
        tbl = pq.read_table(c, columns=["episode_index",
                                        "observation.state",
                                        "action"])
        eps.extend(tbl.column("episode_index").to_pylist())
        states.extend(tbl.column("observation.state").to_pylist())
        actions.extend(tbl.column("action").to_pylist())
    by_ep: dict[int, list[int]] = {}
    for i, ep in enumerate(eps):
        by_ep.setdefault(ep, []).append(i)

    rows = []
    for ep in sorted(by_ep):
        idxs = by_ep[ep]
        n = len(idxs)
        # Rotation columns at first frame of this episode.
        s0 = states[idxs[0]]
        a0 = actions[idxs[0]]
        def col_norm(v: list[float], offs: int) -> float:
            return math.sqrt(sum(x * x for x in v[offs:offs + 3]))
        rot_ok = all(
            abs(c - 1.0) < 1e-3
            for c in (col_norm(s0, 3), col_norm(s0, 6),
                      col_norm(a0, 3), col_norm(a0, 6))
        )
        # action.z range over this episode.
        z_vals = [actions[i][2] for i in idxs]
        z_range = max(z_vals) - min(z_vals) if z_vals else 0.0
        rows.append({
            "episode_index": ep,
            "frames": n,
            "rot_unit_ok": rot_ok,
            "action_z_range_m": z_range,
            "ok": rot_ok and z_range > 0.05,
        })
    return rows


def first_parquet(ds: Path) -> Path | None:
    chunks = sorted((ds / "data").glob("chunk-*/file-*.parquet"))
    return chunks[0] if chunks else None


def check_dataset(ds: Path) -> dict:
    info_path = ds / "meta" / "info.json"
    if not info_path.exists():
        return {"ok": False, "reason": "no meta/info.json"}
    info = json.loads(info_path.read_text())
    feats = info.get("features", {})
    state_shape = feats.get("observation.state", {}).get("shape", [])
    action_shape = feats.get("action", {}).get("shape", [])
    stiff_shape = feats.get("action.stiffness_diag", {}).get("shape", [])
    video_keys = sorted(k for k in feats if k.startswith("observation.images."))
    total_frames = info.get("total_frames", 0)
    total_episodes = info.get("total_episodes", 0)

    schema_ok = (
        state_shape == [27]
        and action_shape == [9]
        and stiff_shape == [6]
        and len(video_keys) >= 1
        and total_frames > 0
        and total_episodes >= 1
    )

    pq_paths = sorted((ds / "data").glob("chunk-*/file-*.parquet"))
    if not pq_paths:
        return {
            "ok": False, "reason": "no parquet",
            "schema_ok": schema_ok, "info": {
                "frames": total_frames, "episodes": total_episodes,
                "state_shape": state_shape, "action_shape": action_shape,
                "video_keys": video_keys,
            },
        }

    try:
        # LeRobot v3.0 writes one parquet per episode in chunk-000/.
        import pyarrow as pa
        tables = [pq.read_table(p, columns=["observation.state", "action"])
                  for p in pq_paths]
        tbl = pa.concat_tables(tables)
    except Exception as e:
        return {"ok": False, "reason": f"parquet read failed: {e!r}",
                "schema_ok": schema_ok}

    n = tbl.num_rows
    parquet_matches_meta = (n == total_frames)

    # 6-D rotation lives at observation.state[3:9] and action[3:9].
    # Each is two stacked column vectors; cols 0,1,2 = first 3-vec, cols
    # 3,4,5 = second. Both should be unit-norm.
    state0 = list(tbl.column("observation.state")[0].as_py())
    action0 = list(tbl.column("action")[0].as_py())

    def col_norm(vec: list[float], offs: int) -> float:
        return math.sqrt(sum(v * v for v in vec[offs:offs + 3]))

    state_col0 = col_norm(state0, 3)
    state_col1 = col_norm(state0, 6)
    action_col0 = col_norm(action0, 3)
    action_col1 = col_norm(action0, 6)

    rot_unit_ok = all(
        abs(v - 1.0) < 1e-3
        for v in (state_col0, state_col1, action_col0, action_col1)
    )

    # Action.z range — sanity that descent happened (z_min should be much
    # lower than z_max). Not a strict monotonicity check: action.z covers
    # the whole approach + descent + release sequence, which has both
    # rising and falling phases.
    actions = tbl.column("action")
    z_samples: list[float] = []
    step = max(1, n // 50)
    for i in range(0, n, step):
        z_samples.append(actions[i].as_py()[2])
    z_max = max(z_samples) if z_samples else None
    z_min = min(z_samples) if z_samples else None
    z_range = (z_max - z_min) if z_samples else 0.0
    descent_observed = z_range > 0.05  # ≥5 cm vertical travel

    return {
        "ok": schema_ok and parquet_matches_meta and rot_unit_ok and descent_observed,
        "schema_ok": schema_ok,
        "parquet_matches_meta": parquet_matches_meta,
        "rot_unit_ok": rot_unit_ok,
        "descent_observed": descent_observed,
        "frames": total_frames,
        "episodes": total_episodes,
        "state_shape": state_shape,
        "action_shape": action_shape,
        "video_keys": video_keys,
        "first_state_rot_norms": (state_col0, state_col1),
        "first_action_rot_norms": (action_col0, action_col1),
        "z_max": z_max,
        "z_min": z_min,
        "z_range_m": z_range,
    }


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: validate_sweep_datasets.py <sweep_dir>", file=sys.stderr)
        return 2
    sweep_dir = Path(sys.argv[1])

    ds = find_dataset(sweep_dir)
    if ds is None:
        print(f"no shared dataset at {sweep_dir}/dataset/", file=sys.stderr)
        return 1

    overall = check_dataset(ds)
    episodes = per_episode_stats(ds)
    n_ep_ok = sum(1 for e in episodes if e.get("ok"))

    out = {
        "dataset": str(ds),
        "overall": overall,
        "n_episodes": len(episodes),
        "n_episodes_ok": n_ep_ok,
        "episodes": episodes,
    }
    (sweep_dir / "dataset_validation.json").write_text(
        json.dumps(out, indent=2) + "\n"
    )
    overall_ok = "ok" if overall.get("ok") else "FAIL"
    print(f"shared dataset: {overall_ok} "
          f"({overall.get('episodes')} episodes, "
          f"{overall.get('frames')} frames)")
    print(f"per-episode: {n_ep_ok}/{len(episodes)} ok")
    for e in episodes:
        if not e.get("ok"):
            print(f"  ep {e['episode_index']:02d}: FAIL — {e}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
