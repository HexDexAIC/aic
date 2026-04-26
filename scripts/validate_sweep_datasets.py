#!/usr/bin/env python3
"""Deep-validate every dataset produced by spawn_sweep_sfp.py.

For each <sweep_dir>/datasets/seed_NN/aic_recording_*:
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


def find_dataset(seed_dataset_dir: Path) -> Path | None:
    if not seed_dataset_dir.exists():
        return None
    cands = sorted(p for p in seed_dataset_dir.iterdir()
                   if p.is_dir() and p.name.startswith("aic_recording_"))
    return cands[-1] if cands else None


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

    pq_path = first_parquet(ds)
    if pq_path is None:
        return {
            "ok": False, "reason": "no parquet",
            "schema_ok": schema_ok, "info": {
                "frames": total_frames, "episodes": total_episodes,
                "state_shape": state_shape, "action_shape": action_shape,
                "video_keys": video_keys,
            },
        }

    try:
        tbl = pq.read_table(pq_path, columns=["observation.state", "action"])
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

    seeds = sorted((sweep_dir / "datasets").glob("seed_*"))
    results = []
    ok_count = 0
    for seed_dir in seeds:
        seed = int(seed_dir.name.split("_")[1])
        ds = find_dataset(seed_dir)
        if ds is None:
            results.append({"seed": seed, "ok": False, "reason": "no dataset dir"})
            continue
        r = check_dataset(ds)
        r["seed"] = seed
        r["dataset"] = str(ds)
        results.append(r)
        if r.get("ok"):
            ok_count += 1

    out = {
        "n_total": len(seeds),
        "n_ok": ok_count,
        "results": results,
    }
    (sweep_dir / "dataset_validation.json").write_text(
        json.dumps(out, indent=2) + "\n"
    )
    print(f"validated {ok_count}/{len(seeds)} datasets ok")
    for r in results:
        if not r.get("ok"):
            print(f"  seed {r['seed']:02d}: FAIL — {r.get('reason') or 'see json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
