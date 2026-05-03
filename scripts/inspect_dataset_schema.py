#!/usr/bin/env python3
"""Inspect the LeRobotDataset for any TF / port-pose ground-truth columns."""
import json
import sys
from pathlib import Path

ROOT = Path.home() / "aic_hexdex_sfp300"
info = json.loads((ROOT / "meta" / "info.json").read_text())

print("info.json keys:", list(info.keys()))
print()
print("=== features ===")
features = info.get("features", {})
for k, v in features.items():
    shape = v.get("shape")
    dtype = v.get("dtype")
    names = v.get("names")
    extra = ""
    if names:
        if isinstance(names, dict):
            for axis, ns in names.items():
                if isinstance(ns, list) and len(ns) <= 12:
                    extra += f"  names[{axis}]={ns}"
        elif isinstance(names, list) and len(names) <= 12:
            extra = f"  names={names}"
    print(f"  {k}: shape={shape} dtype={dtype}{extra}")

print()
print("=== keys with 'tf' / 'pose' / 'port' / 'gt' / 'transform' ===")
for k in features:
    kl = k.lower()
    if any(s in kl for s in ("tf", "pose", "port", "gt", "transform", "ground")):
        print(f"  {k}")

print()
print("=== sample episode metadata (first row of episodes/) ===")
ep_dir = ROOT / "meta" / "episodes"
if ep_dir.exists():
    for sub in sorted(ep_dir.glob("**/*"))[:5]:
        print(f"  {sub}")

# Try reading the first parquet file's columns
import pyarrow.parquet as pq
pq_paths = sorted((ROOT / "data").rglob("*.parquet"))
if pq_paths:
    p = pq_paths[0]
    print(f"\n=== columns in {p.name} ===")
    pf = pq.ParquetFile(p)
    print(pf.schema_arrow)
    print()
    print("first row sample:")
    tbl = pf.read_row_group(0, columns=None)
    df = tbl.to_pandas()
    print(df.head(1).T.to_string())
