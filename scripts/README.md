# scripts/

Helper tooling for running batch scoring sweeps against the AIC eval stack
and post-processing the results.

## Layout

| file | what |
|---|---|
| `run_scoring_loop.sh` | Orchestrator — runs POLICY × N through the full eval stack, writes per-run artifacts + sweep-level summary. Entry point for normal use. |
| `summarize_sweep.py` | Aggregator — parses each `run_NNN/scoring.yaml` into `summary.csv` + `summary.md` (mean/stdev/per-trial/Tier-3 rate). Auto-called by `run_scoring_loop.sh`. Uses system `python3` + `python3-yaml`. |
| `extract_initial_poses.py` | Post-processor — pulls task_board / port / plug / cable / TCP / joint-state starting poses out of each trial's `bag_trial_N/*.mcap` so variance across runs can be visualized. Currently slow (see wiki). Run via the venv below. |
| `.venv/` | (gitignored) Python venv with `mcap`, `mcap-ros2-support`, `pyyaml` for `extract_initial_poses.py`. Recreate with the command below if absent. |

## Usage

```bash
scripts/run_scoring_loop.sh POLICY N [--ground-truth] [--no-gui] [--no-rviz] [--headless] [--timeout SEC] [--output-dir DIR]
```

See `aic_wiki/wiki/methodology/running-sweeps.md` for the full runbook.

## Setting up `.venv`

Required only for `extract_initial_poses.py`. One-time:

```bash
cd "$(git rev-parse --show-toplevel)/scripts"   # or wherever this lives
python3 -m venv .venv
.venv/bin/pip install --quiet --upgrade pip
.venv/bin/pip install --quiet mcap mcap-ros2-support pyyaml
```

Run the extractor with:

```bash
.venv/bin/python extract_initial_poses.py <sweep_dir>
```
