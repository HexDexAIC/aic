# AIC Eval Scoring Guide

A plain-English explanation of how submissions are scored in the AI for Industry Challenge qualification phase.

---

## The Task

The robot (UR5e arm + Robotiq Hand-E gripper) starts with a cable plug already grasped and positioned a few centimeters from a target port. Your policy must insert it.

**Plug/port combinations used:**
- `SFP_MODULE` → `SFP_PORT` on a NIC card (Trials 1 & 2)
- `SC_PLUG` → `SC_PORT` fiber optic connector (Trial 3 — generalization test)

**Per submission: 3 trials, each scored independently (max 100 pts each).**

- Trial 1: SFP insertion, NIC card on `nic_rail_0`
- Trial 2: SFP insertion, NIC card on `nic_rail_1` (different position — tests convergence)
- Trial 3: SC insertion, different board pose, reversed cable orientation — tests generalization

Your policy receives: 3 wrist cameras (1152×1024 @ 20 Hz), joint states, force/torque sensor, and TF frames. It outputs Cartesian pose commands or joint commands to an impedance controller.

---

## Score Structure (100 pts max per trial)

```
Total = Tier 1 (0–1) + Tier 2 (up to +24, down to −36) + Tier 3 (up to 75)
```

---

## Tier 1 — Model Validity (0 or 1 pt)

**A pure pass/fail gate. Failing this scores 0 for the entire trial.**

Your container must:
- Start as a ROS 2 Lifecycle node named `aic_model`
- Successfully configure (load model) within 60 seconds
- Successfully activate within 60 seconds
- Accept and respond to `/insert_cable` action goals when active
- Reject `/insert_cable` goals when not yet active
- Send valid `MotionUpdate` or `JointMotionUpdate` commands

If your container crashes, times out, or doesn't respond to the action server, Tier 1 = 0 and the trial ends there.

---

## Tier 2 — Performance Metrics (up to +24, down to −36 pts)

These metrics reward *how well* you moved, and penalize unsafe behavior.

**Important:** The three positive metrics (smoothness, duration, efficiency) only award points if your Tier 3 score is > 0 — i.e., you got the plug at least close to the port. Moving gracefully in the wrong direction earns nothing.

### Positive Metrics

| Metric | Max | How it's measured |
|--------|-----|-------------------|
| Trajectory Smoothness | 6 pts | Time-weighted average linear jerk (m/s³), computed via Savitzky-Golay filter (15-sample window, local quadratic fit). Only accumulated when arm is moving (speed > 0.01 m/s). 0 jerk → 6 pts; ≥50 m/s³ → 0 pts; linear between. |
| Task Duration | 12 pts | Elapsed time from task start to end. ≤5 sec → 12 pts; ≥60 sec → 0 pts; linear between. |
| Trajectory Efficiency | 6 pts | Total EE path length. ≤ initial plug-port distance → 6 pts; ≥ that + 1 m → 0 pts; linear between. |

### Penalties

| Penalty | Amount | Trigger |
|---------|--------|---------|
| Excessive Insertion Force | −12 pts | Force/torque sensor exceeds **20 N for longer than 1 second** |
| Off-Limit Contact | −24 pts | Any robot link contacts the enclosure (floor, walls, posts, ceiling) or the task board / any mounted component. Cable contacts are not penalized. |

Both penalties are **flat cliffs** — not gradients. A single contact event or sustained high-force episode costs the full penalty regardless of how brief or how much force.

---

## Tier 3 — Task Success (up to 75 pts, or −12 for wrong port)

This is the dominant tier — it drives 75% of the maximum score.

| Outcome | Points |
|---------|--------|
| Full insertion into the **correct** port | **75 pts** |
| Insertion into the **wrong** port | **−12 pts** |
| Partial insertion (plug inside port bounding box, ≤5 mm XY tolerance) | **38–50 pts**, proportional to insertion depth |
| Proximity (plug not in port, but within half the initial plug-port distance from the port entrance) | **0–25 pts**, linear — 25 pts at the entrance, 0 pts at the max distance boundary |
| None of the above | 0 pts |

"Full insertion" is detected when the plug crosses the full insertion depth inside the port. "Partial insertion" requires the plug to be inside the port bounding box (between entrance and bottom) within 5 mm lateral tolerance.

---

## Score Interactions to Keep in Mind

**Getting the plug fully in is worth 75/100 pts.** Everything else is secondary. Tier 2 metrics exist to differentiate policies that all achieve insertion.

**Tier 2 positive metrics are conditional on Tier 3.** If the plug never gets near the port, smoothness/duration/efficiency all score 0 regardless of how elegantly the arm moved.

**Penalties can sink a successful insertion.** A policy that fully inserts but touches the enclosure wall (−24) and sustains >20 N (−12) ends up at 75 − 24 − 12 = 39 pts — worse than a policy that gently gets partial insertion.

**Trial 3 is a deliberate generalization test.** It uses a different plug type, different board orientation, and reversed cable. Overfitting to SFP will likely score 0 on Trial 3.

**Speed has diminishing returns.** The full 12 duration points require insertion in ≤5 seconds. But a 30-second clean insertion still earns ~6 duration points while avoiding the risk of excessive force.

---

## Worked Examples

| Policy behavior | Tier 1 | Tier 2 | Tier 3 | Total |
|-----------------|--------|--------|--------|-------|
| Container fails to start | 0 | 0 | 0 | 0 |
| Arm waves but never approaches port | 1 | 0 (no Tier 3) | 0 | 1 |
| Smooth approach, partial insertion, no penalties | 1 | ~18 | ~44 | ~63 |
| Full insertion in 10s, smooth, no collisions | 1 | ~20 | 75 | ~96 |
| Full insertion but touched enclosure wall | 1 | ~20 − 24 | 75 | ~72 |
| Full insertion but sustained >20 N force | 1 | ~20 − 12 | 75 | ~84 |
| Inserted into wrong port | 1 | 0 | −12 | ~−11 |

---

## Eval Infrastructure

1. Your Docker container starts and initializes the `aic_model` lifecycle node
2. The eval engine spawns the Gazebo scene with the task board and cable
3. The engine sends an `/insert_cable` action goal (180-second time limit)
4. Scoring nodes monitor: EE trajectory, force/torque, contact events, command stream
5. After completion or timeout, scores are written to `$AIC_RESULTS_DIR/scoring.yaml`
6. Zenoh ACL blocks your container from accessing `/gazebo/*`, `/scoring/*`, `/gz_server/*` — simulator internals are not accessible to your policy

Results file location: `~/aic_results/scoring.yaml` (override with `$AIC_RESULTS_DIR`).
