# pi0.5 → AIC Integration Plan

Status: design doc (not yet implemented)
Target: run Physical Intelligence's pi0.5 policy as an AIC challenge policy for UR5e cable insertion.

---

## 1. High-level architecture

Two processes, connected by a websocket:

```
+-----------------------------+             +--------------------------+
|  AIC eval container         |             |  openpi inference server |
|  (Docker, Gazebo + aic_engine)             |  (WSL2 Ubuntu-24.04,     |
|                             |             |   JAX + CUDA, port 8000) |
|  /observations  20 Hz       |             |                          |
|       |                     |             |                          |
|       v                     |             |                          |
|  aic_model (lifecycle node) |             |                          |
|  (runs in WSL2 via pixi)    |             |                          |
|       |                     |             |                          |
|       | PiZeroPolicy class  |             |                          |
|       |   - translate obs   |  websocket  |                          |
|       |   - ----------------+-----------> |  WebsocketPolicyServer   |
|       |   - <---------------+------------ |  pi0.5 inference         |
|       |   - send joint cmds |             |                          |
|       v                     |             |                          |
|  aic_controller -> robot    |             |                          |
+-----------------------------+             +--------------------------+
```

Rationale for splitting:
- The openpi server is a standalone Python process; no ROS2 dependency.
- The AIC side only needs a thin wrapper that implements the `Policy` interface.
- Fine-tuning and inference can run on any machine with the right GPU; the policy wrapper stays tiny.

---

## 2. Why pi0.5, and which checkpoint

- pi0.5 is the latest generation: diffusion VLA with adaRMS time conditioning, trained with knowledge insulation. Best zero-shot generalization of the three variants.
- Closest available pretrained config for our setup: `pi05_droid` (trained on Franka + UR5-family data with a DROID-style obs dict). This is the checkpoint we will load on the server.
- Available at `gs://openpi-assets/checkpoints/pi05_droid` (Google Cloud Storage). Downloadable via `gsutil` or a Hugging Face mirror.

### Why zero-shot pi0.5 will not work on AIC

Two separate reasons, stacked on top of each other:

**2.1 The task is not in pi0's training data.**

pi0.5 was pretrained on ~10,000 hours of robot manipulation data. The tasks look like:
- "pick up the red cup and put it in the bowl"
- "fold the towel"
- "open the drawer"
- "stack the blocks"

These are **pick-and-place / gross manipulation** tasks. The model learned how to see an object in a workspace, move the gripper toward it, close the gripper and lift. It has NOT seen at training scale:
- Cable insertion (connector → port alignment)
- Sub-millimeter precision positioning
- Compliance / force-modulated contact

Cable insertion is a **fine-motor, haptic-dominant task**. Even humans can't do it reliably with vision alone — you always feel the plug as you insert it. The training data simply doesn't teach the model what successful insertion looks like.

**2.2 The camera geometry is wrong.**

pi0 DROID was trained on setups with two distinct camera roles:

```
              [ceiling]
                 |
                 v
           ┌─────────────┐
           │ TRIPOD CAM  │   ← "exterior_image_1_left"
           │             │     stationary, watches whole scene
           │             │     sees: robot + workspace + target
           └──────┬──────┘
                  │ (mounted on tripod, NEVER moves)
                  v
     ┌────────────────────────┐
     │                        │
     │     [robot arm]        │
     │          │             │
     │     [gripper]──[cam]   ← "wrist_image_left"
     │          │             │   bolted to gripper, moves with it
     │     [workspace]        │   sees: what's directly in front of tool
     │                        │
     └────────────────────────┘
```

During training the model learned to use these two views for different purposes:
- **Exterior** → where am I relative to the target? spatial awareness.
- **Wrist** → precise local alignment, what am I about to touch?

The model internally has **different learned representations** for these two camera roles.

AIC looks like this instead:

```
     ┌────────────────────────┐
     │                        │
     │     [robot arm]        │
     │          │             │
     │   [camera RING]        │   all three cameras:
     │    L    C    R         │   bolted to end-effector
     │          │             │   all move when arm moves
     │          v             │   all look at same region
     │     [workspace,        │   (~10 cm apart, similar POV)
     │      partial view]     │
     │                        │
     └────────────────────────┘
```

All three AIC cameras are **clustered on the end-effector** — a camera ring that moves with the arm. None is a stationary third-person view. When you feed AIC's `left_image` into pi0's `exterior_image_1_left` slot, the model expects a static overview shot and instead gets a jiggling close-up. The learned visual features won't match the input.

**Analogy.** Imagine teaching someone to insert a USB-C plug into a laptop by: (1) showing them thousands of videos of people picking up cups and stacking blocks (never showing connector insertion), (2) handing them a VR headset with three GoPros velcroed to their wrist and no overview camera, (3) saying "now insert this plug." They'd wave the plug around in the right general area, but wouldn't reliably insert it.

**Conclusion.** Zero-shot is useful only for validating the plumbing — websocket round-trip, ROS2 lifecycle, action command path. Do not expect the arm to actually complete the task.

---

## 3. Where each component runs

| Component | Host | Notes |
|---|---|---|
| AIC eval container | WSL2 (Docker Desktop) | Already working with fixed18 image |
| aic_model policy process | WSL2 Ubuntu-24.04 (via pixi) | Already working for WaveArm |
| openpi server | WSL2 Ubuntu-24.04 | JAX + CUDA, 8GB+ VRAM needed |
| Connection | `ws://localhost:8000` | Policy and server share localhost |

Reasons for WSL2 over Windows for the server:
- JAX's official GPU build is Linux-only. Windows = CPU fallback = unusable.
- `uv` (openpi's package manager) is Linux-first.
- Already-working CUDA passthrough in WSL2 (your `--gpus all` eval container proves this).
- No port forwarding needed between policy and server.

Fallback if GPU VRAM is tight (eval container already uses some for Gazebo rendering):
- Run openpi server on Windows with PyTorch instead of JAX (openpi has PyTorch support).
- Policy connects via `ws://host.docker.internal:8000` or the Windows host IP.

---

## 4. What AIC actually publishes

On every tick (up to 20 Hz), `aic_adapter` emits one `aic_model_interfaces/Observation` message on `/observations`. Contents:

### Images (3 cameras, all on a camera ring near the end-effector)

| Field | Resolution | Encoding | Role |
|---|---|---|---|
| `left_image` | 1152 × 1024 | R8G8B8 (uint8 RGB) | Left-angled workspace view |
| `center_image` | 1152 × 1024 | R8G8B8 (uint8 RGB) | Straight-ahead workspace view |
| `right_image` | 1152 × 1024 | R8G8B8 (uint8 RGB) | Right-angled workspace view |

Important: these cameras move with the end-effector. There is no static third-person scene camera.

Raw-pixel access in Python:
```python
img = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, 3)
# (1024, 1152, 3) uint8 RGB
```

### Joint state

```python
obs.joint_states.position  # float64[7]
# [0] shoulder_pan_joint   (radians)
# [1] shoulder_lift_joint  (radians)
# [2] elbow_joint          (radians)
# [3] wrist_1_joint        (radians)
# [4] wrist_2_joint        (radians)
# [5] wrist_3_joint        (radians)
# [6] gripper              (meters, finger-to-finger distance)
```

- First 6 are standard UR5e order — matches what pi0 expects.
- Gripper is a LENGTH in meters, not an angle. The AIC adapter divides the raw finger joint by 2 so this value is finger-to-finger separation. Max open ≈ 0.08 m (verify from URDF).

### Other fields (not consumed by pi0)

| Field | Type |
|---|---|
| `wrist_wrench` | geometry_msgs/WrenchStamped — force + torque at tool |
| `controller_state.tcp_pose` | TCP pose (position + quaternion) |
| `controller_state.tcp_velocity` | TCP twist |
| `controller_state.tcp_error` | 6-DOF tracking error |

Most important discard: `wrist_wrench`. Cable insertion is force-sensitive. pi0 has no force input channel — this is a known limitation for insertion-style tasks.

---

## 5. What pi0.5 (DROID config) expects

Over the websocket, per inference request:

```python
{
  "observation/exterior_image_1_left": np.ndarray,  # (224, 224, 3) uint8 RGB
  "observation/wrist_image_left":      np.ndarray,  # (224, 224, 3) uint8 RGB
  "observation/joint_position":        np.ndarray,  # (7,) float32, radians
  "observation/gripper_position":      np.ndarray,  # (1,) float32, [0.0, 1.0]
  "prompt":                            str,
}
```

- Images: uint8, HWC layout, raw RGB in [0, 255]. No normalization at transmit time (server handles the [-1, 1] conversion internally).
- `joint_position` is 7D (6 arm joints + gripper slot). Units: radians.
- `gripper_position` is a 1-element array, continuous in [0, 1].
- `prompt` is natural language.

### Mental picture: what pi0.5 "sees" vs what AIC gives

Think of pi0.5 as a function with a **fixed, narrow input shape**:

```
pi0.5(inputs) → joint velocities

inputs = {
    two specific 224×224 images (one overview, one close-up),
    7 numbers about the robot's joints,
    a sentence describing the task
}
```

That's the entire sensory world of pi0.5. Anything outside these three slots — force, torque, a third camera, Cartesian pose, tracking error, camera intrinsics — is invisible to it. There is no input projection layer in the neural network for these signals.

Meanwhile, AIC gives us a much richer stream:

```
AIC observation = {
    3 × 1152×1024 images (left / center / right, all wrist-mounted),
    7 joint positions + velocities + efforts,
    wrench: (Fx, Fy, Fz, Tx, Ty, Tz)  ← force/torque at the tool,
    TCP pose (x, y, z, qx, qy, qz, qw),
    TCP velocity (linear + angular),
    TCP tracking error (6-DOF),
    camera intrinsics for all 3 cameras,
    timestamps for synchronization
}
```

Wiring pi0.5 into AIC is therefore two operations:
1. **Drop** everything pi0.5 doesn't have a slot for (force, 3rd camera, controller state).
2. **Reshape** the rest to fit pi0.5's expected input shape (resize images, pick 2 of 3 cameras, concatenate state into a 7-vector).

The "drop" step is what makes zero-shot performance bad — we're throwing away the most informative signals for insertion.

---

## 6. Field-by-field mapping

### 6.1 Arm joint positions — direct copy

```python
arm_joints = np.array(obs.joint_states.position[:6], dtype=np.float32)
# AIC UR5e joint order == pi0 expected order. Radians. No normalization.
```

### 6.2 Gripper — meters → [0, 1]

```python
GRIPPER_MAX_METERS = 0.08  # confirm from URDF joint limit
raw = obs.joint_states.position[6]
gripper_norm = np.clip(raw / GRIPPER_MAX_METERS, 0.0, 1.0).astype(np.float32)
gripper_position = np.array([gripper_norm], dtype=np.float32)  # shape (1,)
```

### 6.3 joint_position (full 7D) sent to server

pi0 DROID treats the gripper as the 7th joint. Two valid patterns:

A. Separate the signals (cleaner):
```python
joint_position = np.concatenate([arm_joints, np.zeros(1, dtype=np.float32)])
# gripper_position carries the real gripper signal in its own field
```

B. Fold gripper into joint_position[6] directly:
```python
joint_position = np.concatenate([arm_joints, [gripper_norm]]).astype(np.float32)
# gripper_position field then duplicates joint_position[6]
```

The DROID example on real hardware uses pattern B effectively. Either works — B is closer to what the model saw in training.

### 6.4 Images — resize with aspect-preserving pad

```python
def ros_img_to_pi(msg) -> np.ndarray:
    img = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, 3)
    h, w = img.shape[:2]
    s = 224.0 / max(h, w)
    nh, nw = int(h * s), int(w * s)
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
    out = np.zeros((224, 224, 3), dtype=np.uint8)
    ph, pw = (224 - nh) // 2, (224 - nw) // 2
    out[ph:ph + nh, pw:pw + nw] = resized
    return out
```

AIC is 1152×1024 (aspect ≈ 1.125). After resize-with-pad, the 224×224 image has a small vertical black band. Acceptable.

### 6.5 Which AIC camera goes where

pi0 DROID expects:
- `exterior_image_1_left` = fixed third-person scene camera
- `wrist_image_left` = gripper-mounted tool camera

AIC has neither exactly. Pragmatic choice for zero-shot / first wiring:

```python
exterior_image_1_left = ros_img_to_pi(obs.left_image)    # side view
wrist_image_left      = ros_img_to_pi(obs.center_image)  # straight-ahead view
```

Alternatives to try later:
- Swap left for right.
- Zero out exterior, keep only center:
  ```python
  exterior_image_1_left = np.zeros((224, 224, 3), dtype=np.uint8)
  ```

None of these match training distribution closely — zero-shot performance will be poor. Fine-tuning on AIC-collected demonstrations is the real fix.

### 6.6 Prompt

Build from the `Task` message the engine sends in the `insert_cable` action goal:

```python
prompt = f"insert {task.cable_name} into {task.port_name}"
# e.g. "insert usb_c_cable into port_a"
```

If available, include target module:
```python
prompt = f"insert {task.cable_name} into {task.port_name} on {task.target_module_name}"
```

---

## 7. What we discard and why

| AIC field | Why discarded | Impact |
|---|---|---|
| `right_image` | pi0 DROID takes 2 images | Lose one useful viewpoint |
| `wrist_wrench` | No force input in pi0 | Major — insertion is force-driven |
| `tcp_pose`, `tcp_velocity` | pi0 uses joint space | Minor — FK-derivable |
| `tcp_error` | Controller-internal | Minor |
| `camera_info` intrinsics | Not used by pi0 | Minor |

### Why `wrist_wrench` is the big one

Force/torque at the wrist is the robot's **sense of touch**. Six numbers per timestep, in Newtons and Newton-meters.

For cable insertion specifically, the sensor tells you things like:
- `Fz = -2.5 N` → the plug is pressing into something (good — we've made contact).
- `Tx = 0.3 Nm` → there's a sideways torque on the wrist (the plug is catching on the rim of the port — back off and realign).
- `|F| < 0.1 N` → no contact (still in free space, keep moving).

Humans do cable insertion partly by closing their eyes and feeling it. Visual alignment gets you to within roughly 5 mm; the last 5 mm is pure haptic feedback. Without force data, the model can't tell "pressing on the rim" from "slotted into the port" — both look visually identical from a wrist camera.

**pi0's architecture has no input channel for this data.** You cannot just add a `force` key to the websocket dict — the model wouldn't know what to do with it, because there's no corresponding input projection layer in the neural network. Adding force would require:
- Modifying the model's input layer to accept a 6-vector of wrench data
- Retraining (or at minimum fine-tuning with a new head) on data that includes force signals

That's a substantive modeling project, not a plumbing change. This is why "the discarded field that matters" is force, not the third camera: missing a camera costs a viewpoint, missing force blinds the model to the one signal that determines whether insertion is succeeding.

### Why the other discards are mostly fine

- `tcp_pose` and `tcp_velocity` are **redundant with joint state**. Forward kinematics maps joint positions → TCP pose deterministically. If the model has joint positions, it has (implicitly) TCP pose.
- `tcp_error` is a controller-internal diagnostic (how far is the actual pose from the commanded pose). Nothing in pi0's training data looked like this signal — even if we found a way to feed it, the model couldn't interpret it.
- `camera_info` intrinsics would matter if we were doing 3D reconstruction. pi0 works on raw pixels end-to-end; calibration is folded into the learned representations during pretraining.

---

## 8. Action output

pi0.5 returns an action chunk per inference:

- Shape: `(10, 8)` — 10 timesteps, 8 action dims, float32.
- Semantics (DROID convention):
  - `action[:, 0:6]`: joint velocity commands (rad/s), clipped to [-1, 1].
  - `action[:, 6]`: gripper target, continuous, thresholded at 0.5 for binary open/close.
  - `action[:, 7]`: unused in DROID pipeline — ignore.
- Execution frequency: 15 Hz (DROID default). Re-query the server every 10 executed steps.

Mapping to AIC output:

pi0 gives joint velocities. AIC's `JointMotionUpdate.msg` supports velocity-mode commands:

```python
from aic_control_interfaces.msg import JointMotionUpdate, TrajectoryGenerationMode
from trajectory_msgs.msg import JointTrajectoryPoint

def pi_action_to_joint_cmd(action_vec):
    msg = JointMotionUpdate()
    point = JointTrajectoryPoint()
    # pi0 outputs velocities for arm joints
    point.velocities = [float(v) for v in action_vec[:6]]
    msg.target_state = point
    msg.trajectory_generation_mode.mode = TrajectoryGenerationMode.MODE_VELOCITY
    return msg
```

Note: AIC's gripper command is separate from arm joints. For the zero-shot phase we can ignore gripper output entirely since cable-insertion usually doesn't need gripper open/close events during the approach. If gripper control becomes needed, plumb `action_vec[6]` through a separate message or include it in the `JointTrajectoryPoint.velocities` as index 6 (the gripper joint slot).

---

## 9. Action chunking and timing loop

pi0 is chunk-based: you infer once, then execute 10 steps before re-inferring. This is different from most ROS controllers which expect a fresh command every tick.

```python
CHUNK = 10
HZ = 15  # DROID default execution rate
PERIOD = 1.0 / HZ

t_in_chunk = 0
chunk = None

while running:
    obs = get_observation()
    if obs is None: continue

    if chunk is None or t_in_chunk >= CHUNK:
        request = aic_obs_to_pi_request(obs, task)
        response = client.infer(request)   # blocks ~50-100ms on GPU
        chunk = response["actions"]        # (10, 8)
        t_in_chunk = 0

    cmd = pi_action_to_joint_cmd(chunk[t_in_chunk])
    move_robot(joint_motion_update=cmd)
    t_in_chunk += 1
    time.sleep(PERIOD)
```

Sanity note: if AIC's sim is running at sim-time != wall-time, `time.sleep` drift matters. Use `node.get_clock().sleep_for()` instead.

---

## 10. PiZeroPolicy class sketch

The AIC `Policy` base class calls `insert_cable()` with four callables. Our implementation:

```python
import numpy as np
import cv2
from openpi_client.websocket_client_policy import WebsocketClientPolicy
from aic_control_interfaces.msg import JointMotionUpdate, TrajectoryGenerationMode
from trajectory_msgs.msg import JointTrajectoryPoint
from aic_model.policy import Policy

GRIPPER_MAX_METERS = 0.08
CHUNK = 10
HZ = 15


class PiZeroPolicy(Policy):
    def __init__(self, node):
        super().__init__(node)
        self._client = WebsocketClientPolicy(host="localhost", port=8000)
        self._node = node

    def insert_cable(self, task, get_observation, move_robot, send_feedback):
        chunk = None
        t_in_chunk = 0
        period = 1.0 / HZ

        while rclpy.ok():
            obs = get_observation()
            if obs is None:
                self._node.get_clock().sleep_for(rclpy.duration.Duration(seconds=period))
                continue

            if chunk is None or t_in_chunk >= CHUNK:
                request = self._build_request(obs, task)
                response = self._client.infer(request)
                chunk = response["actions"]  # (10, 8)
                t_in_chunk = 0

            cmd = self._action_to_cmd(chunk[t_in_chunk])
            ok = move_robot(joint_motion_update=cmd)
            if not ok:
                return False
            t_in_chunk += 1
            self._node.get_clock().sleep_for(rclpy.duration.Duration(seconds=period))

        return True

    def _build_request(self, obs, task):
        arm = np.array(obs.joint_states.position[:6], dtype=np.float32)
        grip_norm = float(np.clip(
            obs.joint_states.position[6] / GRIPPER_MAX_METERS, 0.0, 1.0))
        return {
            "observation/exterior_image_1_left": self._img(obs.left_image),
            "observation/wrist_image_left":      self._img(obs.center_image),
            "observation/joint_position":        np.concatenate([arm, [grip_norm]]).astype(np.float32),
            "observation/gripper_position":      np.array([grip_norm], dtype=np.float32),
            "prompt": f"insert {task.cable_name} into {task.port_name}",
        }

    @staticmethod
    def _img(msg):
        img = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, 3)
        h, w = img.shape[:2]
        s = 224.0 / max(h, w)
        nh, nw = int(h * s), int(w * s)
        resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
        out = np.zeros((224, 224, 3), dtype=np.uint8)
        ph, pw = (224 - nh) // 2, (224 - nw) // 2
        out[ph:ph + nh, pw:pw + nw] = resized
        return out

    @staticmethod
    def _action_to_cmd(action):
        msg = JointMotionUpdate()
        point = JointTrajectoryPoint()
        point.velocities = [float(v) for v in np.clip(action[:6], -1.0, 1.0)]
        msg.target_state = point
        msg.trajectory_generation_mode.mode = TrajectoryGenerationMode.MODE_VELOCITY
        return msg
```

Package location: add under a new module, e.g. `aic_example_policies/aic_example_policies/ros/PiZero.py`, following the same layout as `WaveArm.py`.

Launch it:
```bash
./run_policy.sh aic_example_policies.ros.PiZero
```

---

## 11. Setup steps (high level)

When we actually implement this, the order will be:

1. Verify observation format with `ros2 topic echo /observations --once` in the running sim. Confirm: joint count = 7, position[6] ≈ 0.04 when idle, image resolution = 1152×1024.
2. Clone openpi in WSL2:
   ```bash
   cd ~
   git clone --recurse-submodules https://github.com/Physical-Intelligence/openpi.git
   cd openpi
   GIT_LFS_SKIP_SMUDGE=1 uv sync
   GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .
   ```
3. Download checkpoint:
   ```bash
   # via gsutil, or from HF mirror
   gsutil -m cp -r gs://openpi-assets/checkpoints/pi05_droid ~/openpi_checkpoints/pi05_droid
   ```
4. Launch server:
   ```bash
   uv run scripts/serve_policy.py policy:checkpoint \
     --policy.config=pi05_droid \
     --policy.dir=~/openpi_checkpoints/pi05_droid
   ```
   Expect startup time ~30-60s (model load). Server listens on `0.0.0.0:8000`.
5. Add PiZero.py to `aic_example_policies/` and `pip install -e` the package in the pixi env (or add to pixi deps).
6. Install `openpi-client` into the pixi env (lightweight — just websocket + msgpack-numpy).
7. Run:
   - Terminal 1: `./run_eval.sh ground_truth:=false start_aic_engine:=true`
   - Terminal 2: `uv run scripts/serve_policy.py ...` (openpi server)
   - Terminal 3: `./run_policy.sh aic_example_policies.ros.PiZero`

---

## 12. Known issues to expect

- **Zero-shot performance will be poor.** Pretraining data doesn't match AIC's camera geometry or the insertion task. The arm will move plausibly but won't finish the task.
- **Force blindness.** pi0 has no force/torque input. Cable insertion cues like "the plug is hitting the edge" are invisible to the model.
- **Camera mismatch.** All AIC cameras are end-effector-mounted; pi0 DROID expected one static exterior + one wrist. Feature maps will be out-of-distribution.
- **GPU contention.** The eval container uses the GPU for Gazebo rendering. pi0 needs ~8GB VRAM on top. Monitor with `nvidia-smi`.
- **Latency.** Each inference is ~50-100ms on a 4090. At 15 Hz with a chunk of 10, that's 10 * 66ms = 660ms of action before the next inference — fine. But if the GPU is busy with Gazebo, inference latency can spike and the 15 Hz loop misses ticks.

---

## 13. Path to fine-tuning

To get actually working behavior:

1. **Collect demonstrations.** Record successful cable-insertion trajectories via scripted / teleoperated policies. Save the observation stream (all 3 cameras + joint state + task description + wrist_wrench) and the corresponding `JointMotionUpdate` commands.
2. **Convert to LeRobot format.** openpi trains from LeRobot datasets. Write a conversion script mapping AIC trial recordings → LeRobot-format parquet + video.
3. **Write an AIC data config** in openpi. Model it on `examples/ur5/` but:
   - 3 cameras instead of 2 (or pick best 2 for initial experiment).
   - Optionally add force/torque into the state vector. This requires minor model surgery — the `state_proj` layer's input dim changes from 8 to 14.
4. **Fine-tune** from `pi0_base` weights:
   ```bash
   uv run scripts/train.py config:pi0_aic --num_train_steps 30000
   ```
5. **Swap the checkpoint** in the server launch command and re-run.

The force-feedback addition is optional for a first fine-tuning pass; could be validated on pure vision first and added if performance plateaus.

---

## 14. Open questions / to confirm before coding

- [ ] What is the actual URDF upper limit of `gripper/left_finger_joint`? Confirms `GRIPPER_MAX_METERS`.
- [ ] Does pi05_droid's server actually accept the DROID obs keys, or does it expect a different key schema? (Run `client.get_server_metadata()` to read off the required input spec.)
- [ ] Is `openpi-client` pip-installable or do we vendor a single websocket-client file? (Check openpi repo for a standalone client package.)
- [ ] Is there a newer pi-variant checkpoint better-suited to industrial tasks (e.g. a specific "insertion" fine-tune released by the openpi team)? — check the openpi repo's release notes.

---

## 15. Summary table (the mapping at a glance)

| AIC source | Transform | pi0.5 destination |
|---|---|---|
| `obs.joint_states.position[0:6]` | direct copy (radians) | `observation/joint_position[0:6]` |
| `obs.joint_states.position[6]` (meters) | `/ GRIPPER_MAX_METERS`, clip [0,1] | `observation/joint_position[6]` and `observation/gripper_position[0]` |
| `obs.center_image` | np.frombuffer + resize-pad to 224×224 | `observation/wrist_image_left` |
| `obs.left_image` | np.frombuffer + resize-pad to 224×224 | `observation/exterior_image_1_left` |
| `obs.right_image` | **discarded** | — |
| `obs.wrist_wrench` | **discarded** (no force input) | — |
| `obs.controller_state.*` | **discarded** | — |
| task description | `f"insert {task.cable_name} into {task.port_name}"` | `prompt` |
| `action[0:6]` (radians/sec) | `JointTrajectoryPoint.velocities[0:6]`, MODE_VELOCITY | `JointMotionUpdate` on aic_controller |
| `action[6]` | optional gripper handling (zero-shot: ignore) | — |
| `action[7]` | discarded | — |
