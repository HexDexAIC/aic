# pi0.5 → AIC Implementation Runbook

Companion to `pi0_5_integration.md`. That doc explains the *what* and *why*; this one is a step-by-step checklist of the *how*, from empty WSL2 shell to a working end-to-end policy loop.

Target environment: WSL2 Ubuntu-24.04 with NVIDIA GPU passthrough (same environment that already runs the AIC eval container).

Each phase has: **Goal**, **Steps**, **Acceptance criteria**, **Failure modes**. Do not move to the next phase until acceptance criteria are met.

---

## Phase 0 — Prerequisites check (15 min)

**Goal:** Confirm the base environment has everything needed before we install anything.

### Steps

```bash
# 0.1 GPU accessible from WSL2
nvidia-smi
# Expect: table showing GPU name, driver, VRAM total and used

# 0.2 Available VRAM
nvidia-smi --query-gpu=memory.total,memory.used,memory.free --format=csv
# Expect: free >= 8000 MiB when eval container is NOT running

# 0.3 Python version
python3 --version
# Expect: 3.11 or later (openpi needs 3.11+)

# 0.4 Pixi working
pixi --version
# Expect: any version; confirms WaveArm setup is intact

# 0.5 Git LFS
git lfs version
# If missing: sudo apt-get install -y git-lfs && git lfs install
```

### Acceptance criteria
- `nvidia-smi` shows the GPU with at least 8 GB free VRAM.
- Python 3.11+ available.
- `git lfs` present.

### Failure modes
- **GPU not visible**: The Docker Desktop WSL integration must be enabled. If `nvidia-smi` fails, restart Docker Desktop → Settings → Resources → WSL Integration.
- **<8 GB VRAM free**: Shut down the eval container and re-measure. If still tight, plan on running inference with `--policy.precision=float16` or falling back to Windows-native PyTorch.

---

## Phase 1 — Install uv and clone openpi (10 min)

**Goal:** Get openpi installed in a standalone location, separate from the AIC workspace.

### Steps

```bash
# 1.1 Install uv (openpi's package manager — NOT pixi, NOT pip directly)
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env   # or restart shell
uv --version
# Expect: uv 0.x.y

# 1.2 Clone openpi
cd ~
git clone --recurse-submodules https://github.com/Physical-Intelligence/openpi.git
cd openpi

# 1.3 Sync dependencies (pinned via uv.lock)
GIT_LFS_SKIP_SMUDGE=1 uv sync
# Expect: ~2-5 min, downloads JAX + CUDA deps

# 1.4 Install openpi itself as editable
GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .
```

### Acceptance criteria
- `uv run python -c "import openpi; print(openpi.__file__)"` prints a path in `~/openpi/src/openpi/...`.
- `uv run python -c "import jax; print(jax.devices())"` lists a CUDA device, not just `[CpuDevice(id=0)]`.

### Failure modes
- **JAX shows only CPU**: Linux CUDA packages not installed. Run `uv pip install -U "jax[cuda12]"` (substitute cuda11 if your driver is older; check `nvidia-smi` top-right).
- **`uv sync` hangs on GitHub auth**: set `GIT_LFS_SKIP_SMUDGE=1` (as above); openpi ships large submodule assets via LFS which we do not need for inference.

---

## Phase 2 — Download the pi05_droid checkpoint (20–40 min, depends on bandwidth)

**Goal:** Get the pretrained weights onto disk.

### Steps

Option A — via `gsutil` (if you have gcloud set up):
```bash
# 2A.1 Check gcloud
gcloud --version || sudo snap install google-cloud-cli --classic

# 2A.2 Auth (browser popup)
gcloud auth login

# 2A.3 Download checkpoint
mkdir -p ~/openpi_checkpoints
gsutil -m cp -r gs://openpi-assets/checkpoints/pi05_droid ~/openpi_checkpoints/
# Expect: ~10-15 GB, 10-30 minutes
```

Option B — via the Hugging Face mirror (no gcloud needed):
```bash
# 2B.1 Install huggingface CLI
uv pip install -U "huggingface_hub[cli]"

# 2B.2 Find the pi05_droid HF repo (check openpi README for current URL)
# As of writing, it's at: https://huggingface.co/physical-intelligence/pi05_droid

# 2B.3 Download
huggingface-cli download physical-intelligence/pi05_droid \
  --local-dir ~/openpi_checkpoints/pi05_droid
```

### Acceptance criteria
- `ls ~/openpi_checkpoints/pi05_droid/` shows `params/`, `assets/`, and `config.json` (structure may vary).
- `du -sh ~/openpi_checkpoints/pi05_droid` is several GB, not a few MB (confirms no LFS skipping happened here).

### Failure modes
- **gsutil auth fails**: Fall back to option B. openpi maintains HF mirrors for the main checkpoints.
- **Download interrupted**: Both `gsutil -m cp` and `huggingface-cli download` are resumable — just rerun.

---

## Phase 3 — Server smoke test (15 min)

**Goal:** Launch the openpi server and confirm it answers inference requests. Do this BEFORE touching AIC — we want to know the openpi stack is healthy on its own.

### Steps

**Terminal A** — launch the server:
```bash
cd ~/openpi
uv run scripts/serve_policy.py policy:checkpoint \
  --policy.config=pi05_droid \
  --policy.dir=$HOME/openpi_checkpoints/pi05_droid
# Expect first-run: 30-90 seconds of model loading
# Then: "Websocket server started on ws://0.0.0.0:8000"
```

**Terminal B** — run the simple client:
```bash
cd ~/openpi

# 3.1 Verify server metadata (exact expected obs keys)
uv run python -c "
from openpi_client.websocket_client_policy import WebsocketClientPolicy
c = WebsocketClientPolicy('localhost', 8000)
print(c.get_server_metadata())
"
# Expect: a dict describing expected observation keys, action spec

# 3.2 Full inference smoke test with dummy data
uv run python examples/simple_client/main.py
# Expect: action chunks returned, no errors, inference_ms < 200
```

### Acceptance criteria
- Server log shows "Websocket server started".
- Client prints server metadata including the exact observation key names and shapes.
- `simple_client` completes at least 10 inference calls.
- `nvidia-smi` (in a third terminal) shows GPU memory allocated by a Python process.

### Write down the server metadata — it's the source of truth

The server's metadata is the **authoritative spec** for what to send. It supersedes anything in this doc or the other. Save it:

```bash
uv run python -c "
from openpi_client.websocket_client_policy import WebsocketClientPolicy
import json
c = WebsocketClientPolicy('localhost', 8000)
print(json.dumps(c.get_server_metadata(), indent=2, default=str))
" > ~/openpi_metadata.json
```

### Failure modes
- **Port 8000 already in use**: Pass `--server.port=8001` and adjust client.
- **Out-of-memory at model load**: Confirm eval container is OFF. If still OOM, try `--policy.precision=bfloat16` or `float16` (smaller flags may vary — check `--help`).
- **Client hangs on connect**: The server takes ~30s to be ready after log says "started." Add a `sleep 30` or use the health-check call in `simple_client`.

---

## Phase 4 — Verify AIC observation format empirically (20 min)

**Goal:** Before writing any translator code, confirm what the sim actually publishes matches the design doc.

### Steps

**Terminal A** — launch the eval:
```bash
cd ~/ws_aic
./run_eval.sh ground_truth:=false start_aic_engine:=false
# Note: start_aic_engine:=false so the trial doesn't end before we're ready
```

**Terminal B** — from the pixi env, inspect the topic:
```bash
cd ~/ws_aic/src/aic
pixi shell  # enters the ROS2 env
# (export Zenoh vars as in run_policy.sh if needed)

# 4.1 Confirm the topic exists
ros2 topic list | grep observ
# Expect: /observations

# 4.2 Publish rate
ros2 topic hz /observations
# Expect: ~20 Hz (let it run for 10s, Ctrl-C)

# 4.3 Dump one message to YAML
ros2 topic echo /observations --once > /tmp/sample_obs.yaml
# Expect: a big YAML file with left_image, joint_states, etc.

# 4.4 Inspect fields programmatically
ros2 topic echo /observations --once --field joint_states.position
# Expect: array of 7 floats

ros2 topic echo /observations --once --field left_image.height
ros2 topic echo /observations --once --field left_image.width
ros2 topic echo /observations --once --field left_image.encoding
# Expect: 1024, 1152, "rgb8" (or "bgr8" — verify which!)

ros2 topic echo /observations --once --field joint_states.name
# Expect: shoulder_pan_joint, shoulder_lift_joint, elbow_joint, wrist_1_joint,
#          wrist_2_joint, wrist_3_joint, gripper/left_finger_joint (order!)
```

### 4.5 Find the URDF gripper limit

```bash
grep -r "gripper" ~/ws_aic/src/aic/aic_description/urdf/ | grep -i "limit\|upper"
# Look for: <limit lower="..." upper="..."/> on the gripper joint
# Record: this is your GRIPPER_MAX_METERS (but note the adapter halves it — see integration doc)
```

### 4.6 Save a sample observation for testing

```bash
# In pixi shell:
python3 <<'EOF'
import rclpy
from rclpy.node import Node
from aic_model_interfaces.msg import Observation
import pickle

rclpy.init()
node = Node("obs_sniffer")
got = []
def cb(msg): got.append(msg)
sub = node.create_subscription(Observation, "observations", cb, 10)
while not got and rclpy.ok():
    rclpy.spin_once(node, timeout_sec=1.0)
with open("/tmp/sample_obs.pkl", "wb") as f:
    pickle.dump(got[0], f)
print("saved")
EOF
```

### Acceptance criteria
- `ros2 topic hz` reports ~20 Hz.
- Image encoding confirmed (probably `rgb8`; if it's `bgr8` we'll need to swap channels in the translator).
- Joint state has exactly 7 positions, in the expected order.
- URDF gripper max limit found and recorded.
- `/tmp/sample_obs.pkl` written — this is our fixture for Phase 6 unit testing.

### Failure modes
- **Topic not publishing**: AIC's observation stream requires the controllers to be active. If `start_aic_engine:=false` prevents publishing, use `true` but kill the engine process manually after observations start.
- **Image encoding is `bgr8`**: Update the translator to do `img[:, :, ::-1]` to convert BGR → RGB.

---

## Phase 5 — Install openpi-client into the pixi env (15 min)

**Goal:** Make `WebsocketClientPolicy` importable from the AIC policy code.

openpi ships the client as a small subpackage; we do NOT want to install all of openpi (JAX, etc.) into the pixi env — too heavy and the AIC Python version may differ from openpi's.

### Steps

```bash
# 5.1 Identify the client package
ls ~/openpi/packages/openpi-client
# Expect: a pyproject.toml, src/openpi_client/...

# 5.2 Install it into the pixi env
cd ~/ws_aic/src/aic
pixi run pip install ~/openpi/packages/openpi-client
# (if openpi's layout differs, adjust — look for pyproject.toml next to `openpi_client/`)

# 5.3 Verify import
pixi run python -c "from openpi_client.websocket_client_policy import WebsocketClientPolicy; print('ok')"
# Expect: ok

# 5.4 Install cv2 if not already present
pixi run python -c "import cv2; print(cv2.__version__)" || pixi run pip install opencv-python-headless
```

### Acceptance criteria
- Both imports succeed in the pixi env.
- The client can instantiate (don't need to connect yet):
  ```bash
  pixi run python -c "from openpi_client.websocket_client_policy import WebsocketClientPolicy; WebsocketClientPolicy('localhost', 8000); print('ok')"
  ```

### Failure modes
- **openpi-client not a separate package**: If openpi doesn't split the client out, copy the 2–3 relevant files into `aic_example_policies/aic_example_policies/` as a vendored dependency. Files needed: `websocket_client_policy.py`, `image_tools.py`, and their minimal imports.

---

## Phase 6 — Translator unit tests (30 min)

**Goal:** Build the AIC-to-pi0 observation translator as a pure function, tested against the Phase 4 fixture. No ROS, no websocket yet — just a function.

### Steps

Create `~/ws_aic/src/aic/aic_example_policies/aic_example_policies/ros/pi_translator.py`:

```python
import numpy as np
import cv2

GRIPPER_MAX_METERS = 0.08  # TODO: replace with URDF value from Phase 4.5


def ros_img_to_pi(msg, swap_bgr_rgb: bool = False) -> np.ndarray:
    img = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, 3)
    if swap_bgr_rgb:
        img = img[:, :, ::-1]
    h, w = img.shape[:2]
    s = 224.0 / max(h, w)
    nh, nw = int(h * s), int(w * s)
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
    out = np.zeros((224, 224, 3), dtype=np.uint8)
    ph, pw = (224 - nh) // 2, (224 - nw) // 2
    out[ph:ph + nh, pw:pw + nw] = resized
    return out


def aic_obs_to_pi_request(obs, task) -> dict:
    arm = np.array(obs.joint_states.position[:6], dtype=np.float32)
    grip_m = float(obs.joint_states.position[6])
    grip_norm = float(np.clip(grip_m / GRIPPER_MAX_METERS, 0.0, 1.0))
    return {
        "observation/exterior_image_1_left": ros_img_to_pi(obs.left_image),
        "observation/wrist_image_left":      ros_img_to_pi(obs.center_image),
        "observation/joint_position":        np.concatenate([arm, [grip_norm]]).astype(np.float32),
        "observation/gripper_position":      np.array([grip_norm], dtype=np.float32),
        "prompt": f"insert {task.cable_name} into {task.port_name}",
    }
```

Create `/tmp/test_translator.py`:

```python
import pickle, numpy as np
from aic_example_policies.ros.pi_translator import aic_obs_to_pi_request
from aic_task_interfaces.msg import Task

with open("/tmp/sample_obs.pkl", "rb") as f:
    obs = pickle.load(f)

task = Task()
task.cable_name = "usb_c_cable"
task.port_name = "port_a"

req = aic_obs_to_pi_request(obs, task)
for k, v in req.items():
    if isinstance(v, np.ndarray):
        print(f"{k}: shape={v.shape}, dtype={v.dtype}, range=[{v.min()}, {v.max()}]")
    else:
        print(f"{k}: {v!r}")
```

Run it:
```bash
cd ~/ws_aic/src/aic
pixi run python /tmp/test_translator.py
```

### Acceptance criteria
Output matches:
```
observation/exterior_image_1_left: shape=(224, 224, 3), dtype=uint8, range=[0, 255]
observation/wrist_image_left:      shape=(224, 224, 3), dtype=uint8, range=[0, 255]
observation/joint_position:        shape=(7,), dtype=float32, range=[-2, 2]   # approx
observation/gripper_position:      shape=(1,), dtype=float32, range=[0, 1]
prompt: 'insert usb_c_cable into port_a'
```

Then send it to the live server:
```bash
# Ensure Phase 3 server is still running in another terminal
pixi run python <<'EOF'
import pickle
from aic_example_policies.ros.pi_translator import aic_obs_to_pi_request
from aic_task_interfaces.msg import Task
from openpi_client.websocket_client_policy import WebsocketClientPolicy

with open("/tmp/sample_obs.pkl", "rb") as f:
    obs = pickle.load(f)
task = Task(); task.cable_name = "usb_c_cable"; task.port_name = "port_a"
req = aic_obs_to_pi_request(obs, task)

c = WebsocketClientPolicy("localhost", 8000)
r = c.infer(req)
print("actions shape:", r["actions"].shape)
print("inference_ms:", r.get("policy_timing", {}))
EOF
```

Expect: `actions shape: (10, 8)`, inference time ~50–200 ms.

### Failure modes
- **Server rejects request with missing key**: Compare against the metadata saved in Phase 3.3. The config may expect different key names (e.g. `observation/joint_position` vs just `joint_position`).
- **Image range wrong**: If output range isn't [0, 255] uint8, you probably resized before the copy-to-uint8 step.

---

## Phase 7 — Policy class implementation (45 min)

**Goal:** Wire the translator into the AIC `Policy` interface so `aic_model` can use it.

### Steps

### 7.1 Add the policy class

Create `~/ws_aic/src/aic/aic_example_policies/aic_example_policies/ros/PiZero.py`:

```python
import time
import numpy as np
import rclpy
from rclpy.duration import Duration

from aic_model.policy import Policy
from aic_control_interfaces.msg import JointMotionUpdate, TrajectoryGenerationMode
from trajectory_msgs.msg import JointTrajectoryPoint

from openpi_client.websocket_client_policy import WebsocketClientPolicy
from .pi_translator import aic_obs_to_pi_request

CHUNK = 10         # pi0 returns 10-step chunks
HZ = 15            # DROID default execution rate
SERVER_HOST = "localhost"
SERVER_PORT = 8000


class PiZero(Policy):
    def __init__(self, node):
        super().__init__(node)
        self._node = node
        self._client = WebsocketClientPolicy(SERVER_HOST, SERVER_PORT)
        self._node.get_logger().info(
            f"PiZero: connecting to openpi at {SERVER_HOST}:{SERVER_PORT}"
        )
        # Warm-up call to measure latency and fail-fast if the server is down
        meta = self._client.get_server_metadata()
        self._node.get_logger().info(f"PiZero: server metadata keys={list(meta.keys())}")

    def insert_cable(self, task, get_observation, move_robot, send_feedback):
        chunk = None
        t_in_chunk = 0
        period_s = 1.0 / HZ
        steps_taken = 0

        while rclpy.ok():
            obs = get_observation()
            if obs is None:
                self._node.get_clock().sleep_for(Duration(seconds=period_s))
                continue

            if chunk is None or t_in_chunk >= CHUNK:
                request = aic_obs_to_pi_request(obs, task)
                t0 = time.monotonic()
                response = self._client.infer(request)
                dt_ms = (time.monotonic() - t0) * 1000
                chunk = response["actions"]  # (10, 8)
                t_in_chunk = 0
                self._node.get_logger().debug(
                    f"PiZero: new chunk, infer={dt_ms:.1f}ms, step={steps_taken}"
                )

            action = chunk[t_in_chunk]  # (8,)
            arm_vel = np.clip(action[:6], -1.0, 1.0).astype(np.float64)

            cmd = JointMotionUpdate()
            point = JointTrajectoryPoint()
            point.velocities = arm_vel.tolist()
            cmd.target_state = point
            cmd.trajectory_generation_mode.mode = TrajectoryGenerationMode.MODE_VELOCITY

            ok = move_robot(joint_motion_update=cmd)
            if not ok:
                self._node.get_logger().error("PiZero: move_robot returned False")
                return False

            t_in_chunk += 1
            steps_taken += 1
            self._node.get_clock().sleep_for(Duration(seconds=period_s))

        return True
```

### 7.2 Rebuild the package

```bash
cd ~/ws_aic
pixi run colcon build --packages-select aic_example_policies --symlink-install
# --symlink-install means future edits don't require rebuild
source install/setup.bash  # or let pixi handle it
```

### Acceptance criteria
- No build errors.
- `pixi run python -c "from aic_example_policies.ros.PiZero import PiZero; print('ok')"` succeeds.
- Running `./run_policy.sh aic_example_policies.ros.PiZero` at least gets to "Loading policy module" and "Using policy: PiZero" (it will then hang waiting for the engine, which is fine for this check).

### Failure modes
- **Import of `aic_model.policy` fails**: That base class is where `Policy` is defined — read it to confirm the signature. Other example policies do this import too; match their pattern.
- **move_robot signature mismatch**: Open `aic_model/aic_model.py` and confirm `move_robot` accepts the kwarg `joint_motion_update`.

---

## Phase 8 — End-to-end zero-shot run (30 min)

**Goal:** Run the full loop and observe behavior.

### Steps

Three terminals.

**Terminal 1 — openpi server:**
```bash
cd ~/openpi
uv run scripts/serve_policy.py policy:checkpoint \
  --policy.config=pi05_droid \
  --policy.dir=$HOME/openpi_checkpoints/pi05_droid
```

Wait for `Websocket server started`.

**Terminal 2 — AIC eval:**
```bash
cd ~/ws_aic
./run_eval.sh ground_truth:=false start_aic_engine:=true
```

Wait for "AIC Engine initialized, waiting for aic_model".

**Terminal 3 — policy:**
```bash
cd ~/ws_aic
./run_policy.sh aic_example_policies.ros.PiZero
```

### What to watch

- Terminal 3 should print:
  - `PiZero: connecting to openpi at localhost:8000`
  - `PiZero: server metadata keys=[...]`
  - `on_configure(...)` from lifecycle (engine drives this)
  - `on_activate()` (if this doesn't appear, activation is still broken — revisit the Zenoh fixes)
  - `PiZero: new chunk, infer=XX.Xms, step=N` at roughly 1.5-second intervals (10 steps / 15 Hz ≈ 0.67 s... but with GPU contention it may be slower)
- Terminal 2 (eval) should NOT show "No node with name 'aic_model' found" anymore.
- Gazebo window: the arm should be moving — probably not toward the port, but moving non-trivially in response to images.

### Acceptance criteria
- Inference latency < 500 ms per call.
- Arm executes at least 50 steps without crashing.
- No Python exceptions in Terminal 3.

### Failure modes and fixes
- **`on_activate()` not called → lifecycle failure**: Same as the WaveArm debugging session; the Zenoh client-mode fix must be active in `run_policy.sh`.
- **Inference latency > 1s consistently**: GPU is contended with Gazebo. Options: lower Gazebo rendering quality, or move openpi server to the Windows host with PyTorch.
- **Arm doesn't move**: The action velocities from pi0 may be near zero for many steps; check `chunk[:, :6]` magnitudes in debug logs.
- **Arm moves violently**: The [-1, 1] clip may need to be tighter. Try `np.clip(action[:6], -0.3, 0.3)` as a first-pass safety limit.

---

## Phase 9 — Debugging aids (install as you go)

### 9.1 Record every inference round-trip

Add to `PiZero.insert_cable()`:
```python
import json, time
log_path = f"/tmp/pi_log_{int(time.time())}.jsonl"
self._log = open(log_path, "a")
# ... in the loop, after each infer:
self._log.write(json.dumps({
    "step": steps_taken,
    "infer_ms": dt_ms,
    "action": action.tolist(),
    "joint_position": obs.joint_states.position[:7],
}) + "\n")
self._log.flush()
```

### 9.2 Visualize what pi0 sees

Save the two 224×224 images to disk every N steps:
```python
import cv2
if steps_taken % 30 == 0:
    cv2.imwrite(f"/tmp/pi_ext_{steps_taken}.png", request["observation/exterior_image_1_left"][:, :, ::-1])
    cv2.imwrite(f"/tmp/pi_wrist_{steps_taken}.png", request["observation/wrist_image_left"][:, :, ::-1])
```
(Note: `cv2.imwrite` wants BGR, so swap channels.)

Then view from Windows: `\\wsl$\Ubuntu-24.04\tmp\pi_wrist_*.png` in any image viewer.

### 9.3 Quick action plot

```bash
pixi run python -c "
import json, numpy as np, matplotlib.pyplot as plt
data = [json.loads(l) for l in open('/tmp/pi_log_XXXX.jsonl')]
a = np.array([d['action'] for d in data])
plt.plot(a[:, :6]); plt.legend([f'j{i}' for i in range(6)]); plt.savefig('/tmp/actions.png')
"
```

---

## Phase 10 — When you hit the wall: fine-tuning path

Expect zero-shot to produce non-task-completing motion. When you want to actually solve the task:

### 10.1 Collect demonstrations

Use an existing scripted / ground-truth policy to generate successful cable-insertion trajectories. Record synchronously:
- Full `Observation` messages (3 cameras + joints + wrench)
- `JointMotionUpdate` commands issued
- Task description
- Success/failure label per trial

Target: 100–500 successful trajectories minimum for meaningful fine-tuning.

Save as rosbag (`ros2 bag record -a`) then convert.

### 10.2 Convert to LeRobot format

openpi trains from LeRobot-format datasets. Write a one-off converter:
```
rosbag → Parquet per episode:
  - frames[t].observation.images.exterior = left_image
  - frames[t].observation.images.wrist    = center_image
  - frames[t].observation.state           = joint_positions (7,)
  - frames[t].action                      = joint_velocities (7,)
  - episode.task = prompt string
```

Reference: `examples/ur5/convert_dataset.py` (or similar) in openpi.

### 10.3 Add an AIC training config

Clone `src/openpi/training/config.py` for an "aic_insertion" entry modeled on `pi0_ur5`:
- Point `repo_id` at your local LeRobot dataset
- Use `weight_loader=CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_base/params")` to start from pretrained weights
- `num_train_steps=20000` is typical for fine-tuning on ~100 trajectories

### 10.4 Train

```bash
cd ~/openpi
uv run scripts/train.py config:pi0_aic_insertion \
  --output_dir=$HOME/openpi_checkpoints/pi0_aic_v1
# Expect: several hours on a 4090
```

### 10.5 Serve the fine-tuned checkpoint

Identical to Phase 3 but point at your new checkpoint:
```bash
uv run scripts/serve_policy.py policy:checkpoint \
  --policy.config=pi0_aic_insertion \
  --policy.dir=$HOME/openpi_checkpoints/pi0_aic_v1
```

No code changes needed on the AIC side — same wrapper, different server checkpoint.

### 10.6 Stretch: add force input

If vision-only fine-tuning plateaus short of reliable insertion, the next step is architectural: modify the state encoder to accept `[joints(7), force(3), torque(3)] = 13D` instead of `7D`. This requires:
- Custom config class subclassing `Pi0Config` with `state_dim=13`
- Modified `UR5Inputs` to include wrench in `state`
- Retraining from scratch (can't reuse pretrained state projection)

Scope this separately; treat it as a distinct project.

---

## Summary — phase dependencies

```
0. Prereqs check
       │
       ▼
1. Install openpi ──► 2. Download checkpoint
       │                    │
       └─────── 3. Server smoke test
                    │
                    ▼
4. Verify AIC obs format
                    │
                    ▼
5. openpi-client in pixi ──► 6. Translator unit test (needs 3 + 4)
                                  │
                                  ▼
                             7. Policy class
                                  │
                                  ▼
                             8. End-to-end run ──► 9. Debug helpers
                                  │
                                  ▼
                             10. Fine-tuning (later, different project)
```

Phases 0–8 are a few hours of focused work. Phase 10 is a multi-day effort and depends on successfully recording demonstrations via AIC — which is itself a prerequisite worth scoping separately before starting it.

---

## Before starting: decisions to make

1. **Checkpoint source** — gsutil (requires gcloud auth) or HuggingFace (requires account)?
2. **Where GPU contention lives** — run openpi in WSL2 alongside Gazebo (simplest), or on Windows (if VRAM is tight)?
3. **Zero-shot first?** — do Phase 8 with the base checkpoint even though it won't solve the task, to validate the pipeline; or skip straight to collecting demos for fine-tuning?
4. **Gripper handling** — for cable insertion does the AIC task even use the gripper, or is the plug rigidly mounted? If rigid, we can fix `gripper_position = 0.5` constant and ignore `action[6]`.

Answer these four before Phase 1 — they shape what Phases 2, 8, and 10 look like.
