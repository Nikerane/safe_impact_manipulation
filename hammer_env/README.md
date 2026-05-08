# hammer_env

Franka Panda robot with a free hammer object that starts pre-grasped between the fingers. Task: use RL to drive a nail into a wooden block.

## Setup

```bash
conda activate franka_arm
cd ~/repos/safe_impact_manipulation
```

## Scene

```
robot base (origin)
  └── 7-DOF arm
        └── hand + two fingers

hammer  free body with freejoint, starts between fingers
        ├── handle     capsule, 22 cm, brown
        └── head       box 8×5×5 cm, grey

block  pos=[0.55, 0, 0.40]  fixed to world
nail   pos=[0.55, 0, 0.46]  slide joint along -Z, range 0–7.5 cm
```

**qpos:** 17 total — 7 arm + 2 fingers + 7 hammer freejoint + 1 nail slide  
**nv:** 16 total — 7 arm + 2 fingers + 6 hammer freejoint + 1 nail slide  
**Actuators:** 9 (arm + fingers; nail is passive — driven by contact only)

## Scripts

| Script | Purpose |
|--------|---------|
| `python hammer_env/scene_info.py` | Print joints, sites, body positions |
| `python hammer_env/view.py` | Open interactive 3D viewer |

### Viewer controls

| Key / Mouse | Action |
|-------------|--------|
| Left-drag | Rotate camera |
| Right-drag | Pan camera |
| Scroll | Zoom |
| Space | Pause / unpause |
| Backspace | Reset simulation |
| Ctrl+A | Toggle actuator sliders (drag `nail_slide` to test nail) |
| F | Toggle contact forces |

## Files

```
hammer_env/
  assets/
    panda_mocap_hammer.xml   Franka robot + free hammer body
    hammer_nail_scene.xml    Full scene (includes robot, adds floor/block/nail)
  env.py                     FrankaHammerEnv (gymnasium GoalEnv)
  __init__.py                gym.register calls
  scene_info.py              Print scene summary
  view.py                    Interactive viewer
```

## Mesh path

Meshes are vendored locally so the env does not depend on sibling repos:

```
hammer_env/assets/meshes/
```

## Gym environment (`env.py`)

`env.py` wraps the scene as a `gymnasium_robotics` `MujocoRobotEnv` (GoalEnv-style):

- **Action space:** 3D hammer-head/tool position delta (x, y, z), range ±1
- **Observation:** ee position/velocity + hammer head position/velocity + nail top position + nail depth (16 dims)
- **achieved_goal:** nail top 3D position
- **desired_goal:** nail top position after being driven 7.5 cm down
- **Reward (dense):** −‖achieved − desired‖; **(sparse):** −1 until nail is within 2 cm of target

```python
import hammer_env  # registers FrankaHammer-v0
import gymnasium as gym
env = gym.make("FrankaHammer-v0")
obs, info = env.reset()
obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
```

## Nail & block physics: how the nail gets driven in

### Scene geometry

```
block  fixed body, no joint — it never moves
nail   slide joint along -Z, range 0–7.5 cm (0 = flush with block surface, 7.5 cm = fully driven)
```

The nail starts with its tip at the block surface. Each hammer blow transfers an impulse to the nail head; the slide joint converts that into downward motion. The nail is fully passive — it has no actuator, it moves only when contact forces from the hammer exceed the resistance in the joint.

### `friction` vs `frictionloss` — what they are and why both exist

| Parameter | Where it lives | What it models |
|-----------|---------------|----------------|
| `friction="1.5 0.01 0.001"` | `<geom>` on the block | Contact friction between surfaces (block ↔ hammer, block ↔ nail shaft) |
| `frictionloss="0.3"` | `<joint>` on the nail slide | Dry Coulomb friction inside the joint DOF itself (wood gripping the nail) |

They live at different layers of MuJoCo's physics:

- **`friction` is a contact property.** When two geoms touch, MuJoCo pairs their friction values (usually taking the geometric mean) to compute tangential contact forces. The three numbers are `[slide, torsional, rolling]`. The block's `slide=1.5` is deliberately high — it means the wood surface strongly resists any object slipping sideways across it. This stops the hammer head from skating off the block on impact and gives the nail shaft realistic grip from the wood walls as it penetrates.

- **`frictionloss` is a joint property.** It is a constant Coulomb (dry-friction) force that always opposes motion along the joint axis, regardless of contact. Think of it as the wood fibers clamping onto the nail shaft: until the net axial force exceeds `frictionloss`, the nail does not move at all. Once it does move, this force continues to resist, so small taps don't drive the nail far.

### How a single hammer blow plays out

```
1. Hammer head contacts nail_head geom
        │
        ▼ contact impulse along -Z
2. MuJoCo noslip solver resolves the impulse
        │
        ▼ net force on nail body projected onto joint axis
3. frictionloss (0.3 N) checked — nail moves only if force exceeds threshold
        │
        ▼ nail accelerates downward along slide joint
4. damping (8 N·s/m) bleeds velocity — nail decelerates and stops
        │
        ▼ nail position increments; frictionloss holds it at new depth
5. Next blow repeats from new resting depth
```

### Why these specific values

| Value | Reasoning |
|-------|-----------|
| `block friction slide = 1.5` | Wood-on-metal contact; higher than typical (≈0.5–0.8) to prevent the hammer skating sideways on impact |
| `frictionloss = 0.3` | Represents wood grain resistance clamping the nail; large enough that gravity alone (nail mass ≈ 7 g, ~0.07 N) cannot drive the nail, but one firm hammer blow can |
| `damping = 8` | Viscous drag that prevents the nail from oscillating or bouncing after impact, mimicking the energy absorbed by wood fibers |

### What "nail driven in" looks like in the joint

The nail's `qpos` value for `nail_slide` goes from `0` (flush) toward `0.075` (7.5 cm deep). The task is complete when this value reaches the target depth, which maps directly to the `achieved_goal` / `desired_goal` positions in the gym env.

---

## End-effector control: mocap + weld

This env uses the same EE control design as `panda_mujoco_gym`.

### How it works

```
PPO output action (Δx, Δy, Δz)
        │
        ▼
  _set_action():
    desired_head = current_hammer_head + action * action_scale
    desired_ee = desired_head - (current_hammer_head - current_ee)
    set mocap_pos[panda_mocap] = desired_ee    ← move the invisible EE handle
        │
        ▼
  MuJoCo constraint solver:
    <weld body1="panda_mocap" body2="ee_center_body"/>
    → forces ee_center_body to follow panda_mocap
        │
        ▼
  Joint actuators (position servos actuator1..7):
    drive joint angles until EE reaches the weld target
```

### Key components

| Component | File | What it does |
|-----------|------|--------------|
| `panda_mocap` body | `panda_mocap_hammer.xml` | Invisible mocap handle; you set its XYZ each step |
| `<weld>` equality | `panda_mocap_hammer.xml` | Constrains `ee_center_body` to rigidly follow `panda_mocap` |
| `actuator1..7` | `panda_mocap_hammer.xml` | Position servos that drive the arm to satisfy the weld |
| `hammer_freejoint` | `panda_mocap_hammer.xml` | Makes the hammer a physical free object held by finger contact |
| `set_mocap_pose()` | `env.py` | Sets `mocap_pos` + `mocap_quat` for `panda_mocap` |
| `_set_action()` | `env.py` | Adds delta, clamps, calls `set_mocap_pose()` |
| `reset_mocap_welds()` | `env.py` | Zeros the weld's stored relative pose at reset (prevents snap-back) |

### Why not just use joint control or IK?
- No IK code needed; MuJoCo's constraint solver handles the arm kinematics internally.
- EE-space actions are easier to learn with PPO than raw joint angles or torques.
- This is standard practice in MuJoCo manipulation benchmarks (Fetch, panda_mujoco_gym, etc.).

### Difference to `panda_mujoco_gym`
`panda_mujoco_gym` is the upstream reference this env is modeled after.

| | `panda_mujoco_gym` | `hammer_env` |
|---|---|---|
| Task | Push / Slide / Pick-and-place | Hammer a nail |
| Action | 3D EE delta (+ gripper) | 3D hammer-head delta |
| Control | mocap + weld | mocap + weld for arm; physical finger contact holds hammer |
| Goal | Move object to target XYZ | Drive nail 7.5 cm into block |
| Reset | Neutral joints + mocap at EE pose | Neutral joints + free hammer pre-grasped between fingers |
