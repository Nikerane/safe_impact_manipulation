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
