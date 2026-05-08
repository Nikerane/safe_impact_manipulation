# hammer_z1_env

Unitree Z1 arm with a rigidly-attached hammer. Task: use RL to drive a nail into a wooden block.

This is the **Z1 variant** of `hammer_env` (Franka Panda). The robot assets live here; the mjlab task environment lives in `unitree_rl_mjlab/src/tasks/hammer/`.

---

## Scene

```
robot base (origin)
  └── Z1 6-DOF arm
        └── ee_center_body
              └── hammer (rigidly attached)
                    ├── handle   capsule, 22 cm, brown
                    └── head     box 5×2×2.5 cm, grey

block  pos=[0.50, 0, 0.40]   fixed to world
nail   pos=[0.50, 0, 0.46]   slide joint along -Z, range 0–7.5 cm
```

**nq / nv:** 7 (arm) + 1 (gripper) = 8 DOF  
The hammer is fixed to the EE — no freejoint, no grasp needed.  
The nail is passive — driven by contact forces only.

---

## Files

```
hammer_z1_env/
  assets/
    z1_hammer_robot.xml      Z1 arm + attached hammer (no <actuator> block)
    z1_mocap_hammer.xml      Same robot with mocap body + weld (for standalone Gym env)
    nail_block_scene.xml     Wooden block + sliding nail (standalone, no robot)
    hammer_nail_scene.xml    Full scene combining robot + block + nail (legacy Gym scene)
    meshes/                  STL mesh files vendored from mujoco_menagerie
```

---

## mjlab integration

The task is implemented in `unitree_rl_mjlab` as a `ManagerBasedRlEnv`:

```
unitree_rl_mjlab/src/
  assets/robots/unitree_z1/
    z1_constants.py         EntityCfg, BuiltinPositionActuatorCfg, joint/site names
  tasks/hammer/
    hammer_env_cfg.py       Robot-agnostic base ManagerBasedRlEnvCfg
    nail_block.py           EntityCfg for the nail+block scene
    mdp/                    observations.py, rewards.py, terminations.py
    config/z1/
      env_cfgs.py           Z1-specific wiring (IK action, site names)
      rl_cfg.py             PPO hyperparameters
    rl/runner.py            HammerOnPolicyRunner
```

Quick test:

```bash
conda activate unitree_mjlab
cd ~/repos/unitree_rl_mjlab
python -c "
from src.tasks.hammer.config.z1.env_cfgs import z1_hammer_env_cfg
from mjlab.envs import ManagerBasedRlEnv
import torch

cfg = z1_hammer_env_cfg()
cfg.scene.num_envs = 1
env = ManagerBasedRlEnv(cfg, device='cpu')
obs, _ = env.reset()
print('obs shape:', obs['actor'].shape)   # (1, 33)
obs, reward, done, trunc, info = env.step(torch.zeros(1, 3))
print('reward:', reward)
env.close()
"
```

---

## Current control design — DifferentialIK

The current mjlab task uses `DifferentialIKActionCfg`:

```
PPO output action (Δx, Δy, Δz)   3D EE-space delta
        │
        ▼
DifferentialIKAction (mjlab built-in)
  - Jacobian-based IK at each step
  - Converts Δpos at hammer_head_site → Δq for joints 1-6
        │
        ▼
BuiltinPositionActuatorCfg (PD servos)
  - joint1, 3-6: stiffness=1000, damping=100, effort=30 N
  - joint2:      stiffness=1500, damping=150, effort=60 N
  - drives joint angles to IK solution each step
```

The robot XML (`z1_hammer_robot.xml`) has **no `<actuator>` block** — mjlab adds `<position>` actuators programmatically from the `BuiltinPositionActuatorCfg` definitions. This is required for `DifferentialIKActionCfg` to resolve joint IDs at construction time (XmlActuator wrappers are not discoverable via `find_joints_by_actuator_names`).

### Why DifferentialIK and not raw joint control?

- EE-space actions are easier to learn with PPO than raw joint angles.
- Consistent with the Franka `hammer_env` approach (both use task-space deltas).
- Simple to implement with mjlab's built-in action term.

---

## Future direction — Impedance control

> **Note:** DifferentialIK + PD servos is a reasonable baseline but may not be the right long-term choice for an impact task. Impact manipulation (hammering) involves intentional high-force transient contacts that stiff position servos handle poorly — they will fight the contact rather than absorb it.
>
> A better fit is **Cartesian impedance control**, where the arm behaves like a spring-damper in task space:
>
> ```
> F_cmd = K_p * (x_des - x) + K_d * (ẋ_des - ẋ)
> τ_cmd = J^T * F_cmd                              (joint-space torques via Jacobian transpose)
> ```
>
> **Why impedance is better for hammering:**
> - Impedance control is inherently compliant — on impact, the arm yields rather than resisting, which prevents joint torque spikes and is safer.
> - The stiffness K_p and damping K_d become tunable RL parameters or curriculum knobs, not just gains for a position loop.
> - Real Z1 deployment likely requires torque-mode control anyway (the hardware API supports this).
>
> **How to switch:**
> - Replace `DifferentialIKActionCfg` with a `CartesianImpedanceActionCfg` (would need to be written or found in mjlab).
> - Alternatively, use `BuiltinMotorActuatorCfg` (torque mode) with a task-space PD computed in the reward/action shaping layer.
> - The XML and EntityCfg infrastructure stays the same — only the action term and actuator configs change.
>
> This is the recommended next step once the DifferentialIK baseline is trained and evaluated.
