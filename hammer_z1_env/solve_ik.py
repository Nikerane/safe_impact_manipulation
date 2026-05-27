"""Damped-least-squares IK: find Z1 joint angles that place hammer_head_site
at a given world position.

Usage:
    python hammer_z1_env/solve_ik.py                   # default target
    python hammer_z1_env/solve_ik.py --z 0.227         # custom z height
    python hammer_z1_env/solve_ik.py --x 0.5 --y 0 --z 0.227
"""
import argparse
import os
from pathlib import Path

import mujoco
import numpy as np

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_DIR = Path(__file__).resolve().parent
XML = str(_DIR / "assets" / "z1_hammer_robot.xml")
MESH_DIR = _DIR / "assets" / "meshes"

# ---------------------------------------------------------------------------
# Initial guess — start from the previous NEAR_NAIL solution.
# ---------------------------------------------------------------------------
INIT_QPOS = np.array([
    -0.01727037,   # joint1
     1.82525484,   # joint2
    -1.03479732,   # joint3
     0.38712980,   # joint4
    -0.00852019,   # joint5
     1.56918241,   # joint6
    -0.00096473,   # jointGripper
])

ARM_JOINTS = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"]

# ---------------------------------------------------------------------------
# Parse args
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--x", type=float, default=0.50)
parser.add_argument("--y", type=float, default=0.00)
parser.add_argument("--z", type=float, default=0.227)
parser.add_argument("--max-iters", type=int, default=2000)
parser.add_argument("--tol", type=float, default=1e-4, help="convergence tolerance (m)")
parser.add_argument("--damping", type=float, default=1e-3)
args = parser.parse_args()

target = np.array([args.x, args.y, args.z])

# ---------------------------------------------------------------------------
# Load model with mesh assets
# ---------------------------------------------------------------------------
assets = {f.name: f.read_bytes() for f in MESH_DIR.glob("*.stl")}
model = mujoco.MjModel.from_xml_path(XML, assets)
data  = mujoco.MjData(model)

# Map joint names → qpos indices
joint_ids  = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n) for n in ARM_JOINTS]
qpos_addrs = [model.jnt_qposadr[jid] for jid in joint_ids]
n_arm      = len(ARM_JOINTS)

site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "hammer_head_site")
assert site_id >= 0, "hammer_head_site not found"

# ---------------------------------------------------------------------------
# Set initial joint configuration
# ---------------------------------------------------------------------------
for i, addr in enumerate(qpos_addrs):
    data.qpos[addr] = INIT_QPOS[i]

# jointGripper
gj_id   = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "jointGripper")
data.qpos[model.jnt_qposadr[gj_id]] = INIT_QPOS[6]

mujoco.mj_forward(model, data)

# ---------------------------------------------------------------------------
# DLS IK loop (position only, 3-DOF task on 6-DOF arm)
# ---------------------------------------------------------------------------
alpha   = 0.5          # step size
damping = args.damping

for i in range(args.max_iters):
    mujoco.mj_forward(model, data)
    err = target - data.site_xpos[site_id]
    if np.linalg.norm(err) < args.tol:
        print(f"Converged in {i} iterations  |  residual = {np.linalg.norm(err)*100:.3f} cm")
        break

    # Full Jacobian (3 x nv), then extract arm columns
    jac_full = np.zeros((3, model.nv))
    mujoco.mj_jacSite(model, data, jac_full, None, site_id)
    J = np.column_stack([jac_full[:, model.jnt_dofadr[jid]] for jid in joint_ids])

    # Damped least squares
    dq = J.T @ np.linalg.solve(J @ J.T + damping**2 * np.eye(3), err)
    for k, addr in enumerate(qpos_addrs):
        data.qpos[addr] += alpha * dq[k]

    # Clip to joint limits
    for k, jid in enumerate(joint_ids):
        lo, hi = model.jnt_range[jid]
        data.qpos[qpos_addrs[k]] = np.clip(data.qpos[qpos_addrs[k]], lo, hi)
else:
    print(f"Did not converge  |  residual = {np.linalg.norm(err)*100:.3f} cm")

mujoco.mj_forward(model, data)
final_pos  = data.site_xpos[site_id]
final_err  = np.linalg.norm(target - final_pos) * 100  # cm

print(f"\nTarget          : {target}")
print(f"hammer_head_site: {final_pos.round(4)}")
print(f"Residual error  : {final_err:.3f} cm")

# nail_top world z for reference
nail_id  = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "nail_top")
# nail_top not in z1_hammer_robot.xml (no scene); report z clearance from target
print(f"nail_top z (scene): 0.102 m  →  clearance at striking face: "
      f"{(final_pos[2] - 0.025 - 0.102)*100:.1f} cm")

print("\nNEAR_NAIL_JOINT_POS = {")
for name, addr in zip(ARM_JOINTS + ["jointGripper"],
                       qpos_addrs + [model.jnt_qposadr[gj_id]]):
    print(f'    "{name}": {data.qpos[addr]: .8f},')
print("}")
