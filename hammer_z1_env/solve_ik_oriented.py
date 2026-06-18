"""Orientation-aware DLS IK: place hammer_head_site at a target AND point the strike axis along a
desired world direction (default: straight down). Optionally lock joint1=0 for an in-plane arm.

Unlike solve_ik.py (position-only, 3-DoF task -> strike-axis orientation is incidental, ~7.4 deg
oblique), this solves position(3) + strike-axis-alignment(2) so the face strikes PERPENDICULAR to
the target surface (max impulse transfer). With --lock-joint1 the base yaw is fixed at 0 and the
WRIST absorbs the lateral face-centroid offset, giving a fully in-plane perpendicular pose.

Usage:
    python hammer_z1_env/solve_ik_oriented.py                 # joint1 free, vertical strike
    python hammer_z1_env/solve_ik_oriented.py --lock-joint1   # in-plane (joint1=0), vertical strike
    python hammer_z1_env/solve_ik_oriented.py --x 0.5 --y 0 --z 0.15 --lock-joint1

Prints the converged NEAR_NAIL_JOINT_POS dict (full precision) for src/.../z1_constants.py.
"""
import argparse
from pathlib import Path

import mujoco
import numpy as np

_DIR = Path(__file__).resolve().parent
XML = str(_DIR / "assets" / "z1_hammer_robot.xml")
MESH_DIR = _DIR / "assets" / "meshes"

ARM_JOINTS = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"]
# Seed from the current NEAR_NAIL_JOINT_POS (z1_constants.py) — a known near-nail face-down config.
SEED = np.array([-0.119749, 1.908621, -1.582969, 1.118154, 0.021694, 1.220820])

ap = argparse.ArgumentParser()
ap.add_argument("--x", type=float, default=0.50)
ap.add_argument("--y", type=float, default=0.00)
ap.add_argument("--z", type=float, default=0.15)
ap.add_argument("--strike-dir", type=float, nargs=3, default=[0.0, 0.0, -1.0],
                help="desired world direction for the strike axis (default: straight down)")
ap.add_argument("--lock-joint1", action="store_true", help="fix joint1=0 (in-plane arm)")
ap.add_argument("--max-iters", type=int, default=8000)
ap.add_argument("--alpha", type=float, default=0.5)
ap.add_argument("--damping", type=float, default=1e-3)
args = ap.parse_args()

target = np.array([args.x, args.y, args.z])
d_des = np.asarray(args.strike_dir, float); d_des /= np.linalg.norm(d_des)

assets = {f.name: f.read_bytes() for f in MESH_DIR.glob("*.stl")}
model = mujoco.MjModel.from_xml_path(XML, assets)
data = mujoco.MjData(model)

jid = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n) for n in ARM_JOINTS]
adr = [model.jnt_qposadr[j] for j in jid]
dof = [model.jnt_dofadr[j] for j in jid]
sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "hammer_head_site")
gj = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "jointGripper")

q = SEED.copy()
if args.lock_joint1:
  q[0] = 0.0
free = [1, 2, 3, 4, 5] if args.lock_joint1 else [0, 1, 2, 3, 4, 5]

# Identify the strike axis: the site-local axis most aligned with d_des in the seed pose.
for k, a in enumerate(adr):
  data.qpos[a] = q[k]
mujoco.mj_forward(model, data)
R0 = data.site_xmat[sid].reshape(3, 3)
strike_axis = int(np.argmax([abs(R0[:, k] @ d_des) for k in range(3)]))
sign = np.sign(R0[:, strike_axis] @ d_des)
print(f"[ik] strike axis = site-local #{strike_axis} (sign {sign:+.0f}); target {target}, dir {d_des}, "
      f"lock_joint1={args.lock_joint1}")

converged = False
for it in range(args.max_iters):
  for k, a in enumerate(adr):
    data.qpos[a] = q[k]
  mujoco.mj_forward(model, data)
  R = data.site_xmat[sid].reshape(3, 3)
  ax = sign * R[:, strike_axis]
  pe = target - data.site_xpos[sid]
  re = np.cross(ax, d_des)                       # align strike axis -> d_des (2 effective DoF)
  if np.linalg.norm(pe) < 1e-4 and np.linalg.norm(re) < 1e-3:
    converged = True
    break
  Jp = np.zeros((3, model.nv)); Jr = np.zeros((3, model.nv))
  mujoco.mj_jacSite(model, data, Jp, Jr, sid)
  J = np.vstack([np.column_stack([Jp[:, dof[k]] for k in free]),
                 np.column_stack([Jr[:, dof[k]] for k in free])])
  e = np.concatenate([pe, 0.6 * re])
  dq = J.T @ np.linalg.solve(J @ J.T + args.damping ** 2 * np.eye(6), e)
  for kk, k in enumerate(free):
    lo, hi = model.jnt_range[jid[k]]
    q[k] = np.clip(q[k] + args.alpha * dq[kk], lo, hi)

for k, a in enumerate(adr):
  data.qpos[a] = q[k]
mujoco.mj_forward(model, data)
R = data.site_xmat[sid].reshape(3, 3)
ax = sign * R[:, strike_axis]
tilt = np.degrees(np.arccos(np.clip(ax @ d_des, -1, 1)))
face = data.site_xpos[sid]
inlim = all(model.jnt_range[jid[k]][0] <= q[k] <= model.jnt_range[jid[k]][1] for k in range(6))

print(f"[ik] converged={converged}  pos residual {np.linalg.norm(target - face) * 100:.4f} cm  "
      f"strike-axis tilt {tilt:.3f} deg  face {face.round(4)}  all-in-limits={inlim}")
print("\nNEAR_NAIL_JOINT_POS = {")
for name, a in zip(ARM_JOINTS, adr):
  print(f'    "{name}": {data.qpos[a]: .8f},')
print(f'    "jointGripper": {data.qpos[model.jnt_qposadr[gj]]: .8f},')
print("}")
