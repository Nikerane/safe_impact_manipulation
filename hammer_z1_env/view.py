"""
Open the MuJoCo viewer for the Z1 hammer-nail scene.
Opens the scene with the hammer rigidly mounted to the gripper.

Usage:
    python hammer_z1_env/view.py
    python hammer_z1_env/view.py --no-weld   (manual joint tuning mode)
    python hammer_z1_env/view.py --dry-run   (initialize and print key positions)

Tip: Ctrl+A opens actuator sliders. Use --no-weld for manual arm tuning.
"""
import os
import argparse
import mujoco
import mujoco.viewer
import numpy as np

XML = os.path.join(os.path.dirname(__file__), "assets", "hammer_nail_scene.xml")

parser = argparse.ArgumentParser()
parser.add_argument(
    "--no-weld",
    action="store_true",
    help="Disable mocap weld equality so actuator sliders freely move the Z1 arm",
)
parser.add_argument(
    "--dry-run",
    action="store_true",
    help="Initialize scene and print key positions without opening the viewer",
)
args = parser.parse_args()

m = mujoco.MjModel.from_xml_path(XML)
d = mujoco.MjData(m)

# Same neutral pose as env.py (radians; MuJoCo XML uses `<compiler angle="radian" ...>`).
neutral = np.array(
    [-0.000297861, 1.30495, -1.54711, 0.197456, 0.000311167, 1.57079632679, -0.000964725]
)
arm_joints = [f"joint{i}" for i in range(1, 7)] + ["jointGripper"]
for name, val in zip(arm_joints, neutral):
    jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, name)
    d.qpos[m.jnt_qposadr[jid]] = val
d.ctrl[:7] = neutral

# Gripper stays closed for the mounted-tool baseline.
gj = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "jointGripper")
d.qpos[m.jnt_qposadr[gj]] = 0.0
d.ctrl[6] = 0.0

if m.eq_data is not None:
    for i in range(m.eq_data.shape[0]):
        if m.eq_type[i] == mujoco.mjtEq.mjEQ_WELD:
            if args.no_weld:
                m.eq_active[i] = 0
            else:
                m.eq_data[i, 3:10] = np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0])

mujoco.mj_forward(m, d)

ee_sid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "ee_center_site")
mujoco.mj_forward(m, d)

if not args.no_weld:
    # Sync mocap to current EE pose so the weld doesn't snap.
    mocap_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "z1_mocap")
    mid = m.body_mocapid[mocap_id]
    d.mocap_pos[mid] = d.site_xpos[ee_sid].copy()
    ee_mat = d.site_xmat[ee_sid].reshape(9, 1)
    ee_quat = np.empty(4)
    mujoco.mju_mat2Quat(ee_quat, ee_mat)
    d.mocap_quat[mid] = ee_quat

print(f"Scene: {XML}")
print(f"EE world pos: {d.site_xpos[ee_sid].round(4).tolist()}")
print("Tip: Ctrl+A opens actuator sliders.")
if args.no_weld:
    print("Manual mode: mocap weld disabled; actuator sliders move EE freely.")

if args.dry_run:
    head_sid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "hammer_head_site")
    handle_sid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "hammer_handle_site")
    nail_sid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "nail_top")
    print(f"hammer_head_site   {d.site_xpos[head_sid].round(4)}")
    print(f"hammer_handle_site {d.site_xpos[handle_sid].round(4)}")
    print(f"nail_top           {d.site_xpos[nail_sid].round(4)}")
    raise SystemExit(0)

mujoco.viewer.launch(m, d)
