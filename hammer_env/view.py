"""
Opens the MuJoCo interactive viewer for the hammer-nail scene.

Usage:
    python hammer_env/view.py
    python hammer_env/view.py --no-weld   (manual joint tuning mode)
    python hammer_env/view.py --auto-place  (mocap places hammer face near nail)
    python hammer_env/view.py --dry-run   (initialize and print key positions)

Controls:
    Left-click + drag   : rotate camera
    Right-click + drag  : pan camera
    Scroll              : zoom
    Space               : pause / unpause
    Backspace           : reset
    Ctrl+A              : actuator sliders (drag nail_slide to test nail)
    F                   : contact forces
    Esc                 : exit
"""
import os
import argparse
import mujoco
import mujoco.viewer  # not auto-imported by 'import mujoco' in 2.3.x
import numpy as np

XML = os.path.join(os.path.dirname(__file__), "assets", "hammer_nail_scene.xml")

parser = argparse.ArgumentParser()
parser.add_argument(
    "--no-weld",
    action="store_true",
    help="Disable mocap weld equality so actuator sliders freely move the arm",
)
parser.add_argument(
    "--auto-place",
    action="store_true",
    help="Use mocap target to place the hammer face near the nail before launching",
)
parser.add_argument(
    "--dry-run",
    action="store_true",
    help="Initialize scene and print key positions without opening the viewer",
)
args = parser.parse_args()

m = mujoco.MjModel.from_xml_path(XML)
d = mujoco.MjData(m)

# Match the RL/reset convention: start from a neutral arm pose and make mocap weld
# consistent so the end-effector doesn't "snap" on the first steps.
neutral = np.array([0.803685, 0.773506, 0.0195586, -0.93819, -1.43596, 0.760135, 0.988909])
gripper_grasp_qpos = 0.004
hammer_hand_offset = np.array([0.0, 0.0, 0.105])
hammer_hand_quat_offset = np.array([np.cos(np.pi / 4), 0.0, np.sin(np.pi / 4), 0.0], dtype=np.float64)
arm_joints = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "joint7"]
for name, val in zip(arm_joints, neutral):
    jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, name)
    qadr = m.jnt_qposadr[jid]
    d.qpos[qadr] = float(val)
d.ctrl[:7] = neutral
for name in ("finger_joint1", "finger_joint2"):
    jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, name)
    qadr = m.jnt_qposadr[jid]
    d.qpos[qadr] = gripper_grasp_qpos
d.ctrl[-2:] = gripper_grasp_qpos

# Either disable the weld (manual tuning), or reset it to a clean target (RL-like).
if m.eq_data is not None:
    for i in range(m.eq_data.shape[0]):
        if m.eq_type[i] == mujoco.mjtEq.mjEQ_WELD:
            if args.no_weld:
                m.eq_active[i] = 0
            else:
                d0 = np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
                m.eq_data[i, 3:10] = d0

mujoco.mj_forward(m, d)


# Quaternion helpers (MuJoCo uses [w, x, y, z]).
def quat_mul(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        dtype=np.float64,
    )


def set_hammer_pose_from_hand():
    hand_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "hand")
    hand_pos = d.xpos[hand_id].copy()
    hand_xmat = d.xmat[hand_id].reshape(3, 3)
    hand_quat = d.xquat[hand_id].copy()
    hammer_quat = quat_mul(hand_quat, hammer_hand_quat_offset)

    hammer_jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "hammer_freejoint")
    qadr = m.jnt_qposadr[hammer_jid]
    vadr = m.jnt_dofadr[hammer_jid]
    d.qpos[qadr:qadr + 3] = hand_pos + hand_xmat @ hammer_hand_offset
    d.qpos[qadr + 3:qadr + 7] = hammer_quat
    d.qvel[vadr:vadr + 6] = 0.0
    mujoco.mj_forward(m, d)


set_hammer_pose_from_hand()
for _ in range(10):
    d.ctrl[-2:] = gripper_grasp_qpos
    mujoco.mj_step(m, d)
d.qvel[:] = 0.0


if not args.no_weld:
    # Keep the mocap target at the current EE pose by default, so the weld does not
    # pull the arm away from the neutral joint configuration.
    mocap_body_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "panda_mocap")
    mid = m.body_mocapid[mocap_body_id]
    ee_site_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "ee_center_site")
    site_mat = d.site_xmat[ee_site_id].reshape(9, 1)
    quat = np.empty(4)
    mujoco.mju_mat2Quat(quat, site_mat)
    d.mocap_pos[mid] = d.site_xpos[ee_site_id].copy()
    d.mocap_quat[mid] = quat

if args.auto_place and not args.no_weld:
    # Put the hammer head face near the nail by moving the mocap target.
    mocap_body_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "panda_mocap")
    mid = m.body_mocapid[mocap_body_id]
    ee_site_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "ee_center_site")
    site_mat = d.site_xmat[ee_site_id].reshape(9, 1)
    quat = np.empty(4)
    mujoco.mju_mat2Quat(quat, site_mat)

    # Rotate about EE local +X so the striking face is oriented downward.
    q_xm90 = np.array([np.cos(np.pi / 4), -np.sin(np.pi / 4), 0.0, 0.0], dtype=np.float64)
    d.mocap_quat[mid] = quat_mul(quat, q_xm90)

    nail_site_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "nail_top")
    head_site_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "hammer_head_site")

    nail_pos = d.site_xpos[nail_site_id].copy()

    # Iterate to make the *actual* hammer face land above the nail.
    head_half_sizes = np.array([0.025, 0.04, 0.025], dtype=np.float64)
    clearance = 0.001
    down = np.array([0.0, 0.0, -1.0], dtype=np.float64)

    for _ in range(120):
        ee_pos = d.site_xpos[ee_site_id].copy()
        head_pos = d.site_xpos[head_site_id].copy()
        ee_to_head_offset = head_pos - ee_pos

        head_xmat = d.site_xmat[head_site_id].reshape(3, 3)
        axes = [head_xmat[:, 0], head_xmat[:, 1], head_xmat[:, 2]]

        best_axis = 0
        best_sign = 1.0
        best_dot = -1e9
        for ai, a in enumerate(axes):
            for s in (1.0, -1.0):
                dot = float(np.dot(s * a, down))
                if dot > best_dot:
                    best_dot = dot
                    best_axis = ai
                    best_sign = s
        face_normal = best_sign * axes[best_axis]

        half_extent = float(head_half_sizes[best_axis])
        desired_head_center = nail_pos - face_normal * (half_extent + clearance)
        target_ee = desired_head_center - ee_to_head_offset
        d.mocap_pos[mid] = target_ee

        mujoco.mj_step(m, d)

        head_pos2 = d.site_xpos[head_site_id].copy()
        if np.linalg.norm(head_pos2 - desired_head_center) < 0.002:
            break

print("Launching viewer — close the window to exit.")
print(f"Scene: {XML}")
print("Tip: Ctrl+A → actuator sliders → move actuator1..7")
if args.no_weld:
    print("Manual mode: mocap weld disabled; actuator sliders move EE freely.")
elif args.auto_place:
    print("Auto-place mode: mocap moved hammer face near the nail before launch.")
else:
    print("Neutral mode: showing neutral joint pose; mocap is only held at current EE.")

if args.dry_run:
    head_site_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "hammer_head_site")
    handle_site_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "hammer_handle_site")
    nail_site_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "nail_top")
    print(f"hammer_head_site   {d.site_xpos[head_site_id].round(4)}")
    print(f"hammer_handle_site {d.site_xpos[handle_site_id].round(4)}")
    print(f"nail_top           {d.site_xpos[nail_site_id].round(4)}")
    raise SystemExit(0)

# launch() is blocking: runs both GUI and physics loop until window closes
mujoco.viewer.launch(m, d)
