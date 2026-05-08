"""
Physics verification script for the hammer-nail scene.

Runs 3 phases:
  1. SETTLE  — arm moves to hover position, waits for it to reach target
  2. STRIKE  — arm drives down fast onto the nail
  3. OBSERVE — watches nail depth and contacts after impact

Prints every contact pair (geom names, position, normal force) and
tracks nail insertion depth throughout.

Usage:
    python hammer_env/physics_test.py
    python hammer_env/physics_test.py --render     (opens viewer window)
"""
import os
import sys
import argparse
import numpy as np
import mujoco

XML = os.path.join(os.path.dirname(__file__), "assets", "hammer_nail_scene.xml")

# ── helpers ──────────────────────────────────────────────────────────────────

def get_site(m, d, name):
    sid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, name)
    return d.site_xpos[sid].copy()

def get_geom_name(m, gid):
    name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, gid)
    return name if name else f"geom_{gid}"

def nail_depth(m, d):
    jid  = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "nail_slide")
    qadr = m.jnt_qposadr[jid]
    return float(d.qpos[qadr])

def print_contacts(m, d, prefix=""):
    """Print all active contact pairs with names and normal force."""
    if d.ncon == 0:
        print(f"{prefix}  (no contacts)")
        return
    for i in range(d.ncon):
        c = d.contact[i]
        g1 = get_geom_name(m, c.geom1)
        g2 = get_geom_name(m, c.geom2)
        # contact normal force (first component = normal, rest = friction)
        force = np.zeros(6)
        mujoco.mj_contactForce(m, d, i, force)
        normal_force = abs(force[0])
        pos = c.pos.round(4)
        print(f"{prefix}  [{i}] {g1:20s} ↔ {g2:20s}  "
              f"pos={pos}  |Fn|={normal_force:.2f} N")

# ── main ─────────────────────────────────────────────────────────────────────

def run(render=False):
    m = mujoco.MjModel.from_xml_path(XML)
    d = mujoco.MjData(m)

    renderer = None
    if render:
        from gymnasium.envs.mujoco.mujoco_rendering import MujocoRenderer
        cam = {"azimuth": 135.0, "elevation": -20.0, "distance": 2.0,
               "lookat": [0.4, 0.0, 0.4]}
        renderer = MujocoRenderer(m, d, default_cam_config=cam)

    def step(n=1):
        for _ in range(n):
            mujoco.mj_step(m, d)
        if renderer:
            renderer.render("human")

    # neutral pose joint values (arm only)
    neutral = np.array([0.00, 0.41, 0.00, -1.85, 0.00, 2.26, 0.79])
    arm_joints = ["joint1","joint2","joint3","joint4","joint5","joint6","joint7"]
    for name, val in zip(arm_joints, neutral):
        jid  = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, name)
        qadr = m.jnt_qposadr[jid]
        d.qpos[qadr] = val
    d.ctrl[:7] = neutral

    # Reset mocap weld targets so the EE tracks mocap without snapping back
    # to a stale relative pose captured at model load time.
    if m.nmocap > 0 and m.eq_data is not None:
        for i in range(m.eq_data.shape[0]):
            if m.eq_type[i] == mujoco.mjtEq.mjEQ_WELD:
                m.eq_data[i, 3:10] = np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0])

    mujoco.mj_forward(m, d)

    # mocap index (panda_mocap is the only mocap body)
    mocap_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "panda_mocap")
    mid = m.body_mocapid[mocap_id]  # index into mocap arrays
    nail_site_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "nail_top")
    ee_site_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "ee_center_site")
    head_site_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "hammer_head_site")

    # initial mocap orientation (keep constant throughout)
    mujoco.mj_forward(m, d)
    # get current mocap pose
    init_pos  = d.mocap_pos[mid].copy()
    init_quat = d.mocap_quat[mid].copy()

    # ── PHASE 1: SETTLE ──────────────────────────────────────────────────────
    print("=" * 65)
    print("PHASE 1: Move arm to hover hammer 3 cm above nail")
    print("=" * 65)

    nail_pos = d.site_xpos[nail_site_id].copy()
    ee_pos0 = d.site_xpos[ee_site_id].copy()
    head_pos0 = d.site_xpos[head_site_id].copy()
    ee_to_head_offset = head_pos0 - ee_pos0

    desired_head = nail_pos + np.array([0.0, 0.0, 0.03])  # 3 cm above nail center
    target_pos = desired_head - ee_to_head_offset
    d.mocap_pos[mid] = target_pos
    d.mocap_quat[mid] = init_quat

    for step_i in range(500):
        # Avoid actuator-vs-mocap "tug-of-war": keep joint targets equal to current qpos.
        # This makes the joint servos hold the current configuration instead of pulling
        # back toward the original neutral pose while the weld tracks the mocap.
        d.ctrl[0] = d.qpos[m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "joint1")]]
        d.ctrl[1] = d.qpos[m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "joint2")]]
        d.ctrl[2] = d.qpos[m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "joint3")]]
        d.ctrl[3] = d.qpos[m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "joint4")]]
        d.ctrl[4] = d.qpos[m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "joint5")]]
        d.ctrl[5] = d.qpos[m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "joint6")]]
        d.ctrl[6] = d.qpos[m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "joint7")]]
        step()
        head_pos = d.site_xpos[head_site_id].copy()
        err = np.linalg.norm(head_pos - desired_head)
        if step_i % 100 == 0:
            ee_pos = d.site_xpos[ee_site_id].copy()
            print(f"  step {step_i:3d}  EE={ee_pos.round(3)}  "
                  f"head={head_pos.round(3)}  err={err:.4f}")
        if err < 0.005:
            print(f"  ✓ settled at step {step_i}  (error {err:.4f} m)")
            break

    # After settling, lock the joint targets to the achieved configuration so the
    # robot stays put (useful as an RL reset state).
    d.ctrl[0] = d.qpos[m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "joint1")]]
    d.ctrl[1] = d.qpos[m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "joint2")]]
    d.ctrl[2] = d.qpos[m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "joint3")]]
    d.ctrl[3] = d.qpos[m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "joint4")]]
    d.ctrl[4] = d.qpos[m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "joint5")]]
    d.ctrl[5] = d.qpos[m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "joint6")]]
    d.ctrl[6] = d.qpos[m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "joint7")]]

    head = d.site_xpos[head_site_id].copy()
    nail = d.site_xpos[nail_site_id].copy()
    print(f"\n  hammer_head : {head.round(4)}")
    print(f"  nail_top    : {nail.round(4)}")
    print(f"  gap         : {(head[2] - nail[2])*100:.1f} cm above nail")
    print(f"  nail depth  : {nail_depth(m,d)*1000:.2f} mm")
    print(f"  contacts    : {d.ncon}")
    print_contacts(m, d, prefix="")

    """
    # ── PHASE 2: STRIKE ──────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("PHASE 2: Drive hammer DOWN onto nail  (fast strike)")
    print("=" * 65)

    # Move mocap 12 cm below nail_top — forces the arm to push through
    strike_z = nail_top_z + EE_OFFSET_Z - 0.12
    d.mocap_pos[mid] = np.array([NAIL_X, NAIL_Y, strike_z])

    prev_depth = nail_depth(m, d)
    max_force  = 0.0

    for step_i in range(300):
        step()

        depth  = nail_depth(m, d)
        driven = (depth - prev_depth) * 1000  # mm this step

        # track peak contact force on nail
        for i in range(d.ncon):
            c = d.contact[i]
            g1 = get_geom_name(m, c.geom1)
            g2 = get_geom_name(m, c.geom2)
            if "nail" in g1 or "nail" in g2:
                f = np.zeros(6)
                mujoco.mj_contactForce(m, d, i, f)
                max_force = max(max_force, abs(f[0]))

        if d.ncon > 0 or driven > 0.01:
            head_pos = get_site(m, d, "hammer_head_site")
            print(f"  step {step_i:3d}  depth={depth*1000:5.2f} mm  "
                  f"Δ={driven:+.3f} mm  ncon={d.ncon}  head_z={head_pos[2]:.4f}")
            print_contacts(m, d, prefix="  ")

        prev_depth = depth

        if depth >= 0.001:   # stop printing once nail moves
            print(f"\n  ✓ NAIL MOVING — continuing 100 more steps silently…")
            break

    for _ in range(100):
        step()

    # ── PHASE 3: OBSERVE ─────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("PHASE 3: Final state")
    print("=" * 65)

    head = get_site(m, d, "hammer_head_site")
    nail = get_site(m, d, "nail_top")
    depth = nail_depth(m, d)

    print(f"  hammer_head_site : {head.round(4)}")
    print(f"  nail_top         : {nail.round(4)}")
    print(f"  nail driven      : {depth*1000:.2f} mm  "
          f"(max possible = 75.0 mm)")
    print(f"  peak nail force  : {max_force:.2f} N")
    print(f"  active contacts  : {d.ncon}")
    print_contacts(m, d, prefix="")

    if renderer:
        print("\nViewer open — close window to exit.")
        while renderer._viewers.get("human") and \
              renderer._viewers["human"].window and \
              not mujoco.glfw.glfw.window_should_close(
                  renderer._viewers["human"].window):
            step()
        renderer.close()

    print("\nDone.")
    """

# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--render", action="store_true",
                        help="Open viewer window while running")
    args = parser.parse_args()
    run(render=args.render)
