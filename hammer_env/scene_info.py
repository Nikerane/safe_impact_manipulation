"""
Prints a structured summary of the hammer-nail scene:
joints, bodies, sites, actuators, and key positions.

Usage:
    python hammer_env/scene_info.py
"""
import os
import mujoco
import numpy as np

XML = os.path.join(os.path.dirname(__file__), "assets", "hammer_nail_scene.xml")

m = mujoco.MjModel.from_xml_path(XML)
d = mujoco.MjData(m)

# Run one forward pass so positions are populated
mujoco.mj_forward(m, d)

def names(obj_type):
    return [mujoco.mj_id2name(m, obj_type, i) for i in range(
        getattr(m, f"n{obj_type.name.lower().removeprefix('mjobj_')}")
    )]

print("=" * 60)
print("HAMMER-NAIL SCENE SUMMARY")
print("=" * 60)

print(f"\n--- Model dimensions ---")
print(f"  nq  (generalized coords)  : {m.nq}")
print(f"  nv  (velocities / DOF)    : {m.nv}")
print(f"  nu  (actuators / controls): {m.nu}")
print(f"  nbody                     : {m.nbody}")
print(f"  ngeom                     : {m.ngeom}")
print(f"  nsite                     : {m.nsite}")

print(f"\n--- Joints ({m.njnt}) ---")
for i in range(m.njnt):
    name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, i)
    jtype = ["free", "ball", "slide", "hinge"][m.jnt_type[i]]
    lo, hi = m.jnt_range[i]
    print(f"  [{i:2d}] {name:<25s} type={jtype:<6s}  range=[{lo:.3f}, {hi:.3f}]")

print(f"\n--- Sites ({m.nsite}) ---")
for i in range(m.nsite):
    name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_SITE, i)
    pos  = d.site_xpos[i]
    print(f"  {name:<28s}  world pos = [{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}]")

print(f"\n--- Actuators ({m.nu}) ---")
for i in range(m.nu):
    name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
    lo, hi = m.actuator_ctrlrange[i]
    print(f"  [{i:2d}] {name:<35s} ctrl range=[{lo:.3f}, {hi:.3f}]")

print(f"\n--- Key body positions at neutral pose ---")
for bname in ["hand", "hammer", "block", "nail"]:
    bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, bname)
    if bid >= 0:
        pos = d.xpos[bid]
        print(f"  {bname:<20s}  [{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}]")

print(f"\n--- Nail slide joint state ---")
nail_jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "nail_slide")
nail_qadr = m.jnt_qposadr[nail_jid]
print(f"  nail_slide qpos = {d.qpos[nail_qadr]:.4f}  (0 = flush, 0.075 = fully driven)")

print("\n" + "=" * 60)
