# safe_impact_manipulation

MuJoCo hammer-nail manipulation environments for RL research.

| Package | Robot | Notes |
|---------|-------|-------|
| [`hammer_env/`](hammer_env/) | Franka Panda | Standalone Gym env + free hammer |
| [`hammer_z1_env/`](hammer_z1_env/) | Unitree Z1 | Assets + standalone Gym env; mjlab training in sibling repo |

## Z1 + mjlab (`unitree_rl_mjlab`, `hammer-z1` branch)

Clone this repo next to [`unitree_rl_mjlab`](https://github.com/Nikerane/unitree_rl_mjlab) on branch `hammer-z1`.
MJCF/meshes under `hammer_z1_env/assets/` are loaded by `unitree_rl_mjlab` as task **`Unitree-Z1-Hammer`**.

```bash
# layout: ~/repos/safe_impact_manipulation  +  ~/repos/unitree_rl_mjlab
cd ~/repos/unitree_rl_mjlab
export MUJOCO_GL=egl
python scripts/train.py Unitree-Z1-Hammer --agent.logger tensorboard --env.scene.num-envs 64
python scripts/play.py Unitree-Z1-Hammer --checkpoint-file logs/rsl_rl/z1_hammer/<run>/model_xx.pt
pytest tests/ -m "not integration"
```

More detail: [`hammer_z1_env/README.md`](hammer_z1_env/README.md) · mjlab docs in `unitree_rl_mjlab/docs/`.
