import os
from typing import Any, Optional, SupportsFloat

import mujoco
import numpy as np
from gymnasium_robotics.envs.robot_env import MujocoRobotEnv


DEFAULT_CAMERA_CONFIG = {
    "distance": 2.5,
    "azimuth": 135.0,
    "elevation": -20.0,
    "lookat": np.array([0.4, 0.0, 0.4]),
}


class Z1HammerEnv(MujocoRobotEnv):
    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 20,
    }

    def __init__(
        self,
        model_path: str | None = None,
        n_substeps: int = 25,
        reward_type: str = "sparse",
        distance_threshold: float = 0.02,
        action_scale: float = 0.05,
        **kwargs: Any,
    ):
        if model_path is None:
            model_path = os.path.join(os.path.dirname(__file__), "assets", "hammer_nail_scene.xml")

        self.model_path = model_path
        self.reward_type = reward_type
        self.distance_threshold = float(distance_threshold)
        self.action_scale = float(action_scale)

        # 6 arm joints + 1 gripper joint. Gripper: -1.51844=open, 0=closed.
        # NOTE: this neutral pose is a starting guess; needs to be tuned so the gripped
        # hammer head ends up above the nail at [0.55, 0, 0.472].
        # Joint angles here are in radians (the MuJoCo XML uses `<compiler angle="radian" ...>`).
        # If you have degrees, convert with: rad = deg * np.pi / 180.
        self.neutral_joint_values = np.array(
            [-0.000297861, 1.30495, -1.54711, 0.197456, 0.000311167, 1.57079632679, -0.000964725]
        )

        # Gripper command targets. Ctrl is in radians for the hinge gripper.
        self.gripper_open_qpos = -1.0   # ~57° open, gives ~85mm pad gap
        self.gripper_grasp_qpos = 0.0   # closed for the mounted-tool baseline

        super().__init__(
            n_actions=3,
            n_substeps=n_substeps,
            model_path=self.model_path,
            initial_qpos=self.neutral_joint_values,
            default_camera_config=DEFAULT_CAMERA_CONFIG,
            **kwargs,
        )

        self.ctrl_range = self.model.actuator_ctrlrange

    # --- MujocoRobotEnv overrides ---
    def _initialize_simulation(self) -> None:
        self.model = self._mujoco.MjModel.from_xml_path(self.fullpath)
        self.data = self._mujoco.MjData(self.model)
        self._model_names = self._utils.MujocoModelNames(self.model)

        self.model.vis.global_.offwidth = self.width
        self.model.vis.global_.offheight = self.height

        self.arm_joint_names = [f"joint{i}" for i in range(1, 7)]   # joint1..joint6
        self.gripper_joint_names = ["jointGripper"]

        self._env_setup(self.neutral_joint_values)
        self.initial_time = self.data.time
        self.initial_qvel = np.copy(self.data.qvel)

    def _env_setup(self, neutral_joint_values: np.ndarray) -> None:
        self.set_joint_neutral()
        self.data.ctrl[0:6] = neutral_joint_values[0:6]
        self.data.ctrl[6] = self.gripper_grasp_qpos
        self.reset_mocap_welds(self.model, self.data)
        self._mujoco.mj_forward(self.model, self.data)

        self.ee_site = "ee_center_site"
        self.head_site = "hammer_head_site"
        self.nail_site = "nail_top"

        self.grasp_site_pose = self.get_ee_orientation().copy()

        self.initial_mocap_position = self._utils.get_site_xpos(
            self.model, self.data, self.ee_site
        ).copy()
        self.set_mocap_pose(self.initial_mocap_position, self.grasp_site_pose)

        self.goal = self._compute_goal()
        self._reset_reward_state()

    def _reset_reward_state(self) -> None:
        head_pos = self._utils.get_site_xpos(self.model, self.data, self.head_site).copy()
        nail_pos = self._utils.get_site_xpos(self.model, self.data, self.nail_site).copy()
        self._best_head_to_nail_dist = float(np.linalg.norm(head_pos - nail_pos))
        self._max_nail_depth = 0.0
        self._prev_action = np.zeros(3)
        self._action_rate_sq = 0.0
        self._nail_was_moving = False

    def _reset_sim(self) -> bool:
        self.data.time = self.initial_time
        self.data.qvel[:] = np.copy(self.initial_qvel)
        if self.model.na != 0:
            self.data.act[:] = None

        self.set_joint_neutral()
        self.data.ctrl[0:6] = self.neutral_joint_values[0:6]
        self.data.ctrl[6] = self.gripper_grasp_qpos
        self.set_mocap_pose(self.initial_mocap_position, self.grasp_site_pose)

        self._mujoco.mj_forward(self.model, self.data)
        self.goal = self._compute_goal()
        self._mujoco.mj_forward(self.model, self.data)
        self._reset_reward_state()
        return True

    def _mujoco_step(self, action: Optional[np.ndarray] = None) -> None:
        for _ in range(10):
            self._mujoco.mj_step(self.model, self.data, nstep=self.n_substeps)

    def _set_action(self, action: np.ndarray) -> None:
        action = action.copy()
        pos_ctrl = np.clip(action, self.action_space.low, self.action_space.high)
        self._action_rate_sq = float(np.sum((pos_ctrl - self._prev_action) ** 2))
        self._prev_action = pos_ctrl.copy()
        pos_ctrl *= self.action_scale

        # Tool-space control: action is Δ(hammer head). Convert to Δ(EE) before mocap.
        ee_pos = self._utils.get_site_xpos(self.model, self.data, self.ee_site).copy()
        head_pos = self._utils.get_site_xpos(self.model, self.data, self.head_site).copy()
        ee_to_head_offset = head_pos - ee_pos

        desired_head_pos = head_pos + pos_ctrl
        desired_head_pos[2] = float(np.max((0.0, desired_head_pos[2])))

        desired_ee_pos = desired_head_pos - ee_to_head_offset
        self.set_mocap_pose(desired_ee_pos, self.grasp_site_pose)

    def compute_reward(self, achieved_goal, desired_goal, info) -> SupportsFloat:
        # achieved_goal = nail_top world position (3D); desired_goal = fully-driven target (3D)
        d = self.goal_distance(achieved_goal, desired_goal)
        if self.reward_type == "sparse":
            return -(d > self.distance_threshold).astype(np.float32)

        if self.reward_type == "dense":
            # Plain distance reward — gives a signal but no approach guidance.
            return -d

        # reward_type == "progress" (recommended)
        #
        # Components:
        #   1. approach_rew     — reward getting hammer head closer to nail (Phase 1),
        #                         softly faded out as nail depth increases (exp decay)
        #   2. depth_rew        — reward driving the nail deeper (Phase 2)
        #   3. impact_bonus     — one-time bonus at moment of first contact, scaled by
        #                         impact speed (teaches the policy to swing, not press)
        #   4. done_bonus       — large one-time bonus on completion
        #   5. joint_vel_penalty  — suppresses fast joint motion (sim-to-real)
        #   6. action_rate_penalty — suppresses jerky action changes (sim-to-real)

        nail_depth = float(self._utils.get_joint_qpos(self.model, self.data, "nail_slide"))
        head_pos = self._utils.get_site_xpos(self.model, self.data, self.head_site).copy()
        nail_pos = achieved_goal  # 3D world position of nail_top

        # --- Phase 1: approach ---
        head_to_nail = float(np.linalg.norm(head_pos - nail_pos))
        approach_delta = max(0.0, self._best_head_to_nail_dist - head_to_nail)
        self._best_head_to_nail_dist = min(self._best_head_to_nail_dist, head_to_nail)

        # Exponential decay: approach reward fades smoothly as nail is driven deeper.
        # At 0mm depth: weight=1.0. At 5mm: ~0.37. At 15mm: ~0.05.
        # Avoids the cliff discontinuity of a hard gate while still shifting focus
        # to depth once contact is established. (k=200 tuned for 75mm nail travel.)
        #
        # TODO (SDF reward): replace this Euclidean point-to-point distance with a
        # Signed Distance Field reward — SDF_nail(hammer_head_pos) — which stays smooth
        # through the contact boundary and handles approach angle properly.
        # See: Tang et al., "IndustReal", RSS 2023, arXiv:2305.17110.
        # Trigger: if training shows the policy oscillating/hovering at the contact
        # boundary and approach_rew never cleanly hands off to depth_rew.
        approach_weight = float(np.exp(-200.0 * nail_depth))
        approach_rew = approach_delta * 50.0 * approach_weight

        # --- Phase 2: impact / depth ---
        depth_delta = max(0.0, nail_depth - self._max_nail_depth)
        self._max_nail_depth = max(self._max_nail_depth, nail_depth)
        depth_rew = depth_delta * 500.0

        # --- Velocity-at-contact bonus ---
        # Fires exactly once, at the step the nail first starts moving.
        # Scales with impact speed so the policy learns to swing rather than press.
        # Without this, mocap-controlled policies tend to discover slow-press solutions
        # that work in sim but fail on hardware where impact momentum is required.
        nail_moving = nail_depth > 1e-4
        if nail_moving and not self._nail_was_moving:
            head_vel = self._utils.get_site_xvelp(self.model, self.data, self.head_site)
            impact_bonus = float(np.linalg.norm(head_vel)) * 10.0
        else:
            impact_bonus = 0.0
        self._nail_was_moving = nail_moving

        # --- Completion bonus ---
        done_bonus = 100.0 if nail_depth >= (self.distance_threshold * 3.75) else 0.0

        # --- Smoothness penalties (discourage jerky motion, help sim-to-real) ---
        joint_vel_penalty = -float(np.sum(np.abs(self.data.qvel[:6]))) * 0.005
        # Penalises sudden changes in the commanded action between consecutive steps.
        # Catches violent reversals that joint-velocity alone misses (RSL-RL pattern).
        action_rate_penalty = -self._action_rate_sq * 0.5

        return float(approach_rew + depth_rew + impact_bonus + done_bonus + joint_vel_penalty + action_rate_penalty)

    def _get_obs(self) -> dict[str, np.ndarray]:
        ee_pos = self._utils.get_site_xpos(self.model, self.data, self.ee_site).copy()
        ee_vel = (self._utils.get_site_xvelp(self.model, self.data, self.ee_site).copy()) * self.dt

        head_pos = self._utils.get_site_xpos(self.model, self.data, self.head_site).copy()
        head_vel = (self._utils.get_site_xvelp(self.model, self.data, self.head_site).copy()) * self.dt

        nail_pos = self._utils.get_site_xpos(self.model, self.data, self.nail_site).copy()
        nail_depth = np.array(
            [float(self._utils.get_joint_qpos(self.model, self.data, "nail_slide"))],
            dtype=np.float64,
        )

        # Scalar distance from hammer head to nail top — makes the approach
        # phase directly observable so the policy doesn't have to infer it
        # from the difference of two 3D positions.
        head_to_nail_dist = np.array(
            [float(np.linalg.norm(head_pos - nail_pos))], dtype=np.float64
        )

        obs = np.concatenate([ee_pos, ee_vel, head_pos, head_vel, nail_pos, nail_depth, head_to_nail_dist]).copy()

        return {
            "observation": obs,
            "achieved_goal": nail_pos.copy(),
            "desired_goal": self.goal.copy(),
        }

    def _is_success(self, achieved_goal, desired_goal) -> np.float32:
        d = self.goal_distance(achieved_goal, desired_goal)
        return (d < self.distance_threshold).astype(np.float32)

    # --- helpers ---
    def goal_distance(self, goal_a, goal_b) -> SupportsFloat:
        assert goal_a.shape == goal_b.shape
        return np.linalg.norm(goal_a - goal_b, axis=-1)

    def reset_mocap_welds(self, model, data) -> None:
        if model.nmocap > 0 and model.eq_data is not None:
            for i in range(model.eq_data.shape[0]):
                if model.eq_type[i] == mujoco.mjtEq.mjEQ_WELD:
                    model.eq_data[i, 3:10] = np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
        self._mujoco.mj_forward(model, data)

    def set_mocap_pose(self, position: np.ndarray, orientation: np.ndarray) -> None:
        self._utils.set_mocap_pos(self.model, self.data, "z1_mocap", position)
        self._utils.set_mocap_quat(self.model, self.data, "z1_mocap", orientation)

    def set_joint_neutral(self) -> None:
        for name, value in zip(self.arm_joint_names, self.neutral_joint_values[0:6]):
            self._utils.set_joint_qpos(self.model, self.data, name, value)
        for name, value in zip(self.gripper_joint_names, self.neutral_joint_values[6:7]):
            self._utils.set_joint_qpos(self.model, self.data, name, value)

    def get_ee_orientation(self) -> np.ndarray:
        site_mat = self._utils.get_site_xmat(self.model, self.data, self.ee_site).reshape(9, 1)
        current_quat = np.empty(4)
        self._mujoco.mju_mat2Quat(current_quat, site_mat)
        return current_quat

    def get_ee_position(self) -> np.ndarray:
        return self._utils.get_site_xpos(self.model, self.data, self.ee_site)

    def _compute_goal(self) -> np.ndarray:
        nail_pos = self._utils.get_site_xpos(self.model, self.data, self.nail_site).copy()
        return nail_pos + np.array([0.0, 0.0, -0.075])

    def _sample_goal(self) -> np.ndarray:
        return self._compute_goal()

    def _sample_object(self) -> None:
        return None
