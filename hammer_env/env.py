import os
from typing import Any, Optional, SupportsFloat

import mujoco
import numpy as np
from gymnasium.core import ObsType
from gymnasium_robotics.envs.robot_env import MujocoRobotEnv
from gymnasium_robotics.utils import rotations


DEFAULT_CAMERA_CONFIG = {
    "distance": 2.5,
    "azimuth": 135.0,
    "elevation": -20.0,
    "lookat": np.array([0.4, 0.0, 0.4]),
}


class FrankaHammerEnv(MujocoRobotEnv):
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

        self.neutral_joint_values = np.array([0.803685, 0.773506, 0.0195586, -0.93819, -1.43596, 0.760135, 0.988909, 0.004, 0.004])
        self.hammer_hand_offset = np.array([0.0, 0.0, 0.105])
        self.hammer_hand_quat_offset = np.array([np.cos(np.pi / 4), 0.0, np.sin(np.pi / 4), 0.0], dtype=np.float64)
        self.gripper_grasp_qpos = 0.004

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

        self.arm_joint_names = [f"joint{i}" for i in range(1, 8)]
        self.gripper_joint_names = ["finger_joint1", "finger_joint2"]

        self._env_setup(self.neutral_joint_values)
        self.initial_time = self.data.time
        self.initial_qvel = np.copy(self.data.qvel)

    def _env_setup(self, neutral_joint_values: np.ndarray) -> None:
        self.set_joint_neutral()
        self.data.ctrl[0:7] = neutral_joint_values[0:7]
        self.reset_mocap_welds(self.model, self.data)
        self._mujoco.mj_forward(self.model, self.data)

        self.ee_site = "ee_center_site"
        self.head_site = "hammer_head_site"
        self.nail_site = "nail_top"
        self.hammer_joint = "hammer_freejoint"

        self.grasp_site_pose = self.get_ee_orientation().copy()

        self.initial_mocap_position = self._utils.get_site_xpos(self.model, self.data, self.ee_site).copy()
        self.set_mocap_pose(self.initial_mocap_position, self.grasp_site_pose)

        self._set_pregrasp_initial_state()
        self.goal = self._compute_goal()

    def _reset_sim(self) -> bool:
        self.data.time = self.initial_time
        self.data.qvel[:] = np.copy(self.initial_qvel)
        if self.model.na != 0:
            self.data.act[:] = None

        self.set_joint_neutral()
        self.data.ctrl[0:7] = self.neutral_joint_values[0:7]
        self.data.ctrl[-2:] = self.gripper_grasp_qpos
        self.set_mocap_pose(self.initial_mocap_position, self.grasp_site_pose)

        self._mujoco.mj_forward(self.model, self.data)
        self._set_pregrasp_initial_state()
        self.goal = self._compute_goal()
        self._mujoco.mj_forward(self.model, self.data)
        return True

    def _mujoco_step(self, action: Optional[np.ndarray] = None) -> None:
        for _ in range(10):
            self._mujoco.mj_step(self.model, self.data, nstep=self.n_substeps)

    def _set_action(self, action: np.ndarray) -> None:
        action = action.copy()
        pos_ctrl = np.clip(action, self.action_space.low, self.action_space.high)

        pos_ctrl *= self.action_scale
        # Tool-space control: interpret actions as Δ(hammer_head position),
        # then convert desired hammer head position -> desired EE (mocap) position.
        ee_pos = self._utils.get_site_xpos(self.model, self.data, self.ee_site).copy()
        head_pos = self._utils.get_site_xpos(self.model, self.data, self.head_site).copy()
        ee_to_head_offset = head_pos - ee_pos

        desired_head_pos = head_pos + pos_ctrl
        desired_head_pos[2] = float(np.max((0.0, desired_head_pos[2])))

        desired_ee_pos = desired_head_pos - ee_to_head_offset

        self.set_mocap_pose(desired_ee_pos, self.grasp_site_pose)

    def compute_reward(self, achieved_goal, desired_goal, info) -> SupportsFloat:
        d = self.goal_distance(achieved_goal, desired_goal)
        if self.reward_type == "sparse":
            return -(d > self.distance_threshold).astype(np.float32)
        return -d

    def _get_obs(self) -> dict[str, np.ndarray]:
        ee_pos = self._utils.get_site_xpos(self.model, self.data, self.ee_site).copy()
        ee_vel = (self._utils.get_site_xvelp(self.model, self.data, self.ee_site).copy()) * self.dt

        head_pos = self._utils.get_site_xpos(self.model, self.data, self.head_site).copy()
        head_vel = (self._utils.get_site_xvelp(self.model, self.data, self.head_site).copy()) * self.dt

        nail_pos = self._utils.get_site_xpos(self.model, self.data, self.nail_site).copy()
        nail_depth = np.array([float(self._utils.get_joint_qpos(self.model, self.data, "nail_slide"))], dtype=np.float64)

        obs = np.concatenate([ee_pos, ee_vel, head_pos, head_vel, nail_pos, nail_depth]).copy()

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

    def _quat_mul(self, q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
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

    def reset_mocap_welds(self, model, data) -> None:
        if model.nmocap > 0 and model.eq_data is not None:
            for i in range(model.eq_data.shape[0]):
                if model.eq_type[i] == mujoco.mjtEq.mjEQ_WELD:
                    model.eq_data[i, 3:10] = np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
        self._mujoco.mj_forward(model, data)

    def set_mocap_pose(self, position: np.ndarray, orientation: np.ndarray) -> None:
        self._utils.set_mocap_pos(self.model, self.data, "panda_mocap", position)
        self._utils.set_mocap_quat(self.model, self.data, "panda_mocap", orientation)

    def set_joint_neutral(self) -> None:
        for name, value in zip(self.arm_joint_names, self.neutral_joint_values[0:7]):
            self._utils.set_joint_qpos(self.model, self.data, name, value)
        for name, value in zip(self.gripper_joint_names, self.neutral_joint_values[7:9]):
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
        # No additional randomizable object in this scene (nail is part of the fixed model).
        return None

    def _set_pregrasp_initial_state(self) -> None:
        self._mujoco.mj_forward(self.model, self.data)

        self._set_hammer_pose_from_hand()
        for name in self.gripper_joint_names:
            self._utils.set_joint_qpos(self.model, self.data, name, self.gripper_grasp_qpos)
        self.data.ctrl[-2:] = self.gripper_grasp_qpos
        self.set_mocap_pose(self.get_ee_position().copy(), self.grasp_site_pose)

        for _ in range(2):
            self._mujoco_step()
            self.data.ctrl[-2:] = self.gripper_grasp_qpos

        self.data.qvel[:] = 0.0

        for j, name in enumerate(self.arm_joint_names):
            q = float(self._utils.get_joint_qpos(self.model, self.data, name))
            self.data.ctrl[j] = q

    def _set_hammer_pose_from_hand(self) -> None:
        hand_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "hand")
        hand_pos = self.data.xpos[hand_id].copy()
        hand_xmat = self.data.xmat[hand_id].reshape(3, 3)
        hand_quat = self.data.xquat[hand_id].copy()

        hammer_pos = hand_pos + hand_xmat @ self.hammer_hand_offset
        hammer_quat = self._quat_mul(hand_quat, self.hammer_hand_quat_offset)

        hammer_jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, self.hammer_joint)
        qadr = self.model.jnt_qposadr[hammer_jid]
        vadr = self.model.jnt_dofadr[hammer_jid]
        self.data.qpos[qadr:qadr + 3] = hammer_pos
        self.data.qpos[qadr + 3:qadr + 7] = hammer_quat
        self.data.qvel[vadr:vadr + 6] = 0.0
        self._mujoco.mj_forward(self.model, self.data)
