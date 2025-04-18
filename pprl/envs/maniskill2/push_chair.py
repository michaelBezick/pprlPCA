from __future__ import annotations

import numpy as np
from mani_skill2 import format_path
from mani_skill2.envs.ms1.push_chair import PushChairEnv
from mani_skill2.utils.io_utils import load_json
from mani_skill2.utils.registration import register_env
from scipy.spatial import distance as sdist


@register_env("PushChair-v2", max_episode_steps=200)
class ConfigurableTurnFaucetEnv(PushChairEnv):
    """This version of PushChair has modified rewards so as not to"""

    def __init__(
        self,
        *args,
        n_models: int | None = None,
        **kwargs,
    ):
        if n_models is not None:
            model_json = format_path(self.DEFAULT_MODEL_JSON)
            model_db: dict[str, dict] = load_json(model_json)
            kwargs["model_ids"] = list(model_db.keys())[:n_models]

        super().__init__(*args, **kwargs)

    def reset(self, seed=None, options=None):
        obs, info = super().reset(seed, options)
        self.prev_chair_pos = self.chair.pose.p[:2].astype(np.float32)
        self.prev_dist_chair_to_target = np.linalg.norm(
            self.prev_chair_pos - self.target_xy
        )
        return obs, info

    def _get_obs_extra(self):
        obs = super()._get_obs_extra()
        obs["target_link_pos"] = self.target_p[:3]
        return obs

    def evaluate(self, **kwargs):
        disp_chair_to_target = self.chair.pose.p[:2] - self.target_xy
        dist_chair_to_target = np.linalg.norm(disp_chair_to_target)

        # z-axis of chair should be upward
        z_axis_chair = self.root_link.pose.to_transformation_matrix()[:3, 2]
        chair_tilt = np.arccos(z_axis_chair[2])

        vel_norm = np.linalg.norm(self.root_link.velocity)
        ang_vel_norm = np.linalg.norm(self.root_link.angular_velocity)

        flags = dict(
            chair_close_to_target=dist_chair_to_target <= 0.2,
            chair_standing=chair_tilt <= 0.05 * np.pi,
            # chair_static=(vel_norm <= 0.1 and ang_vel_norm <= 0.2),
            chair_static=self.check_actor_static(
                self.root_link, max_v=0.1, max_ang_v=0.2
            ),
        )
        return dict(
            success=all(flags.values()),
            **flags,
            dist_chair_to_target=dist_chair_to_target,
            chair_tilt=chair_tilt,
            chair_vel_norm=vel_norm,
            chair_ang_vel_norm=ang_vel_norm,
        )

    def compute_dense_reward(self, action: np.ndarray, info: dict, **kwargs):
        reward = 0

        # Compute distance between end-effectors and chair surface
        ee_coords = np.array(self.agent.get_ee_coords())  # [M, 3]
        chair_pcd = self._get_chair_pcd()  # [N, 3]

        # EE approach chair
        dist_ees_to_chair = sdist.cdist(ee_coords, chair_pcd)  # [M, N]
        dist_ees_to_chair = dist_ees_to_chair.min(1)  # [M]
        dist_ee_to_chair = dist_ees_to_chair.mean()
        log_dist_ee_to_chair = np.log(dist_ee_to_chair + 1e-5)
        reward += -dist_ee_to_chair - np.clip(log_dist_ee_to_chair, -10, 0)

        # Keep chair standing
        chair_tilt = info["chair_tilt"]
        reward += -chair_tilt * 0.2

        # # Penalize action
        # # Assume action is relative and normalized.
        action_norm = np.linalg.norm(action)
        reward -= action_norm * 1e-6

        # Chair velocity
        # Legacy version uses full velocity instead of xy-plane velocity
        chair_vel = self.root_link.velocity[:2]
        chair_vel_norm = np.linalg.norm(chair_vel)
        disp_chair_to_target = self.root_link.get_pose().p[:2] - self.target_xy
        cos_chair_vel_to_target = sdist.cosine(disp_chair_to_target, chair_vel)

        # Stage reward
        # NOTE(jigu): stage reward can also be used to debug which stage it is
        stage_reward = -10
        # -18 can guarantee the reward is negative

        chair_pos = self.chair.pose.p[:2].astype(np.float32)
        delta_chair_pos = chair_pos - self.prev_chair_pos
        vec_prev_chair_pos_to_target = self.target_xy - self.prev_chair_pos
        dist_chair_to_target = info["dist_chair_to_target"]

        abs_cos_similarity = np.abs(
            np.dot(delta_chair_pos, vec_prev_chair_pos_to_target)
        ) / (
            np.linalg.norm(delta_chair_pos)
            * np.linalg.norm(vec_prev_chair_pos_to_target)
            + 1e-8
        )

        delta_reward = (
            (self.prev_dist_chair_to_target - dist_chair_to_target)
            * abs_cos_similarity
            * 100.0
        )

        reward += delta_reward

        self.prev_chair_pos = chair_pos
        self.prev_dist_chair_to_target = dist_chair_to_target

        if chair_tilt < 0.2 * np.pi:
            # Chair is standing
            if dist_ee_to_chair < 0.1:
                # EE is close to chair
                stage_reward += 2
                if dist_chair_to_target <= 0.2:
                    # Chair is close to target
                    stage_reward += 2
                    # Try to keep chair static
                    reward += np.exp(-chair_vel_norm * 10) * 2
                    # # Legacy: Note that the static condition here is different from success metric
                    # if chair_vel_norm <= 0.1 and chair_ang_vel_norm <= 0.2:
                    #     stage_reward += 2

        reward = reward + stage_reward

        # Update info
        info.update(
            dist_ee_to_chair=dist_ee_to_chair,
            action_norm=action_norm,
            chair_vel_norm=chair_vel_norm,
            cos_chair_vel_to_target=cos_chair_vel_to_target,
            stage_reward=stage_reward,
        )
        return reward
