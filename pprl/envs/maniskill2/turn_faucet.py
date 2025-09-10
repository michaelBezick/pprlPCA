from __future__ import annotations

from typing import Sequence

from mani_skill2 import format_path
from mani_skill2.envs.misc.turn_faucet import TurnFaucetEnv
from mani_skill2.sensors.camera import CameraConfig
from mani_skill2.utils.io_utils import load_json
from mani_skill2.utils.registration import register_env

from mani_skill2.sensors.camera import CameraConfig

TRIPOD_UIDS = {"render_camera"}

@register_env("TurnFaucet-v1", max_episode_steps=200)
class ConfigurableTurnFaucetEnv(TurnFaucetEnv):
    """This version of the environment provides more options for camera
    configuration and the number of faucet models used.
    """

    def __init__(
        self,
        *args,
        observe_render_cam: bool = False,
        robot_cameras: Sequence[str] | None = None,
        n_models: int | None = None,
        **kwargs,
    ) -> None:
        if n_models is not None:
            model_json = format_path(
                "{PACKAGE_ASSET_DIR}/partnet_mobility/meta/info_faucet_train.json"
            )
            model_db: dict[str, dict] = load_json(model_json)
            kwargs["model_ids"] = list(model_db.keys())[:n_models]

        self.observe_render_cam = observe_render_cam
        # self.robot_cameras = robot_cameras
        self.robot_cameras = robot_cameras
        super().__init__(*args, **kwargs)

    def _register_cameras(self) -> list[CameraConfig]:
        cams = super()._register_cameras()
        if not isinstance(cams, (list, tuple)):
            cams = [cams]

        # Keep only world/tripod cameras if present
        kept = [c for c in cams if getattr(c, "uid", None) in TRIPOD_UIDS]

        # If no tripod exists but we’re allowed to add one, add the render camera
        if not kept and getattr(self, "observe_render_cam", True):
            rc = self._register_render_cameras()  # ManiSkill helper that returns a CameraConfig
            kept = [rc]

        # Fallback: if still empty but cams exist, keep the first to avoid downstream asserts
        return kept or [cams[0]]

    def _configure_cameras(self) -> None:
        super()._configure_cameras()
        # prune ego cams from the final registry too
        if hasattr(self, "_camera_cfgs") and isinstance(self._camera_cfgs, dict):
            self._camera_cfgs = {
                uid: cfg for uid, cfg in self._camera_cfgs.items()
                if uid in TRIPOD_UIDS
            } or dict(list(self._camera_cfgs.items())[:1])

