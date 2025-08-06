from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Iterator
from pprl.envs.sofaenv.pointcloud_obs import SofaEnvPointCloudObservations as PCObs
from pprl.observation.hpr import hpr_partial, random_eye
import functools

import gymnasium.spaces as spaces
import hydra
import parllel.logger as logger
import torch
import wandb
from hydra.utils import get_class, instantiate
from omegaconf import DictConfig, OmegaConf, open_dict
from parllel import Array, ArrayDict, dict_map
from parllel.callbacks.recording_schedule import RecordingSchedule
from parllel.logger import Verbosity
from parllel.patterns import build_cages, build_sample_tree
from parllel.replays.replay import ReplayBuffer
from parllel.runners import RLRunner
from parllel.samplers import BasicSampler
from parllel.samplers.eval import EvalSampler
from parllel.torch.agents.sac_agent import SacAgent
from parllel.torch.algos.sac import build_replay_buffer_tree
from parllel.torch.distributions.squashed_gaussian import SquashedGaussian
from parllel.transforms.vectorized_video import RecordVectorizedVideo
from parllel.types import BatchSpec
import numpy as np

from pprl.utils.array_dict import build_obs_array
from pprl.observation.hpr import EpisodeHPR, random_eye, perfect_eye

from omegaconf import OmegaConf

MAX_PTS = 1000

# ------------------------------------------------------------------ #
# choose the camera rings once, reuse every build() call             #
# ------------------------------------------------------------------ #
TRAIN_CAMERAS = [
    {"position": [   0, 175, 120], "lookAt": [10,  0, 55]},
    # {"position": [-175,    0, 120], "lookAt": [ 10,  0, 55]},
    # {"position": [   0,  175, 120], "lookAt": [10,  0, 55]},
    # {"position": [ 0,    0, 200], "lookAt": [ 10,  0, 55]},
]

# """MOVING CAMERA BACK ALONG ITS AXIS"""
#
# old_position = np.array([0, 175, 120])
# lookat = np.array([10, 0, 55])
#
# v = lookat - old_position 
# unit_v = v / np.linalg.norm(v)
#
# MOVE_DISTANCE_CM = 50
#
# new_position = old_position - MOVE_DISTANCE_CM * unit_v

EVAL_CAMERA  = [
    {"position": [0, 175, 170], "lookAt": [10,  0, 105]},
]

hpr_fn = EpisodeHPR(perfect_eye)

import copy

def _pylist_of_dicts(seq):
    return [copy.deepcopy(d) for d in seq]

def _ensure_pylist(cam_cfgs):
    if isinstance(cam_cfgs, np.ndarray):
        cam_cfgs = cam_cfgs.tolist()        # array(dtype=object) → list
    return cam_cfgs


def build_train_env(base_env_factory, **env_kwargs):
    env = base_env_factory(**env_kwargs)     # forward extras
    env.unwrapped.create_scene_kwargs["camera_configs"] = _ensure_pylist(
        env.unwrapped.create_scene_kwargs.get("camera_configs")
    )
    wrapped = PCObs(
        env,
        obs_frame="camera",
        random_downsample=MAX_PTS-3,
        post_processing_functions=[hpr_fn],
        max_expected_num_points=MAX_PTS,
        voxel_grid_size=5,
    )

    return wrapped

def build_eval_env(base_env_factory, **env_kwargs):
    env = base_env_factory(**env_kwargs)     # forward extras
    env.unwrapped.create_scene_kwargs["camera_configs"] = _ensure_pylist(
        env.unwrapped.create_scene_kwargs.get("camera_configs")
    )
    return PCObs(
        env,
        obs_frame="camera",
        random_downsample=MAX_PTS-3,
        post_processing_functions=[],        # no HPR
        max_expected_num_points=MAX_PTS,
        voxel_grid_size=5,
    )


@contextmanager
def build(config: DictConfig) -> Iterator[RLRunner]:

    parallel = config.parallel
    discount = config.algo.discount
    batch_spec = BatchSpec(config.batch_T, config.batch_B)
    storage = "shared" if parallel else "local"

    """TRAIN CAGE"""

    with open_dict(config.env):
        config.env.pop("name")
        traj_info = config.env.pop("traj_info")

    TrajInfoClass = get_class(traj_info)
    TrajInfoClass.set_discount(discount)

    # ----- TRAIN CAMERAS ----------------------------------------------------
    with open_dict(config.env.create_scene_kwargs):
        config.env.create_scene_kwargs.pop("camera_config", None)   # remove old key
        config.env.create_scene_kwargs["camera_configs"] = _pylist_of_dicts(TRAIN_CAMERAS)


    base_train_factory = instantiate(config.env, _convert_="partial", _partial_=True)

    cages, metadata = build_cages(
        # EnvClass=functools.partial(build_train_env, base_train_factory),
        EnvClass=base_train_factory,
        n_envs=batch_spec.B,
        TrajInfoClass=TrajInfoClass,
        parallel=parallel,
    )


    # ----- EVAL CAMERA ------------------------------------------------------
    with open_dict(config.env.create_scene_kwargs):
        config.env.create_scene_kwargs.pop("camera_config", None)   # remove old key
        config.env.create_scene_kwargs["camera_configs"] = _pylist_of_dicts(EVAL_CAMERA)

        config.env.mode = "eval"

        # config.env.create_scene_kwargs.pop("mode", None)   # remove old key
        # config.env.create_scene_kwargs["mode"] = "eval"


    base_eval_factory = instantiate(config.env, _convert_="partial", _partial_=True)

    """EVAL CAGE"""

    eval_cages, eval_metadata = build_cages(
        # EnvClass=functools.partial(build_eval_env, base_eval_factory),
        EnvClass=base_eval_factory,
        n_envs=config.eval.n_eval_envs,
        env_kwargs={"add_rendering_to_info": True,
                    },
        TrajInfoClass=TrajInfoClass,
        parallel=parallel,
    )

    replay_length = int(config.algo.replay_length) // batch_spec.B
    replay_length = (replay_length // batch_spec.T) * batch_spec.T
    sample_tree, metadata = build_sample_tree(
        env_metadata=metadata,
        batch_spec=batch_spec,
        parallel=parallel,
        full_size=replay_length,
        keys_to_skip=("obs", "next_obs"),
    )

    obs_space, action_space = metadata.obs_space, metadata.action_space

    sample_tree["observation"] = build_obs_array(
        metadata.example_obs,
        obs_space,
        batch_shape=tuple(batch_spec),
        storage=storage,
        padding=1,
        full_size=replay_length,
    )

    sample_tree["next_observation"] = sample_tree["observation"].new_array(
        padding=0,
        inherit_full_size=True,
    )

    assert isinstance(action_space, spaces.Box)
    n_actions = action_space.shape[0]

    # create model
    model = torch.nn.ModuleDict()

    with open_dict(config.model):
        encoder_name = config.model.pop("name")

    if encoder_name != "Passthru":
        encoder = instantiate(
            config.model,
            _convert_="partial",
            obs_space=obs_space,
        )
        model["encoder"] = encoder
        embedding_size = encoder.embed_dim
    else:
        embedding_size = spaces.flatdim(obs_space)

    model["pi"] = instantiate(
        config.pi_mlp_head,
        input_size=embedding_size,
        action_size=n_actions,
        action_space=action_space,
        _convert_="partial",
    )
    model["q1"] = instantiate(
        config.q_mlp_head,
        input_size=embedding_size,
        action_size=n_actions,
        _convert_="partial",
    )
    model["q2"] = instantiate(
        config.q_mlp_head,
        input_size=embedding_size,
        action_size=n_actions,
        _convert_="partial",
    )

    distribution = SquashedGaussian(
        dim=n_actions,
        scale=action_space.high[0],
    )
    device = config.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    wandb.config.update({"device": device}, allow_val_change=True)
    device = torch.device(device)

    # instantiate agent
    agent = SacAgent(
        model=model,
        distribution=distribution,
        device=device,
        learning_starts=config.algo.learning_starts,
    )

    sampler = BasicSampler(
        batch_spec=batch_spec,
        envs=cages,
        agent=agent,
        sample_tree=sample_tree,
    )

    # create replay buffer
    replay_buffer_tree = build_replay_buffer_tree(sample_tree)

    def batch_transform(tree: ArrayDict[Array]) -> ArrayDict[torch.Tensor]:
        tree = tree.to_ndarray()  # type: ignore
        tree = tree.apply(torch.from_numpy)
        return tree.to(device=device)

    replay_buffer = ReplayBuffer(
        tree=replay_buffer_tree,
        sampler_batch_spec=batch_spec,
        size_T=replay_length,
        replay_batch_size=config.algo.batch_size,
        newest_n_samples_invalid=0,
        oldest_n_samples_invalid=1,
        batch_transform=batch_transform,
    )

    # create optimizers
    with open_dict(config.optimizer):
        q_optim_conf = config.optimizer.pop("q", {}) or {}
        pi_optim_conf = config.optimizer.pop("pi", {}) or {}
        encoder_optim_conf = config.optimizer.pop("encoder", {}) or {}

    pi_optimizer = instantiate(
        config.optimizer,
        [{"params": agent.model["pi"].parameters(), **pi_optim_conf}],
    )

    q_optimizer = instantiate(
        config.optimizer,
        [
            {"params": agent.model["q1"].parameters(), **q_optim_conf},
            {"params": agent.model["q2"].parameters(), **q_optim_conf},
        ],
    )
    if "encoder" in agent.model:
        q_optimizer.add_param_group(
            {"params": agent.model["encoder"].parameters(), **encoder_optim_conf}
        )

    # create learning rate schedulers
    if gamma := config.get("lr_scheduler_gamma") is not None:
        pi_scheduler = torch.optim.lr_scheduler.ExponentialLR(pi_optimizer, gamma=gamma)
        q_scheduler = torch.optim.lr_scheduler.ExponentialLR(q_optimizer, gamma=gamma)
        lr_schedulers = [pi_scheduler, q_scheduler]
    else:
        lr_schedulers = None

    # create algorithm
    algorithm = instantiate(
        config.algo,
        _convert_="partial",
        batch_spec=batch_spec,
        agent=agent,
        replay_buffer=replay_buffer,
        q_optimizer=q_optimizer,
        pi_optimizer=pi_optimizer,
        learning_rate_schedulers=lr_schedulers,
        action_space=action_space,
    )

    logger.debug("Allocating eval sample tree...")

    eval_tree_keys = [
        "action",
        "agent_info",
        "observation",
        "reward",
        "terminated",
        "truncated",
        "done",
    ]
    eval_tree_example = ArrayDict({key: sample_tree[key] for key in eval_tree_keys})
    # create a new tree with leading dimensions (1, B_eval)
    eval_sample_tree = eval_tree_example.new_array(
        batch_shape=(1, config.eval.n_eval_envs)
    )

    # add env_info according to template from eval envs, since the env_info is different
    eval_sample_tree["env_info"] = dict_map(
        Array.from_numpy,
        eval_metadata.example_info,
        batch_shape=(1, config.eval.n_eval_envs),
        storage=storage,
    )

    video_recorder = RecordVectorizedVideo(
        sample_tree=eval_sample_tree,
        buffer_key_to_record="env_info.rendering",
        env_fps=50,
        output_dir=Path(config.video_path),
        video_length=config.env.max_episode_steps,
        use_wandb=True,
    )
    recording_schedule = RecordingSchedule(video_recorder, trigger="on_eval")

    eval_sampler = EvalSampler(
        max_traj_length=config.env.max_episode_steps,
        max_trajectories=config.eval.max_trajectories,
        envs=eval_cages,
        agent=agent,
        sample_tree=eval_sample_tree,
        step_transforms=[video_recorder],
    )

    # create runner
    runner = RLRunner(
        sampler=sampler,
        agent=agent,
        algorithm=algorithm,
        batch_spec=batch_spec,
        eval_sampler=eval_sampler,
        n_steps=config.runner.n_steps,
        log_interval_steps=config.runner.log_interval_steps,
        eval_interval_steps=config.runner.eval_interval_steps,
        callbacks=[recording_schedule],
    )

    try:
        yield runner

    finally:
        eval_sampler.close()
        eval_sample_tree.close()
        sampler.close()
        sample_tree.close()
        agent.close()
        for eval_cage in eval_cages:
            eval_cage.close()
        for cage in cages:
            cage.close()


@hydra.main(version_base=None, config_path="../conf", config_name="train_sac")
def main(config: DictConfig) -> None:
    with open_dict(config):
        wandb_config = config.pop("wandb", {})
        notes = wandb_config.pop("notes", None)
        tags = wandb_config.pop("tags", None)
        group_name = wandb_config.pop("group_name", None)
        print("Group name is: ", group_name)

    run = wandb.init(
        project="pprl",
        config=OmegaConf.to_container(config, resolve=True, throw_on_missing=True),  # type: ignore
        sync_tensorboard=True,  # auto-upload any values logged to tensorboard
        save_code=True,  # save script used to start training, git commit, and patch
        reinit=True,  # required for hydra sweeps with default launcher
        tags=tags,
        notes=notes,
        group=group_name,
    )

    logger.init(
        wandb_run=run,
        # this log_dir is used if wandb is disabled (using `wandb disabled`)
        log_dir=Path(f"log_data/sac/{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"),
        tensorboard=True,
        output_files={
            "txt": "log.txt",
            # "csv": "progress.csv",
        },  # type: ignore
        config=OmegaConf.to_container(config, resolve=True, throw_on_missing=True),  # type: ignore
        model_save_path=Path("model.pt"),
        # verbosity=Verbosity.DEBUG,
    )

    video_path = (
        Path(config.video_path)
        / f"{datetime.now().strftime('%Y-%m-%d')}/{run.id}"  # type: ignore
    )
    config.update({"video_path": video_path})

    with build(config) as runner:
        runner.run()

    logger.close()
    run.finish()  # type: ignore


if __name__ == "__main__":
    main()
