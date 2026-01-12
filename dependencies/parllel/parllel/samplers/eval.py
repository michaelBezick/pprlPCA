from __future__ import annotations

from typing import Sequence

import numpy as np

from parllel import Array, ArrayDict
from parllel.agents import Agent
from parllel.cages import Cage, TrajInfo
from parllel.transforms import Transform
from parllel.types import BatchSpec

from .sampler import Sampler
import wandb


class MultiEvalSampler:
    """
    A drop‑in replacement for EvalSampler that loops over many samplers,
    one per camera mismatch, and logs with a prefix.
    """
    def __init__(self, samplers: dict[str, EvalSampler]):
        self.samplers = samplers                        # {"back50cm": EvalSampler, …}
        self.num_samples = 2

    def _traj_stats(self, trajs):
        stats = {}
        if hasattr(trajs[0], "DiscountedReturn"):
            stats["discounted_return"] = np.mean([t.DiscountedReturn for t in trajs])
        stats["return"]    = np.mean([t.Return for t in trajs])
        stats["num_trajs"] = len(trajs)
        # Success flag is often named "success" *or* "Success"
        flag = "success" if hasattr(trajs[0], "success") else (
               "Success" if hasattr(trajs[0], "Success") else None)
        if flag is not None:
            stats["success_rate"] = np.mean([getattr(t, flag) for t in trajs])
        return stats
    def collect_batch(self, elapsed_steps: int):


        all_trajs = []

        for name, sampler in self.samplers.items():
            success_rate = 0
            for i in range(self.num_samples):
                trajs = sampler.collect_batch(elapsed_steps)
                # ------- aggregate & log -------------
                s = self._traj_stats(trajs)
                # wandb.log({f"eval/{name}/{k}": v for k, v in s.items()},
                #           step=elapsed_steps)
                all_trajs.extend(trajs)
                success_rate += s['success_rate']
            success_rate /= self.num_samples
            wandb.log({f"eval/{name}/success_rate": success_rate},
                      step=elapsed_steps)
        return all_trajs

    def close(self):
        for s in self.samplers.values():
            s.close()


class EvalSampler(Sampler):
    def __init__(
        self,
        max_traj_length: int,
        max_trajectories: int,
        envs: Sequence[Cage],
        agent: Agent,
        sample_tree: ArrayDict[Array],
        step_transforms: Sequence[Transform] | None = None,
    ) -> None:
        for cage in envs:
            if not cage.reset_automatically:
                raise ValueError(
                    "EvalSampler expects cages that reset environments "
                    "automatically. Set `reset_automatically=True`."
                )

        super().__init__(
            batch_spec=BatchSpec(1, len(envs)),
            envs=envs,
            agent=agent,
            sample_tree=sample_tree,
        )

        self.max_traj_length = max_traj_length
        self.max_trajectories = max_trajectories
        self.step_transforms = step_transforms if step_transforms is not None else []

    def collect_batch(self, elapsed_steps: int) -> list[TrajInfo]:
        # get references to sample tree elements
        action = self.sample_tree["action"][0]
        agent_info = self.sample_tree["agent_info"][0]
        observation = self.sample_tree["observation"][0]
        reward = self.sample_tree["reward"][0]
        done = self.sample_tree["done"][0]
        terminated = self.sample_tree["terminated"][0]
        truncated = self.sample_tree["truncated"][0]
        env_info = self.sample_tree["env_info"][0]
        sample_tree = self.sample_tree

        # set agent to eval mode, preventing sampler states from being overwritten
        self.agent.eval_mode(elapsed_steps)

        # reset all environments and agent recurrent state
        self.reset()

        # rotate reset observations to be current values
        sample_tree.rotate()

        # TODO: freeze statistics in obs normalization

        n_completed_trajs = 0

        # main sampling loop
        for _ in range(self.max_traj_length):
            # apply any transforms to the observation before the agent steps
            for transform in self.step_transforms:
                # apply in-place to avoid redundant array write operation
                transform(sample_tree[0])

            # agent observes environment and outputs actions
            action[...], agent_info[...] = self.agent.step(observation)

            for b, env in enumerate(self.envs):
                env.step_async(
                    action[b],
                    out_obs=observation[b],
                    out_reward=reward[b],
                    out_terminated=terminated[b],
                    out_truncated=truncated[b],
                    out_info=env_info[b],
                )

            for b, env in enumerate(self.envs):
                env.await_step()

            done[:] = np.logical_or(terminated, truncated)

            # if environment is done, reset agent
            # environment has already been reset inside cage
            if np.any(done):
                n_completed_trajs += np.sum(done)
                if n_completed_trajs >= self.max_trajectories:
                    break
                self.agent.reset_one(np.asarray(done))

        # collect all completed trajectories from envs
        completed_trajectories = [
            traj for env in self.envs for traj in env.collect_completed_trajs()
        ]

        return completed_trajectories
