from .snake_env import SnakeEnv
import numpy as np


class CoopSnakeEnv(SnakeEnv):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def step(self, actions):
        """
        Identical to SnakeEnv's step function except for the terminating 
        condition. if all(dones) -> any(dones)
        This is for finishing an episode if there's at least one dead snake.
        step() returns done = [True] * num_snakes if there's a dead snake.
        """

        obs, rews, dones, info = super().step(actions)

        if any(dones):
            dones = [True] * self.num_snakes
            sorted_scores = np.unique(np.sort(self.epi_scores)[::-1])
            ranks = np.array([0 for _ in range(self.num_snakes)])
            base_rank = 1
            for score in sorted_scores[::-1]:
                idx = np.where(np.array(self.epi_scores) == score)[0]
                ranks[idx] = base_rank
                base_rank += len(idx)
            info['rank'] = list(ranks)

            info.update(
                {'episode_scores': self.epi_scores,
                 'episode_steps': self.epi_steps,
                 'episode_fruits': self.epi_fruits,
                 'episode_kills': self.epi_kills})

            self._reset_epi_stats()

        return obs, rews, dones, info
