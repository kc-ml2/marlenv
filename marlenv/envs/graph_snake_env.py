import math
from collections import deque

import numpy as np

from marlenv.core.grid_util import rgb_from_grid
from marlenv.core.snake import Cell, CellColors
from .snake_env import SnakeEnv

"""
    UP = (-1, 0)
    RIGHT = (0, 1)
    DOWN = (1, 0)
    LEFT = (0, -1)
"""


class GraphSnakeEnv(SnakeEnv):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _init_obs(self):
        if self.image_obs:
            self.obs = deque(maxlen=self.frame_stack)
            for _ in range(self.frame_stack):
                self.obs.append(rgb_from_grid(self.grid, Cell, CellColors))
            obs = [np.concatenate(list(self.obs), axis=-1)
                   for _ in range(self.num_snakes)]
        else:
            self.obs = deque(maxlen=self.frame_stack)
            _obs = self._encode(self.grid, vision_range=self.vision_range)
            for _ in range(self.frame_stack):
                self.obs.append(_obs)
            obs = list(zip(*list(self.obs)))
            obs = [np.concatenate(o, axis=-1) for o in obs]

        proc_obs = self._process_obs(obs)

        return proc_obs

    def _get_obs(self):
        obs = super()._get_obs()
        obs = self._process_obs(obs)

        return obs

    def _process_obs(self, obs):

        if not self.observer == 'snake':
            raise ValueError(
                "This is not yet implemented for 'human' observers.")
        if self.image_obs:
            raise ValueError(
                "This is not yet implemented for 'image' observation.")

        vision_range = 5  # range of vision in default five
        if self.vision_range:
            vision_range = self.vision_range
        proc_obs = []
        sqrt2 = math.sqrt(2)
        snake_idx = 0
        for snake in self.snakes:  # for each snake
            if not snake.alive:
                continue
            proc_ob = []
            angle = math.atan2(snake.direction.value[1],
                               snake.direction.value[0])
            head = snake.head_coord
            if self.vision_range:  # if so, the head is at the center
                head = (self.vision_range, self.vision_range)
            for l in range(3):  # for each of three directions except backward
                dx = (int(math.cos(angle + self.action_dict[l])),
                      int(math.sin(angle + self.action_dict[l])))
                proc_ob.append(np.zeros((self.obs_ch,)))
                for i in range(vision_range):
                    temp_ob = obs[snake_idx][head[0] + dx[0] * (i + 1)][
                        head[1] + dx[1] * (i + 1)]
                    proc_ob[-1] += temp_ob / (i + 1)
                    if temp_ob[0] == 1:  # up to the wall
                        break
            for l in [(0, 1), (0, 2)]:  # each of two diagonal directions
                dx = [(int(math.cos(angle + self.action_dict[l[q]])),
                       int(math.sin(angle + self.action_dict[l[q]]))) for q in
                      range(2)]
                proc_ob.append(np.zeros((self.obs_ch,)))
                for i in range(vision_range):
                    temp_ob = \
                    obs[snake_idx][head[0] + (dx[0][0] + dx[1][0]) * (i + 1)][
                        head[1] + (dx[0][1] + dx[1][1]) * (i + 1)]
                    if temp_ob[0] == 1:
                        proc_ob[-1] += temp_ob / ((i + 1) * sqrt2)
                        break
                    proc_ob[-1] += temp_ob / ((i + 1) * sqrt2)
            proc_obs.append(np.array(proc_ob))
            snake_idx += 1

        return np.array(proc_obs)
