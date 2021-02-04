from collections import defaultdict
from typing import List, Tuple

import gym
import math
import numpy as np
from gym.utils import seeding

from marlenv.core.grid_util import (
    random_empty_coords, random_empty_coord, draw, make_grid, dfs_sweep_empty)
from marlenv.core.snake import Direction, Snake, Cell


class SnakeEnv(gym.Env):
    default_action_dict = {
        'noop': 0,
        'left': 1,
        'right': 2,
    }
    action_keys = default_action_dict.keys()

    action_angle_dict = {
        0: 0.0,
        1: -math.pi/2.0,
        2: math.pi/2.0
    }

    default_reward_dict = {
        'fruit': 1.0,
        'kill': 0,
        'lose': -1.0,
        'win': 0.,
        'time': 0.,
    }
    reward_keys = default_reward_dict.keys()

    def __init__(
            self,
            num_snakes=4,
            height=20,
            width=20,
            snake_length=4,
            *args,
            **kwargs
    ):
        """
        kwargs
        'reward_dict', 'num_fruits'
        """
        self.action_dict = SnakeEnv.default_action_dict

        reward_dict = kwargs.pop('reward_dict', SnakeEnv.default_reward_dict)
        if reward_dict.keys() != SnakeEnv.reward_keys:
            raise KeyError(f'reward dict keys must correspond to {SnakeEnv.reward_keys}')
        else:
            self.reward_dict = reward_dict

        self.num_snakes = num_snakes
        self.num_fruits = kwargs.pop('num_fruits', int(round(num_snakes * 0.8)))
        self.grid_shape: Tuple = (height, width)
        self.grid: np.ndarray
        self.snakes: List[Snake]
        self.snake_length = snake_length

        low = 0
        high = 255
        self.action_space = gym.spaces.Discrete(len(self.action_dict))
        self.observation_space = gym.spaces.Box(
            low, high, shape=self.grid_shape, dtype=np.uint8)

    def reset(self):
        self.grid = make_grid(*self.grid_shape,
                              empty_value=Cell.EMPTY.value,
                              wall_value=Cell.WALL.value)
        self.snakes = self._generate_snakes()
        for snake in self.snakes:
            coords = snake.coords
            for coord in coords:
                self.grid[coord] = Cell.BODY.value + 10 * snake.idx
            self.grid[snake.head_coord] = Cell.HEAD.value + 10 * snake.idx
            self.grid[snake.tail_coord] = Cell.TAIL.value + 10 * snake.idx

        xs, ys = self._generate_fruits(self.num_fruits)
        self.grid[xs, ys] = Cell.FRUIT.value

    def seed(self, seed=42):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def render(self, mode='human'):
        # just for debugging
        SYM2CHR = {
            Cell.EMPTY.value: '.',
            Cell.WALL.value: '#',
            Cell.FRUIT.value: 'o',
            Cell.BODY.value: 'b',
            Cell.HEAD.value: 'H',
            Cell.TAIL.value: 't'
        }

        CHR2SYM = {v: k for k, v in SYM2CHR.items()}

        data = np.vectorize(SYM2CHR.get)(self.grid % 10)
        data = [''.join(i) for i in data]
        data = '\n'.join(data)
        print(data)

    def close(self):
        pass

    def step(self, actions):
        """
        Time Complexity:
            O(3*num_snakes)
            + (O(H*W) w/ fruits(maybe better approach, like random polling))

        updates snake status and grid
        all snake statuses are updated only in step method
        """

        if isinstance(actions, int):
            actions = [actions]
        # preprocess
        next_head_coords = defaultdict(list)
        for snake, action in zip(self.snakes, actions):
            if snake.alive:
                snake.direction = self._next_direction(snake.direction, action)
                next_head_coords[snake.head_coord + snake.direction].append(snake.idx)
        dead_idxes, fruit_idxes = self._check_collision(next_head_coords)

        for idx in dead_idxes:
            self.snakes[idx].death = True
            self.snakes[idx].alive = False
        for idx in fruit_idxes:
            tail_coord = self.snakes[idx].tail_coord
            if tail_coord in next_head_coords.keys():
                for s in [self.snakes[di] for di in next_head_coords[tail_coord]]:
                    s.death = True
                    s.alive = False
                    self.snakes[idx].kill += 1
            self.snakes[idx].fruit = True

        rews = []
        dones = []
        # postprocess
        for snake in self.snakes:
            if not snake.death and not snake.alive:
                snake.reward = 0.
            else:
                snake.reward = self.reward_dict['time'] * snake.alive
                snake.reward += self.reward_dict['fruit'] * snake.fruit
                snake.reward += self.reward_dict['lose'] * snake.death
                snake.reward += self.reward_dict['kill'] * snake.kills
                # TODO
                # snake.reward += self.reward_dict['win'] * snake.win
                self._update_grid(snake)

            rews.append(snake.reward)
            dones.append(not snake.alive)

        obs = self._encode(self.grid)

        return obs, rews, dones, None

    def _encode(self, obs):

        return obs

    def _check_collision(self, next_head_coords):
        dead_idxes = []
        fruit_idxes = []
        for coord, idxes in next_head_coords.items():
            cell_value = self.grid[coord] % 10
            if len(idxes) > 1 or cell_value in (Cell.WALL.value,
                                                Cell.BODY.value):
                dead_idxes.extend(idxes)
                if cell_value == Cell.BODY.value:
                    self.snakes[self.grid[coord] // 10].kills += 1
            elif len(idxes) == 1 and cell_value == Cell.FRUIT.value:
                fruit_idxes.extend(idxes)
        dead_idxes = list(set(dead_idxes))
        fruit_idxes = fruit_idxes
        return dead_idxes, fruit_idxes

    def _update_grid(self, snake):
        if snake.alive:
            self.grid[snake.head_coord] = Cell.BODY.value + 10 * snake.idx
            # could be current or prev if ate fruit
            prev_tail_coord = snake.move()
            if prev_tail_coord:
                # Didn't eat the fruit
                self.grid[prev_tail_coord] = Cell.EMPTY.value
            self.grid[snake.head_coord] = Cell.HEAD.value + 10 * snake.idx

            self.grid[snake.tail_coord] = Cell.TAIL.value + 10 * snake.idx
        else:
            if draw(self.grid, snake.coords, Cell.EMPTY.value) is False:
                print('draw failed')
            snake.move()

    def _check_overlap(self, list_of_coords):
        flat_list = []
        for element in list_of_coords:
            flat_list.extend(element)
        unique_list = list(set(flat_list))
        return len(unique_list) == len(flat_list)

    def _generate_snakes(self):
        candidates = dfs_sweep_empty(self.grid, self.snake_length)
        sample_idx = np.random.permutation(len(candidates))[:self.num_snakes]
        samples = [candidates[si] for si in sample_idx]
        while not self._check_overlap(samples):
            sample_idx = np.random.permutation(len(candidates))[:self.num_snakes]
            samples = [candidates[si] for si in sample_idx]
        snakes = [Snake(idx, coords) for idx, coords in enumerate(samples)]

        return snakes

        # snakes = []
        # for idx in range(self.num_snakes):
        #     coord = random_empty_coord(self.grid)
        #     snakes.append(Snake(idx, coord, Direction.RIGHT))

        # return snakes

    def _generate_fruits(self, num_fruits=1):
        xs, ys = random_empty_coords(self.grid, num_coords=num_fruits)

        return xs, ys

    def _next_direction(self, direction, action):
        """
        0 == noop
        1 == left
        2 == right
        Change direction by +-90 degrees
        """
        angle = math.atan2(direction.value[0], direction.value[1])
        new_coord = (int(math.sin(angle + self.action_angle_dict[action])),
                     int(math.cos(angle + self.action_angle_dict[action])))
        return Direction(new_coord)



# TODO:
# encoding ->
# reward engineering -> wrapper
# fruits -> cell list
# coord vs pos(position) vs coord(coordinate)
# pygame
# 1player survive win
# body hit other head kill

# custom reward structure
# obs -> cell states
# win terminal condition setting

