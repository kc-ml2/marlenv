from collections import defaultdict
from .snake_env import SnakeEnv
import numpy as np
from marlenv.core.snake import Cell


class CoopSnakeEnv(SnakeEnv):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # TODO: allow choice for action_dict in SnakeEnv.__init__()
        self.action_dict = SnakeEnv.action_angle_dict

    def step(self, actions):
        """
        Identical to SnakeEnv's step function except for the terminating 
        condition. if all(dones) -> any(dones)
        This is for finishing an episode if there's at least one dead snake.
        step() returns done = [True] * num_snakes if there's a dead snake.
        """

        if isinstance(actions, int):
            actions = [actions]
        assert len(actions) == self.num_snakes
        for i, ac in enumerate(actions):
            if isinstance(ac, np.ndarray):
                actions[i] = ac.item()
        # preprocess
        next_head_coords = defaultdict(list)
        alive_snakes = []
        for snake, action in zip(self.snakes, actions):
            if snake.alive:
                snake.direction = self._next_direction(snake.direction, action)
                new_head_coord = snake.head_coord + snake.direction
                next_head_coords[new_head_coord].append(snake.idx)
                alive_snakes.append(snake.idx)
        dead_idxes, fruit_idxes, fruit_taken = self._check_collision(
            next_head_coords)

        self.alive_snakes -= len(dead_idxes)
        for idx in dead_idxes:
            self.snakes[idx].death = True
            self.snakes[idx].alive = False
        for idx in fruit_idxes:
            tail_coord = self.snakes[idx].tail_coord
            if tail_coord in next_head_coords.keys():
                for di in next_head_coords[tail_coord]:
                    self.snakes[di].death = True
                    self.snakes[di].alive = False
                    self.alive_snakes -= 1
                    self.snakes[idx].kills += 1
            self.snakes[idx].fruit = True
        if self.alive_snakes == 1 and self.num_snakes > 1:
            for snake in self.snakes:
                if snake.alive:
                    snake.win = True
                    break

        rews = []
        dones = []
        fruits, kills = [], []
        # Update snake rews, stats, etc., and update the grid accordingly
        for snake in self.snakes:
            if not snake.death and not snake.alive:
                snake.reward = 0.
                rews.append(snake.reward)
                fruits.append(0)
                kills.append(0)
            else:
                snake.reward = self.reward_dict['time'] * snake.alive
                snake.reward += self.reward_dict['fruit'] * snake.fruit
                snake.reward += self.reward_dict['lose'] * snake.death
                snake.reward += self.reward_dict['kill'] * snake.kills
                snake.reward += self.reward_dict['win'] * snake.win
                rews.append(snake.reward)
                fruits.append(float(snake.fruit))
                kills.append(float(snake.kills))
                self._update_grid(snake)
            dones.append(not snake.alive)

        # Generate fruit and add to the grid
        xs, ys = self._generate_fruits(fruit_taken)
        if xs is not None:
            self.grid[xs, ys] = Cell.FRUIT.value

        obs = self._get_obs()

        done_mask = 1. - np.asarray(dones)
        self.epi_scores = self.epi_scores + done_mask * np.asarray(rews)
        self.epi_steps = self.epi_steps + done_mask * np.ones(len(dones))
        self.epi_fruits = self.epi_fruits + done_mask * np.asarray(fruits)
        self.epi_kills = self.epi_kills + done_mask * np.asarray(kills)

        info = {}
        self.episode_length += 1
        if self.episode_length >= self.max_episode_steps:
            dones = [True] * self.num_snakes

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
