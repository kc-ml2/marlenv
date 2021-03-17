import io
import os
import sys

from PIL import Image

import gym
import pytest

from marlenv.envs.snake_env import SnakeEnv


@pytest.fixture
def snake_env():
    custom_rew = {
        'fruit': 1.0,
        'kill': 2.0,
        'lose': 3.0,
        'win': 4.0,
        'time': 0.1,
    }
    env = gym.make('Snake-v1', num_fruits=4, num_snakes=1, reward_dict=custom_rew)

    return env


def rollout(env, n=100, render_mode=None):
    num_snakes = env.num_snakes
    obs = env.reset()
    dones = [False] * num_snakes

    for _ in range(n):
        if not all(dones):
            if render_mode:
                env.render(render_mode)
            ac = [env.action_space.sample() for _ in range(num_snakes)]
            obs, rews, dones, _ = env.step(ac)


def test_rollout(snake_env):
    n_rollouts = 100
    rollout(snake_env, n=n_rollouts)


@pytest.fixture
def processed_snake_env(snake_env):
    n_rollouts = 100
    rollout(snake_env, n=n_rollouts, render_mode='gif')

    return snake_env


def test_save_gif_default(processed_snake_env):
    env = processed_snake_env
    image_dir = env.save_gif()
    assert os.path.exists(image_dir)

    gif = Image.open(image_dir)
    gif.seek(1)

    os.remove(image_dir)


def test_save_gif_fileobj(processed_snake_env):
    env = processed_snake_env

    with io.BytesIO() as fileobj:
        output = env.save_gif(fileobj)

    assert sys.getsizeof(output) > 0


