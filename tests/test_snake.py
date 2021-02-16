import gym
import numpy as np

from marlenv.envs.snake_env import SnakeEnv

# test_env = SnakeEnv()
#
#
# def test_step():
#     test_env = SnakeEnv()
#
#
# def test_reset():
#     assert False
#
#
# def test_render():
#     assert False
num_snake = 8
# rewards = np.zeros(num_snake)

custom_rew = {
    'fruit': 1.0,
    'kill': 2.0,
    'lose': 3.0,
    'win': 4.0,
    'time': 0.1,
}

def test():
    env = gym.make('Snake-v1', height=10, width=10, num_fruits=4,
                   num_snakes=num_snake, reward_dict=custom_rew)
    print(env.num_fruits)
    obs = env.reset()
    dones = [False] * num_snake
    for _ in range(30):
        if not all(dones):
            env.render()
            ac = [env.action_space.sample() for _ in range(num_snake)]
            obs, rews, dones, _ = env.step(ac)
            print(rews)
