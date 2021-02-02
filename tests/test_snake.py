import gym

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

def test():
    env = gym.make('Snake-v1', height=10, width=10, num_fruits=4)
    print(env.num_fruits)
    env.reset()
    dones = [False] * 4
    for _ in range(30):
        if not all(dones):
            env.render()
            _, _, dones, _ = env.step([0, 0, 0, 0])
