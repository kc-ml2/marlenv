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
num_snake = 8

def test():
    env = gym.make('Snake-v1', height=20, width=20, num_fruits=4,
                   num_snakes=num_snake)
    print(env.num_fruits)
    env.reset()
    dones = [False] * num_snake
    for _ in range(30):
        if not all(dones):
            env.render()
            ac = [env.action_space.sample() for _ in range(num_snake)]
            _, _, dones, _ = env.step(ac)
