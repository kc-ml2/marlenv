from gym.envs.registration import register

register(
    id='Snake-v0',
    entry_point='marlenv.envs.snake:SnakeEnv',
    max_episode_steps=1e12,
)
