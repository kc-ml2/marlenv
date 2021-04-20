from gym.envs.registration import register

register(
    id='Snake-v1',
    entry_point='marlenv.envs.snake_env:SnakeEnv',
)
#     max_episode_steps=1e4,


# register(
#     id='SnakeLong-v1',
#     entry_point='marlenv.envs.snake_env:SnakeEnv',
#     max_episode_steps=1e6,
# )
