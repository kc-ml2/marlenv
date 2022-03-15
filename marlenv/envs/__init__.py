from gym.envs.registration import register

register(
    id='Snake-v1',
    entry_point='marlenv.envs.snake_env:SnakeEnv',
)

register(
    id='CoopSnake-v1',
    entry_point='marlenv.envs.coop_snake_env:CoopSnakeEnv'
)