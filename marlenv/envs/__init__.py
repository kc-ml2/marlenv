from gym.envs.registration import register

register(
    id='snake-v1',
    entry_point='marlenv.envs.snake_env:SnakeEnv',
)

register(
    id='snake-graph-v1',
    entry_point='marlenv.envs.graph_snake_env:GraphSnakeEnv',
)

register(
    id='snake-coop-v1',
    entry_point='marlenv.envs.coop_snake_env:CoopSnakeEnv'
)