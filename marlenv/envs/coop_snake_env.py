from .snake_env import SnakeEnv


class CoopSnakeEnv(SnakeEnv):
    """
        Identical to SnakeEnv's step function except for the terminating
        condition. if all(dones) -> any(dones)
        This is for finishing an episode if there's at least one dead snake.
        step() returns done = [True] * num_snakes if there's a dead snake.
        """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
    def step(self, actions):
        obs, rews, dones, info = super().step(actions)
        if self._done_fn(dones):
            dones = [True] * self.num_snakes

        return obs, rews, dones, info

    def _done_fn(self, dones):
        return any(dones)