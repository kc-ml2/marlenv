# Marlenv

---

Marlenv is a multi-agent environment for reinforcement learning, based on the OpenAI [gym](https://github.com/openai/gym) convention. 

The function names such as reset(), step() are consistent but the return format is different. Unlike the single agent environments, the multi-agent environments included in this repo formats all returns in a list format, where each element corresponds to each agent in the environment. A similar rule applies to the input action where the action should be a list of actions with a length of number of agents. 

Marlenv is an ongoing project and modifications and new environments are expected in the future. 

![marlenv_0](https://user-images.githubusercontent.com/5464491/116667372-10367800-a9d7-11eb-8098-4bfbd93e9970.gif)


### Installation

---

clone marlenv repo and use pip to install

```bash
git clone https://github.com/kc-ml2/marlenv.git
cd marlenv
pip install -e .
```

### Rules

---

**Snake Game**

Multiple snakes battle on a fixed size grid map.

Each snake is spawned at a random location on the map, with a random pose and direction at reset().

The map may be initialized with a different walls upon instantiation of the environment.

Snake dies when its head hits a wall or body of another snake. Here, the other snake receives a reward for kill and the dead snake receives a reward for death ('lose').

When multiple snakes collide head to head, all dies without receiving the kill score. 

When there is only one snake remaining, it receives a win reward for every unit time of survival.

The snake grows by one pixel when it has eatten a fruit. 

### Examples Input Arguments

---

**Snake Game** 

Creating an environment

```python
import gym
import marlenv
env = gym.make(
    'Snake-v1',
		height=20,       # Height of the grid map
		width=20,        # Width of the grid map
		num_snakes=4,    # Number of snakes to spawn on grid
		snake_length=3,  # Initail length of the snake at spawn time
		vision_range=5,  # Vision range (both width height), map returned if None
	  frame_stack=1,   # Number of observations to stack on return
		*args,
		**kwargs
)
```

Single-agent wrapper

```python
env = gym.make('Sanke-v1', num_snakes=1)
env = marlenv.wrappers.SingleAgent(env)
```

This will unwrap the returned the observation, reward, etc from a list

Using the make_snake() function

```python
# Automatically chooses wrappers to handle single agent, multi-agent, vector_env, etc.
env, observation_space, action_space, properties = marlenv.wrappers.make_snake(
	  n_env=1,      # Number of environments. Used to decided vector env or not
	  num_snakes=1, # Number of players. Used to determine single/multi agent
	  **kwargs      # Other input parameters to the environment
)
```

The returned values are

- env : The environment object
- observation_space : The processed observation space (according to env type)
- action_space : The processed action space
- properties : The properties is EasyDict object that includes
    - high: highest value that observation can have
    - low: lowest value that the observation can have
    - n_env: number of environments
    - num_snakes: number of snakes to be spawned
    - reorder: True if observation is given 'NHWC', False if 'NCHW'
    - discrete: True if action space is discrete, categorical
    - action_info
        - {action_high, action_low} if continuous action or {action_n} if discrete

**Custom reward function**

The user can change the reward function structure of the snake-game upon instantiation. 

The reward function can be defined using python dictionary as the following

```python
custom_rewardf = {
	  'fruit': 1.0,
	  'kill': 0.0,
	  'lose': 0.0,
	  'time': 0.0,
	  'win': 0.0
}
env = gym.make('Snake-v1', reward_func=custom_rewardf)
```

Each of the each of the keys represent

- fruit : reward received when the snake eats a fruit
- kill : reward received when the snake kills another snake
- lose : reward (or penalty) received when the snake dies
- time : reward received for each unit of time of survival
- win : reward received during the snake's time of survival as the last one standing

Each reward can be both + and - float number

### Testing

---

```python
pytest
```

### Citation

---

```python
@MISC{marlenv2021,
author =   {ML2},
title =    {Marlenv, Multi-agent Reinforcement Learning Environment},
howpublished = {\url{http://github.com/kc-ml2/marlenv}},
year = {2021}
}
```

### Updates

---

Currently, there is only one environment of multi-agent snake game.
