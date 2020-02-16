<p align="center">
  <a href="" rel="noopener">
 <img width=auto height=200px src="https://imgur.com/4Ocdxhz.jpg" alt="Project logo"></a>
</p>

<h3 align="center">ML2-MARL-ENV</h3>

<div align="center">

<!-- [![Status](https://img.shields.io/badge/status-active-success.svg)]() -->
<!-- [![License](https://img.shields.io/badge/license-MIT-blue.svg)](/LICENSE) -->

</div>

---

<p align="center"> Multi-agent Reinforcement learning environments
    <br> 
</p>

## 📝 Table of Contents

- [About](#about)
- [Getting Started](#getting_started)
- [Deployment](#deployment)
- [Usage](#usage)
- [Dependencies](#Dependencies)
<!-- - [TODO](../TODO.md) -->
<!-- - [Contributing](../CONTRIBUTING.md) -->
<!-- - [Authors](#authors) -->
<!-- - [Acknowledgments](#acknowledgement) -->

## 🧐 About <a name = "about"></a>

Winter Intership program for `tmha@ML2`. [Tae Min Ha / github](https://www.github.com/taemin410)

## 🏁 Getting Started <a name = "getting_started"></a>


### Prerequisites

To install requirements for this repo : 

```
pip install -e .
```

To test the envs 
---
## ⛏️ Usage <a name = "Usage"></a>


```
env = gym.make('python_1p-v0')

env.reset()
done = False
while not done:
    obs, reward, done, info = 
      env.step([env.action_space.sample()])
  
    env.render()

env.close()

```

## ⛏️ Dependencies <a name = "Dependencies"></a>

- [Gym](https://gym.openai.com/) - gym environment for RL envs
- [baselines](https://github.com/openai/baselines/) - Reinforcement learning framework baselines
- [pygames](https://www.pygame.org/) - Pygames framework for games ran in python
- tensorflow, numpy, pytorch

## 🔧 Running the tests <a name = "tests"></a>


```
pytest
```


## 🎈 Fully-trained snake game  
### Competition setting
- Each of the agents' goal is to maximize its own reward
- Snake reward is only given when it eats an apple(+1) and when it collides(-1). When a snake hits the wall or the body of the other snake it terminates the agent's game.


### To run trained model via example.py file

```

python example.py --mode=runGUI

```

### To run trained model via docker

```

docker pull taemin410/ml2-python:terminal

docker run taemin410/ml2-python:terminal

-----------------------------------------------
# To save gif image

docker pull taemin410/ml2-python:saveImage

docker run taemin410/ml2-python:saveImage

```

