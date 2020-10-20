import random
import envs
import gym
import argparse
import time
import tensorflow as tf
import numpy as np
from stable_baselines import DQN, PPO2
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines.common.policies import FeedForwardPolicy, register_policy
from stable_baselines.deepq.policies import FeedForwardPolicy as DqnFFPolicy
from stable_baselines.a2c.utils import conv, linear, conv_to_fc


def custom_cnn(scaled_images, **kwargs):
    activ = tf.nn.relu
    layer_1 = activ(conv(scaled_images, 'c1', n_filters=32, filter_size=2,
                         stride=1, init_scale=np.sqrt(2), **kwargs))
    layer_2 = activ(conv(layer_1, 'c2', n_filters=32, filter_size=2,
                         stride=1, init_scale=np.sqrt(2), **kwargs))
    layer_3 = conv_to_fc(layer_2)
    return activ(linear(layer_3, 'fc1', n_hidden=512, init_scale=np.sqrt(2)))


class CustomPolicy(FeedForwardPolicy):
    def __init__(self, *args, **kwargs):
        super(CustomPolicy, self).__init__(*args, **kwargs,
                                           net_arch=[dict(pi=[64],
                                                          vf=[64])],
                                           feature_extraction="cnn",
                                           cnn_extractor=custom_cnn)


class DqnCnnPolicy(DqnFFPolicy):
    def __init__(self, *args, **kwargs):
        super(DqnCnnPolicy, self).__init__(*args, **kwargs,
                                           feature_extraction="cnn",
                                           cnn_extractor=custom_cnn)


# Register the policy, it will check that the name is not already taken
register_policy('CustomPolicy', CustomPolicy)
register_policy('DqnCnnPolicy', DqnCnnPolicy)


def traindqn(args):
    with tf.device('/gpu:0'):
        env = gym.make('python_1p-v0')
        env = DummyVecEnv([lambda: env])

        model = DQN(DqnCnnPolicy, env, verbose=1, learning_rate=0.0001,
                    exploration_fraction=0.4, train_freq=10)
        model.learn(5000000)
        model.save("dqnwithcnn.pth")


def runGUI(args):
    env = gym.make('python_4p-v1', full_observation=False, vision_range=10)
    net = PPO2.load("4p_compete.pth", policy=CustomPolicy)
    gui = envs.ML2PythonGUI(env, args)

    # gui.baselines_run(net)
    gui.run_model(net)


def runGUImulti(args):
    env = gym.make('python_4p-v1', full_observation=False, vision_range=10)
    net = PPO2.load("4p_compete.pth", policy=CustomPolicy)
    # net = PPO2.load("4p_map_32b_2kstep.pth")

    obs = env.reset()

    done_n = [False for _ in range(4)]

    while not all(done_n):
        actions = []
        for i in range(4):
            action, _ = net.predict(obs[i])
            actions.append(action)

        obs, reward, done_n, info = env.step(actions)

        env.render()
        time.sleep(0.05)

    env.close()


def saveImage(args):
    import imageio
    images = []

    env = gym.make('python_4p-v1', full_observation=False, vision_range=10)
    net = PPO2.load("4p_compete.pth", policy=CustomPolicy)
    # net = PPO2.load("4p_map_32b_2kstep.pth")

    obs = env.reset()

    done_n = [False for _ in range(4)]

    while not all(done_n):
        actions = []
        for i in range(4):
            action, _ = net.predict(obs[i])
            actions.append(action)

        obs, reward, done_n, info = env.step(actions)
        img = env.render()

        images.append(img)

    env.close()

    imageio.mimsave('result.gif', images)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ml2_snake")

    parser.add_argument("--mode", type=str, default='runGUI')
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument_group("interface options")
    parser.add_argument("--human", action='store_true')
    parser.add_argument("--cell_size", type=int, default=20)

    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    '''
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    '''
    globals()[args.mode](args)
