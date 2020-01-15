import gym
import envs
import argparse
import numpy as np
import random
import torch
import torch.nn as nn
import time
from PPO import Memory, ActorCritic, ConvNet, PPO
from torch.utils.tensorboard import SummaryWriter

import torchvision

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class queue():
    def __init__(self, size=20):
        self.mem = []
        self.size = size 

    def push(self, item):
        self.mem.append(item)
        if len(self.mem) > self.size:
            self.mem.pop(0)
            assert len(self.mem) == self.size 

        return sum(self.mem) / len(self.mem) 

def gridpath(args):
        
    env = gym.make('GridPath-v0')

    env.reset()
    done_n = [False for _ in range(env.n_agents)]
    env.action_space[0].np_random.seed(123)
    totalreward = np.zeros(4)

    while not all(done_n):
        
        actions = []
        if args.render:  
            env.render_graphic()
        env.render()

        for i in range(env.n_agents):
            actions.append(env.action_space[i].sample())

        s, r, done_n, _ = env.step(actions)
        time.sleep(0.05)
    
    print("REWARDS: " , totalreward)
    if args.render:  
        env.render_graphic()

    env.render()

    env.close()


def gridexplore(args):
        
    env = gym.make('GridExplore-v1')

    env.reset()
    done_n = [False for _ in range(env.n_agents)]
    env.action_space[0].np_random.seed(123)
    totalreward = np.zeros(4)

    while not all(done_n):
        
        actions = []
        if args.render:  
            env.render_graphic()

        env.render()
        for i in range(env.n_agents):
            actions.append(env.action_space[i].sample())

        s, r, done_n, _ = env.step(actions)

        totalreward  = totalreward + r 
        time.sleep(0.05)
    
    print("REWARDS: " , totalreward)
    if args.render:  
        env.render_graphic()

    env.render()

    env.close()
      
def test(args):

    env = gym.make('GridExplore-v0')
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    state_dim = env.observation_space[0].shape[0]

    action_dim = 5
    

    render = args.render
    max_timesteps = 500
    n_latent_var = 512           # number of variables in hidden layer
    lr = 0.001
    betas = (0.9, 0.999)
    gamma = 0.99                # discount factor
    K_epochs = 2                # update policy for K epochs
    eps_clip = 0.2              # clip parameter for PPO
    #############################################

    
    filename = str(input("filename: "))
    
    memory = Memory()
    ppo = PPO(state_dim, action_dim, n_latent_var, lr, betas, gamma, K_epochs, eps_clip)
    
    print(ppo.policy_old.state_dict)
    ppo.policy_old.load_state_dict(torch.load(filename, map_location=torch.device('cpu')))
    
    avg=0
    for i in range(10):
            
        s = env.reset()
        done_n = [False for _ in range(env.n_agents)]

        totalreward = 0 
        t= 0
        while not all(done_n):
            t+=1

            actions = []
            env.render()
            if render:
                env.render_graphic()        
            state=np.array([s])
            
            state = torch.from_numpy(state).float().to(device)

            action = ppo.policy_old.act(state, memory)
            state, r, done_n, _ = env.step([action])

            totalreward  = totalreward + r 
            time.sleep(0.01)
            if t > 500:
                break

        print("REWARDS: " , totalreward)
        avg += totalreward
    
    if render:
        env.render_graphic()        

    env.render()

    env.close()

    print("AVG REWARD: " , avg/10)

def train(args):
    ############## Hyperparameters ##############
    env_name = "GridExplore-v0"
    # creating environment
    env = gym.make(env_name)
    state_dim = env.observation_space[0].shape[0]

    action_dim = 5
    model = ConvNet(action_dim).to(device)

    render = False
    solved_reward = 50         # stop training if avg_reward > solved_reward
    log_interval = 20           # print avg reward in the interval
    max_episodes = 10000        # max training episodes
    max_timesteps = 500         # max timesteps in one episode
    n_latent_var = 64           # number of variables in hidden layer
    update_timestep = 2400      # update policy every n timesteps
    lr = 0.0001
    betas = (0.9, 0.999)
    gamma = 0.99                # discount factor
    K_epochs = 2                # update policy for K epochs
    eps_clip = 0.2              # clip parameter for PPO
    random_seed = None
    mini_batch_size = 32
    #############################################
    
    if random_seed:
        torch.manual_seed(random_seed)
        env.seed(random_seed)
    
    ppo = PPO(state_dim, action_dim, n_latent_var, lr, betas, gamma, K_epochs, eps_clip)
    print(lr,betas)

    # buffer = {key:value for key,value in memory.__dict__.items() if not key.startswith('__') and not callable(key)}

    
  
    # logging variables
    running_reward = 0
    avg_length = 0
    timestep = 0
    writer = SummaryWriter("logs")

    memory = Memory()
    q = queue(20)

    # training loop
    for i_episode in range(1, max_episodes+1):
        state = env.reset()
        
        # print("length of state arr is : " ,type(state))
        for t in range(max_timesteps):
            timestep += 1
           # env.render()

            state = np.array([state])

            outputs = torch.from_numpy(state).float().to(device)

            # Running policy_old:
            action = ppo.policy_old.act(outputs, memory)
            state, reward, done, _ = env.step([action])

            # Saving reward and is_terminal:
            memory.rewards.append(reward)
            memory.dones.append(done[0])
            
            # update if its time
            if timestep % update_timestep == 0:
                ppo.update(memory)
                memory.clear_memory()
                timestep = 0
            
            running_reward += reward[0]
            if render:
                env.render()
            if all(done):
                break
        
        avg = q.push(running_reward)
        avg_length += t
        
        writer.add_scalar('i_episode/avg_reward', avg , i_episode)
        
        grid = torchvision.utils.make_grid(torch.tensor(env.grid))
        writer.add_image('images', grid, max_timesteps)


        # stop training if avg_reward > solved_reward
        if running_reward > (log_interval*solved_reward):
            print("########## Solved! ##########")
            torch.save(ppo.policy.state_dict(), './PPO_{}.pth'.format(env_name))
            torch.save(ppo.policy.state_dict(), './savedmodels/PPO_{}.pth'.format(time.strftime("%Y%m%d-%H%M%S")))
            break
            
        # logging
        if i_episode % log_interval == 0:
            avg_length = int(avg_length/log_interval)
            running_reward = int((running_reward/log_interval))
            
            print('Episode {} \t avg length: {} \t reward: {}'.format(i_episode, avg_length, running_reward))
            running_reward = 0
            avg_length = 0

    writer.close()
    torch.save(ppo.policy.state_dict(), './PPO_NOTSOLVED_{}.pth'.format(env_name))
    torch.save(ppo.policy.state_dict(), './savedmodels/PPO_NOTSOLVED_{}.pth'.format(time.strftime("%Y%m%d-%H%M%S")))


def mptrain(args):
    ############## Hyperparameters ##############
    env_name = "GridExplore-v0"
    # creating environment
    env = gym.make(env_name)
    state_dim = env.observation_space[0].shape[0]

    action_dim = 5
    model = ConvNet(action_dim).to(device)

    render = False
    solved_reward = 200         # stop training if avg_reward > solved_reward
    log_interval = 20           # print avg reward in the interval
    max_episodes = 500        # max training episodes
    max_timesteps = 500         # max timesteps in one episode
    n_latent_var = 128           # number of variables in hidden layer
    update_timestep = 600      # update policy every n timesteps
    lr = 1e-4
    betas = (0.9, 0.999)
    gamma = 0.99                # discount factor
    K_epochs = 4                # update policy for K epochs
    eps_clip = 0.2              # clip parameter for PPO
    random_seed = None
    mini_batch_size = 32
    #############################################
    
    if random_seed:
        torch.manual_seed(random_seed)
        env.seed(random_seed)
    
    ppo = PPO(state_dim, action_dim, n_latent_var, lr, betas, gamma, K_epochs, eps_clip)
    print(lr,betas)

    # buffer = {key:value for key,value in memory.__dict__.items() if not key.startswith('__') and not callable(key)}

    
    num_processes = 4
    multi_envs = [gym.make(env_name) for i in range(num_processes)] 
    multi_mem = []
    for i in range(num_processes):
        multi_mem.append(Memory())


    # logging variables
    running_reward = 0
    avg_length = 0
    timestep = 0
    writer = SummaryWriter("logs/" + time.strftime("%Y%m%d-%H%M%S"))
    q = queue()

    # training loop
    for i_episode in range(1, max_episodes+1):
        
        states = [multi_envs[i].reset() for i in range(num_processes)]        

        for t in range(max_timesteps):
            timestep += 1

            for k in range(num_processes):
                state = np.array([states[k]])
                outputs = torch.from_numpy(state).float().to(device)

                # Running policy_old:
                action = ppo.policy_old.act(outputs, multi_mem[k])
                state, reward, done, _ = multi_envs[k].step([action])

                # Saving reward and is_terminal:
                multi_mem[k].rewards.append(reward)
                multi_mem[k].dones.append(done[0])

                running_reward += reward[0]
                
                if done:
                    states[k] = multi_envs[k].reset()
                    avg = q.push(running_reward)

            # update if its time
            if timestep % update_timestep == 0:

                for k in range(num_processes):
                    memory = multi_mem[k]
                    # memory = multi_mem.flatten().tolist()
                    ppo.update(memory)
                    # for k in range(num_processes):
                    multi_mem[k].clear_memory()
                    timestep = 0
            
            
            if render:
                env.render()
            if all(done):
                break
                
        avg_length += t 
        
        running_reward /= num_processes
        avg = q.push(running_reward)

        
        # grid = torchvision.utils.make_grid(torch.tensor(env.grid))
        # writer.add_image('images', grid, max_timesteps)

        
        
        # stop training if avg_reward > solved_reward
        if running_reward > (log_interval*solved_reward):
            print("########## Solved! ##########")
            torch.save(ppo.policy.state_dict(), './PPO_{}.pth'.format(env_name))
            torch.save(ppo.policy.state_dict(), './savedmodels/PPO_{}.pth'.format(time.strftime("%Y%m%d-%H%M%S")))
            break
            
        # logging
        if i_episode % log_interval == 0:
            avg_length = int(avg_length/log_interval)
            running_reward = int((running_reward/log_interval))
            writer.add_scalar('episode/average_reward', avg, i_episode)

            print('Episode {} \t avg length: {} \t reward: {}'.format(i_episode, avg_length, running_reward))
            running_reward = 0
            avg_length = 0

    writer.close()
    torch.save(ppo.policy.state_dict(), './PPO_NOTSOLVED_{}.pth'.format(env_name))
    torch.save(ppo.policy.state_dict(), './savedmodels/PPO_NOTSOLVED_{}.pth'.format(time.strftime("%Y%m%d-%H%M%S")))



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ML2-MULTIAGENT ENVS")

    parser.add_argument("--mode", type=str, default='gridexplore')
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument('--render', dest='render', action='store_true')

    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    globals()[args.mode](args)


