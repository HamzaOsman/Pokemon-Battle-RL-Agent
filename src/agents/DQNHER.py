import gymnasium as gym
import math
import random
from collections import namedtuple, deque
import time

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from pokemon_battle_env import PokemonBattleEnv
import numpy as np
from helpers import evaluate, featurize
from matplotlib import pyplot as plt
import configparser

config = configparser.ConfigParser()
config.read('config.ini')

# if GPU is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward', 'goal', 'done'))
print(device)

class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)
    
class DQNHER(nn.Module):

    def __init__(self, n_observations, n_actions):
        super(DQNHER, self).__init__()
        self.layer1 = nn.Linear(n_observations, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, n_actions)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)

# async def trainModel(env: PokemonBattleEnv, gen=1, max_episode=1, learnFromPrevModel=False):  
async def trainModel(env: PokemonBattleEnv, 
                     gen=1, 
                     max_episode=1, 
                     learnFromPrevModel=False,
                     GAMMA = 0.99,
                     EPS_START = 0.9,
                     LR = 1e-4):  

    def optimize_model():
        if len(memory) < BATCH_SIZE:
            return
        transitions = memory.sample(BATCH_SIZE)

        batch = Transition(*zip(*transitions))

        state_batch = torch.cat(batch.state)
        next_state_batch = torch.cat(batch.next_state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)
        goal_batch = torch.cat(batch.goal)
        done_batch = torch.cat(batch.done)

        Q_values = policy_net(torch.cat([state_batch, goal_batch], dim=-1)).gather(1, action_batch)

        with torch.no_grad():
            Q_values_target = reward_batch + GAMMA*target_net(torch.cat([next_state_batch, goal_batch], dim=-1)).max(1).values*(~done_batch)
        
        # Compute Huber loss
        criterion = nn.SmoothL1Loss()
        loss = criterion(Q_values, Q_values_target.unsqueeze(1))

        # Optimize the model
        optimizer.zero_grad()
        loss.backward()
        
        # In-place gradient clipping
        torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
        optimizer.step()

    n_actions = env.action_space.n
    n_state_observations = env.observation_space.shape[0]
    n_goal_observations = env.goal_space.shape[0]

    if(learnFromPrevModel and gen > 1):
        policy_net = loadModel(env, f'./models/DQNHER_model_gen{gen-1}.pth')
        target_net = DQNHER(n_state_observations+n_goal_observations, n_actions).to(device)
        target_net.load_state_dict(policy_net.state_dict())
    else:
        policy_net = DQNHER(n_state_observations+n_goal_observations, n_actions).to(device)
        target_net = DQNHER(n_state_observations+n_goal_observations, n_actions).to(device)
        target_net.load_state_dict(policy_net.state_dict())
        
    #Hyper parameters
    BATCH_SIZE = 128
    EPS_END = 0.05
    EPS_DECAY = max_episode*2
    TAU = 0.005
    K_FUTURE = 4
    
    optimizer = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)
    memory = ReplayMemory(10000)
    steps_done = 0

    evaluate_every = int(config.get("Agent Configuration", "evaluate_every"))
    evaluation_runs = int(config.get("Agent Configuration", "evaluation_runs"))
    eval_returns = []
    eval_winrates = []

    wins = 0
    losses = 0
    ties = 0

    start_time = time.time()
    for i in range(max_episode):
        state, info = await env.reset()
        goal = env.goal_featurizer(env.goal_space.sample())
        goal = torch.tensor(goal, dtype=torch.float32, device=device).unsqueeze(0)
        state = featurize(env, state)
        state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        episode = []
        terminated = truncated = False
        while not terminated and not truncated:
            state_goal_pair = torch.cat([state, goal], dim=-1)
            action = select_action(env, state_goal_pair, policy_net, EPS_START, EPS_END, EPS_DECAY, steps_done)
            steps_done+=1
            next_state, _, terminated, truncated, info = await env.step(action.item())
            reward = compute_reward(env.goal_mapping(state), goal)
            reward = torch.tensor([reward], device=device)
            next_state = featurize(env, next_state)
            next_state = torch.tensor(next_state, dtype=torch.float32, device=device).unsqueeze(0)
            done = torch.tensor([terminated], device=device)

            episode.append((state, action, next_state, reward, done))

            # Store the transition in memory
            memory.push(state, action, next_state, reward, goal, done)

            # Move to the next state
            state = next_state

            # Perform one step of the optimization (on the policy network)
            optimize_model()

            # Soft update of the target network's weights
            target_net_state_dict = target_net.state_dict()
            policy_net_state_dict = policy_net.state_dict()
            for key in policy_net_state_dict:
                target_net_state_dict[key] = policy_net_state_dict[key]*TAU + target_net_state_dict[key]*(1-TAU)
            target_net.load_state_dict(target_net_state_dict)

        # HER
        for j, transition in enumerate(episode):
            state, action, next_state, reward, done = transition
            future_transitions = random.choices(episode[j:], k=K_FUTURE) # randomly sample k future states as new goals
            for future_transition in future_transitions:
                new_goal = env.goal_mapping(future_transition[2])
                reward = compute_reward(env.goal_mapping(next_state), new_goal)
                reward = torch.tensor([reward], device=device)
                memory.push(state, action, next_state, reward, new_goal, done)

        if (i+1) % evaluate_every == 0:
            #TODO define main goal
            params = {'model':policy_net, 'goal': env.goal_featurizer(env.goal_space.sample())}
            progress_msg = "DQNHER evaluation ("+str(len(eval_returns)+1)+"/"+str(max_episode//evaluate_every)+")"
            eval_return, win_rate = await evaluate(env, params, greedyPolicy, progress_msg, evaluation_runs)
            eval_returns.append(eval_return)
            eval_winrates.append(win_rate)

        wins, losses, ties = wins+info["result"][0], losses+info["result"][1], ties+info["result"][2]
    
    plt.figure()
    plt.title('DQNHER Returns')
    plt.xlabel("Evaluation Steps")
    plt.ylabel("Evaluation Results")
    plt.plot(eval_returns)
    plt.savefig(f'./plots/DQNHER_plot_returns{gen}.png')
    plt.close()

    plt.figure()
    plt.title('DQNHER Winrate')
    plt.xlabel("Evaluation Steps")
    plt.ylabel("Win Rate %")
    plt.ylim(0, 1)
    plt.plot(eval_winrates)
    plt.savefig(f'./plots/DQNHER_plot_winrate{gen}.png')
    plt.close()

    policy_net = policy_net.to('cpu')
    torch.save(policy_net.state_dict(), f'./models/DQNHER_model_gen{gen}.pth')

    await env.close()

# TODO: better reward structure
def compute_reward(achieved_state, desired_state):
    achieved_state, desired_state = achieved_state.to(device), desired_state.to(device)
    if torch.sum(torch.eq(achieved_state, desired_state)) == desired_state.size(dim=1):
        reward = 0
    else:
        reward = -1
    return reward

def greedyPolicy(state, params, valid_action_mask):
    with torch.no_grad():
        model = params['model']
        goal = torch.tensor(params['goal'], dtype=torch.float32, device=device)
        state = torch.tensor(state, dtype=torch.float32, device=device)
        state_goal_pair = torch.cat([state, goal], dim=-1)
        action_values = model(state_goal_pair).squeeze()
        valid_actions = np.where(valid_action_mask)[0]
        valid_actions = torch.tensor(valid_actions, device=device)
        a = valid_actions[torch.argmax(action_values[valid_actions])]
        return a.view(1, 1).item()

def select_action(env:PokemonBattleEnv, state, policy_net, EPS_START=0, EPS_END=0, EPS_DECAY=1, steps_done=1):

    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * steps_done / EPS_DECAY)        

    if sample >= eps_threshold:
        with torch.no_grad():
            action_values = policy_net(state).squeeze()
            valid_actions = np.where(env.valid_action_space_mask())[0]
            valid_actions = torch.tensor(valid_actions, device=device)
            a = valid_actions[torch.argmax(action_values[valid_actions])]
            return a.view(1, 1)
    else:
        return torch.tensor([[env.action_space.sample(env.valid_action_space_mask())]], device=device, dtype=torch.long)

async def runGreedyDQNHERAgent(env: PokemonBattleEnv, gen=1,  max_episodes=1):
    wins = losses = ties = 0
    
    model_file = f'./models/DQNHER_model_gen{gen}.pth'
    model = loadModel(env, model_file)

    model.eval()    
    goal = env.goal_featurizer(env.goal_space.sample())
    goal = torch.tensor(goal, dtype=torch.float32, device=device).unsqueeze(0)
    for i in range(max_episodes):
        state, info = await env.reset()
        state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        terminated = truncated = False
        while not (terminated or truncated):
            state_goal_pair = torch.cat([state, goal], dim=-1)
            action = select_action(env, state_goal_pair, model)
            observation, _, terminated, truncated, info = await env.step(action.item())
            state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)

        wins, losses, ties = wins+info["result"][0], losses+info["result"][1], ties+info["result"][2]

    await env.close()
    return wins, losses, ties

def loadModel(env: PokemonBattleEnv, model_path: str):
    try:
        model = DQNHER(env.observation_space.shape[0]+env.goal_space.shape[0], env.action_space.n).to(device)
        model_state = torch.load(model_path, map_location=device)
        model.load_state_dict(model_state)
        return model
    except Exception as e:
        print(e)
        print(f"Error loading DQNHER model")
        exit()