import gymnasium as gym
import math
import random
from collections import namedtuple, deque

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from pokemon_battle_env import PokemonBattleEnv
import numpy as np

# if GPU is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))
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
    
class DQN(nn.Module):

    def __init__(self, n_observations, n_actions):
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(n_observations, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, n_actions)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)

async def trainModel(env: PokemonBattleEnv, max_episode=1, previousModel=None):  
    def optimize_model():
        if len(memory) < BATCH_SIZE:
            return
        transitions = memory.sample(BATCH_SIZE)

        batch = Transition(*zip(*transitions))

        # Compute a mask of non-final states and concatenate the batch elements
        # (a final state would've been the one after which simulation ended)
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                            batch.next_state)), device=device, dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in batch.next_state
                                                    if s is not None])
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        # Compute Q(s_t, a) - the model computes Q(s_t)
        state_action_values = policy_net(state_batch).gather(1, action_batch)

        # Compute V(s_{t+1}) for all next states.
        next_state_values = torch.zeros(BATCH_SIZE, device=device)
        with torch.no_grad():
            next_state_values[non_final_mask] = target_net(non_final_next_states).max(1).values
        
        # Compute the expected Q values
        expected_state_action_values = (next_state_values * GAMMA) + reward_batch

        # Compute Huber loss
        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

        # Optimize the model
        optimizer.zero_grad()
        loss.backward()
        
        # In-place gradient clipping
        torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
        optimizer.step()

    n_actions = env.action_space.n
    n_observations = env.observation_space.shape[0]

    if(previousModel):
        policy_net = loadModel(env, previousModel)
        target_net = DQN(n_observations, n_actions).to(device)
        target_net.load_state_dict(policy_net.state_dict())
    else:
        policy_net = DQN(n_observations, n_actions).to(device)
        target_net = DQN(n_observations, n_actions).to(device)
        target_net.load_state_dict(policy_net.state_dict())
        
    BATCH_SIZE = 128
    GAMMA = 0.99
    EPS_START = 0.9
    EPS_END = 0.05
    EPS_DECAY = max_episode*2
    TAU = 0.005
    LR = 1e-4
    
    optimizer = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)
    memory = ReplayMemory(10000)
    steps_done = 0
    
    wins = 0
    losses = 0
    ties = 0
    for i_episode in range(max_episode):
        state, info = await env.reset()
        state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)

        terminated = truncated = False
        while not terminated and not truncated:
            action = select_action(env, state, policy_net, EPS_START, EPS_END, EPS_DECAY, steps_done)
            steps_done+=1
            observation, reward, terminated, truncated, info = await env.step(action.item())
            reward = torch.tensor([reward], device=device)
            next_state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)

            # Store the transition in memory
            memory.push(state, action, next_state, reward)

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

        wins, losses, ties = wins+info["result"][0], losses+info["result"][1], ties+(info['result'] == (0,0))
    
    await env.close()
    print(f"DQNAgent record:\ngames played: {max_episode}, wins: {wins}, losses: {losses}, win percentage: {wins/max_episode}")
    
    policy_net = policy_net.to('cpu')
    torch.save(policy_net.state_dict(), './models/DQN_model.pth')

def select_action(env, state, policy_net, EPS_START=0, EPS_END=0, EPS_DECAY=1, steps_done=1):
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

async def testModel(env: PokemonBattleEnv, model: DQN, max_episodes=1):
    wins = losses = ties = 0
    model.eval()    
    for i in range(max_episodes):
        state, info = await env.reset()
        state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)

        terminated = truncated = False
        while not (terminated or truncated):
            action = select_action(env, state, model)
            observation, _, terminated, truncated, info = await env.step(action.item())
            state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)

        wins, losses, ties  = wins+info["result"][0], losses+info["result"][1], ties+(info['result'] == (0,0))

    return wins, losses, ties

def loadModel(env: PokemonBattleEnv, model_path: str):
    try:
        model = DQN(env.observation_space.shape[0], env.action_space.n)
        model_state = torch.load(model_path, map_location='cpu')
        model.load_state_dict(model_state)
        return model.to(device)
    except Exception as e:
        print(f"Error loading DQN model")
        exit()