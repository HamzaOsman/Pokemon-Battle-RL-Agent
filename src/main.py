import asyncio
import threading
import time
from A2C_Agent import learnA2C

import websockets
# from a2c import runAdvantageActorCritic
from actor_critic import learnActorCritic, runActorCritic
from agent import Agent
from engine import Engine
from helpers import desynchronize
from pokemon_battle_env import PokemonBattleEnv
import numpy as np

import nest_asyncio
nest_asyncio.apply()


# with open('teams/team1.txt') as f:
#     teamstr2 = f.read()

with open('teams/starters.txt') as f:
    teamstr1 = f.read()

async def main():
    max_episode = 100

    # group together the tasks for building the battles and wait
    tasks = [buildEnv(i) for i in range(max_episode)]
    environmentsList = await asyncio.gather(*tasks)

    # then group together the actual battles and wait
    tasks = []
    for agentEnv, opponentEnv in environmentsList:
        tasks += [runRandomAgent(agentEnv), runRandomAgent(opponentEnv)]

    await asyncio.gather(*tasks)
    
def runAgents(agentFunc, opponentFunc, max_episode=3000):
    websocketUrl = "ws://localhost:8000/showdown/websocket"
    agentSocket =  desynchronize(websockets.connect(websocketUrl))
    opponentSocket =  desynchronize(websockets.connect(websocketUrl))
    print("connection went fine")

    agentEnv, opponentEnv = desynchronize(buildEnv(0, False, agentSocket, opponentSocket))
    print("building env went fine!!")
    tasks = [
        agentFunc(agentEnv, max_episode), 
        opponentFunc(opponentEnv, max_episode)
    ]
    desynchronize(asyncio.gather(*tasks))

def mainSynchronous():
    print("LEARNING AC AGAINST RANDOM")
    runAgents(learnA2C, runRandomAgent, 6000)



async def buildEnv(i: int, isSeparate: bool = True, agentSocket: websockets.WebSocketClientProtocol = None, opponentSocket: websockets.WebSocketClientProtocol = None):
    agentUsername = "agent"
    opponentUsername = "opponent"
    if isSeparate:
        agentUsername += str(i)
        opponentUsername += str(i)
    
    agent = Agent(agentUsername, True, teamstr1)
    opponent = Agent(opponentUsername, False, teamstr1)

    agentEngine = Engine(agent, opponent.username, agentSocket)
    opponentEngine = Engine(opponent, agent.username, opponentSocket)

    await agentEngine.init(i == 0 or isSeparate)
    await opponentEngine.init(i == 0 or isSeparate)
    print("init went fine!!")
    # agentEnv = PokemonBattleEnv(agentEngine, "human")
    agentEnv = PokemonBattleEnv(agentEngine)
    opponentEnv = PokemonBattleEnv(opponentEngine)

    return agentEnv, opponentEnv

async def runRandomAgent(env: PokemonBattleEnv, max_episode=1):
    print("running random agent??")
    wins = 0
    losses = 0
    for ep in range(max_episode):
        print("running random agent??", ep)
        observation, info = desynchronize(env.reset())
        while True:
            randoAction = env.action_space.sample(env.valid_action_space_mask())
            observation, reward, terminated, t, info = desynchronize(env.step(randoAction))
            if terminated: 
                wins, losses = wins+info["result"][0], losses+info["result"][1]
                break
    desynchronize(env.close())
    print(f"runRandomAgent record:\ngames played: {max_episode}, wins: {wins}, losses: {losses}, win percentage: {wins/max_episode}")

def greedyPolicy(x, W, action_mask):
    valid_actions = np.where(action_mask)[0]
    return valid_actions[np.argmax((W.T @ x)[valid_actions])]

def featurize(env: PokemonBattleEnv, x):
    x_normalized = (x - env.observation_space.low) / (env.observation_space.high - env.observation_space.low) # min-max normalization
    return x_normalized

async def runGreedyAgent(env: PokemonBattleEnv, model_file, max_episode=1):
    wins = 0
    losses = 0

    try:
        W = np.load(model_file)
    except:
        print(f"model file \'{model_file}\' not found!")
        exit()

    for i in range(max_episode):
        s, info = await env.reset()
        terminated = truncated = False
        while not (terminated or truncated):
            a = greedyPolicy(s, W, env.valid_action_space_mask())
            s_next, r, terminated, truncated, info = await env.step(a)
            s = s_next
        wins, losses = wins+info["result"][0], losses+info["result"][1]

    await env.close()

    print(f"runGreedyAgent record:\ngames played: {max_episode}, wins: {wins}, losses: {losses}, win percentage: {wins/max_episode}")


async def runQLAgent(env: PokemonBattleEnv, max_episode=1, gamma=0.99, step_size=0.005, epsilon=0.1):
    wins = 0
    losses = 0

    W = np.random.rand(env.observation_space.shape[0], env.action_space.n)

    for i in range(max_episode):
        s, info = await env.reset()
        s = featurize(env, s)
        terminated = truncated = False
        while not (terminated or truncated):
            a = greedyPolicy(s, W, env.valid_action_space_mask()) if np.random.random() < 1-epsilon else env.action_space.sample(env.valid_action_space_mask())
            s_next, r, terminated, truncated, info = await env.step(a)
            s_next = featurize(env, s_next)
            W[:, a] = W[:, a] + step_size * (r+gamma*(1-terminated)*np.amax(W.T @ s_next) - (W.T@s)[a]) * s
            s = s_next
        wins, losses = wins+info["result"][0], losses+info["result"][1]

    await env.close()

    print(f"runQLAgent record:\ngames played: {max_episode}, wins: {wins}, losses: {losses}, win percentage: {wins/max_episode}")

    np.save('./models/QL_model.npy', W)

if __name__ == "__main__":
    start_time = time.time()
    # asyncio.run(main())
    mainSynchronous()
    print(f"Elapsed time:", time.time()-start_time, "s")


