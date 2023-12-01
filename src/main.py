import asyncio
import time

import websockets
from agent import Agent
from engine import Engine
from pokemon_battle_env import PokemonBattleEnv
import numpy as np

with open('teams/team1.txt') as f:
    teamstr2 = f.read()

with open('teams/team2.txt') as f:
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
    
async def mainSynchronous():
    websocketUrl = "ws://localhost:8000/showdown/websocket"
    agentSocket =  await websockets.connect(websocketUrl)
    opponentSocket =  await websockets.connect(websocketUrl)
    print("connection went fine")

    agentEnv, opponentEnv = await buildEnv(0, False, agentSocket, opponentSocket)
    
    max_episode = 1000
    gamma = 0.99
    step_size = 0.001
    epsilon = 0.5

    tasks = []

    # tasks.append(runQLAgent(agentEnv, max_episode, gamma, step_size, epsilon))
    tasks.append(runGreedyAgent(agentEnv, './models/QL_model.npy', max_episode))
    # tasks.append(runRandomAgent(agentEnv, max_episode))
    tasks.append(runRandomAgent(opponentEnv, max_episode))
        
    await asyncio.gather(*tasks)

async def buildEnv(i: int, isSeparate: bool = True, agentSocket: websockets.WebSocketClientProtocol = None, opponentSocket: websockets.WebSocketClientProtocol = None):
    agentUsername = "agent"
    opponentUsername = "opponent"
    if isSeparate:
        agentUsername += str(i)
        opponentUsername += str(i)
    
    agent = Agent(agentUsername, True, teamstr1)
    opponent = Agent(opponentUsername, False, teamstr2)

    agentEngine = Engine(agent, opponent.username, agentSocket)
    opponentEngine = Engine(opponent, agent.username, opponentSocket)

    await agentEngine.init(i == 0 or isSeparate)
    await opponentEngine.init(i == 0 or isSeparate)

    # agentEnv = PokemonBattleEnv(agentEngine, "human")
    agentEnv = PokemonBattleEnv(agentEngine)
    opponentEnv = PokemonBattleEnv(opponentEngine)

    return agentEnv, opponentEnv

async def runRandomAgent(env: PokemonBattleEnv, max_episode=1):
    wins = 0
    losses = 0
    for ep in range(max_episode):
        observation, info = await env.reset()
        while True:
            randoAction = env.action_space.sample(env.valid_action_space_mask())
            observation, reward, terminated, t, info = await env.step(randoAction)
            if terminated: 
                wins, losses = wins+info["result"][0], losses+info["result"][1]
                break
    await env.close()
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
    asyncio.run(mainSynchronous())
    print(f"Elapsed time:", time.time()-start_time, "s")


