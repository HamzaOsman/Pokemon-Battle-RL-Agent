import asyncio
import time

import websockets

from agents import DQN, actor_critic as AC, qlearning as QL

from player_config import PlayerConfig
from engine import Engine
from pokemon_battle_env import PokemonBattleEnv
import numpy as np

with open('teams/team3.txt') as f:
    teamstr1 = f.read()

with open('teams/team3.txt') as f:
    teamstr2 = f.read()

async def main():
    for g in range(2):
        print("\nGeneration: ", g+1)
        websocketUrl = "ws://localhost:8000/showdown/websocket"
        # agentSocket =  await websockets.connect(websocketUrl)
        # opponentSocket =  await websockets.connect(websocketUrl)
        print("connection went fine")

        # dqnAgentEnv, dqnOpponentEnv = await buildEnv("DQN", await websockets.connect(websocketUrl), await websockets.connect(websocketUrl))
        acAgentEnv, acOpponentEnv = await buildEnv("AC", await websockets.connect(websocketUrl), await websockets.connect(websocketUrl))
        #qlAgentEnv, qlOpponentEnv = await buildEnv("QL", await websockets.connect(websocketUrl), await websockets.connect(websocketUrl))
        
        max_episode = 100

        tasks = []
        
        # TODO: Every time we start 
        if g == 0:
            # tasks.append(DQN.trainModel(dqnAgentEnv, max_episode))
            # tasks.append(runRandomAgent(dqnOpponentEnv, max_episode))

            tasks.append(AC.learnActorCritic(acAgentEnv, max_episode, learnFromPrevModel=False))
            tasks.append(runRandomAgent(acOpponentEnv, max_episode))

            # TODO: QL algo

        else:
            # tasks.append(DQN.trainModel(dqnAgentEnv, max_episode, './models/DQN_model.pth'))
            # tasks.append(runGreedyDQNAgent(dqnOpponentEnv, './models/DQN_model.pth', max_episode))

            tasks.append(AC.learnActorCritic(acAgentEnv, max_episode, learnFromPrevModel=True))
            tasks.append(AC.runActorCritic(acOpponentEnv, max_episode))

            # TODO: QL Algo

            
        await asyncio.gather(*tasks)



async def buildEnv(algoName: str, agentSocket: websockets.WebSocketClientProtocol = None, opponentSocket: websockets.WebSocketClientProtocol = None):
    agentUsername = algoName + "-agent"
    opponentUsername = algoName + "-opponent"
    
    agent = PlayerConfig(agentUsername, True, teamstr1)
    opponent = PlayerConfig(opponentUsername, False, teamstr2)

    agentEngine = Engine(agent, opponent.username, agentSocket)
    opponentEngine = Engine(opponent, agent.username, opponentSocket)

    await agentEngine.init(True)
    await opponentEngine.init(True)

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

async def runGreedyDQNAgent(env: PokemonBattleEnv, model_file, max_episode=1):
    wins = 0
    losses = 0
    ties = 0

    model = DQN.loadModel(env, model_file)
    wins, losses, ties  = await DQN.testModel(env, model, max_episode)
    await env.close()

    print(f"runGreedyNNAgent({env.engine.agent.username}) record:\ngames played: {max_episode}, wins: {wins}, losses {losses}, ties: {ties}: win percentage: {wins/max_episode}")
    

if __name__ == "__main__":
    start_time = time.time()
    asyncio.run(main())
    print(f"Elapsed time:", time.time()-start_time, "s")


