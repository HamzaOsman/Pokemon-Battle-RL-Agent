import asyncio
import time
from typing import List

import websockets

from agents import DQN, actor_critic as AC, qlearning as QL

from player_config import PlayerConfig
from engine import Engine
from pokemon_battle_env import PokemonBattleEnv
import numpy as np
import configparser

config = configparser.ConfigParser()
config.read('config.ini')

try:
    with open(config.get("Battle Configuration", "team1_file_path")) as f:
        teamstr1 = f.read()

    with open(config.get("Battle Configuration", "team2_file_path")) as f:
        teamstr2 = f.read()
except:
    teamstr1 = None
    teamstr2 = None

async def main():
    agentMode = config.get("Agent Configuration", "mode")
    agentAlgos = config.get("Agent Configuration", "algorithms").split(", ")

    if  agentMode == "play":
        humanPlayerUsername = config.get("Player Configuration", "player_username")
        await playVsAgent(agentAlgos, humanPlayerUsername)
    elif agentMode == "train":
        numGens = int(config.get("Agent Configuration", "num_generations"))
        numBattles = int(config.get("Agent Configuration", "num_episodes"))
        await trainAgents(agentAlgos, numGens, numBattles)


async def buildEnv(algoName: str, agentSocket: websockets.WebSocketClientProtocol = None, opponentSocket: websockets.WebSocketClientProtocol = None):
    agentTeamSize = int(config.get("Battle Configuration", "agent_team_size"))
    opponentTeamSize = int(config.get("Battle Configuration", "opponent_team_size"))
    battleFormat = config.get("Battle Configuration", "battle_format")

    agentUsername = algoName + "-agent"
    opponentUsername = algoName + "-opponent"
    
    agent = PlayerConfig(agentUsername, True, teamstr1)
    opponent = PlayerConfig(opponentUsername, False, teamstr2)

    agentEngine = Engine(agent, opponent.username, battleFormat, agentSocket)
    opponentEngine = Engine(opponent, agent.username, battleFormat, opponentSocket)

    await agentEngine.init(True)
    await opponentEngine.init(True)

    # agentEnv = PokemonBattleEnv(agentEngine, "human")
    agentEnv = PokemonBattleEnv(agentEngine, agentTeamSize, opponentTeamSize)
    opponentEnv = PokemonBattleEnv(opponentEngine, opponentTeamSize, agentTeamSize)

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


def featurize(env: PokemonBattleEnv, x):
    x_normalized = (x - env.observation_space.low) / (env.observation_space.high - env.observation_space.low) # min-max normalization
    return x_normalized


# Human playing against agent(s)
async def playVsAgent(algos: List[str], opponentUsername: str):
    agentTeamSize = int(config.get("Battle Configuration", "agent_team_size"))
    opponentTeamSize = int(config.get("Battle Configuration", "opponent_team_size"))
    battleFormat = config.get("Battle Configuration", "battle_format")
    agentTasks = []
    # AC, DQN, QL
    for algo in algos:
        agentUsername = algo + "-agent"
        
        agent = PlayerConfig(agentUsername, True, teamstr1)
        agentEngine = Engine(agent, opponentUsername, battleFormat, await websockets.connect("ws://localhost:8000/showdown/websocket"))

        await agentEngine.init(True)

        agentEnv = PokemonBattleEnv(agentEngine, agentTeamSize, opponentTeamSize)    

        if algo == "DQN":
            agentTasks.append(DQN.runGreedyDQNAgent(agentEnv, './models/DQN_model.pth', 1))
        elif algo == "AC":
            agentTasks.append(AC.runActorCritic(agentEnv, 1))
        elif algo == "QL":
            agentTasks.append(QL.runGreedyQLAgent(agentEnv, "./models/QL_model.npy", 1))
        else:
            raise Exception("Trying to play against an unknown agent algorithm!")
        
    await asyncio.gather(*agentTasks)
    
# Agents learning generationally
async def trainAgents(algos, numGenerations = 2, numBattles = 100):
    for g in range(numGenerations):
        print("\nGeneration: ", g+1)
        websocketUrl = "ws://localhost:8000/showdown/websocket"

        if("DQN" in algos):
            dqnAgentEnv, dqnOpponentEnv = await buildEnv("DQN", await websockets.connect(websocketUrl), await websockets.connect(websocketUrl))
        if("AC" in algos):
            acAgentEnv, acOpponentEnv = await buildEnv("AC", await websockets.connect(websocketUrl), await websockets.connect(websocketUrl))
        if("QL" in algos):
            qlAgentEnv, qlOpponentEnv = await buildEnv("QL", await websockets.connect(websocketUrl), await websockets.connect(websocketUrl))
        print("connection went fine, probably")
        
        tasks = []
        
        # TODO: Every time we start 
        if g == 0:
            if("DQN" in algos):
                tasks.append(DQN.trainModel(dqnAgentEnv, numBattles))
                tasks.append(runRandomAgent(dqnOpponentEnv, numBattles))

            if("AC" in algos):
                tasks.append(AC.learnActorCritic(acAgentEnv, numBattles, learnFromPrevModel=False))
                tasks.append(runRandomAgent(acOpponentEnv, numBattles))

            if("QL" in algos):
                tasks.append(QL.runQLAgent(qlAgentEnv, numBattles))
                tasks.append(runRandomAgent(qlOpponentEnv, numBattles))

        else:
            print("not gen 0")
            if("DQN" in algos):
                tasks.append(DQN.trainModel(dqnAgentEnv, numBattles, './models/DQN_model.pth'))
                tasks.append(DQN.runGreedyDQNAgent(dqnOpponentEnv, './models/DQN_model.pth', numBattles))

            if("AC" in algos):
                tasks.append(AC.learnActorCritic(acAgentEnv, numBattles, learnFromPrevModel=True))
                tasks.append(AC.runActorCritic(acOpponentEnv, numBattles))

            if("QL" in algos):
                # TODO: QL should have a way to start learning from a saved model
                tasks.append(QL.runQLAgent(qlAgentEnv, numBattles))
                tasks.append(QL.runGreedyQLAgent(qlOpponentEnv, "./models/QL_model.npy", numBattles))
        await asyncio.gather(*tasks)

if __name__ == "__main__":
    start_time = time.time()
    asyncio.run(main())
    print(f"Elapsed time:", time.time()-start_time, "s")
