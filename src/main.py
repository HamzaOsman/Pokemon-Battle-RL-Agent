import asyncio
import time
from typing import List
import random
import numpy as np

import websockets

from agents import DQN, actor_critic as AC, qlearning as QL, DQNHER

from player_config import PlayerConfig
from engine import Engine
from pokemon_battle_env import PokemonBattleEnv
import configparser

config = configparser.ConfigParser()
config.read('config.ini')

seed_value = 23
random.seed(seed_value)
np.random.seed(seed_value)

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
    elif agentMode == "test":
        numBattles = int(config.get("Agent Configuration", "num_episodes"))
        await testAgents(agentAlgos, numBattles)
    elif agentMode == "experiment":
        numGens = int(config.get("Agent Configuration", "num_generations"))
        numBattles = int(config.get("Agent Configuration", "num_episodes"))
        await runExperiments(agentAlgos, numGens, numBattles)
        

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


# Human playing against agent(s)
async def playVsAgent(algos: List[str], opponentUsername: str):
    agentTeamSize = int(config.get("Battle Configuration", "agent_team_size"))
    opponentTeamSize = int(config.get("Battle Configuration", "opponent_team_size"))
    battleFormat = config.get("Battle Configuration", "battle_format")
    numGens = int(config.get("Agent Configuration", "num_generations"))

    agentTasks = []
    # AC, DQN, QL, DQNHER
    for algo in algos:
        agentUsername = algo + "-agent"
        
        agent = PlayerConfig(agentUsername, True, teamstr1)
        agentEngine = Engine(agent, opponentUsername, battleFormat, await websockets.connect("ws://localhost:8000/showdown/websocket"))

        await agentEngine.init(True)

        agentEnv = PokemonBattleEnv(agentEngine, agentTeamSize, opponentTeamSize)    

        if algo == "DQN":
            agentTasks.append(DQN.runGreedyDQNAgent(agentEnv, numGens, 1))
        elif algo == "AC":
            agentTasks.append(AC.runActorCritic(agentEnv, numGens, 1))
        elif algo == "QL":
            agentTasks.append(QL.runGreedyQLAgent(agentEnv, numGens, max_episode=1))
        elif algo == "DQNHER":
            agentTasks.append(DQNHER.runGreedyDQNHERAgent(agentEnv, numGens, 1))
        # elif algo == "SAC":
        #     agentTasks.append(SAC.runGreedySACAgent(agentEnv, "./models/SAC_model.npy", 1))
        else:
            raise Exception("Trying to play against an unknown agent algorithm!")
        
    await asyncio.gather(*agentTasks)
    
# Agents learning generationally
async def trainAgents(algos, numGenerations = 2, numBattles = 100):
    evaluate_every = int(config.get("Agent Configuration", "evaluate_every"))
    evaluation_runs = int(config.get("Agent Configuration", "evaluation_runs"))
    
    for gen in range(1, numGenerations+1):
        print("\nGeneration:", gen)
        websocketUrl = "ws://localhost:8000/showdown/websocket"

        if("DQN" in algos):
            dqnAgentEnv, dqnOpponentEnv = await buildEnv("DQN", await websockets.connect(websocketUrl), await websockets.connect(websocketUrl))
        if("AC" in algos):
            acAgentEnv, acOpponentEnv = await buildEnv("AC", await websockets.connect(websocketUrl), await websockets.connect(websocketUrl))
        if("QL" in algos):
            qlAgentEnv, qlOpponentEnv = await buildEnv("QL", await websockets.connect(websocketUrl), await websockets.connect(websocketUrl))
        if("DQNHER" in algos):
            dqnherAgentEnv, dqnherOpponentEnv = await buildEnv("DQNHER", await websockets.connect(websocketUrl), await websockets.connect(websocketUrl))
        print("connection went fine, probably")
        
        tasks = []
        
        # TODO: Every time we start 
        if gen == 1:
            if("DQN" in algos):
                tasks.append(DQN.trainModel(dqnAgentEnv, gen, numBattles))
                tasks.append(runRandomAgent(dqnOpponentEnv, numBattles+((numBattles//evaluate_every)*evaluation_runs)))

            if("AC" in algos):
                tasks.append(AC.learnActorCritic(acAgentEnv, gen, numBattles, learnFromPrevModel=False))
                tasks.append(runRandomAgent(acOpponentEnv, numBattles+((numBattles//evaluate_every)*evaluation_runs)))

            if("QL" in algos):
                tasks.append(QL.runQLAgent(qlAgentEnv, gen, numBattles))
                tasks.append(runRandomAgent(qlOpponentEnv, numBattles+((numBattles//evaluate_every)*evaluation_runs)))

            if("DQNHER" in algos):
                tasks.append(DQNHER.trainModel(dqnherAgentEnv, gen, numBattles))
                tasks.append(runRandomAgent(dqnherOpponentEnv, numBattles+((numBattles//evaluate_every)*evaluation_runs)))

        else:
            if("DQN" in algos):
                tasks.append(DQN.trainModel(dqnAgentEnv, gen, numBattles, learnFromPrevModel=True))
                tasks.append(DQN.runGreedyDQNAgent(dqnOpponentEnv, gen-1, numBattles+((numBattles//evaluate_every)*evaluation_runs)))

            if("AC" in algos):
                tasks.append(AC.learnActorCritic(acAgentEnv, gen, numBattles, learnFromPrevModel=True))
                tasks.append(AC.runActorCritic(acOpponentEnv, gen-1, numBattles+((numBattles//evaluate_every)*evaluation_runs)))

            if("QL" in algos):
                tasks.append(QL.runQLAgent(qlAgentEnv, gen, numBattles, learnFromPrevModel=True))
                tasks.append(QL.runGreedyQLAgent(qlOpponentEnv, gen-1, numBattles+((numBattles//evaluate_every)*evaluation_runs)))

            if("DQNHER" in algos):
                tasks.append(DQNHER.trainModel(dqnherAgentEnv, gen, numBattles, learnFromPrevModel=True))
                tasks.append(DQNHER.runGreedyDQNHERAgent(dqnherOpponentEnv, gen-1, numBattles+((numBattles//evaluate_every)*evaluation_runs)))
        await asyncio.gather(*tasks)

async def testAgents(algos, numBattles=200):
    def createTask(algo, agent):
        if(algo == "QL"):
            tasks.append(QL.runGreedyQLAgent(agent, numGens, numBattles))
        elif(algo == "DQN"):
            tasks.append(DQN.runGreedyDQNAgent(agent, numGens, numBattles))
        elif(algo == "AC"):
            tasks.append(AC.runActorCritic(agent, numGens, numBattles))
        elif(algo == "DQNHER"):
            tasks.append(DQNHER.runGreedyDQNHERAgent(agent, numGens, numBattles))

    websocketUrl = "ws://localhost:8000/showdown/websocket"
    numGens = int(config.get("Agent Configuration", "num_generations"))
    tasks = []

    for i, algo1 in enumerate(algos):
        for j, algo2 in enumerate(algos):
            agent1, agent2 = await buildEnv(f'{algo1}-{algo2}', await websockets.connect(websocketUrl), await websockets.connect(websocketUrl))

            createTask(algo1, agent1)

            if(algo1 == algo2): #Default (mirror matchup case)
                tasks.append(runRandomAgent(agent2, numBattles))
                #tasks.append(DQN.testModel(agent2, './models/DQN_model_2.pth', numBattles))
                #tasks.append(AC.runActorCritic(acOpponentEnv, numBattles))
                #tasks.append(QL.runGreedyQLAgent(agent2, models.get("QL"), numBattles))
                continue

            createTask(algo2, agent2)
    
    await asyncio.gather(*tasks)


async def runExperiments(algos, numGens=2, numBattles=1000):
    evaluate_every = int(config.get("Agent Configuration", "evaluate_every"))
    evaluation_runs = int(config.get("Agent Configuration", "evaluation_runs"))
    websocketUrl = "ws://localhost:8000/showdown/websocket"
    
    print("running eXperiments")

    # DQN and AC
    gammas = [0.9, 0.95, 0.99]

    # DQN
    lrs = [1e-6, 1e-5, 1e-4]
    epsilons = [0.25, 0.5, 0.9]
    
    # AC
    actorLRs = [0.0005, 0.005, 0.05]
    criticLRs = [0.0005, 0.005, 0.05]

    for gen in range(1, numGens+1):
        print("gen", gen)

        if("DQN" in algos):
            print("doing dqn!")
            for gamma in gammas:
                for lr in lrs:
                    for epsilon in epsilons:
                        tasks = []
                        dqnAgentEnv, dqnOpponentEnv = await buildEnv(f"DQN", await websockets.connect(websocketUrl), await websockets.connect(websocketUrl))
                        tasks.append(DQN.trainModel(dqnAgentEnv, gen, numBattles, GAMMA=gamma, LR=lr, EPS_START=epsilon))
                        tasks.append(runRandomAgent(dqnOpponentEnv, numBattles+((numBattles//evaluate_every)*evaluation_runs)))
                        await asyncio.gather(*tasks)

        if("AC" in algos):
            print("doing AC!")
            for gamma in gammas:
                for actorLR in actorLRs:
                    for criticLR in criticLRs:
                        tasks = []
                        acAgentEnv, acOpponentEnv = await buildEnv(f"AC", await websockets.connect(websocketUrl), await websockets.connect(websocketUrl))                        
                        tasks.append(AC.learnActorCritic(acAgentEnv, gen, numBattles, gamma, actorLR, criticLR, learnFromPrevModel=False))
                        tasks.append(runRandomAgent(acOpponentEnv, numBattles+((numBattles//evaluate_every)*evaluation_runs)))
                        await asyncio.gather(*tasks)

        if("DQNHER" in algos):
            for gamma in gammas:
                for lr in lrs:
                    for epsilon in epsilons:
                        tasks = []
                        dqnAgentEnv, dqnOpponentEnv = await buildEnv(f"DQNHER", await websockets.connect(websocketUrl), await websockets.connect(websocketUrl))
                        tasks.append(DQNHER.trainModel(dqnAgentEnv, gen, numBattles, GAMMA=gamma, LR=lr, EPS_START=epsilon))
                        tasks.append(runRandomAgent(dqnOpponentEnv, numBattles+((numBattles//evaluate_every)*evaluation_runs)))
                        await asyncio.gather(*tasks)


if __name__ == "__main__":
    start_time = time.time()
    asyncio.run(main())
    print(f"Elapsed time:", time.time()-start_time, "s")
