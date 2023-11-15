import asyncio
import time

import websockets
from agent import Agent
from engine import Engine
from pokemon_battle_env import PokemonBattleEnv

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

    environments = await buildEnv(0, False, agentSocket, opponentSocket)
    
    max_episode = 1

    tasks = []
    for env in environments:
        tasks.append(runRandomAgent(env, max_episode))
        
    await asyncio.gather(*tasks)

async def buildEnv(i: int, isSeparate: bool = True, agentSocket: websockets.WebSocketClientProtocol = None, opponentSocket: websockets.WebSocketClientProtocol = None):
    with open('teams/team1.txt') as f:
        teamstr1 = f.read()

    with open('teams/team2.txt') as f:
        teamstr2 = f.read()

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
    for ep in range(max_episode):
        observation, i = await env.reset()
        while True:
            randoAction = env.action_space.sample(env.valid_action_space_mask())
            observation, reward, terminated, t, i = await env.step(randoAction)
            if terminated: 
                break
    await env.close()

if __name__ == "__main__":
    start_time = time.time()
    # asyncio.run(main())
    asyncio.run(mainSynchronous())
    print(f"Elapsed time:", time.time()-start_time, "s")


