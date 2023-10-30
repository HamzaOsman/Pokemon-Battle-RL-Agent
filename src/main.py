import asyncio
import time

import websockets
from agent import Agent
from engine import Engine
from pokemon_battle_env import PokemonBattleEnv

async def main():
    # group together the tasks for building the battles and wait
    tasks = [buildBattle(i) for i in range(1000)]
    results = await asyncio.gather(*tasks)

    # then group together the actual battles and wait
    tasks = []
    for result in results:
        tasks += result

    results = await asyncio.gather(*tasks)
    

async def mainSynchronous():
    websocketUrl = "ws://localhost:8000/showdown/websocket"
    agentSocket =  await websockets.connect(websocketUrl)
    opponentSocket =  await websockets.connect(websocketUrl)
    print("connection went fine")
    # wait for each battle to complete
    for i in range(1000):
        battleTasks = await buildBattle(i, False, agentSocket, opponentSocket)
        print("making battles fine")
        await asyncio.gather(*battleTasks)
        print("battles completed fine")
    
    await agentSocket.close()
    await opponentSocket.close()

async def buildBattle(i: int, isSeparate: bool = True, agentSocket: websockets.WebSocketClientProtocol = None, opponentSocket: websockets.WebSocketClientProtocol = None):
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

    tasks = [runAgent(agentEngine, agentSocket is None), runAgent(opponentEngine, opponentSocket is None)]
    return tasks

async def runAgent(engine: Engine, closeSocket: bool = True):
    env = PokemonBattleEnv(engine)
    observation = await env.reset()
    reward = 0
    while True:
        randoAction = engine.agent.choose_action(env.engine.battle)
        observation, reward, terminated, t, i = await env.step(randoAction)
        if terminated: break
    # print(reward)
    if closeSocket:
        await env.engine.socket.close()

if __name__ == "__main__":
    start_time = time.time()
    asyncio.get_event_loop().run_until_complete(mainSynchronous())
    print(f"Elapsed time:", time.time()-start_time, "s")


