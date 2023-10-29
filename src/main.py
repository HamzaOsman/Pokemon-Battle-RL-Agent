import asyncio
import time
from agent import Agent
from engine import Engine
from pokemon_battle_env import PokemonBattleEnv

async def main():
    start_time = time.time()

    # group together the tasks for building the battles and wait
    tasks = [buildBattle(i) for i in range(100)]
    results = await asyncio.gather(*tasks)
    # then group together the actual battles and wait
    tasks = []
    for result in results:
        tasks += result

    results = await asyncio.gather(*tasks)
    

    print(f"Elapsed time:", time.time()-start_time, "s")

async def buildBattle(i: int):
    teamstr1= '''Monika (Dugtrio) (F) @ Choice Band
Ability: Arena Trap
EVs: 4 HP / 252 Atk / 252 Spe
Adamant Nature
- Earthquake
- Rock Slide
- Aerial Ace
- Dig

Jesse Pinkman (Claydol) @ Leftovers
Ability: Levitate
EVs: 252 HP / 240 Atk / 16 Def
Adamant Nature
- Earthquake
- Shadow Ball
- Rapid Spin
- Explosion

Walter White (Skarmory) (M) @ Leftovers
Ability: Keen Eye
EVs: 252 HP / 4 Def / 252 SpD
Careful Nature
- Spikes
- Roar
- Toxic
- Drill Peck'''

    teamstr2 = '''Skylar White (Blissey) (F) @ Leftovers
Ability: Natural Cure
EVs: 44 HP / 252 Def / 212 SpA
Bold Nature
IVs: 2 Atk / 30 SpA
- Soft-Boiled
- Wish
- Ice Beam
- Hidden Power [Grass]

Bluetooth Speaker (Metagross) @ Choice Band
Ability: Clear Body
EVs: 252 Atk / 176 Def / 80 SpD
Adamant Nature
- Meteor Mash
- Earthquake
- Rock Slide
- Explosion

Meth in Weed (Tyranitar) (M) @ Leftovers
Ability: Sand Stream
EVs: 244 Atk / 12 SpA / 252 Spe
Naive Nature
- Dragon Dance
- Rock Slide
- Earthquake
- Ice Beam'''

    agent = Agent(f"agent{i}", True, teamstr1)
    opponent = Agent(f"opponent{i}", False, teamstr2)
    agentEngine = Engine(agent, opponent.username)
    opponentEngine = Engine(opponent, agent.username)
    await agentEngine.init()
    await opponentEngine.init()

    tasks = [runAgent(agentEngine), runAgent(opponentEngine)]
    return tasks

async def runAgent(engine: Engine):
    env = PokemonBattleEnv(engine)
    observation = await env.reset()
    reward = 0
    while True:
        randoAction = engine.agent.choose_action(env.engine.battle)
        observation, reward, terminated, t, i = await env.step(randoAction)
        if terminated: break
    # print(reward)

    await env.engine.socket.close()

if __name__ == "__main__":
    asyncio.get_event_loop().run_until_complete(main())

