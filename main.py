import asyncio
import time
from battle_strategy import RandomStrategy
from engine import Engine
from player_model import PlayerModel
from pokemon_battle_env import PokemonBattleEnv
from player_model import PlayerModel
from poke_env import PlayerConfiguration

async def main():
    start_time = time.time()

    # Theory: errors regarding moves being made when theyre not supposed to are actually caused by race conditions
    # the more concurrent agents the heavier the strain on this app and the server
    # which means its possible for the socket timeout to occur before all of the battle data has actually been read in

    # Additionally there are still errors, particularly when a pokemon is trapped.
    # but this code does work concurrently, and i do seem to have solved the most scenarios caused by the agent/opponent
    # acting when they shouldnt

    tasks = [fullRun(i) for i in range(100)]

    # Use asyncio.gather to await all tasks at once
    await asyncio.gather(*tasks)

    print(f"Elapsed time:", time.time()-start_time, "s")

async def fullRun(i):
    env = PokemonBattleEnv(i)
    await env.reset()
    reward = 0
    while True:
        randoAction = RandomStrategy.choose_action(env.engine.agentBattle)
        observation, reward, terminated, t, i = await env.step(randoAction)
        if terminated: break
    print(reward)
    
    await env.engine.agentSocket.close()
    await env.engine.opponentSocket.close()


if __name__ == "__main__":
    asyncio.get_event_loop().run_until_complete(main())

