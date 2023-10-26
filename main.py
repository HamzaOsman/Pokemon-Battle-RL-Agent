import asyncio
from battle_strategy import RandomStrategy
from engine import Engine
from player_model import PlayerModel
from pokemon_battle_env import PokemonBattleEnv
from player_model import PlayerModel
from poke_env import PlayerConfiguration

async def main():
    # engine connects agent and opponent
    # engine = Engine(PlayerModel("agent"), PlayerModel("opponent"))
    # await engine.start()
    # print(engine.agentBattle.active_pokemon.base_stats)

    env = PokemonBattleEnv()
    await env.reset()
    while True:
        randoAction = RandomStrategy.choose_action(env.engine.agentBattle)
        observation, reward, terminated, t, i = await env.step(randoAction)
        print(reward)
        if terminated: break
        


    await env.engine.agentSocket.close()
    await env.engine.opponentSocket.close()



if __name__ == "__main__":
    asyncio.get_event_loop().run_until_complete(main())

