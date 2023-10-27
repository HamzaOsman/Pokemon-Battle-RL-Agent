# import json
import sys
import os
import gymnasium as gymnasium
from gymnasium.spaces import * 

# def pokemonSpecies(pokemon: Pokemon):
#     return pokemon.

# current_directory = os.path.dirname(__file__)
# parent_directory = os.path.abspath(os.path.join(current_directory, '..'))
# sys.path.append(parent_directory)
from ..showdown_master.showdown.run_battle import *


class PokemonBattleEnv(gymnasium.Env):
    def __init__(self, envNum: int):

        self.ps_websocket_client = 1
        self.pokemon_battle_type = 1
        self.battle = 1
        
        self.observation_space = Dict({
            "agentActivePokemons": Discrete(386),
            "enemyActivePokemon": Discrete(386),
        })

        # so what im thinking is that initially this will be probably like 6, but hopefully at the end of the step function we can update it?
        self.action_space = Discrete(9)
        self.reward_range = (-1, 1)

    async def reset(self, seed=None):
        # super().reset(seed=seed)
        self.battle = await start_battle(self.ps_websocket_client, self.pokemon_battle_type)

        return self.battle

        
    async def step(self, action):
        # await self.engine.doAction(action)
        # observation = self._buildObservation()
        # reward = self._determineReward()
        # return observation, reward, self.engine.agentBattle._finished, False, None
        msg = await self.ps_websocket_client.receive_message()
        if battle_is_finished(self.battle.battle_tag, msg):
            if constants.WIN_STRING in msg:
                winner = msg.split(constants.WIN_STRING)[-1].split('\n')[0].strip()
            else:
                winner = None
            logger.debug("Winner: {}".format(winner))
            # await ps_websocket_client.send_message(battle.battle_tag, ["gg"])
            await self.ps_websocket_client.leave_battle(self.battle.battle_tag, save_replay=ShowdownConfig.save_replay)
            return winner
        else:
            action_required = await async_update_battle(self.battle, msg)
            if action_required and not self.battle.wait:
                best_move = await async_pick_move(self.battle)
                await self.ps_websocket_client.send_message(self.battle.battle_tag, best_move)


    def render(self): pass
        # print(self.engine.agentBattle.active_pokemon)
    
    def _createMove(self, move: Move):
        file = open("./data/type_nums.json", 'r')
        # Parse the JSON data and load it into a Python dictionary
        typeNums = json.load(file)
        return {
            # fire, water, etc curse has ??? type, which is unhandled by poke_env
            "type": typeNums[move.entry["type"].title()], 
            "category": move.category.value, # physical, special, status
            "pp": move.current_pp,
            "power": move.base_power,
            "accuracy": move.accuracy
        }

    def _buildObservation(self):
        battleState = self.engine.agentBattle

        friendlyPokemon = []
        for name, pokemon in battleState.team.items():
            friendlyPokemon.append({
                "types": (pokemon.types[0].value, pokemon.types[1].value if pokemon.types[1] is not None else 0),
                "stats": {
                    "hp": pokemon.current_hp,
                    "atk": pokemon.stats["atk"],
                    "def": pokemon.stats["def"],
                    "spa": pokemon.stats["spa"],
                    "spd": pokemon.stats["spd"],
                    "spe": pokemon.stats["spe"]
                },
                "status": pokemon.status.value if (pokemon.status is not None) else 0,
                "boosts": pokemon.boosts,
                "moves": tuple(self._createMove(value) for value in pokemon.moves.values()) #this is gona be death
            })
            
        enemyPokemon = battleState.opponent_active_pokemon
        enemyPokemon = {
            "types": (enemyPokemon.types[0].value, enemyPokemon.types[1].value if enemyPokemon.types[1] is not None else 0),
            "status": enemyPokemon.status.value if (enemyPokemon.status is not None) else 0,
            "boosts": enemyPokemon.boosts
        }

        # TODO: THERE ARE SUBFORMS - pokemon with 2 names but 1 id bruh
        file = open("./data/pokemon_nums.json", 'r')
        # Parse the JSON data and load it into a Python dictionary
        pokeNums = json.load(file)

        return {   
            "agentPokemons": friendlyPokemon,
            "enemyPokemon": enemyPokemon,
            "game": {
                "agentActivePokemon": pokeNums[battleState.active_pokemon.species],
                "enemyActivePokemon": pokeNums[battleState.opponent_active_pokemon.species]
            }
        }
        
    def _determineReward(self):
        if self.engine.agentBattle.won is None:
            return 0
        elif self.engine.agentBattle.won:
            return 1
        else:
            return -1