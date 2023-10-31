import json
import gymnasium as gymnasium
from gymnasium.spaces import * 
from poke_env.player import BattleOrder, DefaultBattleOrder
from engine import Engine
from poke_env.environment.move import Move
import numpy as np
import webbrowser

# def pokemonSpecies(pokemon: Pokemon):
#     return pokemon.

class PokemonBattleEnv(gymnasium.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self, engine: Engine, render_mode=None):
        self.engine = engine

        boostSpace = Dict({
            "accuracy": Discrete(13, start=-6),
            "atk": Discrete(13, start=-6), # -6 -> 6
            "def": Discrete(13, start=-6),
            "evasion": Discrete(13, start=-6),
            "spa": Discrete(13, start=-6),
            "spd": Discrete(13, start=-6),
            "spe": Discrete(13, start=-6),
        })

        pokemonMoveSpace = Dict({
            "type": Discrete(18, start=0), # fire, water, etc
            "category": Discrete(3,start=0), # physical, special, status
            "pp": Discrete(65, start=0),
            "power": Discrete(250, start=0),
            "accuracy": Discrete(70, start=30)
        })

        # agent knows all details about each friendly pokemon
        friendlyPokemonSpace = Dict({
            "types": Tuple((Discrete(18, start=0), Discrete(18, start=0))),
            "stats": Dict({
                "hp": Discrete(1000, start=0),
                "atk": Discrete(1000, start=0),
                "def": Discrete(1000, start=0),
                "spa": Discrete(1000, start=0),
                "spd": Discrete(1000, start=0),
                "spe": Discrete(1000, start=0)
            }),
            "status": Discrete(8, start=0),
            "boosts": boostSpace,
            "moves": Tuple((pokemonMoveSpace,pokemonMoveSpace,pokemonMoveSpace,pokemonMoveSpace))
        })

        # agent knows about the active enemy pokemon
        enemyPokemonSpace = Dict({
            "types": Tuple((Discrete(18, start=0), Discrete(18, start=0))),
            "status": Discrete(8, start=0),
            "boosts": boostSpace,
        })

        # agent knows own pokemon, one enemy pokemon, which two pokemon are active ig
        self.observation_space = Dict({
            "agentPokemons": Tuple((friendlyPokemonSpace,friendlyPokemonSpace,friendlyPokemonSpace)),
            "enemyPokemon": enemyPokemonSpace,
            "game": Dict({
                "agentActivePokemon": Discrete(386, start=1),
                "enemyActivePokemon": Discrete(386, start=1)
            })
        })

        self.action_space = Discrete(6+1) # +1 representing defaultAction (struggle) when no other actions are valid 
        self.reward_range = (-1, 1)
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode
        self._rendered = False

    async def reset(self, seed=None):
        super().reset(seed=seed)
        # if self.engine.socket is not None:
        #     await self.engine.socket.close()

        await self.engine.startBattle()

        self.render()

        return self._buildObservation(), {}

    async def step(self, action):
        battleOrder = self._action_to_battleOrder(action)
        await self.engine.doAction(battleOrder)
        
        observation = self._buildObservation()
        reward = self._determineReward()
        
        self.render()
        
        return observation, reward, self.engine.battle._finished, False, {}
    
    def _action_to_battleOrder(self, action):
        if (0 < action < 5):
            return BattleOrder(self.engine.battle.available_moves[action-1])
        elif (4 < action < 7):
            return BattleOrder(self.engine.battle.available_switches[action-5])
        else:
            return DefaultBattleOrder()
        
    def valid_action_space_mask(self):
        """
        :return: Mask for valid moves used by agent to pick an action
        """
        action_mask = np.zeros(7, np.int8)
        if (len(self.engine.battle.available_moves) > 0):
            action_mask[1:len(self.engine.battle.available_moves)+1] = 1
        else:
            # no valid moves so add defaultBattleOrder (struggle)
            action_mask[0] = 1
        if (len(self.engine.battle.available_switches) > 0):
            action_mask[5:len(self.engine.battle.available_switches)+5] = 1

        # ex: [0, 1,1,0,0, 1,0] => can make available moves #1, #2, or available switch #1
        # ex: [1, 0,0,0,0, 1,0] => can make defaultMove (struggle), or available switch #1

        return action_mask
    
    def render(self):
        if (not self.render_mode): 
            return
        elif (self.render_mode == "human" and not self._rendered):
            url = f"http://localhost:8000/{self.engine.battle.battle_tag}"
            # ONLY UNCOMMENT WHEN # OF EPISODES IS SMALL
            # webbrowser.open(url) 
            self._rendered = True

    async def close(self):
        await self.engine.socket.close()
    
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
        battleState = self.engine.battle

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
        if self.engine.battle.won is None:
            return 0
        elif self.engine.battle.won:
            return 1
        else:
            return -1
