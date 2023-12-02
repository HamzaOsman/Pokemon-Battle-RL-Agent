import json
import gymnasium as gymnasium
from gymnasium.spaces import *
from poke_env.player import BattleOrder, DefaultBattleOrder
from engine import Engine
from poke_env.environment.move import Move
from poke_env.environment.side_condition import SideCondition
from poke_env.data import GenData
import numpy as np
import webbrowser
import asyncio

# AUTO enum starts at 1
NONE_ITEM = 0
NONE_TYPE = 0
NONE_STATUS = 0
NONE_WEATHER = 0
NONE_SIDE_CONDITION = 0
UNKNOWN_TYPE = -1
UNKNOWN_ID = 0
UNKNOWN_ABILITY = 0
UNKNOWN_ITEM = -1
UNKNOWN_STAT = 0
UNKNOWN_CATEGORY = -1
UNKNOWN_PP = -1
UNKNOWN_POWER = -1
UNKNOWN_ACCURACY = 29
UNKNOWN_PRIORITY = -7

class PokemonBattleEnv(gymnasium.Env):

    metadata = {"render_modes": ["human"], "min_rate": 8}
    pokeNums = json.load(open("./data/pokemon_nums.json", 'r'))
    abilityNums = json.load(open("./data/ability_nums.json", 'r'))
    itemNums = json.load(open("./data/item_nums.json", 'r'))
    
    def __init__(self, engine: Engine, render_mode=None):
        self.engine = engine
        
        #WIP
        #Game state
            # effects

        # for each pokemon also add
            # weight
            # gender

        move_min = [
            UNKNOWN_TYPE,     # Type
            UNKNOWN_CATEGORY, # Category (physical, special, status)
            UNKNOWN_PP,       # PP
            UNKNOWN_POWER,    # Power
            UNKNOWN_ACCURACY, # Accuracy
            UNKNOWN_PRIORITY  # Priority
        ]  
        move_max = [18, 3, 64, 250, 100, 5]

        boosts_min = 7*[-6] # attack, special attack, defense, special defense, speed, accuracy, evasiveness
        boosts_max = 7*[6]
        
        friendlyPokemon_min = [
            0, # item
            1, # ability
            1, # type1
            0, # type2
            0, # status
            1, # hp
            1, # atk
            1, # def
            1, # spa
            1, # spd
            1  # spe
        ] + 4*move_min
        friendlyPokemon_max = [117, 76, 18, 18, 7, 999, 999, 999, 999, 999, 999] + 4*move_max
        
        activeFriendlyPokemon_min = boosts_min + friendlyPokemon_min
        activeFriendlyPokemon_max = boosts_max + friendlyPokemon_max

        enemyPokemon_min = [
            UNKNOWN_ID,      # id
            UNKNOWN_ITEM,    # item
            UNKNOWN_ABILITY, # ability
            UNKNOWN_TYPE,    # type1
            UNKNOWN_TYPE,    # type2
            0,               # status
            0,               # hp percent
            # base hp ???
            UNKNOWN_STAT,    # base atk
            UNKNOWN_STAT,    # base def
            UNKNOWN_STAT,    # base spa
            UNKNOWN_STAT,    # base spd
            UNKNOWN_STAT,    # base spe
        ] + 4*move_min
        enemyPokemon_max = [386, 117, 76, 18, 18, 7, 100, 255, 255, 255, 255, 255] + 4*move_max

        activeEnemyPokemon_min = boosts_min + enemyPokemon_min
        activeEnemyPokemon_max = boosts_max + enemyPokemon_max

        sideConditions_min = [
            NONE_SIDE_CONDITION, # LIGHT_SCREEN
            NONE_SIDE_CONDITION, # MIST
            NONE_SIDE_CONDITION, # REFLECT
            NONE_SIDE_CONDITION, # SAFEGUARD
            NONE_SIDE_CONDITION, # SPIKES
        ]
        sideConditions_max = [4, 4, 4, 4, 3]

        game_min = [
            NONE_WEATHER # Weather
        ] + 2*sideConditions_min
        game_max = [9] + 2*sideConditions_max
                         
        features_min = np.array(activeFriendlyPokemon_min+2*friendlyPokemon_min+activeEnemyPokemon_min+2*enemyPokemon_min+game_min)
        features_max = np.array(activeFriendlyPokemon_max+2*friendlyPokemon_max+activeEnemyPokemon_max+2*enemyPokemon_max+game_max)

        self.observation_space = gymnasium.spaces.Box(low=features_min, high=features_max, dtype=np.int16)
        
        # +1 representing defaultAction (struggle) when no other actions are valid
        self.action_space = Discrete(6+1)
        self.reward_range = (-16, 16)

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode
        self._rendered = False

    async def reset(self, seed=None):
        super().reset(seed=seed)
        # if self.engine.socket is not None:
        #     await self.engine.socket.close()

        self._rendered = False

        self.engine.resetBattle()

        await self.engine.startBattle()

        await self.render()

        return self._buildObservation(), {}


    async def step(self, action):
        battleOrder = self._action_to_battleOrder(action)
        await self.engine.doAction(battleOrder)

        observation = self._buildObservation()

        reward = self._determineReward()

        await self.render()

        return observation, reward, self.engine.battle._finished, False, { "result": self._result() }


    def _action_to_battleOrder(self, action):
        if (1 <= action <= 4):
            return BattleOrder(list(self.engine.battle.active_pokemon.moves.values())[action-1])
        elif (5 <= action <= 6):
            return BattleOrder(self.engine.orderedPartyPokemon[action-5])
        else:
            # action index 0
            return DefaultBattleOrder()


    def valid_action_space_mask(self):
        """
        :return: Mask for valid moves used by agent to pick an action
        """
        action_mask = np.zeros(7, np.int8)
        if (not self.engine.battle.force_switch and len(self.engine.battle.available_moves) > 0):
            available_moves_ids = [move.id for move in self.engine.battle.available_moves]
            if ("struggle" not in available_moves_ids):
                action_mask[1:5] = [1 if move.id in available_moves_ids else 0 for move in self.engine.battle.active_pokemon.moves.values()]
            else:
                # add defaultBattleOrder (struggle) when no valid moves
                action_mask[0] = 1 
        if (len(self.engine.battle.available_switches) > 0):
            action_mask[5:] = [1 if p in self.engine.battle.available_switches else 0 for p in self.engine.orderedPartyPokemon]

        # ex: [0, 1,0,1,1, 0,1] => can make moves #1, #3, #4, or switch #2
        # ex: [1, 0,0,0,0, 1,0] => can make defaultMove or switch #1
        
        return action_mask

    async def render(self):
        if (not self.render_mode):
            return
        elif (self.render_mode == "human" and not self._rendered):
            url = f"http://localhost:8000/{self.engine.battle.battle_tag}"
            # ONLY UNCOMMENT WHEN # OF EPISODES IS SMALL
            # webbrowser.open(url)
            self._rendered = True

        await asyncio.sleep(self.metadata["min_rate"])

    async def close(self):
        await self.engine.socket.close()

    def _createMove(self, move: Move):
        return [
            #TODO: add move id (to learn unknown info like move side effects)
            # fire, water, etc curse has ??? type, which is unhandled by poke_env
            NONE_TYPE if move.entry["type"].upper() == "???" else move.type.value,
            move.category.value,  # physical, special, status
            move.current_pp,
            move.base_power,
            int(move.accuracy*100),
            move.priority
        ]

    def _buildObservation(self):
        battleState = self.engine.battle
        # if self.engine.agent.isChallenger:
            
        #     print("the pokemon order is:")
        #     for pokemon in [battleState.active_pokemon] + self.engine.orderedPartyPokemon:
        #         print(pokemon)
        #         print("their moves are:")
        #         print(pokemon.moves)
            
        #     print("\n\n")

        # friendly pokemon
        activeFriendlyPokemon = []
        friendlyPartyPokemon = []

        for pokemon in [battleState.active_pokemon]+self.engine.orderedPartyPokemon:
            # build friendly pokemon observation
            friendlyPokemon = [
                PokemonBattleEnv.itemNums[pokemon.item] if (pokemon.item is not None) else NONE_ITEM,
                PokemonBattleEnv.abilityNums[pokemon.ability],
                pokemon.types[0].value, 
                pokemon.types[1].value if (pokemon.types[1] is not None) else NONE_TYPE,
                pokemon.status.value if (pokemon.status is not None) else NONE_STATUS,
                pokemon.current_hp,
                pokemon.stats["atk"],
                pokemon.stats["def"],
                pokemon.stats["spa"], 
                pokemon.stats["spd"],
                pokemon.stats["spe"]
            ] 
            # include all 4 moves
            for value in pokemon.moves.values():
                friendlyPokemon+= self._createMove(value)            

            if (pokemon.species == battleState.active_pokemon.species):
                activeFriendlyPokemon = list(pokemon.boosts.values()) + friendlyPokemon
            else:
                friendlyPartyPokemon += friendlyPokemon

        # observation of enemy's pokemon
        activeEnemyPokemon = []
        enemyPartyPokemon = []
        for name, pokemon in battleState.opponent_team.items():
            # build enemy pokemon observation
            enemyPokemon = [
                PokemonBattleEnv.pokeNums[pokemon.species],
                NONE_ITEM if (pokemon.item is None) else PokemonBattleEnv.itemNums[pokemon.item] if (pokemon.item != GenData.UNKNOWN_ITEM) else UNKNOWN_ITEM,
                PokemonBattleEnv.abilityNums[pokemon.ability] if(pokemon.ability is not None) else UNKNOWN_ABILITY,
                pokemon.types[0].value, 
                pokemon.types[1].value if (pokemon.types[1] is not None) else NONE_TYPE,
                pokemon.status.value if (pokemon.status is not None) else NONE_STATUS,
                pokemon.current_hp, 
                pokemon.base_stats["atk"],
                pokemon.base_stats["def"],
                pokemon.base_stats["spa"],
                pokemon.base_stats["spd"],
                pokemon.base_stats["spe"]
            ] 
            # include all 4 moves
            for value in pokemon.moves.values():
                enemyPokemon += self._createMove(value)
                
            enemyPokemon+= (4-len(pokemon.moves)) * [UNKNOWN_TYPE, UNKNOWN_CATEGORY, UNKNOWN_PP, UNKNOWN_POWER, UNKNOWN_ACCURACY, UNKNOWN_PRIORITY] 

            if (pokemon.species == battleState.opponent_active_pokemon.species):
                activeEnemyPokemon = list(pokemon.boosts.values()) + enemyPokemon
            else:
                enemyPartyPokemon += enemyPokemon

        for _ in range (3-len(battleState.opponent_team.items())):
            enemyPartyPokemon += [UNKNOWN_ID, UNKNOWN_ITEM, UNKNOWN_ABILITY, UNKNOWN_TYPE, UNKNOWN_TYPE, NONE_STATUS, 100] 
            enemyPartyPokemon += 5*[UNKNOWN_STAT] 
            enemyPartyPokemon += 4*[UNKNOWN_TYPE, UNKNOWN_CATEGORY, UNKNOWN_PP, UNKNOWN_POWER, UNKNOWN_ACCURACY, UNKNOWN_PRIORITY] 
        
        
        playerSide = [
            battleState.turn - battleState.side_conditions[SideCondition.LIGHT_SCREEN] if (SideCondition.LIGHT_SCREEN in battleState.side_conditions) else NONE_SIDE_CONDITION,
            battleState.turn - battleState.side_conditions[SideCondition.MIST] if (SideCondition.MIST in battleState.side_conditions) else NONE_SIDE_CONDITION,
            battleState.turn - battleState.side_conditions[SideCondition.REFLECT] if (SideCondition.REFLECT in battleState.side_conditions) else NONE_SIDE_CONDITION,
            battleState.turn - battleState.side_conditions[SideCondition.SAFEGUARD] if (SideCondition.SAFEGUARD in battleState.side_conditions) else NONE_SIDE_CONDITION,
            battleState.side_conditions[SideCondition.SPIKES] if (SideCondition.SPIKES in battleState.side_conditions) else NONE_SIDE_CONDITION
        ]

        opponentSide = [
            battleState.turn - battleState.opponent_side_conditions[SideCondition.LIGHT_SCREEN] if (SideCondition.LIGHT_SCREEN in battleState.opponent_side_conditions) else NONE_SIDE_CONDITION,
            battleState.turn - battleState.opponent_side_conditions[SideCondition.MIST] if (SideCondition.MIST in battleState.opponent_side_conditions) else NONE_SIDE_CONDITION,
            battleState.turn - battleState.opponent_side_conditions[SideCondition.REFLECT] if (SideCondition.REFLECT in battleState.opponent_side_conditions) else NONE_SIDE_CONDITION,
            battleState.turn - battleState.opponent_side_conditions[SideCondition.SAFEGUARD] if (SideCondition.SAFEGUARD in battleState.opponent_side_conditions) else NONE_SIDE_CONDITION,
            battleState.opponent_side_conditions[SideCondition.SPIKES] if (SideCondition.SPIKES in battleState.opponent_side_conditions) else NONE_SIDE_CONDITION
        ]

        # observation of game
        game = [
            NONE_WEATHER if not battleState.weather else list(battleState.weather.keys())[0].value,
            *playerSide,
            *opponentSide
        ]

        return np.array(activeFriendlyPokemon + friendlyPartyPokemon + activeEnemyPokemon + enemyPartyPokemon + game)

    def _determineReward(self):
        # TODO: intermittent rewards? need to keep track of previous state to compare to

        if not self.engine.battle._finished: 
            return 0
       
        reward = 0
        
        for name, friendly in self.engine.battle.team.items():
            # punished for fainted friendly
            if friendly.fainted:
                reward -= 1
            # reward proportional to keeping friendly alive
            else:
                reward += 1 * (friendly.current_hp/friendly.max_hp)
        for name, enemy in self.engine.battle.opponent_team.items():
            # rewarded for fainted enemy
            if enemy.fainted:
                reward += 1
            # punishment proportional to how close to taking down each enemy
            else:
                reward -= 1 * (enemy.current_hp/enemy.max_hp)

        if self.engine.battle.won is None:
            reward += 0
        elif self.engine.battle.won:
            reward += 10 # TODO: reward for win/loss
        else:
            reward -= 10
        # on ties this would also do -100?
        # reward += 100 if self.engine.battle.won else -100

        return reward
    
    def _result(self):
        if self.engine.battle.won is None:
            return (0, 0, 1)
        elif self.engine.battle.won:
            return (1, 0, 0)
        else:
            return (0, 1, 0)