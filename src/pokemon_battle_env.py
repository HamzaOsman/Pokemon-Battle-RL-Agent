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

        move = [
            Discrete(19, start=UNKNOWN_TYPE),       # Type
            Discrete(4, start=UNKNOWN_CATEGORY),    # Category (physical, special, status)
            Discrete(66, start=UNKNOWN_PP),         # PP
            Discrete(251, start=UNKNOWN_POWER),     # Power
            Discrete(71, start=UNKNOWN_ACCURACY),   # Accuracy (29 - 100)
            Discrete(13, start=UNKNOWN_PRIORITY)    # Priority  
        ]

        boosts = 7*[Discrete(13, start=-6)] # attack, special attack, defense, special, defense, speed, accuracy, evasiveness
        
        friendlyPokemon = [          
            Discrete(118, start=0), # item
            Discrete(76, start=1),  # ability
            Discrete(18, start=0),  # type1
            Discrete(18, start=0),  # type2
            Discrete(8, start=0),   # status
            Discrete(999, start=1), # hp
            Discrete(999, start=1), # atk
            Discrete(999, start=1), # def
            Discrete(999, start=1), # spa
            Discrete(999, start=1), # spd
            Discrete(999, start=1)  # spe
        ] + 4*move
        
        activeFriendlyPokemon = boosts + friendlyPokemon

        enemyPokemon = [
            Discrete(386, start=UNKNOWN_ID),       # id, 0 for unknown
            Discrete(119, start=UNKNOWN_ITEM),     # item, -1 for unknown
            Discrete(77, start=UNKNOWN_ABILITY),   # ability, 0 for unknown
            Discrete(19, start=UNKNOWN_TYPE),      # types, -1 for unknown, 0 for no type
            Discrete(19, start=UNKNOWN_TYPE),
            Discrete(8, start=0),                  # status
            Discrete(101, start=0),                # hp percent
            Discrete(256, start=UNKNOWN_STAT),     # base atk
            Discrete(256, start=UNKNOWN_STAT),     # base def
            Discrete(256, start=UNKNOWN_STAT),     # base spa
            Discrete(256, start=UNKNOWN_STAT),     # base spd
            Discrete(256, start=UNKNOWN_STAT),     # base spe
        ] + 4*move

        sideConditions = [
            Discrete(5, start=NONE_SIDE_CONDITION), #LIGHT_SCREEN 
            Discrete(5, start=NONE_SIDE_CONDITION), #MIST
            Discrete(5, start=NONE_SIDE_CONDITION), #REFLECT
            Discrete(5, start=NONE_SIDE_CONDITION), #SAFEGUARD 
            Discrete(4, start=NONE_SIDE_CONDITION)  #SPIKES
        ] 

        game = [
            Discrete(10, start=NONE_WEATHER)    # Weather
        ] + 2*sideConditions
        
        activeEnemyPokemon = boosts + enemyPokemon
        
        self.observation_space = gymnasium.spaces.Tuple((activeFriendlyPokemon + 2 * friendlyPokemon + activeEnemyPokemon + 2 * enemyPokemon + game))
        
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

        return observation, reward, self.engine.battle._finished, False, {}


    def _action_to_battleOrder(self, action):
        if (1 < action < 5):
            return BattleOrder(list(self.engine.battle.active_pokemon.moves.values())[action-1])
        elif (4 < action < 7):
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

        # battleState.fields
        # print(len(activeFriendlyPokemon))
        # print(len(friendlyPartyPokemon))
        # print(len(activeEnemyPokemon))
        # print(len(enemyPartyPokemon))
        # print(len(game))
        # print(len(activeFriendlyPokemon + friendlyPartyPokemon + activeEnemyPokemon + enemyPartyPokemon + game))
        # print("\n\n")
        # if (len(activeFriendlyPokemon + friendlyPartyPokemon + activeEnemyPokemon + enemyPartyPokemon + game)) != 238:
        #     print("activeFriendlyPokemon")
        #     print(activeFriendlyPokemon)
        #     print("friendlyPartyPokemon")
        #     print(friendlyPartyPokemon)
        #     print("activeEnemyPokemon")
        #     print(activeEnemyPokemon)
        #     print("enemyPartyPokemon")
        #     print(enemyPartyPokemon)
        #     print("game")
        #     print(game)
        #     print(len(activeFriendlyPokemon + friendlyPartyPokemon + activeEnemyPokemon + enemyPartyPokemon + game))
        #     print(len(activeFriendlyPokemon))
        #     print(len(friendlyPartyPokemon))
        #     print(len(activeEnemyPokemon))
        #     print(len(enemyPartyPokemon))
        #     print(len(game))
        #     print("\n\n")

        return activeFriendlyPokemon + friendlyPartyPokemon + activeEnemyPokemon + enemyPartyPokemon + game

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