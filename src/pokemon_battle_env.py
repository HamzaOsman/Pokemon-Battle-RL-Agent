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
    
    def __init__(self, engine: Engine, agent_team_size: int, opponent_team_size: int, render_mode=None):
        self.engine = engine
        self.TEAM_SIZE = agent_team_size
        self.ENEMY_TEAM_SIZE = opponent_team_size

        #WIP
        #Game state
            # effects

        # for each pokemon also add
            # weight
            # gender

        move_min = [
            UNKNOWN_TYPE,     # Type
            UNKNOWN_POWER,    # Power
            UNKNOWN_ACCURACY, # Accuracy
        ]  
        move_max = [18, 250, 100]

        boosts_min = 7*[-6] # attack, special attack, defense, special defense, speed, accuracy, evasiveness
        boosts_max = 7*[6]
        
        friendlyPokemon_min = [
            UNKNOWN_ID,      # id
            1, # type1
            0, # type2
            0, # status
            1, # hp
        ] + 4*move_min
        friendlyPokemon_max = [386, 18, 18, 7, 999] + 4*move_max
        
        activeFriendlyPokemon_min = boosts_min + friendlyPokemon_min
        activeFriendlyPokemon_max = boosts_max + friendlyPokemon_max

        enemyPokemon_min = [
            UNKNOWN_ID,      # id
            UNKNOWN_TYPE,    # type1
            UNKNOWN_TYPE,    # type2
            0,               # status
            0,               # hp percent
        ]
        enemyPokemon_max = [386, 18, 18, 7, 100]

        activeEnemyPokemon_min = boosts_min + enemyPokemon_min
        activeEnemyPokemon_max = boosts_max + enemyPokemon_max
                         
        features_min = np.array(activeFriendlyPokemon_min+(self.TEAM_SIZE-1)*friendlyPokemon_min+activeEnemyPokemon_min+(self.ENEMY_TEAM_SIZE-1)*enemyPokemon_min)
        features_max = np.array(activeFriendlyPokemon_max+(self.TEAM_SIZE-1)*friendlyPokemon_max+activeEnemyPokemon_max+(self.ENEMY_TEAM_SIZE-1)*enemyPokemon_max)

        self.observation_space = gymnasium.spaces.Box(low=features_min, high=features_max, dtype=np.int16)

        # +1 representing defaultAction (struggle) when no other actions are valid
        self.action_space = Discrete(4+(self.TEAM_SIZE-1)+1)
        # self.reward_range = (-50-self.TEAM_SIZE-self.ENEMY_TEAM_SIZE, 50+self.TEAM_SIZE+self.ENEMY_TEAM_SIZE)

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode
        self._rendered = False
        

    async def reset(self, seed=None):
        super().reset(seed=seed)

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
        elif (5 <= action < self.action_space.n):
            return BattleOrder(self.engine.orderedPartyPokemon[action-5])
        else:
            # action index 0
            return DefaultBattleOrder()


    def valid_action_space_mask(self):
        """
        :return: Mask for valid moves used by agent to pick an action
        """
        action_mask = np.zeros(self.action_space.n, np.int8)
        if (not self.engine.battle.force_switch and len(self.engine.battle.available_moves) > 0):
            available_moves_ids = [move.id for move in self.engine.battle.available_moves]
            if ("struggle" not in available_moves_ids):
                i = 0
                for i, move in enumerate(self.engine.battle.active_pokemon.moves.values()):
                    action_mask[1+i] = 1 if move.id in available_moves_ids else 0
                i+=1
                action_mask[1+i:5] = [0] #Pokemons with < 4 moves
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
            NONE_TYPE if move.entry["type"].upper() == "???" else move.type.value,
            move.base_power,
            int(move.accuracy*100),
        ]
    
    def _buildObservation(self):
        battleState = self.engine.battle
        
        # FIRST: friendly pokemon
        activeFriendlyPokemon = []
        friendlyPartyPokemon = []
        for pokemon in [battleState.active_pokemon]+self.engine.orderedPartyPokemon:
            # build friendly pokemon observation
            friendlyPokemon = [
                PokemonBattleEnv.pokeNums[pokemon.species],
                pokemon.types[0].value, 
                pokemon.types[1].value if (pokemon.types[1] is not None) else NONE_TYPE,
                pokemon.status.value if (pokemon.status is not None) else NONE_STATUS,
                pokemon.current_hp,
            ] 
            # include all moves
            for move in pokemon.moves.values():
                friendlyPokemon+= self._createMove(move)     
            # if fewer than 4 moves, use unknowns    
            friendlyPokemon+= (4-len(pokemon.moves)) * [UNKNOWN_TYPE, UNKNOWN_POWER, UNKNOWN_ACCURACY] 

            if (pokemon.species == battleState.active_pokemon.species):
                activeFriendlyPokemon = list(pokemon.boosts.values()) + friendlyPokemon
            else:
                friendlyPartyPokemon += friendlyPokemon

        # SECOND: enemy pokemon
        activeEnemyPokemon = []
        enemyPartyPokemon = []
        for name, pokemon in battleState.opponent_team.items():
            # build enemy pokemon observation
            enemyPokemon = [
                PokemonBattleEnv.pokeNums[pokemon.species],
                pokemon.types[0].value, 
                pokemon.types[1].value if (pokemon.types[1] is not None) else NONE_TYPE,
                pokemon.status.value if (pokemon.status is not None) else NONE_STATUS,
                pokemon.current_hp
            ] 

            if (pokemon.species == battleState.opponent_active_pokemon.species):
                activeEnemyPokemon = list(pokemon.boosts.values()) + enemyPokemon
            else:
                enemyPartyPokemon += enemyPokemon

        for _ in range (self.ENEMY_TEAM_SIZE-len(battleState.opponent_team.items())):
            enemyPartyPokemon += [UNKNOWN_ID, UNKNOWN_TYPE, UNKNOWN_TYPE, NONE_STATUS, 100] 
        

        obs = np.array(activeFriendlyPokemon + friendlyPartyPokemon + activeEnemyPokemon + enemyPartyPokemon)
        return obs

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
                reward += 1 * (friendly.current_hp_fraction)
                
        for name, enemy in self.engine.battle.opponent_team.items():
            # rewarded for fainted enemy
            if enemy.fainted:
                reward += 1
            # punishment proportional to how close to taking down each enemy
            else:
                reward -= 1 * (enemy.current_hp_fraction)
        
        reward -= self.ENEMY_TEAM_SIZE-len(self.engine.battle.opponent_team.items()) #Unseen full HP enemy pokemons 

        if self.engine.battle.won is None:
            reward -= 5
        elif self.engine.battle.won:
            reward += 20
        else:
            reward -= 20


        return reward
    
    def _result(self):
        if self.engine.battle.won is None:
            return (0, 0, 1)
        elif self.engine.battle.won:
            return (1, 0, 0)
        else:
            return (0, 1, 0)