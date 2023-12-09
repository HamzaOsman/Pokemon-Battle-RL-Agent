import json
import gymnasium as gymnasium
from gymnasium.spaces import *
from poke_env.player import BattleOrder, DefaultBattleOrder
from engine import Engine
from poke_env.environment.move import Move
from poke_env.environment.side_condition import SideCondition
from poke_env.environment.status import Status
from poke_env.data import GenData
import numpy as np
import webbrowser
import asyncio
from poke_env.environment.battle import Battle

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
            0, # hp percent
            1, # atk
            1, # def
            1, # spa
            1, # spd
            1  # spe
        ] + 4*move_min
        friendlyPokemon_max = [117, 76, 18, 18, 7, 100, 999, 999, 999, 999, 999] + 4*move_max
        
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
                         
        features_min = np.array(activeFriendlyPokemon_min+(self.TEAM_SIZE-1)*friendlyPokemon_min+activeEnemyPokemon_min+(self.ENEMY_TEAM_SIZE-1)*enemyPokemon_min+game_min)
        features_max = np.array(activeFriendlyPokemon_max+(self.TEAM_SIZE-1)*friendlyPokemon_max+activeEnemyPokemon_max+(self.ENEMY_TEAM_SIZE-1)*enemyPokemon_max+game_max)

        self.observation_space = gymnasium.spaces.Box(low=features_min, high=features_max, dtype=np.int16)

        # only 1 desired goal (enemy pkmn status = fainted) atm
        self.goal_space = gymnasium.spaces.Box(low=np.array([Status.FNT.value]*self.ENEMY_TEAM_SIZE), high=np.array([Status.FNT.value]*self.ENEMY_TEAM_SIZE), dtype=np.int16)
        
        self.goal_featurizer = lambda g : (g-NONE_STATUS) / (7-NONE_STATUS)

        def _goal_mapping(state):
            enemy_status_indices = []
            offset = len(activeFriendlyPokemon_min+(self.TEAM_SIZE-1)*friendlyPokemon_min+boosts_min)+5
            enemy_status_indices.append(offset)
            for i in range(self.ENEMY_TEAM_SIZE-1):
                offset += len(enemyPokemon_min)
                enemy_status_indices.append(offset)
            return state[:, enemy_status_indices]

        self.goal_mapping = _goal_mapping

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

        if (not np.any(action_mask)):
            print("\n\n\n\n\nno actions!\n\n\n\n\n")
            print("battle:", self.engine.battle.battle_tag)
            print("turn:", self.engine.battle.turn)
            print("force_switch?:", self.engine.battle.force_switch)
            print("available_moves:", self.engine.battle.available_moves)
            print("active_pokemon.moves", self.engine.battle.active_pokemon.moves.values())
            print("available_switches", self.engine.battle.available_switches)
            action_mask[0] = 1 
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

        # friendly pokemon
        activeFriendlyPokemon = []
        friendlyPartyPokemon = []

        for pokemon in [battleState.active_pokemon]+self.engine.orderedPartyPokemon:
            # build friendly pokemon observation
            friendlyPokemon = [
                PokemonBattleEnv.itemNums[pokemon.item] if (pokemon.item is not None and pokemon.item != '') else NONE_ITEM,
                PokemonBattleEnv.abilityNums[pokemon.ability],
                pokemon.types[0].value, 
                pokemon.types[1].value if (pokemon.types[1] is not None) else NONE_TYPE,
                pokemon.status.value if (pokemon.status is not None) else NONE_STATUS,
                round(100*pokemon.current_hp/pokemon.max_hp),
                pokemon.stats["atk"],
                pokemon.stats["def"],
                pokemon.stats["spa"], 
                pokemon.stats["spd"],
                pokemon.stats["spe"]
            ] 
            # include all 4 moves
            for value in pokemon.moves.values():
                friendlyPokemon+= self._createMove(value)            

            friendlyPokemon+= (4-len(pokemon.moves)) * [UNKNOWN_TYPE, UNKNOWN_CATEGORY, UNKNOWN_PP, UNKNOWN_POWER, UNKNOWN_ACCURACY, UNKNOWN_PRIORITY] 

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

        for _ in range (self.ENEMY_TEAM_SIZE-len(battleState.opponent_team.items())):
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

        obs = np.array(activeFriendlyPokemon + friendlyPartyPokemon + activeEnemyPokemon + enemyPartyPokemon + game)
        return obs

    def _measureTypeAdvantage(self, battle: Battle):
        reward = 0
        agentPkmn = battle.active_pokemon
        opponentPkmn = battle.opponent_active_pokemon

        reward += opponentPkmn.damage_multiplier(agentPkmn.type_1)
        if agentPkmn.type_2 != None:
            reward += opponentPkmn.damage_multiplier(agentPkmn._type_2)

        for move in agentPkmn.moves.values():
            reward += 0.5 * opponentPkmn.damage_multiplier(move)

        return reward
        
    def _measureNewState(self):
        newBattleState = self.engine.battle
        oldBattleState = self.engine.prevBattleState
        reward = 0

        newAgentPkmn = newBattleState.active_pokemon
        newOpponentPkmn = newBattleState.opponent_active_pokemon
        oldAgentPkmn = oldBattleState.active_pokemon
        oldOpponentPkmn = oldBattleState.opponent_active_pokemon

        # + if your hp goes up, - if your hp goes down
        # don't want to punish or reward swapping through this metric though
        if newAgentPkmn.species == oldAgentPkmn.species:
            reward += (newAgentPkmn.current_hp_fraction 
                   - oldAgentPkmn.current_hp_fraction) * 2
        # + if their hp goes down, - if their hp goes up
        if newOpponentPkmn.species == oldOpponentPkmn.species:
            reward += (oldOpponentPkmn.current_hp_fraction
                   - newOpponentPkmn.current_hp_fraction) * 2
        
        # you lose a status effect
        if newAgentPkmn.species == oldAgentPkmn.species and newAgentPkmn.status == None and oldAgentPkmn.status != None:
            reward += 0.5
        # they gain a status effect
        if newOpponentPkmn.species == oldOpponentPkmn.species and oldOpponentPkmn.status == None and newOpponentPkmn.status != None:
            reward += 0.5

        # do NOT measure type advantage when their pokemon changes
        # this is because type advantage should reflect how good your action was
        # if you defeat an enemy and they swap in a well typed pokemon you should not be punished
        # but you should be rewarded for swapping in a good pokemon, or if you keep a good pokemon
        if oldOpponentPkmn.species == newOpponentPkmn.species:
            reward += 0.1 * self._measureTypeAdvantage(newBattleState)

        # stat boosts, this can punish or reward swapping
        # reward our stats going up
        for stat, newBoost in newAgentPkmn.boosts.items():
            reward += 0.1 * (newBoost - oldAgentPkmn.boosts.get(stat))
        # punish their stats going up
        for stat, newBoost in newOpponentPkmn.boosts.items():
            reward -= 0.1 * (newBoost - oldOpponentPkmn.boosts.get(stat))

        # you defeat one of their pokemon
        for name, pokemon in newBattleState.opponent_team.items():
            if pokemon.fainted and not oldBattleState.opponent_team[name].fainted:
                reward += 3

        # one of your pokemon is defeated
        for name, pokemon in newBattleState.team.items():
            if pokemon.fainted and not oldBattleState.team[name].fainted:
                reward -= 3

        return reward

    def _determineReward(self):     
        if not self.engine.battle._finished:
            return self._measureNewState()

        # final reward dependent on performance
        reward = 0
        for name, friendly in self.engine.battle.team.items():
            # punished for fainted friendly
            if friendly.fainted:
                reward -= 1
            # reward proportional to keeping friendly alive
            else:
                reward += 1
        
        numEnemiesRemaining = self.ENEMY_TEAM_SIZE 
        for name, enemy in self.engine.battle.opponent_team.items():
            # rewarded for fainted enemy
            if enemy.fainted:
                numEnemiesRemaining -= 1
        
        reward -= numEnemiesRemaining
        turnRatio = self.engine.battle.turn / 1000

        if self.engine.battle.won is None:
            # tie :hmm:
            reward -= 0
        elif self.engine.battle.won:
            reward += 10
            # win faster == bigger return
            reward += 1/turnRatio
        else:
            reward -= 10

        return reward
    
    def _result(self):
        if self.engine.battle.won is None:
            return (0, 0, 1)
        elif self.engine.battle.won:
            return (1, 0, 0)
        else:
            return (0, 1, 0)