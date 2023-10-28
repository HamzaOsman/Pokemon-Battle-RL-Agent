from datetime import datetime
from showdown_master.showdown.run_battle import *
from showdown_master.teams import load_team

import gymnasium as gymnasium
from gymnasium.spaces import * 

class PokemonBattleEnv(gymnasium.Env):
    def __init__(self, ShowdownConfig, ps_websocket_client, render_mode=None):

        self.ShowdownConfig = ShowdownConfig
        self.ps_websocket_client = ps_websocket_client
        
        self.observation_space = Dict({
            "agentActivePokemons": Discrete(386),
            "enemyActivePokemon": Discrete(386),
        })

        self.action_space = Discrete(9)
        self.reward_range = (-1, 1)
        self.render_mode = render_mode

        self._battle_state = None

    async def reset(self, seed=None):
        # super().reset(seed=seed)
        if ShowdownConfig.log_to_file:
                ShowdownConfig.log_handler.do_rollover(datetime.now().strftime("%Y-%m-%dT%H:%M:%S.log"))
        team = load_team(ShowdownConfig.team)
        if ShowdownConfig.bot_mode == constants.CHALLENGE_USER:
            await self.ps_websocket_client.challenge_user(
                ShowdownConfig.user_to_challenge,
                ShowdownConfig.pokemon_mode,
                team
            )
        elif ShowdownConfig.bot_mode == constants.ACCEPT_CHALLENGE:
            await self.ps_websocket_client.accept_challenge(
                ShowdownConfig.pokemon_mode,
                team,
                ShowdownConfig.room_name
            )
        elif ShowdownConfig.bot_mode == constants.SEARCH_LADDER:
            await self.ps_websocket_client.search_for_match(ShowdownConfig.pokemon_mode, team)
        else:
            raise ValueError("Invalid Bot Mode: {}".format(ShowdownConfig.bot_mode))

        self._battle_state = await start_battle(self.ps_websocket_client, ShowdownConfig.pokemon_mode)

        if self.render_mode == "human":
            self.render()

        # TODO: convert _battle_state to observation
        obs = {}

        return obs, {}

        
    async def step(self, action):
        terminated = False

        # TODO: convert action to move
        # TODO: remove once we get a proper move based on action
        move = await async_pick_move(self._battle_state)

        # TODO: reformat
        msg = await self.ps_websocket_client.receive_message()
        if battle_is_finished(self._battle_state.battle_tag, msg):
            terminated = True
            if constants.WIN_STRING in msg:
                winner = msg.split(constants.WIN_STRING)[-1].split('\n')[0].strip()
            else:
                winner = None
            logger.debug("Winner: {}".format(winner))
            # await ps_websocket_client.send_message(self._battle_state.battle_tag, ["gg"])
            await self.ps_websocket_client.leave_battle(self._battle_state.battle_tag, save_replay=ShowdownConfig.save_replay)
        else:
            action_required = await async_update_battle(self._battle_state, msg)
            if action_required and not self._battle_state.wait:
                await self.ps_websocket_client.send_message(self._battle_state.battle_tag, move)
            # TODO: do we have to do smth if not the case, otherwise we wouldnt have 'taken' the action? idk

        if self.render_mode == "human":
            self.render()

        reward = 0
        if (terminated):
            reward = 1 if winner == ShowdownConfig.username else -1

        # TODO: convert _battle_state to observation
        obs = {}

        return obs, reward, terminated, False, {}


    def render(self): 
        # TODO: add human render (display showdown link?)
        return
    
    # def _createMove(self, move: Move):
    #     file = open("./data/type_nums.json", 'r')
    #     # Parse the JSON data and load it into a Python dictionary
    #     typeNums = json.load(file)
    #     return {
    #         # fire, water, etc curse has ??? type, which is unhandled by poke_env
    #         "type": typeNums[move.entry["type"].title()], 
    #         "category": move.category.value, # physical, special, status
    #         "pp": move.current_pp,
    #         "power": move.base_power,
    #         "accuracy": move.accuracy
    #     }

    # def _buildObservation(self):
    #     battleState = self.engine.agentBattle

    #     friendlyPokemon = []
    #     for name, pokemon in battleState.team.items():
    #         friendlyPokemon.append({
    #             "types": (pokemon.types[0].value, pokemon.types[1].value if pokemon.types[1] is not None else 0),
    #             "stats": {
    #                 "hp": pokemon.current_hp,
    #                 "atk": pokemon.stats["atk"],
    #                 "def": pokemon.stats["def"],
    #                 "spa": pokemon.stats["spa"],
    #                 "spd": pokemon.stats["spd"],
    #                 "spe": pokemon.stats["spe"]
    #             },
    #             "status": pokemon.status.value if (pokemon.status is not None) else 0,
    #             "boosts": pokemon.boosts,
    #             "moves": tuple(self._createMove(value) for value in pokemon.moves.values()) #this is gona be death
    #         })
            
    #     enemyPokemon = battleState.opponent_active_pokemon
    #     enemyPokemon = {
    #         "types": (enemyPokemon.types[0].value, enemyPokemon.types[1].value if enemyPokemon.types[1] is not None else 0),
    #         "status": enemyPokemon.status.value if (enemyPokemon.status is not None) else 0,
    #         "boosts": enemyPokemon.boosts
    #     }

    #     # TODO: THERE ARE SUBFORMS - pokemon with 2 names but 1 id bruh
    #     file = open("./data/pokemon_nums.json", 'r')
    #     # Parse the JSON data and load it into a Python dictionary
    #     pokeNums = json.load(file)

    #     return {   
    #         "agentPokemons": friendlyPokemon,
    #         "enemyPokemon": enemyPokemon,
    #         "game": {
    #             "agentActivePokemon": pokeNums[battleState.active_pokemon.species],
    #             "enemyActivePokemon": pokeNums[battleState.opponent_active_pokemon.species]
    #         }
    #     }
        
    # def _determineReward(self):
    #     if self.engine.agentBattle.won is None:
    #         return 0
    #     elif self.engine.agentBattle.won:
    #         return 1
    #     else:
    #         return -1