import asyncio
import json
import logging
import traceback
from copy import deepcopy

from showdown_master.config import ShowdownConfig, init_logging
from showdown_master.showdown.websocket_client import PSWebsocketClient
from showdown_master.data import all_move_json
from showdown_master.data import pokedex
from showdown_master.data.mods.apply_mods import apply_mods

from pokemon_battle_env import PokemonBattleEnv
from agents.random_agent import RandomAgent

logger = logging.getLogger(__name__)

# TODO: Can probably delete this later
def check_dictionaries_are_unmodified(original_pokedex, original_move_json):
    # The bot should not modify the data dictionaries
    # This is a "just-in-case" check to make sure and will stop the bot if it mutates either of them
    if original_move_json != all_move_json:
        logger.critical("Move JSON changed!\nDumping modified version to `modified_moves.json`")
        with open("modified_moves.json", 'w') as f:
            json.dump(all_move_json, f, indent=4)
        exit(1)
    else:
        logger.debug("Move JSON unmodified!")

    if original_pokedex != pokedex:
        logger.critical(
            "Pokedex JSON changed!\nDumping modified version to `modified_pokedex.json`"
        )
        with open("modified_pokedex.json", 'w') as f:
            json.dump(pokedex, f, indent=4)
        exit(1)
    else:
        logger.debug("Pokedex JSON unmodified!")

async def showdown():
    ShowdownConfig.configure()
    init_logging(
        ShowdownConfig.log_level,
        ShowdownConfig.log_to_file
    )
    apply_mods(ShowdownConfig.pokemon_mode)

    original_pokedex = deepcopy(pokedex)
    original_move_json = deepcopy(all_move_json)

    ps_websocket_client = await PSWebsocketClient.create(
        ShowdownConfig.username,
        ShowdownConfig.password,
        ShowdownConfig.websocket_uri
    )
    await ps_websocket_client.login()

    env = PokemonBattleEnv(ShowdownConfig, ps_websocket_client)
    random_agent = RandomAgent(env)
    wins, losses = await random_agent.train(2)

    logger.info("W: {}\tL: {}".format(wins, losses))

    check_dictionaries_are_unmodified(original_pokedex, original_move_json)

if __name__ == "__main__":
    try:
        asyncio.run(showdown())
    except Exception as e:
        logger.error(traceback.format_exc())
        raise

# TODO: add self-play, maybe using 2 threads like https://github.com/hsahovic/poke-env/blob/f627cc20092f16869503ca79f42f6c82f1f29cdf/examples/experimental-self-play.py