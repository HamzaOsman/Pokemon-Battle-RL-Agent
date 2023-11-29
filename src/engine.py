import asyncio
from logging import Logger
import orjson
from poke_env import ShowdownException
from poke_env.player import BattleOrder
from poke_env.environment.battle import Battle
from poke_env.data import GenData

from inspect import isawaitable
from typing import List
import websockets
from agent import Agent

# engine which manages the battle
class Engine:
    # battleFormat: str = "gen3randombattle"
    battleFormat: str = "gen3ou"

    def __init__(self, agent: Agent, opponentUsername: str, socket: websockets.WebSocketClientProtocol = None):
        self.battle: Battle = None
        self.agent = agent
        self.orderedPartyPokemon = []
        self.opponentUsername = opponentUsername
        self.socket = socket

    async def init(self, logPlayerIn: bool):
        websocketUrl = "ws://localhost:8000/showdown/websocket"
        if self.socket is None:
            self.socket =  await websockets.connect(websocketUrl)
        if logPlayerIn:
            await self._logPlayerIn()
        
    async def _waitUntilChallenge(self):
        async for message in self.socket:
            # print(self.agent.username, "getting msg...", message)
            # pipe-separated sequences
            messageSplit = message.split("|")
            if(messageSplit[1] == "pm" and messageSplit[4].startswith("/challenge")):
                return
            
    def resetBattle(self):
        self.battle = None
        self.orderedPartyPokemon = []

    async def _sendMessage(self, message: str, room: str = ""):
        await self.socket.send("|".join([room, message]))

    async def _logPlayerIn(self):
        await self._sendMessage(f"/trn {self.agent.username},0,")

    async def _setTeam(self):
        # print("setting team:", "/utm %s" % player.team)
        await self._sendMessage("/utm %s" % self.agent.team)

    async def startBattle(self):
        await self._setTeam()
        if self.agent.isChallenger:
            challengeMsg = f"/challenge {self.opponentUsername}, {Engine.battleFormat}"
        else:
            challengeMsg = "/accept %s" % self.opponentUsername
            await self._waitUntilChallenge()

        await self._sendMessage(challengeMsg)

        await self.parseInitialBattle()
        # print("battle started!")

    async def parseInitialBattle(self):
        isInit = False
        while True:
            message = await self.socket.recv()
            messageSplit = message.split("|")
            isInit = messageSplit[1] == "init" or isInit
            # if not isInit:
            #     print("this is not init:")
            #     print(message)
            # ignore non battle messages, and anything else in the socket before the initialization
            if(messageSplit[0].startswith(">battle") and isInit):
                # if(messageSplit[1].startswith("error")):
                #     print(self.agent.username, message)
                self._handle_battle_message(message)

                # p2, the accepter, gets displayed last
                p2User = self.opponentUsername if self.agent.isChallenger else self.agent.username
                if(messageSplit[1] == "player" and p2User in message):
                    return

    async def _parseBattle(self):
        moveMade = True
        i = 0
        while i < 2:
            message = await self.socket.recv()
            messageSplit = message.split("|")
            # print(self.agent.username, messageSplit)
            # print()
            # ignore non battle messages, and info about spectators looking at battles
            if(messageSplit[0].startswith(">battle") and messageSplit[1] not in {"j", "l"}):
                i += 1
                # if(messageSplit[1].startswith("error")):
                #     print(self.battle.active_pokemon, ") invalid move made!", self.agent.username, message)
                try:
                    self._handle_battle_message(message)
                except:
                    # print("errored and movemade:", moveMade)
                    break

                if(self.battle._finished):
                    break
        # print("active pokemon for", self.agent.username, self.battle.active_pokemon)


    async def doAction(self, action: BattleOrder) -> Battle:
        roomId = self.battle.battle_tag
        agentMsg = action.message
       
        # send the agent and possibly the opponents decision to the socket
        await self._sendMessage(agentMsg, roomId)
        await self._parseBattle()
        if self.battle._wait and not self.battle.finished:
            await self._parseBattle()

        # print()
        return self.battle
    
    # THE FOLLOWING FUNCTIONS ARE HEAVILY BASED ON POKE ENV CODE, citation provided
    # code is provided under MIT license, the library wasn't quite what we needed
    '''
    @misc{poke_env,
        author       = {Haris Sahovic},
        title        = {Poke-env: pokemon AI in python},
        url          = {https://github.com/hsahovic/poke-env}
    }
    '''
    def _handle_battle_message(self, message: str) -> None:  # pragma: no cover
        # Battle messages can be multiline
        split_messages = [m.split("|") for m in message.split("\n")]
        battle: Battle = None
        if (
            len(split_messages) > 1
            and len(split_messages[1]) > 1
            and split_messages[1][1] == "init"
        ):
            battle_info = split_messages[0][0].split("-")
            battle = self._create_battle(battle_info)
        else:
            battle = self.battle


        battle.trapped = False
        for split_message in split_messages[1:]:
            if len(split_message) <= 1:
                continue
            elif split_message[1] == "request":
                if split_message[2]:
                    # Since Python 3.7 json.loads preserves dict order by default.
                    request = orjson.loads(split_message[2])
                    battle._parse_request(request)
                    
                    side = request["side"]
                    self.orderedPartyPokemon = []
                    # comes from pokemon showdown, true order
                    for pokemon in side["pokemon"]:
                        if pokemon:
                            pokemon = battle.team[pokemon["ident"]]
                            # print("active?", pokemon.active, "fainted?", pokemon.fainted, pokemon.species)
                            if not pokemon.active:
                                self.orderedPartyPokemon.append(pokemon)

            elif split_message[1] == "win" or split_message[1] == "tie":
                if split_message[1] == "win":
                    battle._won_by(split_message[2])
                else:
                    battle._tied()
            elif split_message[1] in {"", "t:", "expire", "uhtmlchange"}:
                pass
            elif split_message[1] == "error" and (split_message[2].startswith(
                    "[Unavailable choice] Can't switch: The active Pokémon is "
                    "trapped"
                ) or split_message[2].startswith(
                    "[Invalid choice] Can't switch: The active Pokémon is trapped"
                )):
                    battle.trapped = True
            else:
                battle._parse_message(split_message)

                # handle case where switch happens but orderedPartyPokemon didnt get updated
                i, activePokemon = next(((i, pokemon) for i, pokemon in enumerate(self.orderedPartyPokemon) if pokemon.species == battle.active_pokemon.species), (None, None))
                if (activePokemon is not None):
                    prevActivePokemo = next((pokemon for pokemon in battle.team.values() if pokemon not in self.orderedPartyPokemon))
                    self.orderedPartyPokemon[i] = prevActivePokemo
                    
        self.battle = battle
        
    def _create_battle(self, split_message: List[str]) -> Battle:
        """Returns battle object corresponding to received message.

        :param split_message: The battle initialisation message.
        :type split_message: List[str]
        :return: The corresponding battle object.
        :rtype: AbstractBattle
        """
        if len(split_message) < 2: raise ShowdownException()
        
        battle_tag = "-".join(split_message)[1:]

        gen = GenData.from_format(Engine.battleFormat).gen
        battle = Battle(
            battle_tag=battle_tag,
            username=self.agent.username,
            logger=Logger("agent"),
            save_replays=False,
            gen=gen,
        )

        return battle