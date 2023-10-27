
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
from battle_strategy import BattleStrategy, RandomStrategy

from player_model import PlayerModel 

# engine which manages the battle
class Engine:
    # battleFormat: str = "gen3randombattle"
    battleFormat: str = "gen3ou"

    def __init__(self, agent: PlayerModel, opponent: PlayerModel, opponentStrategy: BattleStrategy = RandomStrategy):
        self.agentBattle: Battle = None
        self.opponentBattle: Battle = None
        self.agentPlayer = agent 
        self.opponentPlayer = opponent
        self.opponentStrategy = opponentStrategy

        # if the agent made a move but it was invalid the opponent needs to be aware
        self.isMakeupMove = False

    async def start(self):
        websocketUrl = "ws://localhost:8000/showdown/websocket"
        self.agentSocket = await websockets.connect(websocketUrl)
        await self._logPlayerIn(self.agentPlayer, self.agentSocket)

        self.opponentSocket = await websockets.connect(websocketUrl)
        await self._logPlayerIn(self.opponentPlayer, self.opponentSocket)

        await self._startBattle()
        
    async def waitUntilChallenge(self, socket: websockets.WebSocketClientProtocol):
        async for message in socket:
            # pipe-separated sequences
            messageSplit = message.split("|")
            if(messageSplit[1] == "pm" and messageSplit[4].startswith("/challenge")):
                return

    async def sendMessageToSocket(self, socket: websockets.WebSocketClientProtocol, message: str, room: str = ""):
        await socket.send("|".join([room, message]))

    async def _logPlayerIn(self, player: PlayerModel, socket: websockets.WebSocketClientProtocol):
        await self.sendMessageToSocket(socket, f"/trn {player.username},0,")

    async def _setTeam(self, player: PlayerModel, socket: websockets.WebSocketClientProtocol):
        # print("setting team:", "/utm %s" % player.team)
        await self.sendMessageToSocket(socket, "/utm %s" % player.team)

    async def _startBattle(self):
        print("battle started")
        await self._setTeam(self.agentPlayer, self.agentSocket)
        await self._setTeam(self.opponentPlayer, self.opponentSocket)

        await self.sendMessageToSocket(self.agentSocket, f"/challenge {self.opponentPlayer.username}, {Engine.battleFormat}") #gen3ou?

        await self.waitUntilChallenge(self.opponentSocket)
        await self.sendMessageToSocket(self.opponentSocket, "/accept %s" % self.agentPlayer.username)

        await self.parseInitialBattle(self.agentSocket, True)
        await self.parseInitialBattle(self.opponentSocket, False)
        if(self.agentBattle is None): print(self.agentPlayer.username, "NO BATTLE!!!")

    async def parseInitialBattle(self, socket: websockets.WebSocketClientProtocol, isAgent: bool):
        while True:
            message = await socket.recv()
            messageSplit = message.split("|")

            # ignore all other messages
            if(messageSplit[0].startswith(">battle")):
                if(messageSplit[1].startswith("error")):
                    print(self.agentPlayer.username, message)

                self._handle_battle_message(message, isAgent)
                # opponent is p2, they will be displayed last
                if(messageSplit[1] == "player" and self.opponentPlayer.username in message):
                    return

    async def parseBattle(self, socket: websockets.WebSocketClientProtocol, isAgent: bool):
        moveMade = True
        i = 0
        while i < 2:
            message = await socket.recv()
            messageSplit = message.split("|")
            # print(isAgent, message)
            # print()
            # ignore all other messages
            if(messageSplit[0].startswith(">battle")):
                i += 1
                if(messageSplit[1].startswith("error")):
                    moveMade = False
                    # print(self.agentPlayer.username, isAgent, message)
                skipNextMsg = self._handle_battle_message(message, isAgent)
                if(skipNextMsg):
                    break
        return moveMade
                
                
    # async def doActionNew(self, action: BattleOrder) -> Battle:
    #     roomId = self.agentBattle.battle_tag

    #     agentMsg = action.message

    #     opponentMsg = self.opponentStrategy.choose_action(self.opponentBattle)
    #     opponentMsg = opponentMsg.message

    #     task1 = asyncio.create_task(self.sendMessageToSocket(self.agentSocket, agentMsg, roomId))
    #     task2 = asyncio.create_task(self.sendMessageToSocket(self.opponentSocket, opponentMsg, roomId))
        
    #     # agent and opponent send their message together
    #     await asyncio.gather((task1, task2))

    #     # agent and opponent wait for results
    #     task1 = asyncio.create_task(self.parseBattle(self.agentSocket, True))
    #     task2 = asyncio.create_task(self.parseBattle(self.opponentSocket, False))

    #     complete, incomplete = await asyncio.wait((task1, task2), return_when=asyncio.FIRST_COMPLETED)

    #     for task in complete:
    #         if task.result:
    #             await 



    async def doAction(self, action: BattleOrder) -> Battle:
        roomId = self.agentBattle.battle_tag

        # agent acts right away
        agentMsg = action.message
        # if(self.agentBattle.trapped):
        #     print(self.agentPlayer.username, "is trapped and wants to use", agentMsg)
        await self.sendMessageToSocket(self.agentSocket, agentMsg, roomId)
        # print(self.agentPlayer.username, "sent", agentMsg)

        # if the agent is making up for failure to move last time then the opponent does not go 
        # the opponent only goes if it is not waiting
        if not self.isMakeupMove and self.opponentBattle._wait == False:
            opponentMsg = self.opponentStrategy.choose_action(self.opponentBattle)
            opponentMsg = opponentMsg.message
            await self.sendMessageToSocket(self.opponentSocket, opponentMsg, roomId)
            # print(self.opponentPlayer.username, "sent", opponentMsg)

        # parse the result 
        # print(self.agentPlayer.username, "before!")
        oMadeMove = await self.parseBattle(self.opponentSocket, False)
        # print(self.agentPlayer.username, "betweee!")
        if(oMadeMove):
            self.isMakeupMove = not await self.parseBattle(self.agentSocket, True)
        # print(self.agentPlayer.username, "after!")

        # print("opponent moved:", oMadeMove)
        # print("agent moved:", not self.isMakeupMove)

        # if the opponent made an invalid move or the agent needs to wait then the opponent must move again
        # (opponent tried to switch when trapped)
        # (agent must wait for opponent to send out a replacement pokemon)

        while((not oMadeMove or self.agentBattle._wait) and not self.agentBattle._finished):
            opponentMsg = self.opponentStrategy.choose_action(self.opponentBattle)
            opponentMsg = opponentMsg.message
            await self.sendMessageToSocket(self.opponentSocket, opponentMsg, roomId)
            # print(self.opponentPlayer.username, "sent on wait:", opponentMsg)
            oMadeMove = await self.parseBattle(self.opponentSocket, False)
            await self.parseBattle(self.agentSocket, True)
            # print("past")

        # print(self.opponentBattle.active_pokemon)
        # print(self.agentBattle.active_pokemon)
        return self.agentBattle



    # THE FOLLOWING FUNCTIONS ARE HEAVILY BASED ON POKE ENV CODE, citation provided
    '''
    @misc{poke_env,
        author       = {Haris Sahovic},
        title        = {Poke-env: pokemon AI in python},
        url          = {https://github.com/hsahovic/poke-env}
    }
    '''
    def _handle_battle_message(self, message: str, isAgentBattle: bool) -> None:  # pragma: no cover
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
            battle = self.agentBattle if isAgentBattle else self.opponentBattle


        battle.trapped = False
        for split_message in split_messages[1:]:
            if len(split_message) <= 1:
                continue
            elif split_message[1] == "request":
                if split_message[2]:
                    request = orjson.loads(split_message[2])
                    battle._parse_request(request)
            elif split_message[1] == "win" or split_message[1] == "tie":
                if split_message[1] == "win":
                    battle._won_by(split_message[2])
                else:
                    battle._tied()
                return True
            elif split_message[1] in {"", "t:", "expire", "uhtmlchange"}:
                pass
            elif split_message[1] == "error" and (split_message[2].startswith(
                    "[Unavailable choice] Can't switch: The active Pokémon is "
                    "trapped"
                ) or split_message[2].startswith(
                    "[Invalid choice] Can't switch: The active Pokémon is trapped"
                )):
                    battle.trapped = True
                    # print("trapped set")
            else:
                battle._parse_message(split_message)
        if isAgentBattle:
            self.agentBattle = battle
        else:
            self.opponentBattle = battle
        
        return False

        
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
            username=self.agentPlayer.username,
            logger=Logger("agent"),
            save_replays=False,
            gen=gen,
        )

        return battle