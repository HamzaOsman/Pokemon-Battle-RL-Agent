import random
from poke_env.player import BattleOrder, DefaultBattleOrder
from poke_env.environment.battle import Battle
from poke_env.teambuilder.constant_teambuilder import ConstantTeambuilder

class Agent():
    def __init__(self, username: str, isChallenger: bool, team: str = None) -> None:
        self.username = username
        self.isChallenger = isChallenger
        self.team = ConstantTeambuilder(team).yield_team() if team else "null"

    def choose_action(self, battle: Battle) -> BattleOrder:
        possibleOrders = [BattleOrder(move) for move in battle.available_moves]
        possibleOrders.extend(
            [BattleOrder(switch) for switch in battle.available_switches]
        )

        if len(possibleOrders) > 0:
            i = int(random.random() * len(possibleOrders))
            # print(self.username, i, len(possibleOrders))
            # print(self.username, i, battle.available_moves)
            if len(possibleOrders) == 0:
                print(self.username, i, battle.active_pokemon.moves.keys())
                print(self.username, i, battle.available_moves)
            return possibleOrders[i]
        else:
            return DefaultBattleOrder()