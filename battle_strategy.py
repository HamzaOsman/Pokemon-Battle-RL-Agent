from abc import ABC, abstractmethod
import random
from poke_env import PlayerConfiguration
from poke_env.player import BattleOrder
from poke_env.player import RandomPlayer
from poke_env.environment.battle import Battle

class BattleStrategy(ABC):
    @abstractmethod
    def choose_action() -> BattleOrder:
        pass

class RandomStrategy(BattleStrategy):
    def choose_action(battle: Battle) -> BattleOrder:
        possibleOrders = [BattleOrder(move) for move in battle.available_moves]
        possibleOrders.extend(
            [BattleOrder(switch) for switch in battle.available_switches]
        )
        return possibleOrders[int(random.random() * len(possibleOrders))]
