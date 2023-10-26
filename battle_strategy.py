from abc import ABC, abstractmethod
from poke_env import PlayerConfiguration
from poke_env.player import BattleOrder
from poke_env.player import RandomPlayer
from poke_env.environment.battle import Battle

class BattleStrategy(ABC):
    @abstractmethod
    def choose_action() -> BattleOrder:
        pass

class RandomStrategy(BattleStrategy):
    randomPlayer = RandomPlayer(PlayerConfiguration("dummy", None))
    def choose_action(battle: Battle) -> BattleOrder:
        return RandomStrategy.randomPlayer.choose_move(battle)