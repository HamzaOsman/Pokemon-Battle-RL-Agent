import random
from poke_env.player import BattleOrder, DefaultBattleOrder
from poke_env.environment.battle import Battle
from poke_env.teambuilder.constant_teambuilder import ConstantTeambuilder

class Agent():
    def __init__(self, username: str, isChallenger: bool, team: str = None) -> None:
        self.username = username
        self.isChallenger = isChallenger
        self.team = ConstantTeambuilder(team).yield_team() if team else "null"