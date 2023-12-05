import random
from poke_env.player import BattleOrder, DefaultBattleOrder
from poke_env.environment.battle import Battle
from poke_env.teambuilder.constant_teambuilder import ConstantTeambuilder

class PlayerConfig():
    def __init__(self, username: str, isChallenger: bool, team: str = None, password=None) -> None:
        self.username = username
        self.password = password
        self.isChallenger = isChallenger
        self.team = ConstantTeambuilder(team).yield_team() if team else "null"