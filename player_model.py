from poke_env.teambuilder.constant_teambuilder import ConstantTeambuilder

class PlayerModel:
    def __init__(self, username:str, team:str = None):
        self.username = username
        self.team = ConstantTeambuilder(team).yield_team() if team else "null"

        