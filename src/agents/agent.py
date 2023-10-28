from abc import ABC, abstractmethod

class Agent(ABC):

    def __init__(self, env):
        self.env = env

    @abstractmethod
    def train(self, *args, **kwargs) -> (int, int):
        # returns wins and losses
        pass