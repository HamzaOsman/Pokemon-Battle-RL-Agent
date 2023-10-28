from agent import Agent

class RandomAgent(Agent):
    
    def __init__(self, env):
        super().__init__(env)

    async def train(self, max_episode=1):
        wins = 0
        losses = 0
        for _ in range(max_episode):
            obs, _ = self.env.reset()
            while True:
                # TODO: check if it's valid somehow?
                action = self.env.action_space.sample()
                obs, reward, terminated, _, _ = await self.env.step(action)
                if (terminated): 
                    if (reward == 1):
                        wins += 1
                    else:
                        losses += 1
                    break
        return wins, losses