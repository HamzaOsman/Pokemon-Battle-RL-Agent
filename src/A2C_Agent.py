import asyncio
from typing import Any, Dict
import gymnasium
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3 import A2C
from stable_baselines3.common.callbacks import BaseCallback
from helpers import desynchronize

from pokemon_battle_env import PokemonBattleEnv

class EvaluationCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(EvaluationCallback, self).__init__(verbose)
        self.wins, self.losses, self.ties = 0, 0, 0

    def _on_step(self) -> bool:
        '''
        called on every training step
        '''
        print("on step callback!!")
        info = self.locals["infos"][0]
        self.wins, self.losses, self.ties = self.wins+info["result"][0], self.losses+info["result"][1], self.ties+info["result"][2]

        # false to continue training, true to stop training
        return False
    def _on_training_start(self, locals_: Dict[str, Any], globals_: Dict[str, Any]) -> None:
        print("training starting?")

    def _on_training_end(self) -> None:
        print("training ending?")

    def _on_rollout_start(self) -> None:
        print("rollout starting?")

    def _on_rollout_end(self) -> None:
        print("rollout ending?")

    
class SynchronousWrapper(gymnasium.Env):
    def __init__(self, pokemonBattleEnv: PokemonBattleEnv):
        self.env = pokemonBattleEnv
        
        self.observation_space = pokemonBattleEnv.observation_space
        self.action_space = pokemonBattleEnv.action_space
        self.reward_range = pokemonBattleEnv.reward_range
        self.render_mode = pokemonBattleEnv.render_mode
        self._rendered = pokemonBattleEnv._rendered

    def reset(self, seed=None):
        print("wrapper reset called")
        result = desynchronize(self.env.reset(seed), "RESETTING!!")
        print("result", result)
        return result

    def step(self, action):
        return desynchronize(self.env.step(action))



async def learnA2C(
        env: PokemonBattleEnv,
        max_episodes=3000
    ):
    callback = EvaluationCallback()
    model = A2C("MlpPolicy", SynchronousWrapper(env), verbose=1)
    print("learning!!!")
    model = model.learn(total_timesteps=max_episodes, callback=callback)
    print("learning!!!")

    desynchronize(env.close())
    model.save("./models/A2C_Model")

    total = callback.wins+callback.losses+callback.tie
    print(f"actor critic record:\ngames played: {total}, wins: {callback.wins}, losses: {callback.losses}, win percentage: {callback.wins/(total)}")
    return model

def runA2C(
        env: PokemonBattleEnv,
        num_battles=3000
    ):
    wins = losses = ties = 0
    model = A2C.load("./models/A2C_Model")

    for i in range(num_battles):
        terminated = truncated = False
        s, info = desynchronize(env.reset())
        while not (terminated or truncated) :
            action, _states = model.predict(obs)
            obs, rewards, terminated, truncated, info = desynchronize(env.step(action))
            print(terminated, truncated)
        wins, losses, ties = wins+info["result"][0], losses+info["result"][1], ties+info["result"][2]
    desynchronize(env.close())
    
    total = wins+losses+ties
    print(f"actor critic record:\ngames played: {total}, wins: {wins}, losses: {losses}, win percentage: {wins/total}")

    