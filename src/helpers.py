import numpy as np
from pokemon_battle_env import PokemonBattleEnv
import torch

def featurize(env: PokemonBattleEnv, x):
    x_normalized = (x - env.observation_space.low) / (env.observation_space.high - env.observation_space.low) # min-max normalization
    return x_normalized


async def evaluate(env: PokemonBattleEnv, model, policy_func, progress_msg, n_runs=10, reward_func=None):
    all_returns = np.zeros([n_runs])
    wins = 0
    losses = 0
    ties = 0
    for i in range(n_runs):
        s, info = await env.reset()
        s = featurize(env, s)
        terminated = truncated = False
        total_return = 0
        while not (terminated or truncated):
            a = policy_func(s, model, env.valid_action_space_mask()) 
            s_next, r, terminated, truncated, info = await env.step(a)
            if (reward_func):
                total_return += reward_func(torch.tensor(s_next, dtype=torch.float32), torch.tensor(model['goal'], dtype=torch.float32))
            else:
                total_return += r
        all_returns[i] = total_return
        wins, losses, ties = wins+info["result"][0], losses+info["result"][1], ties+info["result"][2]
    win_rate = wins / n_runs
    avg_return = np.mean(all_returns)
    print("\n", progress_msg)
    print(f"last_battle: {env.engine.battle.battle_tag}, avg_return: {np.round(avg_return, 3)}, W/L/T: {wins}/{losses}/{ties}, win_rate: {np.round(win_rate, 3)}")
    return avg_return, win_rate