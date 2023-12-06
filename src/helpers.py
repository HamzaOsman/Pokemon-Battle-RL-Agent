import numpy as np
from pokemon_battle_env import PokemonBattleEnv

def featurize(env: PokemonBattleEnv, x):
    x_normalized = (x - env.observation_space.low) / (env.observation_space.high - env.observation_space.low) # min-max normalization
    return x_normalized


async def evaluate(env, model, policy_func, n_runs=10):
    all_returns = np.zeros([n_runs])
    wins = 0
    for i in range(n_runs):
        s, info = await env.reset()
        s = featurize(env, s)
        terminated = truncated = False
        total_return = 0
        while not (terminated or truncated):
            a = policy_func(s, model, env.valid_action_space_mask()) 
            s_next, r, terminated, truncated, info = await env.step(a)
            total_return += r
        all_returns[i] = total_return # only final rewards atm (no intermittent)
        wins += info["result"][0]
    win_rate = wins / n_runs
    return np.mean(all_returns), win_rate