import numpy as np
from pokemon_battle_env import PokemonBattleEnv
from helpers import featurize, evaluate
from player_config import PlayerConfig
import configparser
from matplotlib import pyplot as plt

config = configparser.ConfigParser()
config.read('config.ini')

def greedyPolicy(x, W, action_mask):
    valid_actions = np.where(action_mask)[0]
    return valid_actions[np.argmax((W.T @ x)[valid_actions])]

async def runQLAgent(env: PokemonBattleEnv, max_episode=1, gamma=0.99, step_size=0.001, epsilon=0.5):
    wins = 0
    losses = 0

    W = np.random.rand(env.observation_space.shape[0], env.action_space.n)

    evaluate_every = int(config.get("Agent Configuration", "evaluate_every"))
    evaluation_runs = int(config.get("Agent Configuration", "evaluation_runs"))
    eval_returns = []
    eval_winrates = []
    
    for i in range(max_episode):
        s, info = await env.reset()
        s = featurize(env, s)
        terminated = truncated = False
        while not (terminated or truncated):
            a = greedyPolicy(s, W, env.valid_action_space_mask()) if np.random.random() < 1-epsilon else env.action_space.sample(env.valid_action_space_mask())
            s_next, r, terminated, truncated, info = await env.step(a)
            s_next = featurize(env, s_next)
            W[:, a] = W[:, a] + step_size * (r+gamma*(1-terminated)*np.amax(W.T @ s_next) - (W.T@s)[a]) * s
            s = s_next
        if i % evaluate_every == 0:
            eval_return, win_rate = await evaluate(env, W, greedyPolicy)
            eval_returns.append(eval_return)
            eval_winrates.append(win_rate)
        wins, losses = wins+info["result"][0], losses+info["result"][1]

    await env.close()

    print(f"runQLAgent record:\ngames played: {max_episode}, wins: {wins}, losses: {losses}, win percentage: {wins/max_episode}")
    print("Evaluated Returns: ", eval_returns)
    np.save('./models/QL_model.npy', W)
    
    plt.figure()
    plt.xlabel("Evaluation Steps")
    plt.ylabel("Evaluation Results")
    plt.plot(np.arange(1, evaluation_runs+1), eval_returns)
    plt.savefig('QL_plot_returns.png')

    plt.figure()
    plt.xlabel("Evaluation Steps")
    plt.ylabel("Win Rate %")
    plt.plot(np.arange(1, evaluation_runs+1), eval_winrates)
    plt.savefig('QL_plot_winrate.png')

async def runGreedyQLAgent(env: PokemonBattleEnv, model_file, max_episode=1):
    wins = 0
    losses = 0

    try:
        W = np.load(model_file)
    except:
        print(f"model file \'{model_file}\' not found!")
        exit()

    for i in range(max_episode):
        s, info = await env.reset()
        terminated = truncated = False
        while not (terminated or truncated):
            a = greedyPolicy(s, W, env.valid_action_space_mask())
            s_next, r, terminated, truncated, info = await env.step(a)
            s = s_next
        wins, losses = wins+info["result"][0], losses+info["result"][1]

    await env.close()

    print(f"runGreedyAgent record:\ngames played: {max_episode}, wins: {wins}, losses: {losses}, win percentage: {wins/max_episode}")
    
    
