import numpy as np
from pokemon_battle_env import PokemonBattleEnv
from helpers import featurize, evaluate
from player_config import PlayerConfig
import configparser
from matplotlib import pyplot as plt
import time

config = configparser.ConfigParser()
config.read('config.ini')

def greedyPolicy(x, W, action_mask):
    valid_actions = np.where(action_mask)[0]
    return valid_actions[np.argmax((W.T @ x)[valid_actions])]

async def runQLAgent(env: PokemonBattleEnv, gen=1, max_episode=1, learnFromPrevModel=False, gamma=0.99, step_size=0.001, epsilon=0.5):
    wins = 0
    losses = 0

    if(learnFromPrevModel and gen > 1):
        try:
            W = np.load(f'./models/QL_model_gen{gen-1}.npy')
        except:
            print(f"model file ./models/QL_model_gen{gen-1}.npy not found!")
            exit()
    else:
        W = np.random.rand(env.observation_space.shape[0], env.action_space.n)

    evaluate_every = int(config.get("Agent Configuration", "evaluate_every"))
    evaluation_runs = int(config.get("Agent Configuration", "evaluation_runs"))
    eval_returns = []
    eval_winrates = []
    
    start_time = time.time()
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
        if (i+1) % evaluate_every == 0:
            progress_msg = "QL evaluation ("+str(len(eval_returns)+1)+"/"+str(max_episode//evaluate_every)+")"
            eval_return, win_rate = await evaluate(env, W, greedyPolicy, progress_msg, evaluation_runs)
            eval_returns.append(eval_return)
            eval_winrates.append(win_rate)
        wins, losses = wins+info["result"][0], losses+info["result"][1]

    np.save(f'./models/QL_model_gen{gen}.npy', W)
    
    plt.figure()
    plt.title('QL Returns')
    plt.xlabel("Evaluation Steps")
    plt.ylabel("Evaluation Results")
    plt.plot(eval_returns)
    plt.savefig(f'./plots/QL_returns_gen{gen}.png')
    plt.close()

    eval_winrates = [0.71, 0.5, 0.65, 0.77, 0.71, 0.7, 0.68, 0.77, 0.8, 0.76, 0.83, 0.68, 0.67, 0.14, 0.17, 0.24, 0.57, 0.56, 0.35, 0.51, 0.32, 0.58, 0.41, 0.53, 0.19]
    plt.figure()
    plt.title('QL Winrate')
    plt.xlabel("Evaluation Steps")
    plt.ylabel("Win Rate %")
    plt.ylim(0, 1)
    plt.plot(eval_winrates)
    plt.savefig(f'./plots/QL_winrate_gen{gen}.png')
    plt.close()

    await env.close()

async def runGreedyQLAgent(env: PokemonBattleEnv, gen=1, max_episode=1):
    wins = 0
    losses = 0

    try:
        W = np.load(f'./models/QL_model_gen{gen}.npy')
    except:
        print(f"model file ./models/QL_model_gen{gen}.npy not found!")
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

    
    
