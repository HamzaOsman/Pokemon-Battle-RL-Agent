from pokemon_battle_env import PokemonBattleEnv
import numpy as np
import matplotlib.pyplot as plt
import configparser
from helpers import featurize, evaluate
import time

config = configparser.ConfigParser()
config.read('config.ini')

def softmaxProb(x: np.ndarray, Theta: np.ndarray):
    '''
    x: d-dimensional state features of some state, S
    Theta: dx|A| actor parameters (one column for each action)
    Returns the softmax probabilities over the actions for S
    '''
    h = Theta.T @ x
    m = np.max(h)
    probs = np.exp(h-m)/np.sum(np.exp(h-m))

    return probs

def softmaxPolicy(x: np.ndarray, Theta: np.ndarray, action_mask: np.ndarray):
    '''
    x: d-dimensional state features of some state, S
    Theta: dx|A| actor parameters (one column for each action)
    Returns an action, a, sampled from the softmax probabilities
    '''
    # set invalid actions to 0
    probs = getFilteredProbabilities(x, Theta, action_mask)

    # re-value the probabilities to sum to 1 by weight of the remaining
    probs = probs/np.sum(probs)

    choice = np.random.choice(range(0,len(probs)), p=probs.flatten())
    return choice
    
def logSoftmaxPolicyGradient(x: np.ndarray, a: int, Theta: np.ndarray, action_mask):
    '''
    x: d-dimensional state features of some state, S
    a: action represented by an integer
    Theta: dx|A| actor parameters (one column for each action)
    returns the dx|A| gradient of the chosen action WRT the Theta parameters
    '''
    d = Theta.shape[0]
    paddedFeature = np.zeros((Theta.shape))
    paddedFeature.T[a] = x
    x = x.reshape((d, 1))

    probs = getFilteredProbabilities(x, Theta, action_mask)

    policyWeighted = x @ probs.T 
    return paddedFeature - policyWeighted

def getFilteredProbabilities(x, Theta, action_mask):
    softmaxProbs = softmaxProb(x, Theta)
    valid_actions = np.where(action_mask)[0]

    valid_action_probs = np.zeros(softmaxProbs.shape)
    valid_action_probs[valid_actions] = softmaxProbs[valid_actions]

    if np.allclose(valid_action_probs, 0, atol=0):
        valid_action_probs[valid_actions] = 1/len(valid_actions)

    return valid_action_probs

async def learnActorCritic(
        env: PokemonBattleEnv,
        gen=1,
        max_episodes=1,
        gamma=0.99,
        actor_step_size=0.008,
        critic_step_size=0.008,                        
        learnFromPrevModel = False
        ):
    thetaModel = f"./models/AC_model_Theta_gen{gen-1}.npy"
    wModel = f"./models/AC_model_w_gen{gen-1}.npy"
    
    wins = 0
    losses = 0
    ties = 0
    
    # initialize theta and w
    if learnFromPrevModel:
        try:
            Theta = np.load(thetaModel)
        except Exception as e:
            print(e)
            print(f"model file \'{thetaModel}\' not found!")
            exit()

        try:
            w = np.load(wModel)
        except:
            print(f"model file \'{wModel}\' not found!")
            exit()
    else:
        Theta = np.random.random((env.observation_space.shape[0], env.action_space.n))
        w =  np.random.random(env.observation_space.shape[0])

    # for each battle
    evaluate_every = int(config.get("Agent Configuration", "evaluate_every"))
    evaluation_runs = int(config.get("Agent Configuration", "evaluation_runs"))
    eval_returns = []
    eval_winrates = []
    returns = np.zeros((max_episodes))
    start_time = time.time()
    for i in range(max_episodes):
        s, info = await env.reset()
        s = featurize(env, s)
        terminated = truncated = False
        actor_discount = 1
        rewardSum = 0
        # every action in battle
        while not (terminated or truncated):
            # choose action
            actionMask = env.valid_action_space_mask()
            action = softmaxPolicy(s, Theta, actionMask)

            # actionCounts[action] += 1
            
            # take step 
            observation, reward, terminated, truncated, info = await env.step(action)
            rewardSum += reward
            sPrime = featurize(env, observation)
            
            vhat = s @ w
            vhatPrime = 0 if terminated else (sPrime @ w)
            delta = reward + gamma*vhatPrime - vhat

            # gradient of linear = s
            w = w + critic_step_size*delta*s
            grad = logSoftmaxPolicyGradient(s, action, Theta, actionMask)
            Theta = Theta + actor_step_size*delta*actor_discount*grad 

            s = sPrime
            actor_discount *= gamma
        if (i+1) % evaluate_every == 0:
            progress_msg = "AC evaluation ("+str(len(eval_returns)+1)+"/"+str(max_episodes//evaluate_every)+")"
            eval_return, win_rate = await evaluate(env, Theta, softmaxPolicy, progress_msg, evaluation_runs)
            eval_returns.append(eval_return)
            eval_winrates.append(win_rate)
        wins, losses, ties = wins+info["result"][0], losses+info["result"][1], ties+info["result"][2]
        returns[i] = rewardSum
        rewardSum = 0

    np.save(f'./models/AC_model_Theta_gen{gen}.npy', Theta)
    np.save(f'./models/AC_model_w_gen{gen}.npy', w)
    
    plt.figure()
    plt.title(f'AC Gen {gen} Returns')
    plt.xlabel("Evaluation Steps")
    plt.ylabel("Evaluation Results")
    plt.plot(eval_returns)
    plt.savefig(f'./plots/AC_returns_gen{gen}.png')
    plt.close()

    plt.figure()
    plt.title(f'AC Gen {gen} Win Rate')
    plt.xlabel("Evaluation Steps")
    plt.ylabel("Win Rate %")
    plt.ylim(0, 1)
    plt.plot(eval_winrates)
    plt.savefig(f'./plots/AC_winrate_gen{gen}.png')
    plt.close()
    
    await env.close()
    
    return Theta, w

async def runActorCritic(env: PokemonBattleEnv, gen=1, numBattles=1000):
    thetaModel = f"./models/AC_model_Theta_gen{gen}.npy"
    wModel = f"./models/AC_model_w_gen{gen}.npy"

    wins = 0
    losses = 0
    ties = 0
    returns = np.zeros((numBattles))
    # actionCounts = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

    try:
        Theta = np.load(thetaModel)
    except:
        print(f"model file \'{thetaModel}\' not found!")
        exit()

    try:
        w = np.load(wModel)
    except:
        print(f"model file \'{wModel}\' not found!")
        exit()

    for i in range(numBattles):
        s, info = await env.reset()
        s = featurize(env, s)
        terminated = truncated = False
        rewardSum = 0
        while not (terminated or truncated):
            # choose action
            actionMask = env.valid_action_space_mask()
            action = softmaxPolicy(s, Theta, actionMask)
            
            # take step 
            observation, reward, terminated, truncated, info = await env.step(action)
            rewardSum += reward
            s = featurize(env, observation)
        wins, losses, ties = wins+info["result"][0], losses+info["result"][1], ties+info["result"][2]
        returns[i] = rewardSum
        rewardSum = 0 

    await env.close()