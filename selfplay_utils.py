import numpy as np
import copy
import torch
import matplotlib.pyplot as plt

from typing import Sequence, Union, Mapping
from collections import OrderedDict
from itertools import product

from future_profit_max import FutureProfitMax

def batchify_obs(obs, device):
    # Convert PZ style observations to batch of torch arrays
    observations = {
        agent: torch.tensor(np.stack((obs[agent]['observation'],))).float().to(device)
        for agent in obs}
    action_mask = {
        agent: torch.tensor(np.stack((obs[agent]['action_mask'],))).float().to(device)
        for agent in obs}
    return observations, action_mask


def batchify(x, device, rl_agent):
    # Convert PZ style returns to batch of torch arrays
    x = torch.tensor(x[rl_agent]).to(device)
    return x


def unbatchify(x):
    # Converts np array to PZ style arguments
    actions = {agent: x[agent].cpu().numpy()[0] for agent in x}
    return actions


def evaluate(env, agents, device) -> Sequence[Union[int, float]]:
    # Evaluate the trained RL agent on the environment

    fixed_policies = ['fixed_future']

    # Initialize lists to store normalized episode rewards and no. of markets
    cumu_profits = OrderedDict({a + '-' + policy: None for a, policy in product(
            env.possible_agents, fixed_policies)})

    cumu_rewards = OrderedDict({a: 0 for a in env.possible_agents})

    markets = OrderedDict({a: list() for a in env.possible_agents + ['common']})

    # For storing all agent actions
    agent_actions = {a: None for a in env.possible_agents}

    obs, info = env.reset()

    # Create training_env clones for future profits policy evaluation
    future_profit_env = copy.deepcopy(env)

    # Future Profit Maximizing Policy
    future_profit_policy = FutureProfitMax(future_profit_env)
    _ = future_profit_policy.simulate(obs)
    for a in env.possible_agents:
        cumu_profits[a + '-' + 'fixed_future'] = (future_profit_env.cumu_profit)[a]

    steps = 0
    terms = [False]
    truncs = [False]

    # The state -> action -> reward, next-state loop
    while not any(terms) and not any(truncs):

        # Convert observations to torch tensors and get the action and logprob for the RL agent
        obs, action_mask = batchify_obs(obs, device)

        # Get actions for  all agent
        for agent in env.possible_agents:
            agent_actions[agent] = agents[agent].get_action(
                obs[agent], action_mask=action_mask[agent], inference=True)

        obs, rewards, terms, truncs, infos = env.step(unbatchify(agent_actions))
        terms = [terms[a] for a in terms]
        truncs = [truncs[a] for a in truncs]

        for a in env.possible_agents:
            cumu_rewards[a] = rewards[a]

        steps += 1

    # Store the cumulative returns in the respective list
    for a in env.possible_agents:
        cumu_profits[a + '-' + 'marl'] = env.cumu_profit[a]

    # Number of markets entered
    for a in env.possible_agents:
        markets[a].append(np.max(np.sum(env.p[a], axis=0)))

    # Common markets entered
    markets['common'].append(
        np.prod(np.hstack([env.p[a][:, -1] for a in env.possible_agents]), axis=-1).sum())

    return cumu_profits, cumu_rewards, markets

def plot_all_marl(
        axs, policies: Mapping[str, float], eval_profits: Mapping[str, float],
        eval_episode_t: Sequence[int], eval_rewards: Mapping[str, float],
        seed: int, market_profits=None,
        path=None, close=False) -> None:
        
    # Cumulative firm returns
    axs[0].clear()
    # MARL Policy
    for i, (firm, policy) in enumerate(policies.items()):
        axs[0].plot(eval_episode_t, eval_profits[f'firm_{i}-{policy}'], color=['red', 'mediumblue'][i], label=f'Firm {i}')

    # Fixed Baselines - Future Profit Maximizing Policy
    for i in range(len(policies)):
        try:
            axs[0].plot(eval_episode_t, eval_profits[f'firm_{i}-fixed_future'], ':', color=['red', 'mediumblue'][i])
        except KeyError:
            pass

    axs[0].set(title='Firm Returns')
    axs[0].set(ylabel='Unit of currency')
    axs[0].set(xlabel='Time-step/Episode')
    axs[0].legend(loc='upper left')

    # Cumulative market returns
    axs[1].clear()
    if market_profits['marl'] is not None:
        axs[1].plot(eval_episode_t, market_profits['marl'], color='red', label=f'MARL')

    if market_profits['future_profits'] is not None:
        axs[1].plot(eval_episode_t, market_profits['future_profits'], ':', color='blue', label=f'Future-Returns')

    axs[1].set(title='Total Market Returns')
    axs[1].set(ylabel='Unit of currency')
    axs[1].set(xlabel='Time-step/Episode')
    axs[1].legend(loc='lower right')

    # Episode Rewards
    axs[2].clear()
    
    for i, (firm, policy) in enumerate(policies.items()):
        axs[2].plot(eval_episode_t, eval_rewards[firm], color=['red', 'mediumblue'][i], label=f'Firm {i}')
    
    axs[2].set(title='Episode Rewards')
    axs[2].set(ylabel='Reward')
    axs[2].set(xlabel='Time-step/Episode')
    axs[2].legend(loc='upper left')

    fig = axs[0].get_figure()

    if close:
        fig.savefig(path + '../learning_curves/' + f"{seed}.png")
        plt.close(fig)