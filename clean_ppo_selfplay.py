import numpy as np
import copy
import gc
import os
import glob
import matplotlib.pyplot as plt
import pandas as pd

from collections import OrderedDict, deque
from itertools import product
from typing import Optional, Mapping

import torch
import torch.nn as nn

from selfplay_utils import batchify_obs, batchify, unbatchify, evaluate, plot_all_marl

def flush_buffer(buffer, path, counter):
    chunk_df = pd.DataFrame(buffer)
    float_cols = chunk_df.select_dtypes(include='float64').columns
    chunk_df[float_cols] = chunk_df[float_cols].astype('float32')
    chunk_df.to_parquet(f"{path}_{counter}.parquet", index=False, compression='snappy')
    buffer.clear()
    gc.collect()
    return counter + 1

class CleanPPOSelfPlay():
    def __init__(self, train_env, eval_env, agents, optimizers, device):

        self.device = device

        """ ENV SETUP """
        self.train_env = train_env
        self.eval_env = eval_env
        self.num_agents = len(train_env.possible_agents)
        self.num_actions = train_env.action_space(train_env.possible_agents[0]).n
        self.observation_size = train_env.observation_space(train_env.possible_agents[0]).shape
        self.num_steps = train_env.years

        self.agents = agents
        self.optimizers = optimizers

        self.policies = OrderedDict({a: 'marl' for a in self.train_env.possible_agents})

    def train(self, ent_coef: Mapping[str, float] = 0.01, ent_decay: float = 0.998,
            vf_coef: float = 0.1, gamma: float = 0.99, clip_coef: float = 0.1,
            batch_size: float = 32, gae_lambda: float = 0.99,
            eval_freq: int = 100, num_episodes: int = 1_000,
            update_epochs: int = 3, training_period: int = 1000, path: Optional[str] = None):

        # Create a matplotlib canvas for plotting learning curves
        fig, axs = plt.subplots(3, figsize=(10, 12), sharey=False, sharex=True)
        entropy_list = list()
        
        # Create a list to store the decision trace of the learning agents for SHAP
        decision_trace_buffer = []
        BUFFER_SIZE = 500  # flush every 500 rows
        flush_counter = 0
        trace_path = path + "models/decision_trace"
        
        # Log the decision trace for the last 1000 episodes (limits storage)
        log_start_episode = max(1, num_episodes - 1000 + 1)

        # ALGO Logic: Storage setup
        obs = {agent: torch.zeros((self.num_steps, *self.observation_size)).to(self.device)
                for agent in self.train_env.possible_agents}
        action_masks = {agent: torch.zeros((self.num_steps, self.num_actions)).to(self.device)
                for agent in self.train_env.possible_agents}
        actions = {agent: torch.zeros((self.num_steps)).to(self.device)
                for agent in self.train_env.possible_agents}
        logprobs = {agent: torch.zeros((self.num_steps,)).to(self.device)
                for agent in self.train_env.possible_agents}
        rewards = {agent: torch.zeros((self.num_steps,)).to(self.device)
                for agent in self.train_env.possible_agents}
        dones = {agent: torch.zeros((self.num_steps,)).to(self.device)
                for agent in self.train_env.possible_agents}
        values = {agent: torch.zeros((self.num_steps,)).to(self.device)
                for agent in self.train_env.possible_agents}

        # For storing plotting data
        eval_policies = ['marl', 'fixed_future']

        eval_profits = OrderedDict({a + '-' + policy: list() for a, policy in
                                    product(self.train_env.possible_agents, eval_policies)})
        eval_rewards = OrderedDict({a: list() for a in self.train_env.possible_agents})
        eval_markets = OrderedDict({a: list() for a in self.train_env.possible_agents + ['common']})
        eval_episode_t = list()

        market_profits = {'marl': None, 'future_profits': None}

        # Initialize an empty agent-actions dict
        agent_actions = {agent: None for agent in self.train_env.possible_agents}

        # make a copy of the learning agents in the agents list
        fixed_agents = deque(copy.deepcopy(self.train_env.possible_agents), maxlen=self.num_agents)
        learning_agent = fixed_agents.popleft()

        # Rollout-store-optimize loop
        for episode in range(1, num_episodes + 1):
            # Switch learning-agent and fixed-agent periodically
            if episode % training_period == 0 and episode != num_episodes:
                fixed_agents.append(learning_agent)
                learning_agent = fixed_agents.popleft()

            # Reset Env at the beginning of each episode
            new_obs, _ = self.train_env.reset(seed=None)

            # Extract the most recent observations and action-masks
            next_obs, next_action_mask = batchify_obs(new_obs, self.device)

            # Init variables for episode loop
            end_step = self.num_steps - 1
            total_episodic_return = 0

            # The episode loop - Rollout and collect transitions
            for step in range(0, self.num_steps):
                # ALGO LOGIC: action logic
                with torch.no_grad():
                    policy_outputs = {}

                    # Learning agent: sampled action + value
                    agent_actions[learning_agent], logprob, _, value = \
                        self.agents[learning_agent].get_action_and_value(
                            next_obs[learning_agent],
                            action_mask=next_action_mask[learning_agent])

                    # Store the value in the learning-agent's buffer
                    values[learning_agent][step] = value.flatten()

                    # Full policy outputs only when we're logging this episode
                    if episode >= log_start_episode:
                        logits, probs, val = self.agents[learning_agent].get_policy_outputs(
                            next_obs[learning_agent],
                            action_mask=next_action_mask[learning_agent])
                        policy_outputs[learning_agent] = {
                            "logits": logits.detach().cpu().reshape(-1).numpy(),
                            "probs": probs.detach().cpu().reshape(-1).numpy(),
                            "value": float(val.detach().cpu().reshape(-1)[0].item()),
                        }
                        del logits, probs, val #free memory

                    for agent in fixed_agents:
                        # fixed-agent action
                        agent_actions[agent] = self.agents[agent].get_action(
                            next_obs[agent],
                            inference=True,
                            action_mask=next_action_mask[agent]
                        )

                        # full policy outputs for logging
                        logits, probs, val = self.agents[agent].get_policy_outputs(
                                next_obs[agent],
                                action_mask=next_action_mask[agent]
                            )

                        policy_outputs[agent] = {
                            "logits": logits.detach().cpu().reshape(-1).numpy(),
                            "probs": probs.detach().cpu().reshape(-1).numpy(),
                            "value": float(val.detach().cpu().reshape(-1)[0].item())
                        }
                        del logits, probs, val  # free memory

                # Store transitions in the learning-agent's buffer
                obs[learning_agent][step] = next_obs[learning_agent]
                action_masks[learning_agent][step] = (next_action_mask)[learning_agent]
                actions[learning_agent][step] = agent_actions[learning_agent]
                logprobs[learning_agent][step] = logprob

                # Perform one step of the simulation
                new_obs, reward, terms, truncs, info = self.train_env.step(unbatchify(agent_actions))

                # Trace logging: one row per agent and year
                done = any([terms[a] for a in terms]) or any([truncs[a] for a in truncs])

                for agent in self.train_env.possible_agents:
                    # Skip fixed agent and non-logging episodes
                    if agent != learning_agent:
                        continue
                    if episode < log_start_episode:
                        continue

                    action_i = agent_actions[agent]
                    if torch.is_tensor(action_i):
                        action_i = int(action_i.detach().cpu().reshape(-1)[0].item())
                    else:
                        action_i = int(action_i)

                    row = {
                        "episode": episode,
                        "seed": getattr(self.train_env, "seed", None),
                        "year": step,
                        "agent": agent,
                        "learning_agent": agent == learning_agent,
                        "chosen_action": action_i,
                        "reward": float(reward[agent]),
                        "done": bool(done),
                        "value": policy_outputs[agent]["value"],
                        "chosen_logit": float(policy_outputs[agent]["logits"][action_i]),
                        "chosen_prob": float(policy_outputs[agent]["probs"][action_i]),
                    }

                    # observation tensor (flattened)
                    obs_np = next_obs[agent].detach().cpu().reshape(-1).numpy()
                    for i, v in enumerate(obs_np):
                        row[f"obs_{i}"] = float(v)

                    # full policy outputs
                    for i, v in enumerate(policy_outputs[agent]["logits"]):
                        row[f"logit_{i}"] = float(v)

                    for i, v in enumerate(policy_outputs[agent]["probs"]):
                        row[f"prob_{i}"] = float(v)

                    decision_trace_buffer.append(row)
                    
                    if len(decision_trace_buffer) >= BUFFER_SIZE:
                        flush_counter = flush_buffer(decision_trace_buffer, trace_path, flush_counter) 
                
                # Store the learning-agent's reward in its buffer
                rewards[learning_agent][step] = batchify(reward, self.device, learning_agent)
                dones[learning_agent][step] = batchify({learning_agent: done}, self.device, learning_agent)

                # Keep track of cumulative episodic reward
                total_episodic_return += reward[learning_agent]

                # If end of episode (terminated or truncated)
                if done:
                    # Set end-of-episode step and exit out of the episode loop
                    end_step = step + 1
                    break
                else:
                    # Bachify observations for the next iteration
                    next_obs, next_action_mask = batchify_obs(new_obs, self.device)

            # Evaluate Policy
            if (episode - 1) % eval_freq == 0 or episode == num_episodes:
                cumu_profits, cumu_rewards, episode_markets = evaluate(
                    env=self.train_env, agents=self.agents, device=self.device)

                total_marl_profit = 0
                total_future_profit = 0

                for a in list(cumu_profits.keys()):
                    eval_profits[a].append(cumu_profits[a])

                    if 'marl' in a:
                        total_marl_profit += cumu_profits[a]
                    elif 'fixed_future' in a:
                        total_future_profit += cumu_profits[a]

                for a in list(cumu_rewards.keys()):
                    eval_rewards[a].append(cumu_rewards[a])

                for a in episode_markets.keys():
                    eval_markets[a].append(episode_markets[a][0])

                # Append total market profits to the respective list in the market_profits dict
                if total_marl_profit > 0:
                    try:
                        market_profits['marl'].append(total_marl_profit)
                    except AttributeError:
                        market_profits['marl'] = [total_marl_profit]

                if total_future_profit > 0:
                    try:
                        market_profits['future_profits'].append(
                            total_future_profit)
                    except AttributeError:
                        market_profits['future_profits'] = [total_future_profit]

                eval_episode_t.append(episode)

                plot_all_marl(
                    axs=axs, policies=self.policies, eval_profits=eval_profits,
                    eval_episode_t=eval_episode_t, eval_rewards=eval_rewards,
                    seed=self.train_env.seed,
                    path=path,
                    close=(episode == num_episodes),
                    market_profits=market_profits)

            # bootstrap returns if not done
            with torch.no_grad():
                advantages = torch.zeros_like(rewards[learning_agent]).to(self.device)
                lastgaelam = 0
                for t in reversed(range(end_step)):
                    delta = rewards[learning_agent][t] + (gamma * values[learning_agent][t + 1] *
                            (1 - dones[learning_agent][t])) - values[learning_agent][t]

                    advantages[t] = lastgaelam = \
                        delta + (gamma * gae_lambda * (1 - dones[learning_agent][t]) * lastgaelam)

                returns = advantages + values[learning_agent]

            # Get the rollouts for the previous episode
            b_obs = obs[learning_agent][:end_step]
            b_logprobs = logprobs[learning_agent][:end_step]
            b_actions = actions[learning_agent][:end_step]
            b_action_masks = action_masks[learning_agent][:end_step]
            b_advantages = advantages[:end_step]
            b_returns = returns[:end_step]
            b_values = values[learning_agent][:end_step]

            # Decay Entropy Coefficient if instructed to do so
            if ent_decay is not None:
                ent_coef[learning_agent] *= ent_decay
            entropy_list.append(ent_coef[learning_agent])

            # Optimizing the policy and value network
            b_inds = np.arange(len(b_obs))
            clip_fracs = []
            for epoch in range(update_epochs):
                np.random.shuffle(b_inds)
                for start in range(0, len(b_obs), batch_size):
                    end = start + batch_size
                    mb_inds = b_inds[start:end]

                    _, newlogprob, entropy, newvalue = \
                        self.agents[learning_agent].get_action_and_value(
                            x=b_obs[mb_inds], action=b_actions.long()[mb_inds],
                            action_mask=b_action_masks[mb_inds])
                    logratio = newlogprob - b_logprobs[mb_inds]
                    ratio = logratio.exp()

                    with torch.no_grad():
                        clip_fracs += [((ratio - 1.0).abs() > clip_coef).float().mean().item()]

                    mb_advantages = b_advantages[mb_inds]

                    # Policy loss
                    pg_loss1 = -mb_advantages * ratio
                    pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - clip_coef, 1 + clip_coef)
                    pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                    # Value loss
                    newvalue = newvalue.view(-1)
                    # Clip value loss (keeps reward in a certain range)
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(newvalue - b_values[mb_inds], -clip_coef, clip_coef,)
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()

                    entropy_loss = entropy.mean()
                    loss = pg_loss - (ent_coef[learning_agent] * entropy_loss) + (v_loss * vf_coef)

                    self.optimizers[learning_agent].zero_grad()
                    loss.backward()

                    self.optimizers[learning_agent].step()

            y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
            var_y = np.var(y_true)
            explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        # Save final models (Policy and Value Networks) of all the agents
        for a in self.agents:
            checkpoint = {
                "model_state_dict": self.agents[a].state_dict(),
                "optimizer_state_dict": self.optimizers[a].state_dict(),
                "observation_size": self.observation_size,
                "num_actions": self.num_actions,
                "num_steps": self.num_steps,
                "agent_name": a,
            }
            torch.save(checkpoint, path + f"models/{a}.pth")

        # Save and flush any remaining rows
        if decision_trace_buffer:
            flush_counter = flush_buffer(decision_trace_buffer, trace_path, flush_counter)

        # Merge all chunks into one final file
        chunks = sorted(glob.glob(f"{trace_path}_*.parquet"))
        pd.concat([pd.read_parquet(f) for f in chunks]).to_parquet(
            path + "models/decision_trace.parquet", index=False)
        # Clean up chunk files
        for f in chunks:
            os.remove(f)

        # """ RENDER THE FINAL POLICY """
        for a in self.agents:
            self.agents[a].eval()

        with torch.no_grad():
            for episode in range(1):
                obs, info = self.eval_env.reset(seed=None)
                terms = [False]
                truncs = [False]
                actions = {a: None for a in self.eval_env.possible_agents}

                while not any(terms) and not any(truncs):
                    obs, action_mask = batchify_obs(obs, self.device)

                    # Get actions for  all agent
                    for a in self.eval_env.possible_agents:
                        actions[a] = self.agents[a].get_action(obs[a], action_mask=action_mask[a], inference=True)

                    # Perform one step of the simulation
                    obs, rewards, terms, truncs, infos = self.eval_env.step(unbatchify(actions))

                    terms = [terms[a] for a in terms]
                    truncs = [truncs[a] for a in truncs]
