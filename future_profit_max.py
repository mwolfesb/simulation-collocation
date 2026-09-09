import numpy as np
from firm_world import FirmWorldEnv


def future_profit_max_strategy(env, agent, obs, year):
    if (env.cumu_profit[agent] > env.min_cap and np.sum(obs['action_mask'][:-1]) > 0):
        
        # Get the average of the past returns of all markets without current presence of agent
        mu_pi_0_t = ((1 - env.p[agent][:, year]) * np.mean(env.Pi[:, :year+1], axis=-1)).reshape(-1, 1)

        # Average of past returns as expected future returns (all future years, incl. current year)
        mu_pi_t_T = np.tile(mu_pi_0_t, (1, env.years - year - 1))

        # Loss due to LOF (accounting for experience in the market)
        years_remaining = env.years - year
        experience_ratio = (np.arange(1, years_remaining) / years_remaining).reshape(1, -1)
        lof_loss = mu_pi_t_T * (env.LoF[:, env.firms[agent]].reshape(-1, 1) * (1 - experience_ratio))

        # Competitor presence in each market
        num_competitors_in_market = np.zeros(env.countries, dtype=env.dtype)
        for a in env.possible_agents:
            if a != agent:
                num_competitors_in_market += env.p[a][:, year].reshape(-1)

        # Calculate competition loss for all markets
        comp_loss = np.sum(
            num_competitors_in_market.reshape(-1, 1) * mu_pi_t_T * env.comp_int, axis=-1, keepdims=True)

        # Calculate potential returns for all markets considering: LOF, experience, and competition loss (based on current market shares)
        potential_future_profits = (np.sum(mu_pi_t_T - lof_loss - comp_loss, axis=-1) - env.min_cap)

        # Apply the action mask and choose market with the highest potential future returns
        with np.errstate(divide='ignore'):
            next_market = np.argmax(potential_future_profits + np.log(obs['action_mask'][:-1]))

        # Enter the market with positive future returns
        if potential_future_profits[next_market] > 0:
            return next_market
        else:
            return env.countries   # The do-nothing action
    else:
        return env.countries  # The do-nothing action


class FutureProfitMax():
    def __init__(self, env: FirmWorldEnv):
        self.env = env

    def simulate(self, obs):
        # Initialize a returns dictionary to None
        profits = {agent: None for agent in self.env.possible_agents}

        for y in range(self.env.years-1):
            # Initialize default actions to None for all agents
            actions = {agent: None for agent in self.env.possible_agents}

            # For each agent, get the action that has the highest potential future return
            for agent in self.env.possible_agents:
                actions[agent] = future_profit_max_strategy(self.env, agent, obs[agent], y)

            # Take a step in the environment
            obs, rewards, terminations, truncations, infos = self.env.step(actions)

            # Accumulate the returns
            for agent in self.env.possible_agents:
                profits[agent] = infos[agent]

        return profits, rewards
