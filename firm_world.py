import numpy as np
import os
import json
import random
import pandas as pd

from typing import Mapping, Sequence, Optional

from gymnasium.spaces import Discrete, Box
from pettingzoo.utils.env import ParallelEnv

class FirmWorldEnv(ParallelEnv):
    def __init__(self, num_countries: int = 50, num_years: int = 50, num_firms: int = 2,
                data_dict: Mapping[str, int | float | Sequence[float]] = {},
                save_results: bool = True,
                comp_int_value: Optional[float] = 0.5, 
                beta: Optional[float] = 0.9, alpha: Optional[float] = 1.0,
                FMA: Optional[int] = 0, entry_barriers: Optional[int] = 0,
                country_spread: Optional[int] = 0, LOF: Optional[float] = 0, 
                LOF_reduction: Optional[float] = 0, path: Optional[str] = None, 
                seed: int = 0
                ):

        # Path for saving results, env parameters, model files, etc.
        self.path = path
        self.save_results = save_results

        # Set random seed for consistency in the created Env
        self.seed = seed
        np.random.seed(seed)
        random.seed(seed)
        
        #FMA - First Mover Advantage Input
        self.FMAinput = FMA
        
        #Entry Barriers Input
        self.entry_barriersinput = entry_barriers
        
        #Country Spread Input
        self.country_spreadinput = country_spread
        
        #alpha Input
        self.alpha_input = alpha
        
        #Beta Input
        self.beta_input = beta  
        
        #LOF - Liability of Foreignness Input
        self.LOFinput = LOF
        
        #LOF reduction due to other firm present in the market
        self.LOF_reductioninput = LOF_reduction
                
        # Competition intensity value range (or fixed value)
        self.comp_int_valueinput = comp_int_value

        # Set a fixed floating-point precision for all the arrays created
        self.dtype = np.float32

        # Data dictionary for setting the parameters of the simulation
        self.data_dict = data_dict

        # Number of countries in the simulated world
        self.countries = (num_countries if self.data_dict.get('num_countries') is None
                        else self.data_dict.get('num_countries'))

        # Time - Number of years the simulation runs for
        self.years = (num_years if self.data_dict.get('num_years') is None
                    else self.data_dict.get('num_years'))

        # Possible agents in the simulation are the number of firms
        self.possible_agents = ["firm_%d" % firm for firm in range(num_firms)]

        self.agents = self.possible_agents[:]

        # Observation Spaces for all agents
        self.observation_spaces = dict(zip(self.agents,
                                        [Box(shape=(4, num_countries),
                                                low=-np.inf,
                                                high=np.inf,
                                                dtype=self.dtype)] * len(self.agents)))

        # Action Spaces for all agents
        self.action_spaces = dict(zip(self.agents, [Discrete(self.countries + 1)] * len(self.agents)))

        # Correlation Matrix
        z = (np.linspace(-0.5, 0.5, self.countries, dtype=self.dtype).reshape(-1, 1) -
            np.linspace(-1.5, -0.5, self.countries, dtype=self.dtype))
        self.correlation = (np.triu(z, 0) + np.triu(z, 0).T - np.eye(self.countries))

        # Compute competitive intensity (0 < c < 1)
        self.comp_int = (np.random.uniform(self.comp_int_valueinput[0], self.comp_int_valueinput[1]) if
                        self.data_dict.get('comp_int') is None else
                        self.data_dict.get('comp_int'))

        # a: First mover's advantage in years
        self.fma = (np.random.randint(self.FMAinput[0], self.FMAinput[1]) if 
                    self.data_dict.get('fma') is None else 
                    self.data_dict.get('fma'))

        # Minimum capital required to enter a new market
        self.min_cap = (np.random.randint(self.entry_barriersinput[0], self.entry_barriersinput[1]) if
                    self.data_dict.get('min_cap') is None else
                    self.data_dict.get('min_cap'))

        # 𝛼: Mean of the normal distribution
        self.alpha = (np.random.uniform(self.alpha_input[0], self.alpha_input[1]) if 
                self.data_dict.get('alpha') is None else 
                self.data_dict.get('alpha'))

        # μ: Size of countries
        self.countries_size_spread = (
                np.random.uniform(self.country_spreadinput[0], self.country_spreadinput[1]) if
                self.data_dict.get('countries_size_spread') is None else
                self.data_dict.get('countries_size_spread'))

        self.countries_size = (
            np.random.uniform(500, 500 + self.countries_size_spread, self.countries) if 
            self.data_dict.get('countries_size') is None else 
            np.array(self.data_dict.get('countries_size')))

        # Λ: Liability of Foreignness (LoF)
        self.LoF_strength = (np.random.uniform(self.LOFinput[0], self.LOFinput[1]) if 
            self.data_dict.get('LoF_strength') is None else 
            self.data_dict.get('LoF_strength'))

        self.LoF = (np.power(1 - self.correlation, 1/self.LoF_strength) if
                    self.data_dict.get('LoF') is None else
                    np.array(self.data_dict.get('LoF')))

        # Λ: Liability of Foreignness Reduction
        self.LOF_ORredux = (np.random.uniform(self.LOF_reductioninput[0], self.LOF_reductioninput[1]) if 
            self.data_dict.get('LOF_ORredux') is None else 
            self.data_dict.get('LOF_ORredux'))

        # β: Auto-Correlation
        self.beta = (np.random.uniform(self.beta_input[0], self.beta_input[1]) if 
                self.data_dict.get('beta') is None else 
                self.data_dict.get('beta'))

        # Π: Market Size for all the countries and all years
        if self.data_dict.get('Pi') is None:
            # Initialize/Reset Market Size - For all the countries and all years
            self.Pi = np.zeros(shape=(self.countries, self.years), dtype=self.dtype)

            # Calculate Market Size for year-0
            self.Pi[:, 0] = np.random.multivariate_normal(
                mean=[self.alpha] * self.countries, cov=self.correlation) * self.countries_size

            # Sample and iteratively calculate Market Size for the entire episode
            for t in range(1, self.years):
                mvnd_sample = np.random.multivariate_normal(mean=[self.alpha] * self.countries, cov=self.correlation)
                self.Pi[:, t] = (self.Pi[:, t-1] * self.beta) + (mvnd_sample * self.countries_size * (1 - self.beta))
        else:
            self.Pi = np.array(self.data_dict.get('Pi'))

        # Randomly initialize country of origin for each firm
        if self.data_dict.get('firm_0') is None:
            self.firms = dict(zip(self.agents, np.random.choice(self.countries, len(self.agents),replace=True)))
        else:
            self.firms = {}
            for agent in self.agents:
                self.firms[agent] = self.data_dict.get(agent)

        # Indicator variables for representing presence of a firm in a country
        self.p = {agent: np.zeros(shape=(self.countries, self.years), dtype=int)
                for agent in self.agents}

        # Action mask for indicating valid actions (countries a firm can enter) at a time-step
        self.action_mask = {agent: np.ones(self.countries + 1, dtype=int)
                            for agent in self.agents}

        # Cumulative returns of firms initialized to 100 (Initial Resources)
        self.cumu_profit = dict(zip(self.agents, [100] * len(self.agents)))

        # Episode time (years) counter
        self.t = 0

        # Standard normalized Market Size, to be used as observations
        self.Pi_norm = ((self.Pi - np.mean(self.Pi, axis=0)) / np.std(self.Pi, axis=0))

        if save_results:
            self.yearly_profits = {a: np.zeros((self.years, 1)) for a in self.agents}
            self.yearly_markets = {a: np.ones((self.years, 1)) for a in self.agents}
            self.yearly_market_entries = {a: np.zeros((self.years, self.num_agents)) for a in self.agents}
            self.markets_entered = {a: np.ones((self.years,), dtype=int) * -1 for a in self.agents}
            self.first_entrants = np.empty((self.countries,),dtype=[('leader_firms', 'O')])
            self.first_entrants[:] = [(-1,)] * self.countries
            self.simultaneous_first_entrants = {a: np.zeros((self.years, 1), dtype=int) for a in self.agents}
            self.following = {a: np.ones(self.years) * -1 for a in self.agents}
        else:
            self.yearly_profits = None
            self.yearly_markets = None
            self.yearly_market_entries = None
            self.markets_entered = None
            self.first_entrants = None
            self.following = None

    def reset(self, seed=None, return_info=False, options=None):
        # Set random seed again for consistency in the created Env
        if seed is not None:
            self.seed = seed
            np.random.seed(seed)
            random.seed(seed)
        
        # Reset indicator variables for representing presence of a firm in a country
        self.p = {agent: np.zeros(shape=(self.countries, self.years), dtype=int)
                for agent in self.agents}

        # Initialize cumulative returns of firms initialized to 100 (initial resources)
        self.cumu_profit = dict(zip(self.agents, [100] * len(self.agents)))

        # Reset episode time (years) at the beginning of the episode
        self.t = 0

        # Action mask for indicating valid actions (countries a firm can enter) at a time-step
        self.action_mask = {agent: np.ones(self.countries + 1, dtype=int)
                            for agent in self.agents}

        # Mark countries of origin for firms
        for agent in self.agents:
            self.p[agent][self.firms[agent], 0] = 1
            self.action_mask[agent][self.firms[agent]] = 0

        # Reset first entrants and followers used for logging
        if self.save_results:
            self.yearly_profits = {a: np.zeros((self.years, 1)) for a in self.agents}
            self.yearly_markets = {a: np.ones((self.years, 1)) for a in self.agents}
            self.yearly_market_entries = {a: np.zeros((self.years, self.num_agents)) for a in self.agents}
            self.markets_entered = {a: np.ones((self.years,), dtype=int) * -1 for a in self.agents}
            self.first_entrants = np.empty((self.countries,), dtype=[('leader_firms', 'O'),])
            self.first_entrants['leader_firms'][:] = [(-1,)]
            self.simultaneous_first_entrants = {a: np.zeros((self.years, 1), dtype=int) for a in self.agents}
            self.following = {a: np.ones(self.years) * np.nan for a in self.agents}
            for agent in self.agents:
                self.first_entrants['leader_firms'][self.firms[agent]] = (self.agents.index(agent),)

        # Standard normalized Market Size, to be used as observations
        self.Pi_norm = ((self.Pi - np.mean(self.Pi, axis=0)) / np.std(self.Pi, axis=0))

        # Create observations and action_mask for all agents
        observations = {a: {"observation": None, "action_mask": None} for a in self.agents}
        current_market_size = self.Pi_norm[:, self.t]
        for agent in self.agents:
            presence = self.p[agent][:, self.t]
            distance = self.correlation[self.firms[agent]]

            # Presence of competitor in a country, as a sum of presence of all other firms in each country
            competitor_presence = np.zeros(self.countries, dtype=self.dtype)
            for a in self.agents:
                if a != agent:
                    competitor_presence += self.p[a][:, self.t]

            # Normalize competitor presence
            competitor_presence /= max(len(self.agents) - 1, 1)

            observations[agent]['observation'] = \
                np.stack([presence, distance, competitor_presence, current_market_size], axis=0, dtype=self.dtype)
            observations[agent]['action_mask'] = self.action_mask[agent]

        # Empty info dicts for all agents
        infos = {agent: {} for agent in self.agents}

        return observations, infos

    def step(self, actions):
        observations = {a: {"observation": None, "action_mask": None} for a in self.agents}
        rewards = dict(zip(self.agents, [0] * len(self.agents)))
        current_profits = dict(zip(self.agents, [0] * len(self.agents)))
        terminations = {a: False for a in self.agents}
        truncations = {a: False for a in self.agents}

        for i, agent in enumerate(self.agents):
            # Carry forward countries-present-at from the previous year
            self.p[agent][:, self.t + 1] = self.p[agent][:, self.t]

            if actions[agent] is None:
                # The agent is dead
                terminations[agent] = True
                current_profits[agent] = None
            elif self.action_mask[agent][actions[agent]] == 0:
                # Invalid action - Firm already present in the country
                current_profits[agent] = -10
                terminations[agent] = True
            else:  # Valid action
                if actions[agent] == self.countries:  # No action - Do nothing
                    pass
                elif self.cumu_profit[agent] >= self.min_cap:
                    # Deduct expansion cost from current resources
                    current_profits[agent] = -self.min_cap

                    # Update the indicator variable and action-mask based on the new country entered
                    self.p[agent][actions[agent], self.t + 1] = 1
                    self.action_mask[agent][actions[agent]] = 0

                    if self.save_results:
                        # Check and update first entrants and followers
                        previous_firms = list(self.first_entrants['leader_firms'][actions[agent]])

                        # Update the specific market entered by the current agent
                        self.markets_entered[agent][self.t + 1] = actions[agent]

                        # If the country is previously unoccupied
                        if previous_firms[0] == -1:
                            # First-mover: Update first entrant by appending the new firm
                            self.first_entrants['leader_firms'][actions[agent]] = tuple(previous_firms + [i])
                        else:
                            # Follower: Update firm to follower
                            self.following[agent][self.t + 1] = previous_firms[0]
                else:  # Insufficient resources for expansion
                    pass

        # Update "following" - markets entered at the same time marked as a negative number of simultaneous entrants
        for agent in self.agents:
            if actions[agent] is not None and actions[agent] < self.countries \
            and self.cumu_profit[agent] >= self.min_cap and self.save_results:
                previous_firms = list(self.first_entrants['leader_firms'][actions[agent]])
                if previous_firms[0] == -1:
                    self.following[agent][self.t + 1] = -(len(previous_firms) - 1)

        # Remove the placeholder '-1' from the first entrants list
        for agent in self.agents:
            if actions[agent] is not None and actions[agent] < self.countries \
            and self.cumu_profit[agent] >= self.min_cap and self.save_results:
                previous_firms = list(self.first_entrants['leader_firms'][actions[agent]])
                if previous_firms[0] == -1:
                    self.first_entrants['leader_firms'][actions[agent]] = (tuple(previous_firms[1:]))

                # Calculate market entry sequence
                current_firms = -1
                for a in self.agents:
                    current_firms += self.p[a][actions[agent], self.t + 1]

                # Exclusive First Entrant
                if current_firms == 0:
                    self.yearly_market_entries[agent][self.t + 1, current_firms] = 1
                else:
                    # Check for simultaneous first entrants
                    if np.sum([self.p[a][actions[agent], self.t] for a in self.agents]) == 0:
                        # Mark as simultaneous first entrants for the current firm
                        self.simultaneous_first_entrants[agent][self.t + 1] = 1
                    else:  # Following entrant cases - 2nd
                        self.yearly_market_entries[agent][self.t + 1, current_firms] = 1

        # Increment simulation time-step (year)
        self.t += 1

        current_market_size = self.Pi_norm[:, self.t]

        # Calculate rewards(rₜ), and create observations (sₜ₊₁) and action_masks ∀ agents
        for agent in self.agents:
            if not terminations[agent]:
                pi_i_t = self.p[agent][:, self.t] * self.Pi[:, self.t]
                experience_ratio = np.sum(self.p[agent], axis=-1) / (self.t + 1)

                # Get all the markets entered by the current agent as first-mover
                focal_firm = self.p[agent][:, :self.t]
                first_to_market = np.ones(self.countries, dtype=int)
                for a in self.agents:
                    if a != agent:
                        competing_firm = self.p[a][:, :self.t]
                        first_to_market *= (np.sum(focal_firm, axis=-1) >
                                            np.sum(competing_firm, axis=-1))

                # Calculate competition loss as a first-mover (if follower present)
                comp_loss_first = 0
                for a in self.agents:
                    if a != agent:
                        competing_firm = self.p[a][:, :self.t]
                        comp_loss_a = np.sum(first_to_market * pi_i_t * self.comp_int *
                            np.clip(np.sum(competing_firm, axis=-1) / self.fma, 0, 1.0), axis=-1)

                        if comp_loss_a > comp_loss_first:
                            comp_loss_first = comp_loss_a

                # Calculate competition loss for follower
                comp_loss_last = np.sum((1 - first_to_market) * pi_i_t * self.comp_int, axis=-1)
                    
                # Calculate LOF reduction due to other firm present in the market (reducing LOF)
                LOF_discount = (1 - first_to_market) * self.LOF_ORredux

                # Calculate current returns for the agent combining all the above factors
                current_profits[agent] += np.sum(pi_i_t) - np.sum(
                    pi_i_t * self.LoF[:, self.firms[agent]] * (1-LOF_discount) * 
                    (1 - experience_ratio)) - comp_loss_first - comp_loss_last

                self.cumu_profit[agent] += current_profits[agent]

                # Create observations and action_mask for the current agent
                presence = self.p[agent][:, self.t]
                distance = self.correlation[self.firms[agent]]

                # Presence of competitor in a country, as a sum of presence of other firm in each country
                competitor_presence = np.zeros(self.countries, dtype=self.dtype)
                for a in self.agents:
                    if a != agent:
                        competitor_presence += self.p[a][:, self.t]

                # Normalize competitor presence - 0 if no competitors, 1 if competitor present
                competitor_presence /= max(len(self.agents) - 1, 1)

                # Stack observations for the current agent as layers of features
                observations[agent]['observation'] = \
                    np.stack([presence, distance, competitor_presence, current_market_size], axis=0, dtype=self.dtype)
                observations[agent]['action_mask'] = self.action_mask[agent]

                # Keep track of yearly returns for logging
                if self.yearly_profits is not None:
                    self.yearly_profits[agent][self.t] = current_profits[agent]

                # Keep track of yearly expansion count for logging
                if self.yearly_markets is not None:
                    self.yearly_markets[agent][self.t] = sum(self.p[agent][:, self.t])

        # Check if the episode is over
        if self.t + 1 == self.years:
            # Indicate end of episode for all agents
            truncations = {a: True for a in self.agents}

            # Avoid negative returns: add |smallest negative return| to both returns and +1 for same home country
            if np.any(np.array([self.cumu_profit[a] for a in self.agents]) < 0):
                min_profit = np.min(np.array([self.cumu_profit[a] for a in self.agents]))
                for a in self.agents:
                    self.cumu_profit[a] += 1+np.abs(min_profit)
            # Calculate rewards based on the cumulative returns of all firms
            total_profit = np.sum(np.clip(np.array([self.cumu_profit[a] for a in self.agents]), 0, None))

            # Assign rewards to all firms based as a ratio of their returns
            for a in self.agents:
                rewards[a] = self.cumu_profit[a] / total_profit

            infos = {a: self.cumu_profit[a] for a in self.agents}
        else:  # Episode is not over
            # Zero rewards for all agents during the episode, and -10 for terminations
            rewards = {agent: -10 if terminations[agent] else 0 for agent in self.agents}

            # Empty info dicts for all agents
            infos = {agent: {} for agent in self.agents}

        return observations, rewards, terminations, truncations, infos

    def get_occupancy_matrix(self, t: int):
        # For each firm get an occupancy map indicating when a market was entered
        occu_mat = {a: np.zeros((self.countries, t+1), dtype=self.dtype)
                    for a in self.agents}

        # For each firm get an occupancy map indicating when a market was entered as a first-mover
        first_to_market = {a: np.zeros((self.countries, t+1), dtype=self.dtype)
                        for a in self.agents}

        # For each firm get an occupancy map indicating when a market was entered as a follower
        follower_to_market = {a: np.zeros((self.countries, t+1), dtype=self.dtype)
                            for a in self.agents}

        # Calculate occupancy matrix for each firm, indicating when a market was entered
        for i, agent in enumerate(self.possible_agents):
            entry_year = np.expand_dims(np.argmax(self.p[agent][:, :t+1] *
                            np.linspace(1, 0.1, t+1), axis=-1), axis=1)

            np.put_along_axis(occu_mat[agent], entry_year, 1, axis=1)
            occu_mat[agent] *= self.p[agent][:, :t+1]

        # Sum the occupancy matrices of all firms to get a combined occupancy matrix
        occu_mat_all_firms = np.sum(np.stack([occu_mat[agent] for agent in self.possible_agents],
                    axis=0), axis=0).astype(bool).astype(int)

        # Extract markets which were entered first by each firm
        for i, agent in enumerate(self.possible_agents):
            entry_year = np.expand_dims(np.argmax(occu_mat_all_firms * np.linspace(1, 0.1, t + 1), axis=-1), axis=1)
            np.put_along_axis(first_to_market[agent], entry_year, 1, axis=1)
            first_to_market[agent] *= occu_mat[agent]
            follower_to_market[agent] = occu_mat[agent] - first_to_market[agent]

        return first_to_market, follower_to_market

    def data_dump(self):
        env_params = np.hstack(
            (np.ones((self.years, 1)) * self.seed,
             np.ones((self.years, 1)) * self.num_agents,
             np.ones((self.years, 1)) * self.comp_int,
             np.ones((self.years, 1)) * self.fma,
             np.ones((self.years, 1)) * self.min_cap,
             np.ones((self.years, 1)) * self.alpha,
             np.ones((self.years, 1)) * self.countries_size_spread,
             np.ones((self.years, 1)) * self.LoF_strength,
             np.ones((self.years, 1)) * self.LOF_ORredux,
             np.ones((self.years, 1)) * self.beta,))
        env_params_columns = [
            'Seed', 'num_firms', 'Comp. Int.', 'FMA', 'Entry barriers', 'alpha',
            'country_spread', 'LoF Strength', 'LOF OR redux', 'beta']

        # Home country
        home_country = np.hstack(
            [np.ones((self.years, 1)) * int(self.firms[agent]) for agent in self.agents])
        home_country_columns = [f'Home_country-{agent}' for agent in self.agents]

        # Home country size
        home_country_size = np.hstack(
            [np.ones((self.years, 1)) * self.countries_size[self.firms[agent]] for agent in self.agents])
        home_country_size_columns = [f'Home_size-{agent}' for agent in self.agents]

        # Home country Pi
        home_country_pi = np.hstack(
            [np.ones((self.years, 1)) * self.Pi[self.firms[agent], 0] for agent in self.agents])
        home_country_pi_columns = [f'Home_pi-{agent}' for agent in self.agents]

        # Markets
        markets = np.hstack(
            [self.yearly_markets[agent] for agent in self.agents])
        markets_columns = [f'Markets-{agent}' for agent in self.agents]

        # Exclusive First-mover
        first_markets = np.hstack([
            np.cumsum(self.yearly_market_entries[agent][:, 0]).reshape(-1, 1) for agent in self.agents])
        first_markets_columns = [f'Exclusive_First_markets-{agent}' for agent in self.agents]

        # Follower 
        second_markets = np.hstack([
            np.cumsum(self.yearly_market_entries[agent][:, 1]).reshape(-1, 1) for agent in self.agents])
        second_markets_columns = [f'Second_markets-{agent}' for agent in self.agents]

        # Simultaneous first entrants
        try:
            simultaneous_first_markets = np.hstack([np.cumsum(
                self.simultaneous_first_entrants[agent]).reshape(-1, 1) for agent in self.agents])

            simultaneous_first_markets_columns = [f'Simultaneous_First_markets-{agent}' for agent in self.agents]
        except IndexError:
            simultaneous_first_markets = None
            simultaneous_first_markets_columns = []

        # Exclusive markets
        firm_presence = np.zeros_like(self.p['firm_0'])
        for agent in self.agents:
            firm_presence += self.p[agent]
        exclusive_markets = np.hstack([
            np.sum((firm_presence * self.p[agent]) == 1, axis=0).reshape(-1, 1) for agent in self.agents])
        exclusive_markets_columns = [f'Exclusive_markets-{agent}' for agent in self.agents]

        # Common markets
        common_markets = np.hstack([
            np.sum((firm_presence * self.p[agent]) >= 2, axis=0).reshape(-1, 1) for agent in self.agents])
        common_markets_columns = [f'Common_markets-{agent}' for agent in self.agents]

        # First-mover or following firm
        following_first_entrant = np.hstack([
            self.following[agent].reshape(-1, 1) for agent in self.agents])
        following_columns = [f'{agent}-Following' for agent in self.agents]

        # Yearly returns
        profits = np.hstack([self.yearly_profits[agent] for agent in self.agents])
        profits_columns = [f'Profit-{agent}' for agent in self.agents]

        # Yearly return share
        with np.errstate(divide='ignore', invalid='ignore'):
            profits_share = profits / np.sum(profits, axis=-1).reshape(-1, 1)
        profits_share_columns = [f'Profit_share-{agent}' for agent in self.agents]

        # Cumulative returns
        cumu_profits = np.hstack(
                [np.cumsum(self.yearly_profits[agent]).reshape(-1, 1) for agent in self.agents])
        cumu_profits_columns = [f'Cumu_profit-{agent}' for agent in self.agents]

        # Cumulative return share
        with np.errstate(divide='ignore', invalid='ignore'):
            cumu_profits_share = cumu_profits / np.sum(cumu_profits, axis=-1).reshape(-1, 1)
        cumu_profits_share_columns = [f'Cumu_profit_share-{agent}' for agent in self.agents]

        # Stack all arrays to create Env parameters array
        env_array = np.hstack((
            env_params, home_country, home_country_size, home_country_pi,
            markets, exclusive_markets, common_markets, following_first_entrant,
            profits, profits_share, cumu_profits, cumu_profits_share,
            first_markets, second_markets))

        if simultaneous_first_markets is not None:
            env_array = np.hstack((env_array, simultaneous_first_markets))

        # Stack all columns to create Env parameters columns array
        env_columns = (
            env_params_columns + home_country_columns +
            home_country_size_columns + home_country_pi_columns +
            markets_columns + exclusive_markets_columns +
            common_markets_columns + following_columns + profits_columns +
            profits_share_columns + cumu_profits_columns +
            cumu_profits_share_columns + first_markets_columns +
            second_markets_columns + simultaneous_first_markets_columns)

        # Convert Env parameters array to Pandas DF
        env_df = pd.DataFrame(env_array, columns=env_columns)

        # Convert Pi array to Pandas DF
        pi_df = pd.DataFrame(self.Pi)

        # Write DFs to CSV
        env_df.to_csv(self.path + 'results/env_data_marl.csv')
        pi_df.to_csv(self.path + 'results/Pi.csv')

        # Convert yearly market entries to Pandas DF and write to CSV
        for agent in self.agents:
            firm_p_df = pd.DataFrame(self.p[agent])
            firm_p_df.to_csv(self.path + f'results/firm{agent}_p.csv')

    def observation_space(self, agent):
        return self.observation_spaces[agent]

    def action_space(self, agent):
        return self.action_spaces[agent]

    def close(self):
        if self.save_results:
            self.data_dump()
