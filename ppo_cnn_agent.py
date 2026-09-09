import numpy as np

import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical

class CNNAgent(nn.Module):
    def __init__(self, num_countries: int, num_actions: int, obs_depth: int,
                l1_kernels: int, l2_kernels: int, fc_size: int):
        super().__init__()

        self.network = nn.Sequential(
            self._layer_init(nn.Conv1d(obs_depth, l1_kernels, 1)), 
            nn.ReLU(),
            self._layer_init(nn.Conv1d(l1_kernels, l2_kernels, 1)), 
            nn.ReLU(),
            nn.Flatten(),
            self._layer_init(nn.Linear(num_countries * l2_kernels, fc_size)), 
            nn.ReLU(),
        )
        self.actor = self._layer_init(nn.Linear(fc_size, num_actions), std=0.01)
        self.critic = self._layer_init(nn.Linear(fc_size, 1))

    def _layer_init(self, layer, std=np.sqrt(2), bias_const=0.0):
        torch.nn.init.orthogonal_(layer.weight, std)
        torch.nn.init.constant_(layer.bias, bias_const)
        return layer

    def get_policy_outputs(self, x, action_mask=None):
        hidden = self.network(x)
        logits = self.actor(hidden)

        if action_mask is not None:
            logits += torch.log(action_mask)

        probs = torch.softmax(logits, dim=-1)
        value = self.critic(hidden)
        return logits, probs, value

    def get_value(self, x):
        return self.critic(self.network(x))

    def get_action(self, x, action_mask=None, inference=False):
        hidden = self.network(x)
        logits = self.actor(hidden)
        if action_mask is not None:
            logits += torch.log(action_mask)
        probs = Categorical(logits=logits)
        if not inference:
            action = probs.sample()
        else:
            action = torch.argmax(probs.probs, dim=-1)
        return action

    def get_action_and_value(self, x, action=None, action_mask=None, inference=False):
        hidden = self.network(x)
        logits = self.actor(hidden)
        if action_mask is not None:
            logits += torch.log(action_mask)
        probs = Categorical(logits=logits)
        if action is None:
            if not inference:
                action = probs.sample()
            else:
                action = torch.argmax(probs.probs, dim=-1)
        return action, probs.log_prob(action), probs.entropy(), self.critic(hidden)
