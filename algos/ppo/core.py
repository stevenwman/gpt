import os

from mani_skill.utils import gym_utils
from mani_skill.utils.wrappers.flatten import FlattenActionSpaceWrapper
from mani_skill.utils.wrappers.record import RecordEpisode
from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv

os.environ["TORCHDYNAMO_INLINE_INBUILT_NN_MODULES"] = "1"

import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import wandb
from torch.distributions.normal import Normal

class Logger:
    def __init__(self, log_wandb=False, tensorboard: SummaryWriter = None) -> None:
        self.writer = tensorboard
        self.log_wandb = log_wandb
    def add_scalar(self, tag, scalar_value, step):
        if self.log_wandb:
            wandb.log({tag: scalar_value}, step=step)
        self.writer.add_scalar(tag, scalar_value, step)
    def close(self):
        self.writer.close()

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class Agent(nn.Module):
    def __init__(self, n_obs, n_act, device=None):
        super().__init__()
        self.critic = nn.Sequential(
            layer_init(nn.Linear(n_obs, 256, device=device)),
            nn.Tanh(),
            layer_init(nn.Linear(256, 256, device=device)),
            nn.Tanh(),
            layer_init(nn.Linear(256, 256, device=device)),
            nn.Tanh(),
            layer_init(nn.Linear(256, 1, device=device)),
        )
        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(n_obs, 256, device=device)),
            nn.Tanh(),
            layer_init(nn.Linear(256, 256, device=device)),
            nn.Tanh(),
            layer_init(nn.Linear(256, 256, device=device)),
            nn.Tanh(),
            layer_init(nn.Linear(256, n_act, device=device), std=0.01*np.sqrt(2)),
        )
        self.actor_logstd = nn.Parameter(torch.zeros(1, n_act, device=device))

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, obs, action=None):
        action_mean = self.actor_mean(obs)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        if action is None:
            action = action_mean + action_std * torch.randn_like(action_mean)
        return action, probs.log_prob(action).sum(1), probs.entropy().sum(1), self.critic(obs)