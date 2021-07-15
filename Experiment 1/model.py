import torch
import torch.nn as nn
from torch.distributions import Categorical


class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, n_latent_var, device):
        super(ActorCritic, self).__init__()
        self.device = device
        # actor
        self.actor = nn.Sequential(
            nn.Linear(state_dim, n_latent_var),
            nn.ReLU(),
            nn.Linear(n_latent_var, n_latent_var),
            nn.ReLU(),
            nn.Linear(n_latent_var, action_dim),
            nn.Softmax(dim=-1)
        )

        # critic
        self.critic = nn.Sequential(
            nn.Linear(state_dim, n_latent_var),
            nn.ReLU(),
            nn.Linear(n_latent_var, n_latent_var),
            nn.ReLU(),
            nn.Linear(n_latent_var, 1)
        )

    def forward(self):
        # for torch nn Module api compatibility only
        raise NotImplementedError

    def act(self, state, memory, action_mask=None):
        state = torch.from_numpy(state).float().to(self.device)
        action_probs = self.actor(state)
        if action_mask is not None:
            action_probs *= action_mask
        dist = Categorical(action_probs)
        action = dist.sample()

        memory.states.append(state)
        memory.actions.append(action)
        memory.logprobs.append(dist.log_prob(action))

        return action.item()

    def evaluate(self, state, action):
        action_probs = self.actor(state) # masking ?
        dist = Categorical(action_probs)

        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()

        state_value = self.critic(state)

        return action_logprobs, torch.squeeze(state_value), dist_entropy
