class Network(object):

  def initial_inference(self, image) -> NetworkOutput:
    # representation + prediction function
    return NetworkOutput(0, 0, {}, [])

  def recurrent_inference(self, hidden_state, action) -> NetworkOutput:
    # dynamics + prediction function
    return NetworkOutput(0, 0, {}, [])
x
  def get_weights(self):
    # Returns the weights of this network.
    return []

  def training_steps(self) -> int:
    # How many steps / batches the network has been trained for.
    return 0
  


  # muzero/network.py
import torch
import torch.nn as nn
from typing import Dict, NamedTuple

class NetworkOutput(NamedTuple):
    value: torch.Tensor          # shape [B, 1]
    reward: torch.Tensor         # shape [B, 1]
    policy_logits: Dict[int, torch.Tensor]  # mapping from action idx to logit
    hidden_state: torch.Tensor   # shape [B, latent_dim]

class RepresentationNet(nn.Module):
    def __init__(self, obs_channels=8, latent_dim=256):
        super().__init__()
        # Example: 3D convs or 2D convs on BEV projection – keep it small at first
        self.conv = nn.Sequential(
            nn.Conv3d(obs_channels, 32, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv3d(32, 64, 3, stride=2, padding=1),
            nn.ReLU(),
        )
        self.fc = nn.Linear(64 * 8 * 8 * 4, latent_dim)  # adjust dims

    def forward(self, obs):
        x = self.conv(obs)        # [B, C, D, H, W]
        x = x.view(x.size(0), -1)
        z = torch.tanh(self.fc(x))  # latent state
        return z

class DynamicsNet(nn.Module):
    def __init__(self, latent_dim=256, action_space_size=15):
        super().__init__()
        self.action_embed = nn.Embedding(action_space_size, 16)
        self.fc = nn.Sequential(
            nn.Linear(latent_dim + 16, 256),
            nn.ReLU(),
            nn.Linear(256, latent_dim + 1),   # next latent + reward
        )

    def forward(self, hidden_state, action_index):
        a_emb = self.action_embed(action_index)  # [B, 16]
        x = torch.cat([hidden_state, a_emb], dim=-1)
        out = self.fc(x)
        next_hidden = torch.tanh(out[..., :-1])
        reward = out[..., -1:]
        return next_hidden, reward

class PredictionNet(nn.Module):
    def __init__(self, latent_dim=256, action_space_size=15):
        super().__init__()
        self.policy_head = nn.Linear(latent_dim, action_space_size)
        self.value_head = nn.Linear(latent_dim, 1)

    def forward(self, hidden_state):
        logits = self.policy_head(hidden_state)   # [B, A]
        value  = self.value_head(hidden_state)    # [B, 1]
        return logits, value

class MuZeroNetwork(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.repr = RepresentationNet()
        self.dyn  = DynamicsNet(action_space_size=config.action_space_size)
        self.pred = PredictionNet(action_space_size=config.action_space_size)
        self._training_steps = 0

    def initial_inference(self, obs) -> NetworkOutput:
        # obs: [C, D, H, W] or [B, ...], wrap to batch
        if obs.dim() == 4:
            obs = obs.unsqueeze(0)
        hidden = self.repr(obs)
        policy_logits, value = self.pred(hidden)
        reward = torch.zeros_like(value)  # no reward at root
        # Map logits to a dict[action_idx] for compatibility with pseudocode
        policy_dict = {i: policy_logits[..., i] for i in range(self.config.action_space_size)}
        return NetworkOutput(value, reward, policy_dict, hidden)

    def recurrent_inference(self, hidden_state, action) -> NetworkOutput:
        # action: single Action index or tensor of indices
        if not torch.is_tensor(action):
            action = torch.tensor([action.index], dtype=torch.long, device=hidden_state.device)
        next_hidden, reward = self.dyn(hidden_state, action)
        policy_logits, value = self.pred(next_hidden)
        policy_dict = {i: policy_logits[..., i] for i in range(self.config.action_space_size)}
        return NetworkOutput(value, reward, policy_dict, next_hidden)

    def training_steps(self):
        return self._training_steps

    def increment_training_steps(self, n=1):
        self._training_steps += n
