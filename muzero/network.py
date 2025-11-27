# class Network(object):

# 	def initial_inference(self, image) -> NetworkOutput:
# 		# representation + prediction function
# 		return NetworkOutput(0, 0, {}, [])

# 	def recurrent_inference(self, hidden_state, action) -> NetworkOutput:
# 		# dynamics + prediction function
# 		return NetworkOutput(0, 0, {}, [])
    
# 	def get_weights(self):
# 		# Returns the weights of this network.
# 		return []

# 	def training_steps(self) -> int:
# 		# How many steps / batches the network has been trained for.
# 		return 0


import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
import numpy as np

from .config import MuZeroConfig


# === Small container for network outputs ===

@dataclass
class NetworkOutput:
    # Scalars are 1D tensors of shape (1,) or (batch,)
    value: torch.Tensor          # shape: (batch,)
    reward: torch.Tensor         # shape: (batch,)
    policy_logits: torch.Tensor  # shape: (batch, action_space_size)
    hidden_state: torch.Tensor   # shape: (batch, latent_dim)


# === Representation network h: observation -> latent state ===

class RepresentationNetwork(nn.Module):
    """
    h(o_t): observation (e.g. 64x64x3) -> latent vector s_0 (latent_dim).
    """

    def __init__(self, latent_dim: int = 128):
        super().__init__()
        # Expect obs as (B, C, H, W), with C=3, H=W=64 (we'll enforce this later)
        #TODO change to simple mlp for cartpole
        # todo might have to change the obs to tensor method
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=5, stride=2, padding=2),  # 64 -> 32
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1), # 32 -> 16
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1), # 16 -> 8
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1), # 8 -> 4
            nn.ReLU(),
        )
        self.fc = nn.Linear(64 * 4 * 4, latent_dim)
        
        # for 4 dim input (cartpole)
        self.net = nn.Sequential(
            nn.Linear(4, 64),
            nn.ReLU(),
            nn.Linear(64, latent_dim)
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """
        obs: (B, 3, 64, 64) float32 in [0,1]
        returns: latent state s_0: (B, latent_dim)
        """
        # x = self.conv(obs)                       # (B, 64, 4, 4)
        # x = x.view(x.size(0), -1)                # (B, 64*4*4)
        # s = self.fc(x)                           # (B, latent_dim)
        # return s
        
        # testing cartpole
        return self.net(obs)


# === Dynamics network g: (state, action) -> (next_state, reward) ===

class DynamicsNetwork(nn.Module):
    """
    g(s_k, a_k): latent state + action -> next latent state s_{k+1}, reward r_k.
    Action is discrete: we embed action index and concat to state.
    """

    def __init__(self, latent_dim: int, action_space_size: int, hidden_dim: int = 128):
        super().__init__()
        self.latent_dim = latent_dim
        self.action_space_size = action_space_size

        # Embed action index into a small vector
        # change to smaller embedding size for cart pole
        action_embedding_size = 4
        self.action_embedding = nn.Embedding(action_space_size, action_embedding_size)

        # MLP for next state
        self.state_mlp = nn.Sequential(
            nn.Linear(latent_dim + action_embedding_size, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
        )

        # MLP for reward scalar
        self.reward_head = nn.Sequential(
            nn.Linear(latent_dim + action_embedding_size, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, s: torch.Tensor, a: torch.Tensor):
        """
        s: (B, latent_dim)
        a: (B,) int64 action indices
        returns:
            s_next: (B, latent_dim)
            reward: (B,)
        """
        a_emb = self.action_embedding(a)         # (B, 16)
        x = torch.cat([s, a_emb], dim=1)         # (B, latent_dim+16)

        s_next = self.state_mlp(x)               # (B, latent_dim)
        reward = self.reward_head(x).squeeze(-1) # (B,)

        return s_next, reward


# === Prediction network f: state -> (value, policy_logits) ===

class PredictionNetwork(nn.Module):
    """
    f(s_k): latent state -> scalar value, policy logits over actions.
    """

    def __init__(self, latent_dim: int, action_space_size: int, hidden_dim: int = 128):
        super().__init__()
        self.latent_dim = latent_dim
        self.action_space_size = action_space_size

        self.mlp = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
        )

        self.value_head = nn.Linear(hidden_dim, 1)
        self.policy_head = nn.Linear(hidden_dim, action_space_size)

    def forward(self, s: torch.Tensor):
        """
        s: (B, latent_dim)
        returns:
            value: (B,)
            policy_logits: (B, action_space_size)
        """
        x = self.mlp(s)
        value = self.value_head(x).squeeze(-1)        # (B,)
        policy_logits = self.policy_head(x)           # (B, A)
        return value, policy_logits


# === Full MuZero network wrapper ===

class MuZeroNetwork(nn.Module):
    """
    Combines representation h, dynamics g, prediction f.
    Provides:
      - initial_inference(obs)
      - recurrent_inference(hidden_state, action)
    """

    def __init__(self, config: MuZeroConfig, device: torch.device | None = None):
        super().__init__()
        self.config = config
        self.latent_dim = 32
        self.action_space_size = config.action_space_size
        self.device = device if device is not None else 'cuda' if torch.cuda.is_available() else 'cpu'

        self.representation_network = RepresentationNetwork(latent_dim=self.latent_dim)
        self.dynamics_network = DynamicsNetwork(
            latent_dim=self.latent_dim,
            action_space_size=self.action_space_size,
            hidden_dim=64,
        )
        self.prediction_network = PredictionNetwork(
            latent_dim=self.latent_dim,
            action_space_size=self.action_space_size,
            hidden_dim=64,
        )
        #changing latent to 32, hidden to 64 (instead of 128) to test cartpole
        self.to(self.device)

        # Optional: track training steps
        self._training_steps = 0

    # --- Helper to ensure tensors are on correct device and batched ---

    def _obs_to_tensor(self, obs):
        """
        obs: either (H,W,C) numpy array or (B,C,H,W) torch tensor
        converts to torch.Tensor on self.device with shape (B,3,64,64)
        """
        if isinstance(obs, torch.Tensor):
            x = obs
            if x.dim() == 3:
                # assume (C,H,W) or (H,W,C); we handle (3,H,W) only here
                if x.shape[0] == 3:   # (3,H,W)
                    x = x.unsqueeze(0)
                else:
                    raise ValueError("Expected tensor obs as (3,H,W) or (B,3,H,W)")
            # (B,3,H,W)
            return x.to(self.device)

        # Assume numpy array (H,W,C) in [0,1] or [0,255]
        
        if isinstance(obs, np.ndarray):
            return torch.from_numpy(obs).float()  # working with carpole for now
        
            if obs.ndim != 3 or obs.shape[2] != 3:
                raise ValueError(f"Expected obs as (H,W,3), got {obs.shape}")
            x = torch.from_numpy(obs).float()          # (H,W,3)
            if x.max() > 1.0:
                x = x / 255.0
            x = x.permute(2, 0, 1).unsqueeze(0)        # (1,3,H,W)
            return x.to(self.device)

        raise TypeError("Unsupported obs type for initial_inference")

    def _action_to_tensor(self, action):
        """
        action: either Action object with .index or int or tensor
        returns LongTensor of shape (B,)
        """
        #TODO: can't take np array input, might have to adjust
        if isinstance(action, torch.Tensor):
            return action.long().to(self.device)

        # Single action case
        if hasattr(action, "index"):
            idx = action.index
        else:
            idx = int(action)

        a = torch.tensor([idx], dtype=torch.long, device=self.device)  # (1,)
        return a

    # --- Core MuZero interface ---

    def initial_inference(self, obs) -> NetworkOutput:
        """
        h(o_t) + f(s_0)
        obs: single observation (H,W,3) or tensor (3,H,W)
        returns NetworkOutput with batch dimension 1.
        """
        obs_tensor = self._obs_to_tensor(obs)          # (1,3,H,W)
        s0 = self.representation_network(obs_tensor)   # (1,latent_dim)

        value, policy_logits = self.prediction_network(s0)  # (1,), (1,A)

        # At root, MuZero usually sets reward = 0 (no previous action)
        reward = torch.zeros_like(value)
        return NetworkOutput(
            value=value.squeeze(0),                    # scalar
            reward=reward.squeeze(0),                  # scalar
            policy_logits=policy_logits.squeeze(0),    # (A,)
            hidden_state=s0.squeeze(0),                # (latent_dim,)
        )

    def recurrent_inference(self, hidden_state, action) -> NetworkOutput:
        """
        g(s_k, a_k) + f(s_{k+1})
        hidden_state: (latent_dim,) or (B, latent_dim)
        action: Action object or int or (B,) tensor
        """
        # Ensure batch
        if isinstance(hidden_state, torch.Tensor):
            s = hidden_state.to(self.device)
            if s.dim() == 1:
                s = s.unsqueeze(0)                     # (1,latent_dim)
        else:
            raise TypeError("hidden_state must be torch.Tensor")

        a = self._action_to_tensor(action)             # (1,) or (B,)

        s_next, reward = self.dynamics_network(s, a)   # (B,latent_dim), (B,)
        value, policy_logits = self.prediction_network(s_next)  # (B,), (B,A)

        return NetworkOutput(
            value=value.squeeze(0),
            reward=reward.squeeze(0),
            policy_logits=policy_logits.squeeze(0),
            hidden_state=s_next.squeeze(0),
        )

    # --- Utility methods used by training loop / MCTS scaffolding ---

    def get_weights(self):
        # Can be used to save checkpoints
        return self.state_dict()

    def set_weights(self, weights):
        self.load_state_dict(weights)

    def training_steps(self) -> int:
        return self._training_steps

    def increment_training_steps(self, n: int = 1):
        self._training_steps += n


# # Optional alias, if other files refer to `Network` type
Network = MuZeroNetwork
