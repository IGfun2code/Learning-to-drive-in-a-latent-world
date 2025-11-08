# baseline.py
from typing import Callable, Dict, Any
import gymnasium as gym
import torch
import torch.nn as nn
from stable_baselines3 import PPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from encoder import ResNetEncoder


class CarlaFeatureExtractor(BaseFeaturesExtractor):
    """
    Wraps our vision encoder so SB3 PPO can use it.
    SB3 will pass observations already as tensors of shape (B, C, H, W)
    if the env is set up correctly.
    """
    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 512):
        super().__init__(observation_space, features_dim)
        # you can change out_dim to features_dim to match PPO expectations
        self.encoder = ResNetEncoder(out_dim=features_dim, freeze=True)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        # observations: (B, C, H, W)
        return self.encoder(observations)


def make_ppo_baseline(
    env: gym.Env,
    learning_rate: float = 3e-4,
    features_dim: int = 512,
    **kwargs: Any,
) -> PPO:
    """
    Create a PPO model with our custom vision encoder.
    """
    policy_kwargs: Dict[str, Any] = dict(
        features_extractor_class=CarlaFeatureExtractor,
        features_extractor_kwargs=dict(features_dim=features_dim),
    )

    model = PPO(
        policy="CnnPolicy",   # we'll override the feature extractor anyway
        env=env,
        learning_rate=learning_rate,
        policy_kwargs=policy_kwargs,
        verbose=1,
        **kwargs,
    )
    return model
