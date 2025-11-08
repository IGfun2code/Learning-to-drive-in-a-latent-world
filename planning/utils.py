# utils.py
from typing import Tuple
import gymnasium as gym
import numpy as np
from stable_baselines3.common.evaluation import evaluate_policy


class ChannelFirstWrapper(gym.ObservationWrapper):
    def observation(self, obs):
        # (H,W,3) -> (3,H,W)
        return np.transpose(obs, (2,0,1))

def make_carla_env(env_id="CarRacing-v3"):
    env = gym.make(env_id, render_mode="rgb_array")
    env = ChannelFirstWrapper(env)
    return env

def evaluate_agent(model, env: gym.Env, n_episodes: int = 5) -> Tuple[float, float]:
    """
    Runs SB3's evaluate_policy helper.
    Returns mean_reward, std_reward.
    """
    mean_reward, std_reward = evaluate_policy(
        model,
        env,
        n_eval_episodes=n_episodes,
        render=False,
    )
    return mean_reward, std_reward


def preprocess_obs(obs):
    """
    In case CARLA gives you HWC uint8, and your model wants CHW float32.
    SB3 can handle some of this automatically if the env's observation_space
    is defined correctly, but you can keep this around as a utility.
    """
    # Placeholder – likely not needed if env is set up correctly
    return obs
