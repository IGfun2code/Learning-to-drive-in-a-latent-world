'''
gym env classes
'''
import gymnasium as gym
import numpy as np
import cv2
class Environement:
    """
    base environemnt class
    meant to be used as subclass for specific envs
    """
    def __init__(self):
        # fill this in when you create specific env
        # ex gym.make("env name") etc
        pass
    def reset(self):
        obs, info = self.env.reset()
        return self._preprocess(obs), info
    
    def step(self, action_index):
        obs, reward, terminated, truncated, info = self.env.step(action_index)
        obs = self._preprocess(obs)
        done = terminated or truncated
        return obs, reward, done, info

    def close(self):
        """Close env"""
        self.env.close()
        
    # for video saving
    def render(self):
        """returns rendered current frame of env"""
        frame = self.env.render() # render returns 400, 600, 3
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)  # convert to bgr for opencv
        return frame_bgr
    # for video saving
    def get_frame_size(self):
        # swap to return expected for opencv h,w,c
        size = self.env.render().shape
        return (size[1], size[0])
        
    def _preprocess(self, obs):
        #do no preprocessing by default
        return obs
    
class DrivingEnvironment(Environement):
    """
    Wrapper around CarRacing-v3 (Discrete Mode).
    Handles:
    - reset()
    - step(action_index)
    - observation preprocessing
    """

    def __init__(self, obs_shape=(64, 64)):
        # Use discrete action mode
        super().__init__()
        self.env = gym.make("CarRacing-v3", render_mode="rgb_array", lap_complete_percent=0.95, domain_randomize=False, continuous=False)
        self.obs_shape = obs_shape

    # def reset(self):
    #     obs, info = self.env.reset()
    #     return self._preprocess(obs), info

    # def step(self, action_index):
    #     obs, reward, terminated, truncated, info = self.env.step(action_index)
    #     obs = self._preprocess(obs)


    #     done = terminated or truncated
    #     return obs, reward, done, info

    # def close(self):
    #     """Close env"""
    #     self.env.close()

    def _preprocess(self, obs):
        # obs shape: (96, 96, 3)
        obs = cv2.resize(obs, self.obs_shape, interpolation=cv2.INTER_AREA)
        obs = obs.astype(np.float32) / 255.0
        return obs
class CartPoleEnvironment(Environement):
    """
    Wrapper around CarRacing-v0 (Discrete Mode).
    Handles:
    - reset()
    - step(action_index)
    - observation preprocessing
    TODO
    """

    def __init__(self, obs_shape=(64, 64)):
        # Use discrete action mode
        self.env = gym.make("CartPole-v1", render_mode="rgb_array")
        self.obs_shape = obs_shape
