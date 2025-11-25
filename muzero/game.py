from typing import List
from muzero.config import MuZeroConfig
from muzero.mcts import Node
import gymnasium as gym
import numpy as np
import cv2

class Player(object):
    #dummy class since it is used in the MuZero MCTS algorithm
    #but we only have on player --> the driver
    pass

class Action(object):

    def __init__(self, index: int):
        self.index = index

    def __hash__(self):
        return self.index

    def __eq__(self, other):
        return self.index == other.index

    def __gt__(self, other):
        return self.index > other.index
    
class ActionHistory(object):
    """Simple history container used inside the search.

    Only used to keep track of the actions executed.
    """

    def __init__(self, history: List[Action], action_space_size: int):
        self.history = list(history)
        self.action_space_size = action_space_size

    def clone(self):
        return ActionHistory(self.history, self.action_space_size)

    def add_action(self, action: Action):
        self.history.append(action)

    def last_action(self) -> Action:
        return self.history[-1]

    def action_space(self) -> List[Action]:
        return [Action(i) for i in range(self.action_space_size)]

    def to_play(self) -> Player:
        return Player()
    
class DrivingEnvironment:
    """
    Wrapper around CarRacing-v3 (Discrete Mode).
    Handles:
    - reset()
    - step(action_index)
    - observation preprocessing
    """

    def __init__(self, obs_shape=(64, 64)):
        # Use discrete action mode
        self.env = gym.make("CarRacing-v3", render_mode="rgb_array", lap_complete_percent=0.95, domain_randomize=False, continuous=False)
        self.obs_shape = obs_shape
        
        # video params
        # self.record_video = record_video
        # self.video_path = video_path
        # self.video_writer = None
        self.frame_size = (600, 400)  # CarRacing's native resolution

    def reset(self):
        obs, info = self.env.reset()

        # init video writer if recording
        # if self.record_video:
        #     if self.video_writer is not None:  # close previous one
        #         self.video_writer.release()

            # self.video_writer = cv2.VideoWriter(
            #     self.video_path,
            #     cv2.VideoWriter_fourcc(*"mp4v"),
            #     30,                      # FPS
            #     self.frame_size        
            # )
        return self._preprocess(obs), info

    def step(self, action_index):
        obs, reward, terminated, truncated, info = self.env.step(action_index)
        obs = self._preprocess(obs)

        # capture frame if recording
        # if self.record_video:
        #     frame = self.env.render() # render returns 400, 600, 3
        #     frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)  # convert to bgr for opencv
        #     self.video_writer.write(frame_bgr)

        done = terminated or truncated
        return obs, reward, done, info

    def close(self):
        """Close env"""
        # if self.record_video and self.video_writer is not None:
        #     self.video_writer.release()
        #     self.video_writer = None
        #     print(f'video saved to {self.video_path}')
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
        # obs shape: (96, 96, 3)
        obs = cv2.resize(obs, self.obs_shape, interpolation=cv2.INTER_AREA)
        obs = obs.astype(np.float32) / 255.0
        return obs


class Game(object):
    """A single episode of interaction with the environment."""

    def __init__(self, config: MuZeroConfig, env, record_video=False, video_path=None):
        self.config = config
        self.environment = env  # Occuworld + Carla Environment
        self.history = [] #list of actions so far
        self.rewards = [] #reward per step
        self.child_visits = [] #visit distributions from MCTS for each step (used as policy targets).
        self.root_values = [] #root value estimates from MCTS (used as value targets)

        #store the observations at each step
        self.observations = [] #raw observations (occupancy grids) from the env
        ob_init, _ = self.environment.reset()
        self.observations.append(ob_init) #store the first observation in the env.
        #boolean variable to determine if the episode is done
        self.done = False

        self.action_space_size = self.config.action_space_size
        self.discount = self.config.discount
        self.max_moves = self.config.max_moves
        
        # recording params
        self.record_video = record_video
        if self.record_video:
            #TODO check if video folder exists
            self.frame_size = self.environment.get_frame_size()
            self.video_path = video_path
            self.video_writer = cv2.VideoWriter(
                self.video_path,
                cv2.VideoWriter_fourcc(*"mp4v"),
                30,                      # FPS
                self.frame_size        
            )
            
        

    def terminal(self) -> bool:
        # #check if the boolean variable is True
        # if self.done == True:
        #     return True
        # #secondary check to make sure the steps in the episode do not exceed your max moves
        # if len(self.history) >= self.max_moves:
        #     return True
        
        # need to check if we reached terminal in env or if we've exceeded max moves
        is_terminal = self.done or len(self.history) >= self.max_moves
        if is_terminal and self.record_video:
            self.video_writer.release()
            print(f'video saved to {self.video_path}')
        return is_terminal

    def legal_actions(self):
        # return a list of legal actions
        #you can filter unwanted actions here

        #This currently assumes that all actions are good since it returns all of them
        return [Action(i) for i in range(self.action_space_size)]

    def apply(self, action: Action):
        #Apply action in env and get the observation, reward, and done from the environment step
        obs, reward, done, _ = self.environment.step(action.index)
        #store the info
        self.rewards.append(float(reward))
        self.history.append(action)
        self.observations.append(obs)
        #since done is sometimes returned as None
        if done:
                self.done = done

    def store_search_statistics(self, root: Node):
        # After MCTS at a given step:
        # Take the visit counts of each child.
        # Convert them into a probability distribution over actions.
        # Store this in the child_visits list --> Target policy at that time
        # Store root.value in root_values --> Target Value at that time
        sum_visits = max(sum(child.visit_count for child in root.children.values()), 1e-8) #max function avoids division by zero
        action_space = (Action(index) for index in range(self.action_space_size))
        self.child_visits.append([
                root.children[a].visit_count / sum_visits if a in root.children else 0
                for a in action_space
        ])
        self.root_values.append(root.value())

    def make_image(self, state_index: int):
        # Just a way to get the observation in the episode
        if state_index == -1:
            obs = self.observations[-1]
        else: 
            obs = self.observations[state_index]
        if self.record_video:
            frame = self.environment.render()
            self.video_writer.write(frame)
        return obs
            

    def make_target(self, state_index: int, num_unroll_steps: int, td_steps: int):
        # The value target is the discounted root value of the search tree N steps
        # into the future, plus the discounted sum of all rewards until then.
        targets = []
        for current_index in range(state_index, state_index + num_unroll_steps + 1):
            bootstrap_index = current_index + td_steps
            if bootstrap_index < len(self.root_values):
                value = self.root_values[bootstrap_index] * self.discount**td_steps
            else:
                value = 0   

            for i, reward in enumerate(self.rewards[current_index:bootstrap_index]):
                value += reward * self.discount**i  # pytype: disable=unsupported-operands

            # For simplicity the network always predicts the most recently received
            # reward, even for the initial representation network where we already
            # know this reward.
            if current_index > 0 and current_index <= len(self.rewards):
                last_reward = self.rewards[current_index - 1]
            else:
                last_reward = 0

            if current_index < len(self.root_values):
                targets.append((value, last_reward, self.child_visits[current_index]))
            else:
                # States past the end of games are treated as absorbing states.
                targets.append((0, last_reward, []))
        return targets

    def to_play(self) -> Player:
        #Would indicate who's turn it is in a multi player game
        return Player()

    def action_history(self) -> ActionHistory:
        return ActionHistory(self.history, self.action_space_size)