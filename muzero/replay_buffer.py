import random
from collections import deque

class ReplayBuffer:
    """
    Stores entire episodes of self-play.
    Supports:
      - adding full episodes
      - sampling (episode, time index) pairs
      - FIFO behavior based on window_size
    """

    def __init__(self, config):
        self.config = config
        self.window_size = config.window_size   # max number of transitions
        self.buffer = deque()                   # stores Game objects
        self.num_transitions = 0                # running count of total transitions

    def add_episode(self, game):
        """Add a completed Game to the buffer."""
        self.buffer.append(game)
        self.num_transitions += len(game.history)

        # If over capacity, pop old episodes
        while self.num_transitions > self.window_size:
            old_game = self.buffer.popleft()
            self.num_transitions -= len(old_game.history)

    def sample_batch(self, batch_size):
        """
        Samples a batch of (game, position) pairs.
        You will later convert these into:
            obs_t, actions_t, target_rewards, target_values, target_policies
        """
        batch = []
        for _ in range(batch_size):
            game = random.choice(self.buffer)
            position = random.randrange(0, len(game.history))  # random time idx
            batch.append((game, position))
        return batch

    def __len__(self):
        return len(self.buffer)
