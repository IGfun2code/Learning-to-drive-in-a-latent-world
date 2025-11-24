from game import ActionHistory, Player, Action
from config import MuZeroConfig
from network import Network
from typing import List
import math
import numpy as np

MAXIMUM_FLOAT_VALUE = float('inf')

class MinMaxStats(object):
  """A class that holds the min-max values of the tree."""

  def __init__(self, config: MuZeroConfig):
    #save the max and min values (currently at -inf, inf)
    self.config = config
    if self.config.known_bounds != None:
      self.maximum = self.config.known_bounds.max
      self.minimum = self.config.known_bounds.min
    else:
      self.maximum = -MAXIMUM_FLOAT_VALUE
      self.minimum = MAXIMUM_FLOAT_VALUE

  def update(self, value: float):
    #update the min and max values
    self.maximum = max(self.maximum, value)
    self.minimum = min(self.minimum, value)

  def normalize(self, value: float) -> float:
    if self.maximum > self.minimum:
      # We normalize only when we have set the maximum and minimum values.
      return (value - self.minimum) / (self.maximum - self.minimum)
    return value
  
class Node(object):

    def __init__(self, prior: float):
        self.visit_count = 0
        #place holder. Will be replaced later on with expand_node
        self.to_play = -1
        self.prior = prior
        self.value_sum = 0
        self.children = {}
        self.hidden_state = None
        self.reward = 0 #the reward predicted by the dynamics model for the action that led to this node

    def expanded(self) -> bool:
        return len(self.children) > 0

    def value(self) -> float:
        if self.visit_count == 0:
            return 0
        return self.value_sum / self.visit_count

# The score for a node is based on its value, plus an exploration bonus based on
# the prior.
def ucb_score(config: MuZeroConfig, parent: Node, child: Node, min_max_stats: MinMaxStats) -> float:
  pb_c = math.log((parent.visit_count + config.pb_c_base + 1) /
                  config.pb_c_base) + config.pb_c_init
  pb_c *= math.sqrt(parent.visit_count) / (child.visit_count + 1)

  prior_score = pb_c * child.prior
  if child.visit_count > 0:
    value_score = child.reward + config.discount * min_max_stats.normalize(
        child.value())
  else:
    value_score = 0
  return prior_score + value_score

# Select the child with the highest UCB score.
def select_child(config: MuZeroConfig, node: Node, min_max_stats: MinMaxStats):
  _, action, child = max(
      (ucb_score(config, node, child, min_max_stats), action,
       child) for action, child in node.children.items())
  return action, child

# We expand a node using the value, reward and policy prediction obtained from
# the neural network.
def expand_node(node: Node, to_play: Player, actions: List[Action], network_output):
  node.to_play = to_play
  node.hidden_state = network_output.hidden_state
  node.reward = float(network_output.reward)

  logits = network_output.policy_logits
  policy = {}
  for action in actions:
    #get the specific 
    logit = float(logits[action.index])
    policy[action] = math.exp(logit)
    
  policy_sum = sum(policy.values())
  for action, p in policy.items():
    node.children[action] = Node(p / (policy_sum + 1e-8))

# At the end of a simulation, we propagate the evaluation all the way up the
# tree to the root.
def backpropagate(search_path: List[Node], value: float, to_play: Player,
                  discount: float, min_max_stats: MinMaxStats):
  for node in reversed(search_path):
    node.value_sum += value if node.to_play == to_play else -value
    node.visit_count += 1
    min_max_stats.update(node.value())

    value = node.reward + discount * value

# At the start of each search, we add dirichlet noise to the prior of the root
# to encourage the search to explore new actions.
def add_exploration_noise(config: MuZeroConfig, node: Node):
  # Add Dirichlet noise to the priors at the root node to encourage exploration.
  actions = list(node.children.keys())
  noise = np.random.dirichlet([config.dirichlet_alpha] * len(actions))
  frac = config.root_exploration_fraction
  for a, n in zip(actions, noise):
    node.children[a].prior = node.children[a].prior * (1 - frac) + n * frac

# Stubs to make the typechecker happy.
def softmax_sample(distribution, temperature: float):
  # Inputs:
  #   distribution: a list like [(visit_count_0, action_0), (visit_count_1, action_1), ...]
  #   temperature:  smaller temp ==> Greedy, larger temp ==> more random

  #make sure the distribution is not zero
  if len(distribution) != 0:
    raise ValueError("softmax_sample called with empty distribution")
  
  #Must determine the sampling based on the temperature
  if temperature <= 1e-8:
    #Temperature of near zero means to be greedy: Pick the action with the most counts
    index, (count, action) = max(enumerate(distribution), key=lambda x: x[1][0])
    return index, action
  else:
    #Temperature is not zero, so pick action based on the temperature (exploitation/exploration)
    counts = np.array([count for count, _ in distribution], dtype=np.float32)
    #calculate ratio of count versus temperature (Higer count means higher probability)
    logits = counts / max(temperature, 1e-8)
    logits = logits - np.max(logits) #makes all the logits be <= 0
    probs = np.exp(logits) #all values are <= 1
    probs = probs / (probs.sum() + 1e-8)

    #sample an index using the probability
    idx = np.random.choice(len(distribution), p=probs)
    return idx, distribution[idx][1] #return the idx and action associated with that idx


def select_action(config: MuZeroConfig, num_moves: int, node: Node,
                  network: Network):
  visit_counts = [
      (child.visit_count, action) for action, child in node.children.items()
  ]
  t = config.visit_softmax_temperature(
      num_moves=num_moves, training_steps=network.training_steps())
  _, action = softmax_sample(visit_counts, t)
  return action


# Core Monte Carlo Tree Search algorithm.
# To decide on an action, we run N simulations, always starting at the root of
# the search tree and traversing the tree according to the UCB formula until we
# reach a leaf node.
def run_mcts(config: MuZeroConfig, root: Node, action_history: ActionHistory,
             network: Network):
  min_max_stats = MinMaxStats(config)

  for _ in range(config.num_simulations):
    history = action_history.clone()
    node = root
    search_path = [node]

    while node.expanded():
      action, node = select_child(config, node, min_max_stats)
      history.add_action(action)
      search_path.append(node)

    # Inside the search tree we use the dynamics function to obtain the next
    # hidden state given an action and the previous hidden state.
    parent = search_path[-2]
    network_output = network.recurrent_inference(parent.hidden_state,
                                                 history.last_action())
    expand_node(node, history.to_play(), history.action_space(), network_output)

    value = float(network_output.value)
    backpropagate(search_path, value, history.to_play(),
                  config.discount, min_max_stats)