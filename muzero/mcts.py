from game import ActionHistory
from config import MuZeroConfig
from typing import List

MAXIMUM_FLOAT_VALUE = float('inf')

KnownBounds = collections.namedtuple('KnownBounds', ['min', 'max'])

class MinMaxStats(object):
	"""A class that holds the min-max values of the tree."""

	def __init__(self, known_bounds: Optional[KnownBounds]):
		self.maximum = known_bounds.max if known_bounds else -MAXIMUM_FLOAT_VALUE
		self.minimum = known_bounds.min if known_bounds else MAXIMUM_FLOAT_VALUE

	def update(self, value: float):
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
			self.to_play = -1
			self.prior = prior
			self.value_sum = 0
			self.children = {}
			self.hidden_state = None
			self.reward = 0

	def expanded(self) -> bool:
			return len(self.children) > 0

	def value(self) -> float:
			if self.visit_count == 0:
					return 0
			return self.value_sum / self.visit_count

# Core Monte Carlo Tree Search algorithm.
# To decide on an action, we run N simulations, always starting at the root of
# the search tree and traversing the tree according to the UCB formula until we
# reach a leaf node.
def run_mcts(config: MuZeroConfig, root: Node, action_history: ActionHistory,
						 network: Network):
	min_max_stats = MinMaxStats(config.known_bounds)

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

		backpropagate(search_path, network_output.value, history.to_play(),
									config.discount, min_max_stats)


class Environment(object):
	"""The environment MuZero is interacting with."""

	def step(self, action):
		pass