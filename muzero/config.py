class MuZeroConfig(object):

	def __init__(self):
		
		### Need to tweak these parameters
		# self.action_space_size = 15 #number of actions we want to control (discrete steering and throttle)
		
		# nothing, right, left, gas, break for discrete carracing-v3
		self.action_space_size = 5


		# will yeild about 30sec per episode if you step through the env at 10Hz (Probably increase it after initial training)
		self.max_moves = 300 #max environment steps per episode

		self.num_simulations = 50 #number of simulations MCTS runs per decision step
		self.td_steps = 10 # num of steps in the future to boot strap (Would be inf for monte carlo)
		self.discount = 0.99 #typical discount rate for future rewards

		#encourages exploration at the root of the search tree
		self.dirichlet_alpha = 0.3
		self.root_exploration_fraction = 0.25 

		# UCB formula - Upper Confidence Bound
		# An algorithm used to solve and manage the fundamental 
		# exploration-exploitation trade-off
		self.pb_c_base = 19652
		self.pb_c_init = 1.25

		# If we already have some information about which values occur in the
		# environment, we can use them to initialize the rescaling.
		self.known_bounds = None #Can change this if 

		### Training
		self.training_steps = 50000
		self.checkpoint_interval = int(1e3)
		self.window_size = int(1e5) # replay buffer size (max number of experience you store)
		self.batch_size = 32 #batch size for trainig
		self.num_unroll_steps = 5 # future steps the dynamics model is trained to model at once

		self.weight_decay = 0.0  #prevents the weights from blowing up or overfitting (can set it to 1e-4 if needed)
		self.momentum = 0.9

		# Exponential learning rate scheduler
		self.lr_init = 1e-3
		self.lr_decay_rate = 0.1
		self.lr_decay_steps = 2000

	
	def visit_softmax_temperature(self, num_moves, training_steps):
		#Determines the exploration rate and exploitation based on what training step we are on
		return 1.0 if training_steps < 10000 else 0.25