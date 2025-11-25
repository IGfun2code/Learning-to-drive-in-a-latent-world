from muzero.config import MuZeroConfig
from muzero.network import MuZeroNetwork
from muzero.mcts import Node, run_mcts, expand_node, add_exploration_noise, select_action
from muzero.game import Game, DrivingEnvironment

def play_game(config, network, env):
    game = Game(config, env)

    while not game.terminal():
        root = Node(0)
        current_obs = game.make_image(-1)          # occupancy grid
        network_output = network.initial_inference(current_obs)
        expand_node(root, game.to_play(), game.legal_actions(), network_output)
        add_exploration_noise(config, root)

        # MCTS
        run_mcts(config, root, game.action_history(), network)

        # Choose action
        action = select_action(config, len(game.history), root, network)

        game.apply(action)
        game.store_search_statistics(root)

    return game
