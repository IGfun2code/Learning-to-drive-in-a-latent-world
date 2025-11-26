from muzero.config import MuZeroConfig
from muzero.network import MuZeroNetwork
from muzero.mcts import Node, run_mcts, expand_node, add_exploration_noise, select_action, select_action_eval
from muzero.game import Game, DrivingEnvironment
import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

from muzero.replay_buffer import ReplayBuffer
from muzero.batch import make_training_batch

def play_game(config, network, env, record_video=False, video_path=None):
    """
    produces a single trajectory, storing game state information in returned Game Object
    """
    game = Game(config, env, record_video, video_path)

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

def evaluate_policy(config, network, env, num_episodes=3, record_video=False, video_path=None):
    """
    similar to play_game, but selects actions greedily and returns eval ave return
    """
    returns = []

    original_noise = config.root_exploration_fraction
    config.root_exploration_fraction = 0.0

    for _ in range(num_episodes):
        game = Game(config, env, record_video, video_path)
        while not game.terminal():
            root = Node(0.0)
            obs = game.make_image(-1)
            out = network.initial_inference(obs)
            expand_node(root, game.to_play(), game.legal_actions(), out)

            # NO exploration noise
            # run fewer sims or same number
            run_mcts(config, root, game.action_history(), network)

            action = select_action_eval(root)   # greedy selection
            game.apply(action)

        episode_return = sum(game.rewards)
        returns.append(episode_return)

    # reset to original value
    config.root_exploration_fraction = original_noise
    return np.mean(returns)



def muzero_update(config, network, replay_buffer, optimizer, device):
    """
    One MuZero training step:
      - sample a batch from replay buffer
      - unroll the network K steps
      - compute value, reward, and policy losses
      - backprop and update network parameters

    Returns a dict of scalar losses for logging.
    """

    if len(replay_buffer) == 0:
        return None  # nothing to train on yet

    # 1. Sample batch from replay buffer
    batch = replay_buffer.sample_batch(config.batch_size)

    # 2. Build training tensors
    obs_batch, action_batch, value_batch, reward_batch, policy_batch = make_training_batch(config, batch)

    # 3. Move to device
    obs_batch = obs_batch.to(device)         # (B, C, H, W)
    action_batch = action_batch.to(device)   # (B, K)
    value_batch = value_batch.to(device)     # (B, K+1)
    reward_batch = reward_batch.to(device)   # (B, K+1)
    policy_batch = policy_batch.to(device)   # (B, K+1, A)

    B, K = action_batch.shape
    A = policy_batch.size(-1)

    # 4. Unroll the network K steps

    # Initial latent state from representation network
    # s: (B, latent_dim)
    s = network.representation_network(obs_batch)

    values_pred = []
    policies_logits = []
    rewards_pred = []

    # We will:
    #  - predict value + policy from s_k at each step k = 0..K
    #  - predict reward from dynamics at each step k = 0..K-1
    for k in range(K + 1):
        # Prediction from current state s_k
        v_pred, p_logits = network.prediction_network(s)   # (B,), (B,A)
        values_pred.append(v_pred.unsqueeze(1))            # -> (B,1)
        policies_logits.append(p_logits.unsqueeze(1))      # -> (B,1,A)

        # For k < K, use dynamics to get s_{k+1} and immediate reward r_k
        if k < K:
            a_k = action_batch[:, k]                       # (B,)
            s, r_pred = network.dynamics_network(s, a_k)   # (B,latent), (B,)
            rewards_pred.append(r_pred.unsqueeze(1))       # -> (B,1)

    # Stack along unroll dimension
    values_pred = torch.cat(values_pred, dim=1)            # (B, K+1)
    policies_logits = torch.cat(policies_logits, dim=1)    # (B, K+1, A)
    rewards_pred = torch.cat(rewards_pred, dim=1)          # (B, K)

    # 5. Compute losses

    # Value loss: MSE over (B, K+1)
    value_loss = F.mse_loss(values_pred, value_batch)

    # Reward loss:
    #   We predicted K rewards: r_t ... r_{t+K-1}
    #   But reward_batch has K+1 "last_reward" targets.
    #   We align predictions with reward targets from index 1..K
    reward_loss = F.mse_loss(rewards_pred, reward_batch[:, 1:])  # (B,K)

    # Policy loss: cross-entropy between visit-count distribution and predicted policy
    log_probs = F.log_softmax(policies_logits, dim=-1)           # (B,K+1,A)
    # Per-step CE: -sum_a pi_target * log pi_pred
    policy_loss_per_step = -(policy_batch * log_probs).sum(dim=-1)  # (B, K+1)

    # Mask out steps where there is no policy target (sum==0) — e.g., beyond end of game
    mask = policy_batch.sum(dim=-1) > 0                           # (B,K+1) bool
    if mask.any():
        policy_loss = policy_loss_per_step[mask].mean()
    else:
        policy_loss = policy_loss_per_step.mean() * 0.0  # no-op if no valid policies

    # Combine losses (you can later add weights if desired)
    total_loss = value_loss + reward_loss + policy_loss

    # 6. Backprop + optimization
    optimizer.zero_grad()
    total_loss.backward()
    torch.nn.utils.clip_grad_norm_(network.parameters(), max_norm=5.0)
    optimizer.step()
    network.increment_training_steps()

    return {
        "total_loss": float(total_loss.item()),
        "value_loss": float(value_loss.item()),
        "reward_loss": float(reward_loss.item()),
        "policy_loss": float(policy_loss.item()),
    }
    
def plot_results(results):
    """plot saved results"""
    #todo move to separate logging class
    for name, data in results.items():
        arr = np.array(data)
        steps = arr[:, 0]
        eval_returns = arr[:, 1]

        plt.figure(figsize=(8, 5))
        plt.plot(steps, eval_returns)
        plt.xlabel("Training Step")
        plt.ylabel(name)
        plt.title(f"MuZero {name}")
        plt.grid(True)
        plt.savefig(f"{name}.jpg", dpi=500)
        plt.close()


def main():
    config = MuZeroConfig()
    #debug
    # config.num_simulations = 5
    # config.training_steps = 200
    # config.max_moves = 50
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Environment and network
    env = DrivingEnvironment()
    network = MuZeroNetwork(config, device=device)

    # Optimizer + LR scheduler
    optimizer = optim.Adam(
        network.parameters(),
        lr=config.lr_init,
        weight_decay=config.weight_decay,
    )

    # Exponential LR decay: lr = lr_init * (decay_rate)^(steps / decay_steps)
    gamma = config.lr_decay_rate ** (1.0 / max(config.lr_decay_steps, 1))
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)

    # Replay buffer
    replay_buffer = ReplayBuffer(config)
    
    # logging TODO make separate class
    logging = {
        "loss": [],
        "eval ave return": []
    }

    # Training loop:
    #  We use network.training_steps() as our global counter.
    while network.training_steps() < config.training_steps:
        step = network.training_steps()

        # 1. Self-play: generate one new episode
        game = play_game(config, network, env)
        replay_buffer.add_episode(game)

        # 2. Training update (one gradient step)
        stats = muzero_update(config, network, replay_buffer, optimizer, device)

        # 3. LR schedule step
        scheduler.step()

        
        
        # 4. Logging
        if stats is not None and step % 1 == 0:
            print(
                f"[Train step {step}] "
                f"total={stats['total_loss']:.4f} "
                f"value={stats['value_loss']:.4f} "
                f"reward={stats['reward_loss']:.4f} "
                f"policy={stats['policy_loss']:.4f}"
            )
            logging["loss"].append((step, stats["total_loss"]))
        # 5. run periodic evals
        if step % config.eval_freq == 0:
            if step % config.video_freq == 0:
                eval_return = evaluate_policy(config, network, env, record_video=True, video_path=f"videos/step_{int(step)}.mp4")
            else:
                eval_return = evaluate_policy(config, network, env)
                 
           
            logging["eval ave return"].append((step, eval_return))
            print(f"eval ave return={eval_return:.4f}")

        # 6. Checkpointing
        if step % config.checkpoint_interval == 0:
            ckpt_path = f"muzero_checkpoint_step_{step}.pt"
            torch.save(
                {
                    "step": step,
                    "network_state_dict": network.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                },
                ckpt_path,
            )
            print(f"Saved checkpoint to {ckpt_path}")

    print("Training complete.")
    env.close()
    plot_results(logging)


if __name__ == "__main__":
    main()