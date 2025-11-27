# batch.py
import torch
import numpy as np

def make_training_batch(config, batch):
    """
    Converts sampled (game, position) pairs into tensors for training.

    Args:
        config: MuZeroConfig
        batch:  list of (game, pos) pairs from replay buffer

    Returns:
        obs_batch:      (B, C, H, W)
        action_batch:   (B, K)         long tensor
        value_batch:    (B, K+1)       float tensor
        reward_batch:   (B, K+1)       float tensor
        policy_batch:   (B, K+1, A)    float tensor
    """

    B = len(batch)
    K = config.num_unroll_steps
    A = config.action_space_size

    # Allocate arrays
    obs_batch     = []
    action_batch  = torch.zeros((B, K), dtype=torch.long)
    value_batch   = torch.zeros((B, K+1), dtype=torch.float32)
    reward_batch  = torch.zeros((B, K+1), dtype=torch.float32)
    policy_batch  = torch.zeros((B, K+1, A), dtype=torch.float32)

    for i, (game, pos) in enumerate(batch):
        # print(f'debug: r: {len(game.rewards)} | o: {len(game.observations)} | a: {len(game.history)} | pi: {len(game.child_visits)} | v: {len(game.root_values)}')

        # ----------------------------
        # 1. Extract initial observation
        # ----------------------------
        obs = game.make_image(pos)    # numpy H,W,C or C,H,W depending on env
        if obs.ndim == 3:
            # convert to CHW
            obs = np.transpose(obs, (2, 0, 1))
        obs_batch.append(obs)

        # ----------------------------
        # 2. Extract targets using Game.make_target
        # ----------------------------
        targets = game.make_target(
            pos,
            config.num_unroll_steps,
            config.td_steps
        )

        # targets[k] = (value, last_reward, policy)
        for k in range(K+1):
            value, reward, policy = targets[k]

            value_batch[i, k]  = float(value)
            reward_batch[i, k] = float(reward)

            if len(policy) > 0:
                policy_batch[i, k, :] = torch.tensor(policy, dtype=torch.float32)

        # ----------------------------
        # 3. Extract actions for unroll
        # ----------------------------
        for k in range(K):
            action_index = pos + k
            if action_index < len(game.history):
                action_batch[i, k] = game.history[action_index].index
            else:
                action_batch[i, k] = 0   # pad with 0 if beyond episode end

    # Convert obs_batch → torch tensor
    obs_batch = torch.tensor(np.array(obs_batch), dtype=torch.float32)

    return obs_batch, action_batch, value_batch, reward_batch, policy_batch
