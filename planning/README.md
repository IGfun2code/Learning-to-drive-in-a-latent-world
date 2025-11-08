# PPO Image-Based Baseline

## Overview
This project trains a PPO reinforcement learning agent on a visual environment using Stable-Baselines3.  
The goal is to establish a working image-based baseline with a pretrained vision encoder before integrating more complex environments like CARLA or adding model-based planning (MuZero/TD-MPC2).

Currently, the environment uses **CarRacing-v3** from Gymnasium, which outputs 96x96 RGB frames similar to CARLA’s camera views.  
All components are modular so that CARLA or other visual environments can be swapped in later with minimal changes.

---

## File Structure
.
├── main.py          # Entry point: sets up environment, loads PPO model, runs training/evaluation loop  
├── baseline.py      # Defines PPOBaseline using Stable-Baselines3 with a custom visual encoder  
├── encoder.py       # Pretrained ResNet encoder for image feature extraction  
├── utils.py         # Environment setup, evaluation, and utility functions  
├── config.yaml      # (optional) Hyperparameters, environment config, etc.  
├── data/            # (optional) Checkpoints, logs, evaluation results  
└── README.md        # Project documentation  

---

## Environment
- **Default:** `CarRacing-v3` (96x96 RGB continuous control)
- **Other Options:** Any Gymnasium environment that outputs image observations  
- The `ChannelFirstWrapper` converts images from (H, W, C) → (C, H, W) for PyTorch compatibility.  
- You can later replace this environment with CARLA by updating only `utils.py`.

---

## Training Flow
1. Create Gym environment (`CarRacing-v3`)
2. Wrap to output (C, H, W) images
3. Encode observations via pretrained ResNet (frozen or fine-tuned)
4. PPO agent outputs continuous control actions
5. Execute action, receive reward, next frame
6. PPO updates policy and value networks automatically
7. Repeat until convergence; evaluate average return and save model

---

## Running Training
```bash
python main.py --env-id CarRacing-v3 --timesteps 100000
