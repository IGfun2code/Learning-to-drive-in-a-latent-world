# main.py
import argparse
from pathlib import Path

from baseline import make_ppo_baseline
from utils import make_carla_env, evaluate_agent


def parse_args():
    parser = argparse.ArgumentParser(description="PPO Baseline")
    parser.add_argument("--env-id", type=str, default="CarRacing-v3")
    parser.add_argument("--timesteps", type=int, default=200_000)
    parser.add_argument("--eval-episodes", type=int, default=5)
    parser.add_argument("--save-dir", type=str, default="./checkpoints")
    return parser.parse_args()

#TODO: figure out gym carla wrapper

def main():
    args = parse_args()

    # 1. create env
    env = make_carla_env(args.env_id)

    # 2. create model
    model = make_ppo_baseline(env)

    # 3. train
    model.learn(total_timesteps=args.timesteps)

    # 4. evaluate
    mean_reward, std_reward = evaluate_agent(model, env, n_episodes=args.eval_episodes)
    print(f"[eval] mean_reward={mean_reward:.2f} ± {std_reward:.2f}")

    # 5. save
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    model_path = save_dir / "ppo_carla_baseline.zip"
    model.save(model_path)
    print(f"[info] saved model to {model_path}")


if __name__ == "__main__":
    main()
