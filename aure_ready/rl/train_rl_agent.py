# rl/train_rl_agent.py

import os
import argparse
from datetime import datetime, timedelta
from rl.trading_env import TradingEnv
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.vec_env import DummyVecEnv

def parse_args():
    parser = argparse.ArgumentParser(description="Training RL Agent per trading")
    parser.add_argument("--symbol", type=str, default="BTCUSDT", help="Simbolo su cui fare trading")
    parser.add_argument("--start", type=str, required=True, help="Data inizio (ISO, es. 2025-01-01T00:00:00Z)")
    parser.add_argument("--stop", type=str, required=True, help="Data fine per training")
    parser.add_argument("--total_timesteps", type=int, default=200_000, help="Numero di step totali per training")
    parser.add_argument("--window_size", type=int, default=50, help="Finestra di osservazione")
    parser.add_argument("--save_dir", type=str, default="rl_models", help="Cartella dove salvare i modelli")
    parser.add_argument("--eval_interval", type=int, default=50_000, help="Timesteps fra un eval e l'altro")
    parser.add_argument("--eval_episodes", type=int, default=5, help="Num episodi di valutazione")
    return parser.parse_args()

def main():
    args = parse_args()
    os.makedirs(args.save_dir, exist_ok=True)

    # Definisco ambiente
    env = DummyVecEnv([lambda: TradingEnv(
        symbol=args.symbol,
        start=args.start,
        stop=args.stop,
        window_size=args.window_size
    )])

    # Callback per checkpoint
    checkpoint_callback = CheckpointCallback(save_freq=50_000, save_path=args.save_dir, name_prefix="rl_checkpoint")

    # Callback per valutazione
    eval_callback = EvalCallback(env, best_model_save_path=args.save_dir,
                                 log_path=args.save_dir, eval_freq=args.eval_interval,
                                 n_eval_episodes=args.eval_episodes, deterministic=True)

    # Inizializzo modello PPO
    model = PPO(
        policy="MlpPolicy",
        env=env,
        verbose=1,
        tensorboard_log=os.path.join(args.save_dir, "tb_logs")
    )

    # Avvio training
    model.learn(total_timesteps=args.total_timesteps, callback=[checkpoint_callback, eval_callback])

    # Salvo modello finale
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    model_path = os.path.join(args.save_dir, f"ppo_{args.symbol}_{timestamp}.zip")
    model.save(model_path)
    print(f"Modello RL salvato in {model_path}")

if __name__ == "__main__":
    # Esempio di utilizzo:
    # python -m rl.train_rl_agent --symbol BTCUSDT --start 2025-01-01T00:00:00Z --stop 2025-04-01T00:00:00Z --total_timesteps 200000
    main()
