"""
Train the PPO allocation policy used by the 'rlbase' allocation mode.

Training metrics (rollout/ep_rew_mean, train/entropy_loss, ...) are written by
stable-baselines3's own logger to <out>_logs/progress.csv, so learning curves
can be plotted from logged data instead of copied by hand.

Usage (from the repository root):
    python train_ppo.py --timesteps 100000 --seed 0 --out qcloud_ppo_model-retrained

Then use it in a simulation with
    QCloudSimEnv(..., allocation_mode="rlbase", rl_model_path="qcloud_ppo_model-retrained")
"""
import argparse
import os

from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.logger import configure

from QCloud import IBM_Brussels, IBM_Kawasaki, IBM_Kyiv, IBM_Quebec, IBM_Strasbourg, QCloudGymEnv

CALI = os.path.join("QCloud", "calibration", "03_2025")


def make_devices():
    return [
        IBM_Strasbourg(env=None, name="ibm_strasbourg", cali_filepath=os.path.join(CALI, "ibm_strasbourg_calibrations_2025-03-29T05_16_26Z.csv"), printlog=False),
        IBM_Brussels(env=None, name="ibm_brussels", cali_filepath=os.path.join(CALI, "ibm_brussels_calibrations_2025-03-30T02_20_13Z.csv"), printlog=False),
        IBM_Kyiv(env=None, name="ibm_kyiv", cali_filepath=os.path.join(CALI, "ibm_kyiv_calibrations_2025-03-30T02_24_45Z.csv"), printlog=False),
        IBM_Quebec(env=None, name="ibm_quebec", cali_filepath=os.path.join(CALI, "ibm_quebec_calibrations_2025-03-30T02_51_33Z.csv"), printlog=False),
        IBM_Kawasaki(env=None, name="ibm_kawasaki", cali_filepath=os.path.join(CALI, "ibm_kawasaki_calibrations_2025-03-30T03_15_05Z.csv"), printlog=False),
    ]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--timesteps", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default="qcloud_ppo_model-retrained",
                        help="output model path (without .zip); kept separate from the shipped model by default")
    args = parser.parse_args()

    env = QCloudGymEnv(make_devices(), job_records_manager=None, printlog=False)
    check_env(env)

    model = PPO("MlpPolicy", env, seed=args.seed, verbose=1)
    log_dir = f"{args.out}_logs"
    model.set_logger(configure(log_dir, ["stdout", "csv"]))
    model.learn(total_timesteps=args.timesteps)
    model.save(args.out)
    print(f"Saved model to {args.out}.zip; training curve in {os.path.join(log_dir, 'progress.csv')}")


if __name__ == "__main__":
    main()
