from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback
from ElectricityMarketEnv import ElectricityMarketEnv
import os
from tqdm import tqdm

TRAIN = True

class TQDMCallback(BaseCallback):
    def __init__(self, total_timesteps, verbose=0):
        super().__init__(verbose)
        self.total_timesteps = total_timesteps
        self.pbar = None

    def _on_training_start(self):
        self.pbar = tqdm(total=self.total_timesteps, desc="Training Progress", unit="step")

    def _on_step(self):
        self.pbar.update(self.training_env.num_envs)  # Update by number of environments
        return True

    def _on_training_end(self):
        self.pbar.close()

def train_model(env, time_steps,path):
    if os.path.exists(path) and not TRAIN:
        model = SAC.load(path)
        print(f'Loaded model from {path}')
        return model
    model = SAC('MultiInputPolicy', env)
    model.learn(total_timesteps=time_steps, callback=TQDMCallback(time_steps))
    model.save(path)
    print(f'Model saved at {path}')

if __name__ == '__main__':
    env = DummyVecEnv([lambda: ElectricityMarketEnv()])
    for TRAIN_STEPS in [1000, 5000, 8760, 10000, 20000,50000]:
        print(f'Training for {TRAIN_STEPS} steps')
        PATH = f'./models/sac_electricity_market_{TRAIN_STEPS}'
        train_model(env,time_steps=TRAIN_STEPS,path=PATH)
