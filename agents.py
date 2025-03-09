import numpy as np

class RandomAgent:
    def __init__(self, action_space):
        self.action_space = action_space

    def get_action(self, obs):
        action = self.action_space.sample()
        return action
    
class RuleBasedAgent:
    def __init__(self, action_space):
        self.action_space = action_space
        self.PRICE_HIGH = 20
        self.PRICE_LOW = 10
    def get_action(self, obs):
        battery_action = 0.0
        demand = obs['demand'][0]
        renewable = obs['renewable'][0]
        price = obs['price'][0]
        soc = obs['soc'][0]

        # If the price is high, we try to sell the renewable energy
        if price > self.PRICE_HIGH:
            battery_action = -min(renewable, soc)
        # If the price is low, we try to charge the battery
        elif price < self.PRICE_LOW:
            battery_action = min(renewable, 20 - soc)
        return np.array([battery_action] + [0.0 for _ in range(self.action_space.shape[0] - 1)], dtype=np.float32)
    