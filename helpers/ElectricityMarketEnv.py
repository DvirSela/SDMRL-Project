import gymnasium as gym
from gymnasium import spaces
import numpy as np
from dotenv import load_dotenv
import os
load_dotenv()
NUM_EPISODES = int(os.getenv('NUM_EPISODES', 10))
NUM_STEPS = int(os.getenv('NUM_STEPS', 8760))
INITIAL_SOC = float(os.getenv('INITIAL_SOC'))
RENEWABLE_SCALE = float(os.getenv('RENEWABLE_SCALE', 25.0))
DEMAND_NOT_MET_FACTOR = float(os.getenv('DEMAND_NOT_MET_FACTOR', 1.3))
BATTERY_CAPACITY = float(os.getenv('BATTERY_CAPACITY', 30.0))

class ElectricityMarketEnv(gym.Env):
    metadata = {"render.modes": ["human"]}

    def __init__(self, battery_capacity=BATTERY_CAPACITY, initial_soc=INITIAL_SOC, renewable_scale=RENEWABLE_SCALE, demand_not_met_factor=DEMAND_NOT_MET_FACTOR, num_steps=NUM_STEPS):
        super(ElectricityMarketEnv, self).__init__()
        # Parameters for battery and market
        self.battery_capacity = BATTERY_CAPACITY
        self.soc = INITIAL_SOC
        self.base_demand = 5.0
        self.base_price = 10.0

            
        # Demand function parameters
        self.demand_peak1_amp = 10.0
        self.demand_peak2_amp = 8.0
        self.demand_peak1_time = 8.0    # morning peak hour
        self.demand_peak2_time = 18.0   # evening peak hour
        self.demand_variance = 2.0
        self.demand_noise_std = 1.0
        
        # Price function parameters
        self.price_amp1 = 5.0
        self.price_amp2 = 3.0
        self.price_noise_std = 1.0
        
        # Renewable generation parameters
        self.renewable_noise_std = 0.5

        # Demand response penalty
        self.demand_response_penalty = -0.9


        self.action_space = spaces.Box(low=-self.battery_capacity, high=self.battery_capacity, shape=(1,), dtype=np.float32)
        self.last_action = None
        self.last_state = None
        self.observation_space = spaces.Dict({
            'soc': spaces.Box(low=0.0, high=self.battery_capacity, shape=(1,), dtype=np.float32),
            'demand': spaces.Box(low=0.0, high=np.inf, shape=(1,), dtype=np.float32),
            'price': spaces.Box(low=0.0, high=np.inf, shape=(1,), dtype=np.float32),
            'renewable': spaces.Box(low=0.0, high=np.inf, shape=(1,), dtype=np.float32),
            'season': spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32),
            'time': spaces.Box(low=0.0, high=np.inf, shape=(1,), dtype=np.float32)
        }
        )
        
        self.time = 0
        self.max_time = NUM_STEPS

    def _compute_demand(self, time):
        """
        Computes household demand based on:
        - Two daily peaks (morning & evening).
        - Seasonal and weekly variations.
        """
        hour = time % 24
        day_of_year = (time // 24) % 365
        weekday = (time // 24) % 7  # 0=Monday, 6=Sunday

        peak1 = self.demand_peak1_amp * np.exp(-((hour - self.demand_peak1_time) ** 2) / (2 * self.demand_variance ** 2))
        peak2 = self.demand_peak2_amp * np.exp(-((hour - self.demand_peak2_time) ** 2) / (2 * self.demand_variance ** 2))
        seasonal_factor = 1 + 0.3 * np.cos(2 * np.pi * day_of_year / 365)
        weekly_factor = 0.9 if weekday in [5, 6] else 1.0
        demand = self.base_demand * seasonal_factor * weekly_factor + peak1 + peak2
        demand += np.random.normal(0, self.demand_noise_std)
        return max(demand, 0)
        

    def _compute_price(self, time):
        """
        Computes electricity price based on:
        - Daily oscillations with peaks.
        - Seasonal and weekly variations.
        - Random market fluctuations.
        """
        hour = time % 24
        day_of_year = (time // 24) % 365
        weekday = (time // 24) % 7  # 0=Monday, 6=Sunday

        daily_variation = (self.price_amp1 * np.sin(2 * np.pi * hour / 24) +
                           self.price_amp2 * np.sin(4 * np.pi * hour / 24))
        seasonal_factor = 1 + 0.4 * np.cos(2 * np.pi * day_of_year / 365)
        weekly_factor = 1.2 if weekday < 5 else 1.0
        market_fluctuation = np.random.normal(0, self.price_noise_std)
        price = self.base_price * seasonal_factor * weekly_factor + daily_variation + market_fluctuation
        return max(price, 0)
        
    def _compute_renewable(self, time):
        """
        Computes renewable generation available between 6 AM and 6 PM,
        includes seasonal variations.
        """
        hour = time % 24
        day_of_year = (time // 24) % 365
        # Seasonal factor: peak generation around day 172 (summer) and lower in winter.
        seasonal_factor = 1 + 0.5 * np.cos(2 * np.pi * (day_of_year - 172) / 365)
        if 6 <= hour <= 18:
            generation = np.sin(np.pi * (hour - 6) / 12) * RENEWABLE_SCALE * seasonal_factor
            generation += np.random.normal(0, self.renewable_noise_std)
            return max(generation, 0)
        else:
            return 0.0

    def _compute_season(self, time):
        """
        Computes the seasonality factor between 0 and 1:
        0 - winter, 1 - summer, between 0 and 1 - transition seasons.
        """
        day_of_year = (time // 24) % 365
        season = (day_of_year % 365) / 365
        return season
    def _get_state(self):
        """Constructs and returns the current state as a dict of numpy arrays."""
        D = self._compute_demand(self.time)
        P = self._compute_price(self.time)
        R = self._compute_renewable(self.time)
        state = {
            'soc': np.array([self.soc], dtype=np.float32),
            'demand': np.array([D], dtype=np.float32),
            'price': np.array([P], dtype=np.float32),
            'renewable': np.array([R], dtype=np.float32),
            'season': np.array([self._compute_season(self.time)], dtype=np.float32),
            'time': np.array([self.time], dtype=np.float32)
        }
        return state
    def step(self, action):
        battery_action = action
        demand_not_met_penalty = 0.0
        battery_to_spend = 0.0
        
        to_sell = 0.0
        # First, we try to meet the demand of the household
        demand = self.last_state['demand'][0]

        renewable = self.last_state['renewable'][0]
        price = self.last_state['price'][0]

        # meet the demand with renewable energy
        remaining_demand = max(demand - renewable, 0)
        remaining_renewable = max(renewable - demand, 0)
        if remaining_demand > 0:
            # If there is still a demand, we use the battery
            battery_to_spend = min(remaining_demand, self.soc)
            remaining_demand = max(remaining_demand - self.soc, 0)
            self.soc = max(self.soc - battery_to_spend, 0)

        if remaining_demand > 0:
            # If there is still a demand, we get penalized by half the price of the demand
            demand_not_met_penalty = remaining_demand * price / DEMAND_NOT_MET_FACTOR
        if battery_action > 0:
            # Charging the battery
            to_charge = min(battery_action, remaining_renewable)
            self.soc = min(self.soc + to_charge, self.battery_capacity)
        else:
            to_sell = remaining_renewable
            to_sell += min(-battery_action, self.soc)
            self.soc = max(self.soc + battery_action, 0)
        reward = to_sell * price - demand_not_met_penalty 
        self.time += 1
        done = self.time >= self.max_time
        obs = self._get_state()
        self.last_state = obs
        self.last_action = action
        info = {'battery_used_in_demand': battery_to_spend / demand if demand > 0 else 0,'sold':to_sell,'penalty':demand_not_met_penalty,'rewards':reward}

        return obs, reward, done, done, info
    def reset(self, seed=None,options=None):
        super().reset(seed=seed)
        self.soc = INITIAL_SOC
        self.time = 0
        obs = self._get_state()
        self.last_state = obs
        self.last_action = None
        return obs, {}

    def render(self, mode='human'):
        print(f"Time: {self.time}, SoC: {self.soc}, demand {self.last_state['demand']}, price {self.last_state['price']}, renewable {self.last_state['renewable']}, action {self.last_action}")
