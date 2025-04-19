
import gymnasium as gym
import torch
from copy import deepcopy

# Wrapper to use learned reward model in environment
class RewardWrapper(gym.Wrapper):
    def __init__(self, env, reward_model):
        super().__init__(env)
        self.reward_model = reward_model
        
    def step(self, action):
        next_state, _, terminated, truncated, info = self.env.step(action)
        
        # Convert state and action to tensors
        state_tensor = torch.FloatTensor(next_state)
        action_tensor = torch.FloatTensor(action)
        
        # Get reward from learned reward model
        with torch.no_grad():
            reward = self.reward_model(state_tensor, action_tensor).item()
            
        return next_state, reward, terminated, truncated, info

# Wrapper to store and set environment state
class SetWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.env = env.unwrapped
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space

    def set_state(self, state):
        self.env = deepcopy(state)

    def get_state(self):
        return deepcopy(self.env)
