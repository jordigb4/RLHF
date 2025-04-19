import numpy as np
import time
import gymnasium as gym
import torch
import torch.optim as optim
from stable_baselines3 import SAC
from stable_baselines3.common.evaluation import evaluate_policy

from reward_model import RewardModel  ## This is a custom class that defines a reward model neural network
from wrappers import RewardWrapper, SetWrapper  ## This is a custom class that defines the RewardWrapper and the SetWrapper classes

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# Agent class
class RLHFPendulum:
    def __init__(self):
        """
        Initializes the RLHF agent for the Pendulum environment. 
        """
        self.env = gym.make('Pendulum-v1')
        self.state_dim = 3
        self.action_dim = 1
        self.reward_model = RewardModel(self.state_dim, self.action_dim) # Reward model to be learned
        self.reward_model_optimizer = optim.Adam(self.reward_model.parameters(), lr=2e-4, weight_decay=0.0001)
        self.model = None  # Will store the SAC agent
        

    def collect_trajectories(self, n_pairs, num_steps=200,  use_policy=False):
        """
        Collects n_pairs pieces of num_steps trajectories from the environment using the current policy 
        (if use_policy is True) or random actions.
        Parameters:
        - n_pairs (int): Number of pairs of trajectories to collect.
        - num_steps (int, optional): Number of steps per trajectory. Default is 200.
        - use_policy (bool, optional): Whether to use the policy model to predict actions. Default is False.
        Returns:
        - A list of tuples, where each tuple contains a trajectory (a list of (state, action, reward) tuples) and the total reward for that trajectory.
        """
    
        assert num_steps <= 200, "num_steps must be less than or equal to 200"

        # Here we will store the collected pairs of trajectories
        trajectories = []
        
        # Use a wrapper to store the environment state and set it back after each trajectory
        env_set = SetWrapper(self.env)

        for _ in range(n_pairs):
            #Let's collect a pair of trajectories from the same initial state

            # Set the initial state of the environment. This state is selected as a random state from
            # a trajectory using the current policy (why?)
            # This will be the initial state for both trajectories in the pair (why?)
            current_state, _ = env_set.reset()
            for _ in range(np.random.randint(200-num_steps+1)):
                if use_policy and self.model is not None:
                        action, _ = self.model.predict(current_state, deterministic=False) # (why deterministic=False?)
                else:
                        action = env_set.action_space.sample()  # Random actions

                # No need to check for terminated or truncated because this env always end in 200 steps
                next_state, _, _, _, _ = env_set.step(action)  
                current_state = next_state
            current_initial_setate = current_state
            initial_state = env_set.get_state()

            # Collect two trajectories from the same initial state
            for _ in range(2):
                trajectory = []
                total_reward = 0.0
                
                # Set initial state the same for both trajectories using the wrapped environment
                env_set.reset()
                env_set.set_state(initial_state)
                current_state = current_initial_setate
                
                # Generate trajectory
                for _ in range(num_steps):
                    # Use model to predict action
                    if use_policy and self.model is not None:
                        action, _ = self.model.predict(current_state, deterministic=False)
                    else:
                        action = env_set.action_space.sample()  # Random actions
                    
                    # Take step in environment
                    next_state, reward, done, truncated, _ = env_set.step(action)
                    
                    # Store (s,a,r) tuple
                    trajectory.append((current_state, action, reward))
                    total_reward += reward
                    
                    current_state = next_state
                    
                    if done or truncated:
                        break
                
                trajectories.append((trajectory, total_reward))
        
        return trajectories


    def generate_preference_data(self, num_pairs=50, num_steps= 200, use_policy=False):
        """
        Collects pairs of trajectories, determines preferences based
        on true rewards, and returns the preference data.
        Parameters:
        - num_pairs (int): Number of pairs of trajectories to collect.
        - num_steps (int, optional): Number of steps per trajectory. Default is 200.
        - use_policy (bool, optional): Whether to use the policy model to predict actions. Default is False.
        Returns:
        - A list of tuples, where each tuple contains two trajectories and a preference value.
        """
        # Collect pairs of trajectories and determine preferences using true reward
        preference_data = []
        trajectories = self.collect_trajectories(num_pairs * 2, num_steps = num_steps, use_policy=use_policy)
        
        
        for i in range(0, len(trajectories), 2):
            traj1, r1 = trajectories[i]
            traj2, r2 = trajectories[i+1]

            best = 1 if r1 > r2 else 0
            preference_data.append((traj1, traj2, best))
            
        return preference_data
    

    def train_reward_model(self, preference_data, epochs=10):
        """
        Trains the reward model using preference data.
        Notice we train the reward model *FROM JUST ONLY PREFERENCE DATA*
        Parameters:
        - preference_data (list): List of tuples, where each tuple contains two trajectories and a preference value.
        - epochs (int, optional): Number of epochs to train the reward model. Default is 10.
        """
        for _ in range(epochs):
            # For each preference, train the reward model to predict the preference
            # [Can be done better with a DataLoader, etc.]
            for traj1, traj2, pref in preference_data:

                # Convert the current pair of trajectories to tensors to compute the reward model loss
                s1 = torch.FloatTensor(np.array([s for s, _, _ in traj1]))
                a1 = torch.FloatTensor(np.array([a for _, a, _ in traj1]))
                s2 = torch.FloatTensor(np.array([s for s, _, _ in traj2]))
                a2 = torch.FloatTensor(np.array([a for _, a, _ in traj2]))
                
                r1 = self.reward_model.forward(s1, a1).mean()
                r2 = self.reward_model.forward(s2, a2).mean()

                diff_rew = r1 - r2 if pref else r2 - r1
                loss = - (torch.log(torch.sigmoid(diff_rew)))

                self.reward_model_optimizer.zero_grad()
                loss.backward()
                self.reward_model_optimizer.step()
                


    def train(self, num_iterations=5, timesteps_per_iteration=10000, num_pairs=50, num_steps=200):
        """
        Trains the agent using the RLHF algorithm.
            1- Collects preference data using the current policy (or random if first iteration).
            2- Trains the reward model using the preference data.
            3- Initializes and trains the SAC agent on the learned reward function.
            4- Evaluates the current policy on the original environment.
        If the policy is good enough, the training stops. In other case, it continues for the given number 
        of iterations.
        
        Parameters:
        - num_iterations (int): Number of iterations to train the agent.
        - timesteps_per_iteration (int): Number of timesteps to train the agent per iteration.
        - num_pairs (int): Number of pairs of trajectories to collect for preference data.
        - num_steps (int, optional): Number of steps per trajectory. Default is 200.
        Returns:
        - The trained agent and the best reward function found.
        """

        # Initialize best reward and model
        best_reward = float('-inf')
        best_model = None
        
        # Iterate for the given number of iterations. Exit if policy is good enough
        for iteration in range(num_iterations):
            print(f"\nIteration {iteration + 1}/{num_iterations}")
            
            # 1- Generate preference data using current policy (or random if first iteration)
            print("Generating preference data...")
            start = time.time()
            preference_data = self.generate_preference_data(
                num_pairs=num_pairs, 
                num_steps = num_steps,
                use_policy=(iteration > 0)  # Use random policy only in first iteration
            )
            print(f"  Time to generate preference data: {time.time()-start:.2f} seconds")

            # 2- Train/update reward model
            print("Training reward model...")
            start = time.time()
            self.train_reward_model(preference_data)
            print(f"  Time to train reward model: {time.time()-start:.2f} seconds")
            
            # 3- Initialize and train SAC agent on the learned reward function
            wrapped_env = RewardWrapper(self.env, self.reward_model) # Create wrapped environment with current reward model
            print("Learning SAC agent on learned reward function from preferences...")
            start = time.time()
            self.model = SAC("MlpPolicy", wrapped_env, learning_rate=0.001,   verbose=0)
            self.model.learn(total_timesteps=timesteps_per_iteration)
            print(f"  Time to train agent: {time.time()-start:.2f} seconds")
            
            # 4- Evaluate current policy on the original environment if it is good enough, exit!
            mean_reward, std_reward = evaluate_policy(self.model, gym.make('Pendulum-v1'), n_eval_episodes=20)
            print(f"Mean reward after iteration {iteration + 1}: {mean_reward:.2f} +/-{std_reward:.2f}")

            # Save best model to fight possible overfitting
            if mean_reward > best_reward:
                best_reward = mean_reward
                best_model = self.model
                best_reward_function = self.reward_model
            # Early stopping when policy is good enough
            if mean_reward > -250:
                print("\nPolicy is good enough. Stopping training.")
                break

        self.model = best_model

        return self.model, best_reward_function


if __name__ == "__main__":

    # Create the agent
    rlhf = RLHFPendulum()
    # Train the agent
    trained_model, _ = rlhf.train(num_iterations=5, timesteps_per_iteration=10000, num_pairs=100, num_steps=20)

    # See performance of the trained agent
    print("\nSeeing trained agent...")
    env = gym.make('Pendulum-v1', render_mode="human")
    state, _ = env.reset()
    total_reward = 0
        
    for _ in range(200):
            action, _ = trained_model.predict(state, deterministic=True)
            state, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            
            if terminated or truncated:
                break
                
    print(f"Episode reward: {total_reward:.2f}")
    env.close()
