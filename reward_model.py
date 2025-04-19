import torch.nn as nn
import torch

## Defining simple reward model, MLP with 3 layers
# Reward obtained for being in "state" and executing "action".
# Other reward models can be implemented by changing this function (for instance arriving to state)
class RewardModel(nn.Module):

    def __init__(self, state_dim, action_dim):
        # state_dim: dimension of the state space
        # action_dim: dimension of the action space
        super(RewardModel, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim + action_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )

    def forward(self, state, action):
        if state.dim() == 1:
            state = state.unsqueeze(0)
        if action.dim() == 1:
            action = action.unsqueeze(0)
        return self.network(torch.cat([state, action], dim=1))