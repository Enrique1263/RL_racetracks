import torch
from models import Actor

# Define dummy state input (batch_size=1, channels=1, height=64, width=48)
state_shape = (1, 64, 48)  # (C, H, W)
action_dim = 4

# Create actor model
actor = Actor(state_shape, action_dim)

# Generate a random grayscale state input
dummy_state = torch.randn(1, *state_shape)  # Shape: (1, 1, 64, 48)

# Get action and log probability
action, log_prob = actor.sample_action(dummy_state)

# Print results
print("Predicted Action:", action)
print("Action Shape:", action.shape)
print("Log Probability:", log_prob)