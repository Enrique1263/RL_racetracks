import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

class R25CamNet(nn.Module):
    def __init__(self, input_shape):
        super(R25CamNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=input_shape[0], out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(in_features=128*80*60, out_features=512)
        self.fc2 = nn.Linear(in_features=512, out_features=4) # throttle, brake, right_steer, left_steer

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, kernel_size=2, stride=2) # (3, 320, 240)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, kernel_size=2, stride=2) # (32, 160, 120)
        x = F.relu(self.conv3(x))
        x = F.max_pool2d(x, kernel_size=2, stride=2) # (64, 80, 60)
        x = x.view(-1, 128*80*60)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        x = torch.sigmoid(x)
        return x

   
class R25MLP(nn.Module):
    def __init__(self, input_shape):
        super(R25MLP, self).__init__()
        self.fc1 = nn.Linear(in_features=input_shape[0], out_features=2048)
        self.fc2 = nn.Linear(in_features=2048, out_features=512)
        self.fc3 = nn.Linear(in_features=512, out_features=4)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        x = torch.sigmoid(x)
        return x


class R25Tiny(nn.Module):
    def __init__(self, input_shape):
        super(R25Tiny, self).__init__()
        self.input_shape = input_shape
        self.conv1 = nn.Conv2d(in_channels=input_shape[0], out_channels=8, kernel_size=3, stride=1, padding=1)
        # self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        # self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        # self.fc1 = nn.Linear(in_features=128*80*60, out_features=512)
        self.fc2 = nn.Linear(in_features=8 * input_shape[1] // 2 * input_shape[2] // 2, out_features=4) # throttle, brake, right_steer, left_steer

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, kernel_size=2, stride=2) # (8, 3, 2)
        # x = F.relu(self.conv2(x))
        # x = F.max_pool2d(x, kernel_size=2, stride=2) # (64, 160, 120)
        # x = F.relu(self.conv3(x))
        # x = F.max_pool2d(x, kernel_size=2, stride=2) # (128, 80, 60)
        x = x.view(-1, 8 * self.input_shape[1] // 2 * self.input_shape[2] // 2)
        # x = F.relu(self.fc1(x))
        x = self.fc2(x)
        x = torch.sigmoid(x)
        return x


class R25TinySimple(nn.Module):
    def __init__(self, input_shape):
        super(R25TinySimple, self).__init__()
        self.input_shape = input_shape
        self.conv1 = nn.Conv2d(in_channels=input_shape[0], out_channels=8, kernel_size=3, stride=1, padding=1)
        # self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        # self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        # self.fc1 = nn.Linear(in_features=128*80*60, out_features=512)
        self.fc2 = nn.Linear(in_features=8 * input_shape[1] // 2 * input_shape[2] // 2, out_features=4) # left, forward, right, brake

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, kernel_size=2, stride=2) # (8, 3, 2)
        # x = F.relu(self.conv2(x))
        # x = F.max_pool2d(x, kernel_size=2, stride=2) # (64, 160, 120)
        # x = F.relu(self.conv3(x))
        # x = F.max_pool2d(x, kernel_size=2, stride=2) # (128, 80, 60)
        x = x.view(-1, 8 * self.input_shape[1] // 2 * self.input_shape[2] // 2)
        # x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    

class Actor(nn.Module):
    def __init__(self, state_shape, action_dim):
        super(Actor, self).__init__()

        input_channels = state_shape[0]

        # CNN Feature Extractor
        self.conv = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU()
        )

        # Get CNN output size dynamically
        with torch.no_grad():
            dummy_input = torch.zeros(1, *state_shape)  # Use batch_size=1 to avoid constraints
            dummy_output = self.conv(dummy_input)
            self.flatten_size = dummy_output.numel() // dummy_output.shape[0]  # Compute dynamically

        # print("flatten_size", self.flatten_size)

        # Fully Connected Layers
        self.fc = nn.Sequential(
            nn.Linear(self.flatten_size, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
        )

        # Output layers for mean & log_std
        self.mean = nn.Linear(128, action_dim)
        self.log_std = nn.Linear(128, action_dim)

    def forward(self, state):
        """Returns action distribution parameters (mean & log_std)"""
        x = self.conv(state)
        x = torch.flatten(x, start_dim=1)
        # print("x", x.shape)
        x = self.fc(x)

        mean = self.mean(x)  # Mean of action distribution
        mean = torch.sigmoid(mean)  # Squash output to [0,1] range
        log_std = self.log_std(x).clamp(-20, 2)  # Clamp log_std for stability
        return mean, log_std

    def sample_action(self, state):
        """Samples an action using reparameterization trick"""
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = Normal(mean, std)
        z = normal.rsample()  # Sample using reparameterization trick
        action = torch.sigmoid(z)  # Squash output to [0,1] range

        # Compute log probability
        log_prob = normal.log_prob(z) - torch.log(action * (1 - action) + 1e-6)  # Log prob adjustment
        log_prob = log_prob.sum(dim=-1, keepdim=True)

        return action, log_prob


class Critic(nn.Module):
    def __init__(self, state_shape, action_dim):
        super(Critic, self).__init__()

        input_channels = state_shape[0]

        # CNN Feature Extractor
        self.conv = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU()
        )

        # Get CNN output size dynamically
        with torch.no_grad():
            dummy_input = torch.zeros(1, *state_shape)  # Use batch_size=1 to avoid constraints
            dummy_output = self.conv(dummy_input)
            self.flatten_size = dummy_output.numel() // dummy_output.shape[0]  # Compute dynamically

        # Fully Connected Layers for State and Action
        self.fc_state = nn.Linear(self.flatten_size, 256)
        self.fc_action = nn.Linear(action_dim, 256)

        # Q-value output layer
        self.fc = nn.Sequential(
            nn.ReLU(),
            nn.Linear(512, 128),  # 256(state) + 256(action)
            nn.ReLU(),
            nn.Linear(128, 1)  # Single Q-value output
        )

    def forward(self, state, action):
        """Returns Q-value for (state, action) pairs"""
        x = self.conv(state)
        x = torch.flatten(x, start_dim=1)

        x_state = self.fc_state(x)
        x_action = self.fc_action(action)

        x = torch.cat([x_state, x_action], dim=-1)  # Concatenate state & action
        return self.fc(x)  # Output Q-value