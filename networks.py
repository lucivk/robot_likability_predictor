import torch
from torch import nn
import torch.nn.functional as f

class BaseNetwork(nn.Module):
    def save(self, base_dir):
        name = self.__class__.__name__
        torch.save(self.state_dict(), f'{base_dir}/{name}.pth')
    def load(self, base_dir):
        name = self.__class__.__name__
        self.load_state_dict(torch.load(f'{base_dir}/{name}.pth'))

class SimpleCNN(BaseNetwork):
    def __init__(self, in_channels):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=8, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(16 * 56 * 56, 1)

    def forward(self, x):
        x = f.relu(self.conv1(x))  # Apply first convolution and ReLU activation
        x = self.pool(x)           # Apply max pooling
        x = f.relu(self.conv2(x))  # Apply second convolution and ReLU activation
        x = self.pool(x)           # Apply max pooling
        x = x.reshape(x.shape[0], -1)  # Flatten the tensor
        x = self.fc1(x)            # Apply fully connected layer
        return x
