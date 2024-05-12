# write an mlp model for mnist here

import torch
import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):

    def __init__(self, input_dim, hidden_dim, num_classes):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_classes)
    def forward(self, X):
        out = X.view(X.shape[0], -1)
        out = F.relu(self.fc1(out))
        out = self.fc2(out)
        return out