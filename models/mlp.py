# write an mlp model for mnist here

import torch
import torch.nn as nn
import torch.nn.functional as F



class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes, depth=1):
        super(MLP, self).__init__()
        self.depth = depth
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.hidden_layers = nn.ModuleList()
        if depth > 1:
            for _ in range(depth - 1):
                self.hidden_layers.append(nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.BatchNorm1d(hidden_dim),
                    nn.ReLU()
                ))
        self.fc2 = nn.Linear(hidden_dim, num_classes)

    def forward(self, X):
        out = X.view(X.shape[0], -1)
        out = F.relu(self.fc1(out))
        for layer in self.hidden_layers:
            out = F.relu(layer(out))
        out = self.fc2(out)
        return out
    

# class MLP(nn.Module):

# def __init__(self, input_dim, hidden_dim, num_classes):
#     super(MLP, self).__init__()
#     self.fc1 = nn.Linear(input_dim, hidden_dim)
#     self.fc2 = nn.Linear(hidden_dim, num_classes)
# def forward(self, X):
#     out = X.view(X.shape[0], -1)
#     out = F.relu(self.fc1(out))
#     out = self.fc2(out)
#     return out