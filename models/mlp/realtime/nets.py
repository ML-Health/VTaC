import torch
import torch.nn as nn


class ClassifierMLP(nn.Module):
    def __init__(self, input_dim, dropout_prob=0.5):
        super(ClassifierMLP, self).__init__()

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_dim, 1024),
            nn.ReLU(),
            nn.Dropout(p=dropout_prob),
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Dropout(p=dropout_prob),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(p=dropout_prob),
            nn.Linear(64, 1),
        )

    def forward(self, signal):
        prob = self.classifier(signal)
        return prob


class MLP(nn.Module):
    def __init__(self, n_components):
        super(MLP, self).__init__()

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(n_components, 64),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward(self, signal):
        prob = self.classifier(signal)
        return prob
