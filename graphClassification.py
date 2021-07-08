import torch.nn as nn


class EmbeddingGraphClassifier(nn.Module):
    def __init__(self, data_size):
        super(EmbeddingGraphClassifier, self).__init__()
        self.data_size = data_size
        self.fc = nn.Linear(self.data_size, 1)

    def forward(self, x):
        x = self.fc(x)
        return x
