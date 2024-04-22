import numpy
import torch
import torch.nn as nn
import esm
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

class FC(nn.Module):
    def __init__(self, feature, dropout=0.5):
        super().__init__()
        self.LastDense = nn.Sequential(
            nn.BatchNorm1d(1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(feature, 1, bias=False),
            nn.Sigmoid()
        )


    def forward(self, x):
        x = x.unsqueeze(2).expand(-1, -1, x.size(1), -1) + x.unsqueeze(1).expand(-1, x.size(1), -1, -1)
        x, _ = torch.max(x, dim=1)
        result = self.LastDense(x)
        return result


class MyModel(nn.Module):
    def __init__(self, pretrained_model):
        super(MyModel, self).__init__()
        self.pretrained_model = pretrained_model
        self.n_layers = n_layers = 2
        self.hidden_dim = hidden_dim = 512
        drop_prob = 0.5
        embedding_dim = pretrained_model.embed_dim
        self.lstm = nn.LSTM(embedding_dim,
                            hidden_dim,
                            n_layers,
                            dropout=drop_prob,
                            batch_first=True,
                            # bidirectional=True
                            )

        self.fc = FC(hidden_dim, drop_prob)
        self.pooling = nn.AdaptiveAvgPool1d(1)  # 1D平均池化层，将整个序列池化成一个值

    def forward(self, tokens, hidden, repr_layers=[30]):
        x = self.pretrained_model(tokens, repr_layers)
        x = x['representations'][30]

        lstm_out, hidden = self.lstm(x, hidden)
        lstm_out = lstm_out.permute(0, 2, 1)

        # 序列池化
        x = self.pooling(lstm_out).permute(0, 2, 1)
        # 线性层进行分类
        x = self.fc(x)
        return x

    def init_hidden(self, batch_size):
        hidden = (torch.zeros(self.n_layers, batch_size, self.hidden_dim),
                  torch.zeros(self.n_layers, batch_size, self.hidden_dim)
                  )
        return hidden


