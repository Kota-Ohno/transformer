import torch
import torch.nn as nn
import math
from config import MAX_SEQ_LENGTH, HIDDEN_SIZE

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, device):
        super(PositionalEncoding, self).__init__()
        self.d_model = d_model
        self.device = device
        self.encoding = self.create_encoding()

    def forward(self, x):
        return x + self.encoding[:, :x.size(1), :].to(self.device)

    def create_encoding(self):
        encoding = torch.zeros(MAX_SEQ_LENGTH, self.d_model, device=self.device)
        position = torch.arange(0, MAX_SEQ_LENGTH, dtype=torch.float, device=self.device).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, self.d_model, 2, device=self.device).float() * (-math.log(10000.0) / self.d_model))
        encoding[:, 0::2] = torch.sin(position * div_term)
        encoding[:, 1::2] = torch.cos(position * div_term)
        encoding = encoding.unsqueeze(0)  # バッチ次元を追加
        return encoding

class FeedForwardNetwork(nn.Module):
    def __init__(self, d_model, d_ff, dropout):
        super(FeedForwardNetwork, self).__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = nn.ReLU()(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x
