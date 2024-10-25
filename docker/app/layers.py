import torch
import torch.nn as nn
import math
from config import MAX_SEQ_LENGTH

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_length=MAX_SEQ_LENGTH):
        super(PositionalEncoding, self).__init__()
        self.d_model = d_model
        self.max_seq_length = max_seq_length
        
        self.encoding = nn.Parameter(self.create_encoding(), requires_grad=False)

    def forward(self, x):
        seq_length = x.size(1)
        return x + self.encoding[:, :seq_length, :]

    def create_encoding(self):
        encoding = torch.zeros(1, self.max_seq_length, self.d_model)
        position = torch.arange(0, self.max_seq_length).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, self.d_model, 2).float() * (-math.log(10000.0) / self.d_model))
        
        encoding[0, :, 0::2] = torch.sin(position * div_term)
        encoding[0, :, 1::2] = torch.cos(position * div_term)
        
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
