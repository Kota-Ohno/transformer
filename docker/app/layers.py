import torch
import torch.nn as nn
import math
from config import MAX_SEQ_LENGTH

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_length=MAX_SEQ_LENGTH):
        super(PositionalEncoding, self).__init__()
        # 位置エンコーディングを事前計算
        pe = torch.zeros(max_seq_length, d_model)
        position = torch.arange(0, max_seq_length).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_seq_length, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        入力テンソルに位置エンコーディングを加算します。
        Args:
            x (torch.Tensor): 入力テンソル (batch_size, seq_len, d_model)
        Returns:
            torch.Tensor: 位置エンコーディングが加算されたテンソル (batch_size, seq_len, d_model)
        """
        # 入力シーケンス長に応じた位置エンコーディングを使用
        return x + self.pe[:, :x.size(1), :]

class FeedForwardNetwork(nn.Module):
    def __init__(self, d_model, d_ff, dropout):
        super(FeedForwardNetwork, self).__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()  # ReLU をモジュールとして定義

    def forward(self, x):
        """
        フィードフォワードネットワークの順伝播
        Args:
            x (torch.Tensor): 入力テンソル (batch_size, seq_len, d_model)
        Returns:
            torch.Tensor: 出力テンソル (batch_size, seq_len, d_model)
        """
        x = self.fc1(x)
        x = self.relu(x)  # nn.Module として呼び出し
        x = self.dropout(x)
        x = self.fc2(x)
        return x
