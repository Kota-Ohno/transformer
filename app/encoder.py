import torch
import torch.nn as nn
from .layers import PositionalEncoding, FeedForwardNetwork
from .attention import MultiHeadAttention
from .utils import add_padding

class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_heads, num_layers, ff_dim, dropout, device):
        super(Encoder, self).__init__()
        self.device = device
        self.embedding = nn.Embedding(input_dim, hidden_dim)
        self.pos_encoding = PositionalEncoding(hidden_dim, device)
        self.layers = nn.ModuleList([
            EncoderLayer(hidden_dim, num_heads, ff_dim, dropout, device) for _ in range(num_layers)
        ])
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = add_padding(x)  # パディングトークンを追加
        x = self.embedding(x) + self.pos_encoding(x)  # トークン埋め込みと位置エンコーディングを追加
        for layer in self.layers:
            x = layer(x)
        return x

class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout, device):
        super(EncoderLayer, self).__init__()
        self.self_attention = MultiHeadAttention(d_model, d_model, d_model, num_heads)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.feed_forward = FeedForwardNetwork(d_model, d_ff, dropout)

    def forward(self, x, mask=None):
        self_attention = self.self_attention(x, x, x, mask)
        x = self.dropout1(self.norm1(x + self_attention))
        feed_forward_output = self.feed_forward(x)
        x = self.dropout1(self.norm2(x + feed_forward_output))
        return x