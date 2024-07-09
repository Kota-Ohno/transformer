import torch
import torch.nn as nn
from layers import PositionalEncoding, FeedForwardNetwork
from attention import MultiHeadAttention

from config import MAX_SEQ_LENGTH

def validate_indices(x, vocab_size):
    if (x >= vocab_size).any() or (x < 0).any():
        raise ValueError("インデックスがエンベディングテーブルの範囲外です。")

class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_heads, num_layers, ff_dim, dropout, device):
        super(Encoder, self).__init__()
        self.device = device
        self.embedding = nn.Embedding(input_dim, hidden_dim, padding_idx=0)
        self.pos_encoding = PositionalEncoding(hidden_dim, device)
        self.layers = nn.ModuleList([
            EncoderLayer(hidden_dim, num_heads, ff_dim, dropout, device) for _ in range(num_layers)
        ])
        self.dropout = nn.Dropout(dropout)

    # エンコーダのforwardメソッド内でチェックを行う
    def forward(self, x):
        validate_indices(x, self.embedding.num_embeddings)
        x = self.embedding(x)
        x = x.squeeze(1)  # 余分な次元を削除
        x = x + self.pos_encoding(x)
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
