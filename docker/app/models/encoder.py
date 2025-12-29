import torch
import torch.nn as nn
from models.attention import MultiHeadAttention
from models.layers import FeedForward, PositionalEncoding
from utils.config import CONFIG

class EncoderLayer(nn.Module):
    """Transformerのエンコーダーレイヤー"""

    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super(EncoderLayer, self).__init__()

        # 自己アテンション
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)

        # フィードフォワードネットワーク
        self.feed_forward = FeedForward(d_model, d_ff, dropout)

        # レイヤー正規化
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        # ドロップアウト
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        """
        Args:
            x: 入力テンソル [batch_size, seq_len, d_model]
            mask: パディングマスク [batch_size, 1, seq_len] または [batch_size, 1, 1, seq_len]

        Returns:
            出力テンソル [batch_size, seq_len, d_model]
        """
        # マスクの次元を調整（必要な場合）
        if mask is not None and mask.dim() == 3:
            mask = mask.unsqueeze(1)

        # 自己アテンション（残差接続とレイヤー正規化）
        attn_output = self.self_attn(
            self.norm1(x), self.norm1(x), self.norm1(x), mask
        )
        x = x + self.dropout1(attn_output)

        # フィードフォワード（残差接続とレイヤー正規化）
        ff_output = self.feed_forward(self.norm2(x))
        x = x + self.dropout2(ff_output)

        return x

class Encoder(nn.Module):
    """Transformerのエンコーダー"""

    def __init__(self, vocab_size, d_model, num_heads, num_layers, d_ff, dropout=0.1, max_seq_length=512):
        super(Encoder, self).__init__()

        # 単語埋め込み
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)

        # 位置エンコーディング
        self.pos_encoding = PositionalEncoding(d_model, max_seq_length)

        # エンコーダーレイヤー
        self.layers = nn.ModuleList([
            EncoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])

        # 最終レイヤー正規化
        self.norm = nn.LayerNorm(d_model)

        # ドロップアウト
        self.dropout = nn.Dropout(dropout)
        self.d_model = d_model

    def forward(self, x, mask=None):
        """
        Args:
            x: 入力テンソル [batch_size, seq_len]
            mask: パディングマスク [batch_size, 1, seq_len]

        Returns:
            出力テンソル [batch_size, seq_len, d_model]
        """
        # 入力バリデーション: 整数型と範囲チェック
        vocab_size = self.embedding.num_embeddings

        # 整数型チェック
        if not x.dtype.is_integer:
            raise TypeError(
                f"Expected integer dtype for token indices, but got {x.dtype}. "
                f"Input shape: {x.shape}"
            )

        # 範囲チェック
        x_min = x.min().item()
        x_max = x.max().item()
        valid_min = 0
        valid_max = vocab_size - 1

        if x_min < valid_min or x_max > valid_max:
            raise ValueError(
                f"Token indices out of valid range [0, {valid_max}]. "
                f"Found range: [{x_min}, {x_max}]. "
                f"Input shape: {x.shape}, vocab_size: {vocab_size}"
            )

        # 埋め込みと位置エンコーディング
        x = self.embedding(x)
        x = self.pos_encoding(x)
        x = self.dropout(x)

        # エンコーダーレイヤーの適用
        for layer in self.layers:
            x = layer(x, mask)

        # 最終正規化
        x = self.norm(x)

        return x
