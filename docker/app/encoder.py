import torch
import torch.nn as nn
from typing import List
from layers import PositionalEncoding, FeedForwardNetwork
from attention import MultiHeadAttention
from torch.utils.checkpoint import checkpoint
from config import LAYER_DROPOUT

def validate_indices(x, vocab_size):
    if (x >= vocab_size).any() or (x < 0).any():
        raise ValueError("インデックスがエンベディングテーブルの範囲外です。")

class Encoder(nn.Module):
    """
    Transformerのエンコーダー部分を実装するクラス。

    入力シーケンスを受け取り、自己注意機構を用いて文脈を考慮した
    表現に変換します。

    Attributes:
        device (str): 使用するデバイス（'cuda' または 'cpu'）
        embedding (nn.Embedding): 入力トークンの埋め込み層
        pos_encoding (PositionalEncoding): 位置エンコーディング層
        layers (nn.ModuleList): エンコーダー層のリスト
        dropout (nn.Dropout): ドロップアウト層

    Args:
        input_dim (int): 入力ボキャブラリーのサイズ
        hidden_dim (int): 隠れ層の次元数
        num_heads (int): マルチヘッドアテンションのヘッド数
        num_layers (int): エンコーダー層の数
        ff_dim (int): フィードフォワードネットワークの中間層の次元数
        dropout (float): ドロップアウト率
        device (str): 使用するデバイス
    """

    def __init__(self, input_dim: int, hidden_dim: int, num_heads: int, num_layers: int, ff_dim: int, dropout: float, device: str):
        super(Encoder, self).__init__()
        self.device = device
        self.embedding = nn.Embedding(input_dim, hidden_dim, padding_idx=0)
        self.use_checkpointing = True  # チェックポイントを使用するかどうか

        # パディングトークンのエンベディングがゼロに初期化されていることを確認
        with torch.no_grad():
            self.embedding.weight[0].fill_(0)

        self.pos_encoding = PositionalEncoding(hidden_dim)
        self.layers = nn.ModuleList([
            EncoderLayer(hidden_dim, num_heads, ff_dim, dropout) for _ in range(num_layers)
        ])
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        エンコーダーの順伝播を行います。

        Args:
            x (torch.Tensor): 入力テンソル。形状は(batch_size, sequence_length)
            mask (torch.Tensor): ソースパディングマスク。形状は(batch_size, 1, 1, sequence_length)

        Returns:
            torch.Tensor: エンコードされた出力テンソル。形状は(batch_size, sequence_length, hidden_dim)

        Raises:
            ValueError: 入力テンソルのインデックスが埋め込み層の範囲外の場合
        """
        validate_indices(x, self.embedding.num_embeddings)
        x = self.embedding(x)
        x = x + self.pos_encoding(x)
        x = self.dropout(x)

        # レイヤードロップの準備（トレーニング時のみ適用）
        layer_dropout = LAYER_DROPOUT if self.training else 0.0
        dropout_probs = torch.empty(len(self.layers)).uniform_()

        # チェックポイントを使用する場合
        if self.use_checkpointing and self.training:
            # 層ごとの順伝播をチェックポイント化する
            for i, layer in enumerate(self.layers):
                # レイヤードロップ - 一部の層をランダムにスキップ
                if dropout_probs[i] < layer_dropout:
                    continue

                # チェックポイント機能はダミー引数が必要
                def custom_forward(module, input_x, mask_):
                    return module(input_x, mask_)

                # トレーニング中のみチェックポイントを使用
                x = checkpoint(custom_forward, layer, x, mask)
        else:
            # 通常の順伝播（評価時など）
            for i, layer in enumerate(self.layers):
                # レイヤードロップ - 一部の層をランダムにスキップ
                if dropout_probs[i] < layer_dropout:
                    continue
                x = layer(x, mask)

        return x

class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attention = MultiHeadAttention(d_model, d_model // num_heads, d_model // num_heads, num_heads)
        self.feed_forward = FeedForwardNetwork(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, mask):
        """
        Args:
            x (torch.Tensor): 入力 (batch, seq_len, d_model)
            mask (torch.Tensor): パディングマスク (batch, 1, 1, seq_len)
        """
        # Pre-LN方式: 正規化を最初に適用
        _x = self.norm1(x)
        attention_output = self.self_attention(_x, _x, _x, mask=mask)
        x = x + self.dropout1(attention_output)  # 残差接続

        _x = self.norm2(x)
        ff_output = self.feed_forward(_x)
        x = x + self.dropout2(ff_output)  # 残差接続
        return x
