import torch
import torch.nn as nn
from config import MAX_SEQ_LENGTH
from layers import PositionalEncoding, FeedForwardNetwork
from attention import MultiHeadAttention

class Decoder(nn.Module):
    """
    Transformerのデコーダー部分を実装するクラス。

    エンコーダーの出力と目標シーケンスを受け取り、
    次のトークンを予測するための表現を生成します。

    Attributes:
        device (str): 使用するデバイス（'cuda' または 'cpu'）
        embedding (nn.Embedding): 入力トークンの埋め込み層
        pos_encoding (PositionalEncoding): 位置エンコーディング層
        layers (nn.ModuleList): デコーダー層のリスト
        dropout (nn.Dropout): ドロップアウト層
        output_layer (nn.Linear): 出力層

    Args:
        input_dim (int): 入力ボキャブラリーのサイズ
        hidden_dim (int): 隠れ層の次元数
        num_heads (int): マルチヘッドアテンションのヘッド数
        num_layers (int): デコーダー層の数
        ff_dim (int): フィードフォワードネットワークの中間層の次元数
        output_dim (int): 出力ボキャブラリーのサイズ
        dropout (float): ドロップアウト率
        device (str): 使用するデバイス
    """

    def __init__(self, input_dim, hidden_dim, num_heads, num_layers, ff_dim, output_dim, dropout, device):
        super(Decoder, self).__init__()
        self.device = device
        self.embedding = nn.Embedding(input_dim, hidden_dim)
        self.pos_encoding = PositionalEncoding(hidden_dim)
        self.layers = nn.ModuleList([
            DecoderLayer(hidden_dim, num_heads, ff_dim, dropout) for _ in range(num_layers)
        ])
        self.dropout = nn.Dropout(dropout)
        self.output_layer = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, encoder_output):
        """
        デコーダーの順伝播を行います。

        Args:
            x (torch.Tensor): 入力テンソル。形状は(batch_size, sequence_length)
            encoder_output (torch.Tensor): エンコーダーの出力テンソル。
                形状は(batch_size, source_sequence_length, hidden_dim)

        Returns:
            torch.Tensor: デコードされた出力テンソル。
                形状は(batch_size, sequence_length, output_dim)
        """
        x = self.embedding(x)
        x = x + self.pos_encoding(x)
        for layer in self.layers:
            x = layer(x, encoder_output)
        x = self.output_layer(x)
        return x

class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout):
        super(DecoderLayer, self).__init__()
        self.self_attention = MultiHeadAttention(d_model, d_model, d_model, num_heads)
        self.encoder_decoder_attention = MultiHeadAttention(d_model, d_model, d_model, num_heads)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.feed_forward = FeedForwardNetwork(d_model, d_ff, dropout)

    def forward(self, x, encoder_output, src_mask=None, tgt_mask=None):
        self_attention = self.self_attention(x, x, x, tgt_mask)
        x = self.dropout1(self.norm1(x + self_attention))
        encoder_decoder_attention = self.encoder_decoder_attention(x, encoder_output, encoder_output, src_mask)
        x = self.dropout1(self.norm2(x + encoder_decoder_attention))
        feed_forward_output = self.feed_forward(x)
        x = self.dropout1(self.norm3(x + feed_forward_output))
        return x
