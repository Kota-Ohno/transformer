import torch
import torch.nn as nn
from config import LAYER_DROPOUT
from layers import PositionalEncoding, FeedForwardNetwork
from attention import MultiHeadAttention
from torch.utils.checkpoint import checkpoint

def validate_indices(x, vocab_size):
    if (x >= vocab_size).any() or (x < 0).any():
        raise ValueError("インデックスがエンベディングテーブルの範囲外です")

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
        self.embedding = nn.Embedding(input_dim, hidden_dim, padding_idx=0)
        self.use_checkpointing = True  # チェックポイントを使用するかどうか

        # パディングトークンのエンベディングがゼロに初期化されていることを確認
        with torch.no_grad():
            self.embedding.weight[0].fill_(0)

        self.pos_encoding = PositionalEncoding(hidden_dim)
        self.layers = nn.ModuleList([
            DecoderLayer(hidden_dim, num_heads, ff_dim, dropout) for _ in range(num_layers)
        ])
        self.dropout = nn.Dropout(dropout)
        self.output_layer = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, encoder_output, tgt_mask, memory_mask, cache=None):
        """
        デコーダーの順伝播を行います。
        Args:
            x (torch.Tensor): ターゲット入力テンソル (batch_size, target_seq_len)
            encoder_output (torch.Tensor): エンコーダー出力 (batch_size, source_seq_len, hidden_dim)
            tgt_mask (torch.Tensor): ターゲット用マスク (パディング + 後続) (batch_size, 1, target_seq_len, target_seq_len)
            memory_mask (torch.Tensor): エンコーダー出力用マスク (ソースパディング) (batch_size, 1, 1, source_seq_len)
            cache (list, optional): 各層のキャッシュを保持するリスト。推論時に使用。
        Returns:
            tuple: (出力テンソル, 更新されたキャッシュ)
        """
        validate_indices(x, self.embedding.num_embeddings)
        x = self.embedding(x)
        x = x + self.pos_encoding(x)
        x = self.dropout(x)

        # キャッシュの初期化
        new_cache = []
        if cache is None:
            cache = [None] * len(self.layers)

        # レイヤードロップの準備（トレーニング時のみ適用）
        layer_dropout = LAYER_DROPOUT if self.training else 0.0
        dropout_probs = torch.empty(len(self.layers)).uniform_()

        # チェックポイントを使用する場合（ただし推論時のキャッシュ使用時は除く）
        if self.use_checkpointing and self.training and cache[0] is None:
            # 層ごとの順伝播をチェックポイント化する
            for i, layer in enumerate(self.layers):
                # レイヤードロップ - 一部の層をランダムにスキップ
                if dropout_probs[i] < layer_dropout:
                    new_cache.append(None)
                    continue

                # チェックポイント対応のために簡易版forward関数を定義
                def custom_forward(module, input_x, enc_out, t_mask, m_mask):
                    return module(input_x, enc_out, t_mask, m_mask, None)[0]

                # チェックポイントを通した順伝播
                x = checkpoint(
                    custom_forward,
                    layer, x, encoder_output, tgt_mask, memory_mask
                )
                new_cache.append(None)  # トレーニング中はキャッシュを使用しない
        else:
            # 通常の順伝播（評価時またはキャッシュ使用時）
            for i, layer in enumerate(self.layers):
                # レイヤードロップは推論時には適用しない（評価時はcache[0] is None）
                if cache[0] is None and dropout_probs[i] < layer_dropout and self.training:
                    new_cache.append(None)
                    continue

                x, layer_cache = layer(x, encoder_output, tgt_mask, memory_mask, cache=cache[i])
                new_cache.append(layer_cache)

        x = self.output_layer(x)
        return x, new_cache

class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout):
        super(DecoderLayer, self).__init__()
        d_k = d_v = d_model // num_heads
        self.self_attention = MultiHeadAttention(d_model, d_k, d_v, num_heads)
        self.encoder_decoder_attention = MultiHeadAttention(d_model, d_k, d_v, num_heads)
        self.feed_forward = FeedForwardNetwork(d_model, d_ff, dropout)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    def forward(self, x, encoder_output, tgt_mask, memory_mask, cache=None):
        """
        Args:
            x (torch.Tensor): デコーダー入力 (batch, tgt_len, d_model)
            encoder_output (torch.Tensor): エンコーダー出力 (batch, src_len, d_model)
            tgt_mask (torch.Tensor): ターゲット用マスク (batch, 1, tgt_len, tgt_len)
            memory_mask (torch.Tensor): エンコーダー出力用マスク (batch, 1, 1, src_len)
            cache (dict, optional): キャッシュ。推論時に使用
                                  {
                                    'self_k': テンソル, 'self_v': テンソル,
                                    'memory_k': テンソル, 'memory_v': テンソル
                                  }
        Returns:
            tuple: (出力テンソル, 更新されたキャッシュ)
        """
        # キャッシュの初期化（Noneの場合）
        if cache is None:
            cache = {}

        # セルフアテンション層 (Pre-LN方式)
        residual = x
        x = self.norm1(x)

        # 推論時に過去のキー・バリューをキャッシュから読み込む
        self_attn_kwargs = {}
        if 'self_k' in cache and 'self_v' in cache:
            self_attn_kwargs['cached_k'] = cache['self_k']
            self_attn_kwargs['cached_v'] = cache['self_v']

        # セルフアテンション計算（キャッシュ付き）
        x_attn, self_attn_weights, new_self_k, new_self_v = self.self_attention(
            x, x, x, mask=tgt_mask, return_cache=True, **self_attn_kwargs
        )

        # キャッシュを更新
        cache['self_k'] = new_self_k
        cache['self_v'] = new_self_v

        x = residual + self.dropout1(x_attn)

        # エンコーダ-デコーダアテンション層 (Pre-LN方式)
        residual = x
        x = self.norm2(x)

        # 推論時にエンコーダ出力のキー・バリューをキャッシュから読み込む
        enc_dec_attn_kwargs = {}
        if 'memory_k' in cache and 'memory_v' in cache:
            enc_dec_attn_kwargs['cached_k'] = cache['memory_k']
            enc_dec_attn_kwargs['cached_v'] = cache['memory_v']
        else:
            # エンコーダ出力のキー・バリューを初めて計算
            _, _, memory_k, memory_v = self.encoder_decoder_attention(
                x, encoder_output, encoder_output,
                mask=memory_mask, return_cache=True
            )
            cache['memory_k'] = memory_k
            cache['memory_v'] = memory_v
            enc_dec_attn_kwargs['cached_k'] = memory_k
            enc_dec_attn_kwargs['cached_v'] = memory_v

        x_attn, _, _, _ = self.encoder_decoder_attention(
            x, encoder_output, encoder_output,
            mask=memory_mask, return_cache=True, **enc_dec_attn_kwargs
        )

        x = residual + self.dropout2(x_attn)

        # フィードフォワード層 (Pre-LN方式)
        residual = x
        x = self.norm3(x)
        x = residual + self.dropout3(self.feed_forward(x))

        return x, cache
