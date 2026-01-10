import torch
import torch.nn as nn
from models.layers import PositionalEncoding, FeedForward
from models.attention import MultiHeadAttention
from utils.validation import validate_token_ids

class DecoderLayer(nn.Module):
    """Transformerのデコーダーレイヤー"""

    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super(DecoderLayer, self).__init__()

        # 自己アテンション
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)

        # エンコーダ-デコーダアテンション
        self.encoder_attn = MultiHeadAttention(d_model, num_heads, dropout)

        # フィードフォワードネットワーク
        self.feed_forward = FeedForward(d_model, d_ff, dropout)

        # レイヤー正規化
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

        # ドロップアウト
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    def forward(self, x, encoder_output, tgt_mask=None, src_mask=None, cache=None):
        """
        Args:
            x: デコーダー入力 [batch_size, tgt_seq_len, d_model]
            encoder_output: エンコーダー出力 [batch_size, src_seq_len, d_model]
            tgt_mask: ターゲットマスク [batch_size, 1, tgt_seq_len, tgt_seq_len]
            src_mask: ソースマスク [batch_size, 1, 1, src_seq_len]
            cache: キャッシュ情報（推論時に使用）

        Returns:
            output: 出力テンソル [batch_size, tgt_seq_len, d_model]
            new_cache: 更新されたキャッシュ情報
        """
        # キャッシュの初期化
        new_cache = {} if cache is None else cache.copy()
        cache = {} if cache is None else cache

        # 自己アテンション
        residual = x
        x_norm = self.norm1(x)

        # キャッシュを使用するパラメータを準備
        self_attn_params = {}
        if 'self_k' in cache and 'self_v' in cache:
            self_attn_params['cached_k'] = cache['self_k']
            self_attn_params['cached_v'] = cache['self_v']

        # 自己アテンションの実行
        self_attn_output, _, self_k, self_v = self.self_attn(
            q=x_norm, k=x_norm, v=x_norm,
            mask=tgt_mask,
            return_cache=True,
            **self_attn_params
        )

        # キャッシュの更新
        new_cache['self_k'] = self_k
        new_cache['self_v'] = self_v

        # 残差接続
        x = residual + self.dropout1(self_attn_output)

        # エンコーダ-デコーダアテンション
        residual = x
        x_norm = self.norm2(x)

        # キャッシュを使用するパラメータを準備
        enc_attn_params = {}
        if 'memory_k' in cache and 'memory_v' in cache:
            enc_attn_params['cached_k'] = cache['memory_k']
            enc_attn_params['cached_v'] = cache['memory_v']

        # エンコーダ-デコーダアテンションの実行
        enc_attn_output, _, memory_k, memory_v = self.encoder_attn(
            q=x_norm, k=encoder_output, v=encoder_output,
            mask=src_mask,
            return_cache=True,
            **enc_attn_params
        )

        # キャッシュの更新
        new_cache['memory_k'] = memory_k
        new_cache['memory_v'] = memory_v

        # 残差接続
        x = residual + self.dropout2(enc_attn_output)

        # フィードフォワード
        residual = x
        x_norm = self.norm3(x)
        x = residual + self.dropout3(self.feed_forward(x_norm))

        return x, new_cache

class Decoder(nn.Module):
    """Transformerのデコーダー"""

    def __init__(self, vocab_size, d_model, num_heads, num_layers, d_ff, dropout=0.1, max_seq_length=512):
        super(Decoder, self).__init__()

        # 単語埋め込み
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)

        # 位置エンコーディング
        self.pos_encoding = PositionalEncoding(d_model, max_seq_length)

        # デコーダーレイヤー
        self.layers = nn.ModuleList([
            DecoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])

        # 最終レイヤー正規化
        self.norm = nn.LayerNorm(d_model)

        # 出力層
        self.output_layer = nn.Linear(d_model, vocab_size)

        # ドロップアウト
        self.dropout = nn.Dropout(dropout)
        self.d_model = d_model

    def forward(self, x, encoder_output, tgt_mask=None, src_mask=None, cache=None):
        """
        Args:
            x: デコーダー入力 [batch_size, tgt_seq_len]
            encoder_output: エンコーダー出力 [batch_size, src_seq_len, d_model]
            tgt_mask: ターゲットマスク [batch_size, 1, tgt_seq_len, tgt_seq_len]
            src_mask: ソースマスク [batch_size, 1, 1, src_seq_len]
            cache: キャッシュ情報のリスト

        Returns:
            output: 出力テンソル [batch_size, tgt_seq_len, vocab_size]
            new_cache: 更新されたキャッシュ情報のリスト
        """
        # 入力検証：トークンIDの範囲と型をチェック
        vocab_size = self.embedding.num_embeddings
        validate_token_ids(x, vocab_size, tensor_name="decoder input")

        # キャッシュの準備
        if cache is None:
            cache = [None] * len(self.layers)
        else:
            # キャッシュサイズの調整
            if len(cache) < len(self.layers):
                cache = cache + [None] * (len(self.layers) - len(cache))
            elif len(cache) > len(self.layers):
                cache = cache[:len(self.layers)]

        # 位置エンコーディングのオフセットを計算
        # キャッシュが存在する場合、既に処理済みのトークン数を取得
        pos_offset = 0
        if len(cache) > 0 and cache[0] is not None:
            # 最初のレイヤーのキャッシュから既に処理済みのシーケンス長を取得
            layer_cache = cache[0]
            if isinstance(layer_cache, dict) and 'self_k' in layer_cache:
                cached_k = layer_cache['self_k']
                if cached_k is not None and (cached_k.dim() == 3 or cached_k.dim() >= 3):
                    # cached_kの形状: [batch_size, cached_seq_len, d_model]
                    pos_offset = cached_k.size(1)

        # 埋め込みと位置エンコーディング
        x = self.embedding(x)
        x = self.pos_encoding(x, offset=pos_offset)
        x = self.dropout(x)

        # 新しいキャッシュを格納するリスト
        new_cache = []

        # デコーダーレイヤーの適用
        for i, layer in enumerate(self.layers):
            layer_cache = cache[i]
            x, layer_new_cache = layer(x, encoder_output, tgt_mask, src_mask, layer_cache)
            new_cache.append(layer_new_cache)

        # 最終正規化と出力層
        x = self.norm(x)
        output = self.output_layer(x)

        return output, new_cache
