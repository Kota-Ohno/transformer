import torch
import torch.nn as nn
from config import LAYER_DROPOUT
from typing import List, Tuple, Dict
from torch.utils.checkpoint import checkpoint

# 高度なレイヤーモジュールをインポート
from advanced_layers import RelativeMultiHeadAttention, EnhancedFeedForward

class EnhancedDecoderLayer(nn.Module):
    """
    相対位置エンコーディングとGLUを使用した改良版デコーダーレイヤー。

    Attributes:
        self_attention (RelativeMultiHeadAttention): 自己アテンション
        cross_attention (RelativeMultiHeadAttention): エンコーダー出力に対するクロスアテンション
        feed_forward (EnhancedFeedForward): 強化版フィードフォワードネットワーク
        norm1 (nn.LayerNorm): 第1層のレイヤー正規化
        norm2 (nn.LayerNorm): 第2層のレイヤー正規化
        norm3 (nn.LayerNorm): 第3層のレイヤー正規化
        dropout1 (nn.Dropout): 第1層のドロップアウト
        dropout2 (nn.Dropout): 第2層のドロップアウト

    Args:
        d_model (int): モデルの次元数
        num_heads (int): アテンションヘッドの数
        d_ff (int): フィードフォワード層の次元数
        dropout (float): ドロップアウト率
        max_dist (int): 相対位置エンコーディングの最大距離
    """
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float, max_dist: int = 64):
        super(EnhancedDecoderLayer, self).__init__()

        # 自己アテンション（相対位置エンコーディング使用）
        self.self_attention = RelativeMultiHeadAttention(
            d_model=d_model,
            num_heads=num_heads,
            dropout=dropout,
            max_dist=max_dist
        )

        # クロスアテンション（エンコーダー出力に対するアテンション）
        self.cross_attention = RelativeMultiHeadAttention(
            d_model=d_model,
            num_heads=num_heads,
            dropout=dropout,
            max_dist=max_dist
        )

        # 強化版フィードフォワードネットワーク
        self.feed_forward = EnhancedFeedForward(
            d_model=d_model,
            d_ff=d_ff,
            dropout=dropout
        )

        # レイヤー正規化
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

        # ドロップアウト
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, encoder_output: torch.Tensor,
                self_attn_mask: torch.Tensor = None, cross_attn_mask: torch.Tensor = None,
                cache: Dict[str, torch.Tensor] = None) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        デコーダーレイヤーの順伝播

        Args:
            x (torch.Tensor): 入力テンソル (batch_size, tgt_len, d_model)
            encoder_output (torch.Tensor): エンコーダー出力 (batch_size, src_len, d_model)
            self_attn_mask (torch.Tensor, optional): 自己アテンションマスク
            cross_attn_mask (torch.Tensor, optional): クロスアテンションマスク
            cache (Dict[str, torch.Tensor], optional): キャッシュ

        Returns:
            Tuple[torch.Tensor, Dict[str, torch.Tensor]]: 出力テンソルとキャッシュ
        """
        # キャッシュの初期化
        if cache is None:
            cache = {}

        # 自己アテンション（Pre-LN方式）
        residual = x
        x = self.norm1(x)

        # キャッシュからキーと値を取得
        self_attn_kwargs = {}
        if 'self_k' in cache and 'self_v' in cache:
            self_attn_kwargs['cached_k'] = cache['self_k']
            self_attn_kwargs['cached_v'] = cache['self_v']

        # 自己アテンション計算（キャッシュ付き）
        self_attn_output, self_attn_weights, new_self_k, new_self_v = self.self_attention(
            q=x, k=x, v=x,
            mask=self_attn_mask,
            return_cache=True,
            **self_attn_kwargs
        )

        # キャッシュを更新
        cache['self_k'] = new_self_k
        cache['self_v'] = new_self_v

        # 残差接続
        x = residual + self.dropout1(self_attn_output)

        # クロスアテンション（Pre-LN方式）
        residual = x
        x = self.norm2(x)

        # エンコーダー出力に対するアテンションのキャッシュ
        cross_attn_kwargs = {}
        if 'cross_k' in cache and 'cross_v' in cache:
            cross_attn_kwargs['cached_k'] = cache['cross_k']
            cross_attn_kwargs['cached_v'] = cache['cross_v']
        else:
            # 初めてクロスアテンションを計算する場合
            cross_output, cross_attn_weights, cross_k, cross_v = self.cross_attention(
                q=x, k=encoder_output, v=encoder_output,
                mask=cross_attn_mask,
                return_cache=True
            )
            cache['cross_k'] = cross_k
            cache['cross_v'] = cross_v
            cross_attn_kwargs['cached_k'] = cross_k
            cross_attn_kwargs['cached_v'] = cross_v

        # クロスアテンション計算
        cross_output, _, _, _ = self.cross_attention(
            q=x, k=encoder_output, v=encoder_output,
            mask=cross_attn_mask,
            return_cache=True,
            **cross_attn_kwargs
        )

        # 残差接続
        x = residual + self.dropout2(cross_output)

        # フィードフォワードネットワーク（すでにPre-LN方式が組み込まれている）
        x = self.feed_forward(x)

        return x, cache


class EnhancedDecoder(nn.Module):
    """
    相対位置エンコーディングとGLUを使用した改良版デコーダー。

    Attributes:
        device (str): 使用するデバイス
        embedding (nn.Embedding): 入力埋め込み層
        layers (nn.ModuleList): デコーダーレイヤーのリスト
        norm (nn.LayerNorm): 最終的なレイヤー正規化
        output_layer (nn.Linear): 出力層
        dropout (nn.Dropout): ドロップアウト層
        use_checkpointing (bool): チェックポイントを使用するかどうか

    Args:
        output_dim (int): 出力ボキャブラリーサイズ
        hidden_dim (int): 隠れ層の次元数
        num_heads (int): アテンションヘッドの数
        num_layers (int): デコーダーレイヤーの数
        ff_dim (int): フィードフォワード層の次元数
        output_dim (int): 出力層の次元数
        dropout (float): ドロップアウト率
        device (str): 使用するデバイス
        max_dist (int): 相対位置エンコーディングの最大距離
    """
    def __init__(self, vocab_dim: int, hidden_dim: int, num_heads: int, num_layers: int,
                 ff_dim: int, output_dim: int, dropout: float, device: str, max_dist: int = 64):
        super(EnhancedDecoder, self).__init__()

        self.device = device
        self.use_checkpointing = True

        # 入力埋め込み層
        self.embedding = nn.Embedding(vocab_dim, hidden_dim, padding_idx=0)

        # パディングトークンのエンベディングをゼロに初期化
        with torch.no_grad():
            self.embedding.weight[0].fill_(0)

        # デコーダーレイヤー
        self.layers = nn.ModuleList([
            EnhancedDecoderLayer(
                d_model=hidden_dim,
                num_heads=num_heads,
                d_ff=ff_dim,
                dropout=dropout,
                max_dist=max_dist
            ) for _ in range(num_layers)
        ])

        # 最終的なレイヤー正規化
        self.norm = nn.LayerNorm(hidden_dim)

        # 出力層
        self.output_layer = nn.Linear(hidden_dim, output_dim)

        # ドロップアウト
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, encoder_output: torch.Tensor,
                self_attn_mask: torch.Tensor = None, cross_attn_mask: torch.Tensor = None,
                cache: List[Dict[str, torch.Tensor]] = None) -> Tuple[torch.Tensor, List[Dict[str, torch.Tensor]]]:
        """
        デコーダーの順伝播

        Args:
            x (torch.Tensor): 入力テンソル (batch_size, tgt_len)
            encoder_output (torch.Tensor): エンコーダー出力 (batch_size, src_len, hidden_dim)
            self_attn_mask (torch.Tensor, optional): 自己アテンションマスク
            cross_attn_mask (torch.Tensor, optional): クロスアテンションマスク
            cache (List[Dict[str, torch.Tensor]], optional): キャッシュ

        Returns:
            Tuple[torch.Tensor, List[Dict[str, torch.Tensor]]]: 出力テンソルとキャッシュ
        """
        # 入力の検証
        if (x >= self.embedding.num_embeddings).any():
            raise ValueError("インデックスがエンベディングテーブルの範囲外です")

        # 入力埋め込み
        x = self.embedding(x)
        x = self.dropout(x)

        # キャッシュの初期化
        new_cache = []
        if cache is None:
            cache = [None] * len(self.layers)

        # レイヤードロップの準備
        layer_dropout = LAYER_DROPOUT if self.training else 0.0
        dropout_probs = torch.empty(len(self.layers)).uniform_()

        # チェックポイントを使用する場合（訓練時のみ、キャッシュなし）
        if self.use_checkpointing and self.training and cache[0] is None:
            for i, layer in enumerate(self.layers):
                # レイヤードロップアウト
                if dropout_probs[i] < layer_dropout:
                    new_cache.append(None)
                    continue

                # チェックポイント対応のカスタム関数
                def custom_forward(module, input_x, enc_out, s_mask, c_mask):
                    return module(input_x, enc_out, s_mask, c_mask, None)[0]

                # チェックポイントを使用
                x = checkpoint(
                    custom_forward,
                    layer, x, encoder_output, self_attn_mask, cross_attn_mask
                )
                new_cache.append(None)  # 訓練中はキャッシュを使用しない
        else:
            # 通常の順伝播（評価時またはキャッシュ使用時）
            for i, layer in enumerate(self.layers):
                # レイヤードロップアウト（訓練時のみ、キャッシュなし）
                if cache[0] is None and dropout_probs[i] < layer_dropout and self.training:
                    new_cache.append(None)
                    continue

                # レイヤーの順伝播
                x, layer_cache = layer(
                    x, encoder_output, self_attn_mask, cross_attn_mask, cache=cache[i]
                )
                new_cache.append(layer_cache)

        # 最終的なレイヤー正規化
        x = self.norm(x)

        # 出力層
        output = self.output_layer(x)

        return output, new_cache
