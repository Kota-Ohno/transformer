import torch
import torch.nn as nn
from config import LAYER_DROPOUT
from torch.utils.checkpoint import checkpoint

# 高度なレイヤーモジュールをインポート
from advanced_layers import RelativeMultiHeadAttention, EnhancedFeedForward

class EnhancedEncoderLayer(nn.Module):
    """
    相対位置エンコーディングとGLUを使用した改良版エンコーダーレイヤー。

    Attributes:
        self_attention (RelativeMultiHeadAttention): 相対位置エンコーディングを使用したマルチヘッドアテンション
        feed_forward (EnhancedFeedForward): 強化版フィードフォワードネットワーク
        norm1 (nn.LayerNorm): 第1層のレイヤー正規化
        norm2 (nn.LayerNorm): 第2層のレイヤー正規化
        dropout (nn.Dropout): ドロップアウト層

    Args:
        d_model (int): モデルの次元数
        num_heads (int): アテンションヘッドの数
        d_ff (int): フィードフォワード層の次元数
        dropout (float): ドロップアウト率
        max_dist (int): 相対位置エンコーディングの最大距離
    """
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float, max_dist: int = 64):
        super(EnhancedEncoderLayer, self).__init__()

        # 相対位置エンコーディングを使用したマルチヘッドアテンション
        self.self_attention = RelativeMultiHeadAttention(
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

        # ドロップアウト
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
        エンコーダーレイヤーの順伝播

        Args:
            x (torch.Tensor): 入力テンソル (batch_size, seq_len, d_model)
            mask (torch.Tensor, optional): アテンションマスク

        Returns:
            torch.Tensor: 出力テンソル (batch_size, seq_len, d_model)
        """
        # マルチヘッドアテンション（Pre-LN方式）
        residual = x
        x = self.norm1(x)
        x = self.self_attention(x, x, x, mask)
        x = self.dropout(x)
        x = residual + x

        # フィードフォワードネットワーク（すでにPre-LN方式が実装されている）
        x = self.feed_forward(x)

        return x


class EnhancedEncoder(nn.Module):
    """
    相対位置エンコーディングとGLUを使用した改良版エンコーダー。

    Attributes:
        device (str): 使用するデバイス
        embedding (nn.Embedding): 入力埋め込み層
        layers (nn.ModuleList): エンコーダーレイヤーのリスト
        norm (nn.LayerNorm): 最終的なレイヤー正規化
        dropout (nn.Dropout): ドロップアウト層
        use_checkpointing (bool): チェックポイントを使用するかどうか

    Args:
        input_dim (int): 入力ボキャブラリーサイズ
        hidden_dim (int): 隠れ層の次元数
        num_heads (int): アテンションヘッドの数
        num_layers (int): エンコーダーレイヤーの数
        ff_dim (int): フィードフォワード層の次元数
        dropout (float): ドロップアウト率
        device (str): 使用するデバイス
        max_dist (int): 相対位置エンコーディングの最大距離
    """
    def __init__(self, input_dim: int, hidden_dim: int, num_heads: int, num_layers: int,
                 ff_dim: int, dropout: float, device: str, max_dist: int = 64):
        super(EnhancedEncoder, self).__init__()

        self.device = device
        self.use_checkpointing = True  # チェックポイントを使用して省メモリ化

        # 入力埋め込み層
        self.embedding = nn.Embedding(input_dim, hidden_dim, padding_idx=0)

        # パディングトークンのエンベディングをゼロに初期化
        with torch.no_grad():
            self.embedding.weight[0].fill_(0)

        # エンコーダーレイヤー
        self.layers = nn.ModuleList([
            EnhancedEncoderLayer(
                d_model=hidden_dim,
                num_heads=num_heads,
                d_ff=ff_dim,
                dropout=dropout,
                max_dist=max_dist
            ) for _ in range(num_layers)
        ])

        # 最終的なレイヤー正規化
        self.norm = nn.LayerNorm(hidden_dim)

        # ドロップアウト
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
        エンコーダーの順伝播

        Args:
            x (torch.Tensor): 入力テンソル (batch_size, seq_len)
            mask (torch.Tensor, optional): アテンションマスク

        Returns:
            torch.Tensor: エンコードされた出力 (batch_size, seq_len, hidden_dim)
        """
        # 入力の検証（範囲外のインデックスを防止）
        x = torch.clamp(x, 0, self.embedding.num_embeddings - 1)

        # 入力埋め込み
        x = self.embedding(x)
        x = self.dropout(x)

        # シーケンス長を制限（メモリ効率化のため）
        max_seq_length = 128
        if x.size(1) > max_seq_length:
            x = x[:, :max_seq_length, :]
            if mask is not None:
                # マスクも調整
                mask = mask[:, :, :max_seq_length, :max_seq_length]

        # レイヤードロップアウトの準備
        layer_dropout = LAYER_DROPOUT if self.training else 0.0
        dropout_probs = torch.empty(len(self.layers)).uniform_()

        # エンコーダーレイヤーの適用
        if self.use_checkpointing and self.training:
            # チェックポイントを使用してメモリ効率を向上
            for i, layer in enumerate(self.layers):
                # レイヤードロップアウト - 訓練時にランダムに層をスキップ
                if dropout_probs[i] < layer_dropout:
                    continue

                # PyTorchのcheckpoint機能を使用
                def custom_forward(module, input_x, input_mask):
                    return module(input_x, input_mask)

                x = checkpoint(custom_forward, layer, x, mask)
        else:
            # 通常の順伝播
            for i, layer in enumerate(self.layers):
                # レイヤードロップアウト - 訓練時にランダムに層をスキップ
                if self.training and dropout_probs[i] < layer_dropout:
                    continue

                x = layer(x, mask)

        # 最終的なレイヤー正規化
        x = self.norm(x)

        return x
