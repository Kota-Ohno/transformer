import torch
import torch.nn as nn
from config import CONFIG, LAYER_DROPOUT
from torch.utils.checkpoint import checkpoint
import logging
from typing import Optional, Tuple

# 高度なレイヤーモジュールをインポート
from advanced_layers import RelativeMultiHeadAttention, EnhancedFeedForward

class EnhancedEncoderLayer(nn.Module):
    """
    相対位置エンコーディングとGLUを使用した強化版エンコーダーレイヤー。

    Attributes:
        self_attention (RelativeMultiHeadAttention): 自己アテンション
        feed_forward (EnhancedFeedForward): 強化版フィードフォワードネットワーク
        norm1 (nn.LayerNorm): 第1層のレイヤー正規化
        dropout (nn.Dropout): ドロップアウト

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

        # レイヤー正規化
        self.norm1 = nn.LayerNorm(d_model)

        # 強化版フィードフォワードネットワーク（すでにPre-LN方式実装済み）
        self.feed_forward = EnhancedFeedForward(
            d_model=d_model,
            d_ff=d_ff,
            dropout=dropout
        )

        # ドロップアウト
        self.dropout = nn.Dropout(dropout)

    def _adjust_mask(self, mask: Optional[torch.Tensor], batch_size: int, seq_len: int) -> Optional[torch.Tensor]:
        """
        マスクの形状を調整する

        Args:
            mask (torch.Tensor): 調整するマスク
            batch_size (int): バッチサイズ
            seq_len (int): シーケンス長

        Returns:
            torch.Tensor: 調整されたマスク
        """
        if mask is None:
            return None

        # マスクの形状を確認
        if mask.dim() == 2:
            # (batch_size, seq_len) → (batch_size, 1, 1, seq_len)
            mask = mask.unsqueeze(1).unsqueeze(1)
        elif mask.dim() == 3:
            # (batch_size, seq_len, seq_len) → (batch_size, 1, seq_len, seq_len)
            mask = mask.unsqueeze(1)

        # シーケンス長が一致しない場合は調整
        if mask.size(-1) != seq_len or mask.size(-2) != seq_len:
            logging.warning(f"マスクサイズ不一致: {mask.size()}, 期待値: (batch_size, 1, {seq_len}, {seq_len})")

            # 新しいマスクを作成
            new_mask = torch.ones(batch_size, 1, seq_len, seq_len, device=mask.device)

            # 共通部分のコピー
            min_seq = min(mask.size(-1), seq_len)
            new_mask[:, :, :min_seq, :min_seq] = mask[:, :, :min_seq, :min_seq]

            return new_mask

        return mask

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        エンコーダーレイヤーの順伝播

        Args:
            x (torch.Tensor): 入力テンソル (batch_size, seq_len, d_model)
            mask (torch.Tensor, optional): アテンションマスク

        Returns:
            torch.Tensor: 出力テンソル (batch_size, seq_len, d_model)
        """
        batch_size, seq_len = x.size(0), x.size(1)

        # マスクの調整
        if mask is not None:
            mask = self._adjust_mask(mask, batch_size, seq_len)

        # マルチヘッドアテンション（Pre-LN方式）
        residual = x
        x_norm = self.norm1(x)
        attn_output = self.self_attention(x_norm, x_norm, x_norm, mask)
        x = residual + self.dropout(attn_output)

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
        self.max_seq_length = CONFIG.get("MAX_SEQ_LENGTH", 512)

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

        logging.info(f"EnhancedEncoder initialized: {num_layers} layers, {hidden_dim} hidden dim, {num_heads} heads")

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
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
        if x.size(1) > self.max_seq_length:
            logging.warning(f"シーケンス長を制限します: {x.size(1)} → {self.max_seq_length}")
            x = x[:, :self.max_seq_length, :]
            if mask is not None:
                # マスクも調整
                mask = mask[:, :self.max_seq_length]

        # レイヤードロップアウトの準備
        layer_dropout = LAYER_DROPOUT if self.training else 0.0
        dropout_probs = torch.empty(len(self.layers)).uniform_()

        # エンコーダーレイヤーの適用
        for i, layer in enumerate(self.layers):
            # レイヤードロップアウト - 訓練時にランダムに層をスキップ
            if self.training and dropout_probs[i] < layer_dropout:
                continue

            # チェックポイント使用時はメモリ効率を向上
            if self.use_checkpointing and self.training:
                x = checkpoint(layer, x, mask)
            else:
                x = layer(x, mask)

        # 最終的なレイヤー正規化
        x = self.norm(x)

        return x
