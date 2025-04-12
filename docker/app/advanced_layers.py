import torch
import torch.nn as nn
import math
import numpy as np
import logging
from typing import Dict, Tuple, Optional

# 共通設定
MAX_CACHE_ENTRIES = 100

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GatedLinearUnit(nn.Module):
    """
    Gated Linear Unitを実装したクラス。
    GLUは通常のフィードフォワードネットワークよりも表現力が高いとされる。

    Attributes:
        fc1 (nn.Linear): 入力をより高次元に変換する線形層
        fc2 (nn.Linear): ゲート制御のための線形層
        dropout (nn.Dropout): 過学習を防ぐためのドロップアウト層

    Args:
        d_model (int): 入力の次元数
        d_ff (int, optional): 中間層の次元数（デフォルトはd_modelの4倍）
        dropout (float, optional): ドロップアウト率（デフォルトは0.1）
    """
    def __init__(self, d_model: int, d_ff: Optional[int] = None, dropout: float = 0.1):
        super(GatedLinearUnit, self).__init__()
        if d_ff is None:
            d_ff = 4 * d_model  # 通常はモデル次元の4倍

        # 線形変換
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_model, d_ff)

        # 出力層
        self.output = nn.Linear(d_ff, d_model)

        # ドロップアウト
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        GLUの順伝播

        Args:
            x (torch.Tensor): 入力テンソル [batch_size, seq_len, d_model]

        Returns:
            torch.Tensor: GLU適用後のテンソル [batch_size, seq_len, d_model]
        """
        # 活性化関数のないfc1出力とGELU活性化fc2出力の要素積
        gelu_output = torch.nn.functional.gelu(self.fc2(x))
        linear_output = self.fc1(x)
        gated_output = linear_output * gelu_output

        # 出力層と残差接続
        output = self.output(gated_output)
        output = self.dropout(output)

        return output


class RelativeMultiHeadAttention(nn.Module):
    """
    相対位置エンコーディングを使用したマルチヘッドアテンション。
    標準的なMultiHeadAttentionに対して、トークン間の相対的な位置情報を考慮できる。

    Attributes:
        d_model (int): モデルの次元数
        num_heads (int): アテンションヘッドの数
        d_k (int): 各ヘッドの次元数（d_model / num_heads）
        max_relative_position (int): 相対位置エンコーディングの最大距離

    Args:
        d_model (int): モデルの次元数
        num_heads (int): アテンションヘッドの数
        dropout (float): ドロップアウト率
        max_dist (int): 相対位置エンコーディングの最大距離
    """
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1, max_dist: int = 64):
        super(RelativeMultiHeadAttention, self).__init__()

        # 引数チェック
        assert d_model % num_heads == 0, "d_modelはnum_headsで割り切れる必要があります"

        # クラス変数の設定
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.max_relative_position = max_dist

        # 線形変換層の定義
        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.output_linear = nn.Linear(d_model, d_model)

        # 相対位置エンコーディングの初期化
        self.relative_key_embedding = nn.Parameter(
            torch.zeros(2 * max_dist + 1, self.d_k)
        )
        nn.init.xavier_uniform_(self.relative_key_embedding)

        # ドロップアウト層
        self.dropout = nn.Dropout(dropout)

        # スケーリング係数
        self.scale = 1.0 / math.sqrt(self.d_k)

    def _get_relative_position_bucket(self, relative_position, max_distance):
        """
        相対位置からバケットインデックスを計算

        Args:
            relative_position (torch.Tensor): 相対位置
            max_distance (int): 最大距離

        Returns:
            torch.Tensor: バケットインデックス
        """
        # [-max_distance, max_distance]の範囲にクリップ
        clipped = torch.clamp(relative_position, -max_distance, max_distance)

        # [0, 2*max_distance]の範囲に変換
        return clipped + max_distance

    def _compute_relative_position_matrix(self, length_q, length_k, max_distance):
        """
        相対位置行列を計算

        Args:
            length_q (int): クエリシーケンス長
            length_k (int): キーシーケンス長
            max_distance (int): 最大距離

        Returns:
            torch.Tensor: 相対位置行列 [length_q, length_k]
        """
        # 位置インデックスの作成
        range_q = torch.arange(length_q, device=self.relative_key_embedding.device)
        range_k = torch.arange(length_k, device=self.relative_key_embedding.device)

        # 相対位置の計算 [length_q, length_k]
        distance_matrix = range_q.unsqueeze(1) - range_k.unsqueeze(0)

        # バケットインデックスに変換
        relative_position_bucket = self._get_relative_position_bucket(
            distance_matrix, max_distance
        )

        return relative_position_bucket

    def forward(self, q, k, v, mask=None, cached_k=None, cached_v=None, return_cache=False):
        """
        マルチヘッドアテンションの順伝播処理

        Args:
            q (torch.Tensor): クエリテンソル [batch_size, seq_len_q, d_model]
            k (torch.Tensor): キーテンソル [batch_size, seq_len_k, d_model]
            v (torch.Tensor): 値テンソル [batch_size, seq_len_k, d_model]
            mask (torch.Tensor, optional): アテンションマスク
            cached_k (torch.Tensor, optional): キャッシュされたキー
            cached_v (torch.Tensor, optional): キャッシュされた値
            return_cache (bool): キャッシュを返すかどうか

        Returns:
            torch.Tensor または Tuple: アテンション出力とオプションのキャッシュ
        """
        batch_size = q.size(0)

        # キャッシュの使用
        k = cached_k if cached_k is not None else k
        v = cached_v if cached_v is not None else v

        # シーケンス長の取得
        seq_len_q, seq_len_k = q.size(1), k.size(1)

        # 1. 線形変換
        q = self.q_linear(q)  # [batch_size, seq_len_q, d_model]
        k = self.k_linear(k)  # [batch_size, seq_len_k, d_model]
        v = self.v_linear(v)  # [batch_size, seq_len_k, d_model]

        # 2. ヘッドに分割
        q = q.view(batch_size, seq_len_q, self.num_heads, self.d_k)
        k = k.view(batch_size, seq_len_k, self.num_heads, self.d_k)
        v = v.view(batch_size, seq_len_k, self.num_heads, self.d_k)

        # 3. 転置してヘッド次元を前に
        q = q.transpose(1, 2)  # [batch_size, num_heads, seq_len_q, d_k]
        k = k.transpose(1, 2)  # [batch_size, num_heads, seq_len_k, d_k]
        v = v.transpose(1, 2)  # [batch_size, num_heads, seq_len_k, d_k]

        # 4. コンテンツベースのアテンションスコア計算
        # [batch_size, num_heads, seq_len_q, seq_len_k]
        content_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        # 5. 相対位置エンコーディングの適用
        # 相対位置行列の計算 [seq_len_q, seq_len_k]
        relative_position_bucket = self._compute_relative_position_matrix(
            seq_len_q, seq_len_k, self.max_relative_position
        )

        # 相対位置エンベッディングの取得 [seq_len_q, seq_len_k, d_k]
        relative_position_embeddings = self.relative_key_embedding[relative_position_bucket]

        # クエリと相対位置エンベッディングの内積 [batch_size, num_heads, seq_len_q, seq_len_k]
        relative_scores = torch.einsum('bhqd,qkd->bhqk', q, relative_position_embeddings)

        # 6. コンテンツスコアと相対位置スコアの結合
        attention_scores = content_scores + relative_scores

        # 7. マスクの適用
        if mask is not None:
            # マスクの形状を調整
            if mask.dim() == 2:
                mask = mask.unsqueeze(1).unsqueeze(1)  # [batch_size, 1, 1, seq_len_k]
            elif mask.dim() == 3:
                mask = mask.unsqueeze(1)  # [batch_size, 1, seq_len_q, seq_len_k]

            # マスクを拡張
            mask = mask.expand(-1, self.num_heads, -1, -1)

            # マスク適用（0の部分は-∞に）
            attention_scores = attention_scores.masked_fill(mask == 0, -1e9)

        # 8. ソフトマックスでアテンション重みを計算
        attention_weights = torch.softmax(attention_scores, dim=-1)
        attention_weights = self.dropout(attention_weights)

        # 9. アテンション重みと値の積
        # [batch_size, num_heads, seq_len_q, d_k]
        context = torch.matmul(attention_weights, v)

        # 10. 転置してヘッド次元を元に戻す
        # [batch_size, seq_len_q, num_heads, d_k]
        context = context.transpose(1, 2).contiguous()

        # 11. ヘッドの結合
        # [batch_size, seq_len_q, d_model]
        context = context.view(batch_size, seq_len_q, self.d_model)

        # 12. 最終的な線形変換
        output = self.output_linear(context)

        # キャッシュを返すかどうか
        if return_cache:
            return output, attention_weights, k, v

        return output


class EnhancedFeedForward(nn.Module):
    """
    強化版フィードフォワードネットワーク。
    GLUとLayerNormをPre-LN方式で組み合わせたもの。

    Attributes:
        layer_norm (nn.LayerNorm): レイヤー正規化
        glu (GatedLinearUnit): ゲート付き線形ユニット

    Args:
        d_model (int): モデルの次元数
        d_ff (int): 中間層の次元数
        dropout (float): ドロップアウト率
    """
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super(EnhancedFeedForward, self).__init__()

        # レイヤー正規化（Pre-LN）
        self.layer_norm = nn.LayerNorm(d_model)

        # Gated Linear Unit
        self.glu = GatedLinearUnit(d_model, d_ff, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        フィードフォワードネットワークの順伝播

        Args:
            x (torch.Tensor): 入力テンソル [batch_size, seq_len, d_model]

        Returns:
            torch.Tensor: 出力テンソル [batch_size, seq_len, d_model]
        """
        # レイヤー正規化（Pre-LN方式）
        normed_x = self.layer_norm(x)

        # GLUの適用
        glu_output = self.glu(normed_x)

        # 残差接続
        return x + glu_output
