import torch
import torch.nn as nn
import math
import numpy as np
import logging
from typing import Dict, Tuple, Optional

# 共通設定
MAX_CACHE_ENTRIES = 100


class GatedLinearUnit(nn.Module):
    """
    Gated Linear Unit (GLU) を実装するクラス。

    Dauphin et al. (2017) "Language Modeling with Gated Convolutional Networks" で提案されたモジュール。
    Transformerのフィードフォワードネットワークの代わりに使用でき、
    特にシーケンスモデリングタスクでの性能向上が報告されています。

    Attributes:
        d_model (int): 入出力の次元数
        d_ff (int): 中間層の次元数
        dropout (float): ドロップアウト率

    Args:
        d_model (int): 入出力の次元数
        d_ff (int): 中間層の次元数（デフォルトはd_model * 4）
        dropout (float): ドロップアウト率
    """
    def __init__(self, d_model, d_ff=None, dropout=0.1):
        super(GatedLinearUnit, self).__init__()
        if d_ff is None:
            d_ff = 4 * d_model  # デフォルト値

        # GLUのための2つの線形投影
        self.linear_value = nn.Linear(d_model, d_ff)
        self.linear_gate = nn.Linear(d_model, d_ff)

        # 出力投影
        self.output_proj = nn.Linear(d_ff, d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        Gated Linear Unitの順伝播

        Args:
            x (torch.Tensor): 入力テンソル (batch_size, seq_len, d_model)

        Returns:
            torch.Tensor: 出力テンソル (batch_size, seq_len, d_model)
        """
        # 値と門の線形投影
        value = self.linear_value(x)
        gate = self.linear_gate(x)

        # GLUの計算：値にシグモイド活性化した門を掛ける
        gated_output = value * torch.sigmoid(gate)

        # 出力投影とドロップアウト
        output = self.output_proj(gated_output)
        output = self.dropout(output)

        return output


class RelativeMultiHeadAttention(nn.Module):
    """
    相対位置エンコーディングを組み込んだマルチヘッドアテンション。

    Attributes:
        d_model (int): モデルの次元数
        num_heads (int): アテンションヘッドの数
        d_k (int): キーの次元数
        q_linear (nn.Linear): クエリの線形変換層
        k_linear (nn.Linear): キーの線形変換層
        v_linear (nn.Linear): 値の線形変換層
        rel_pos_enc (RelativePositionalEncoding): 相対位置エンコーディング
        pos_proj (nn.Linear): 位置エンコーディングの投影層
        final_linear (nn.Linear): 最終的な線形変換層

    Args:
        d_model (int): モデルの次元数
        num_heads (int): アテンションヘッドの数
        dropout (float): ドロップアウト率
        max_dist (int): 相対位置エンコーディングの最大距離
    """
    def __init__(self, d_model, num_heads, dropout=0.1, max_dist=64):
        super(RelativeMultiHeadAttention, self).__init__()

        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        # 線形変換層
        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.output_linear = nn.Linear(d_model, d_model)

        # 相対位置エンコーディング
        self.rel_pos_enc = nn.Parameter(torch.randn(max_dist * 2 + 1, self.d_k))

        # ドロップアウト
        self.dropout = nn.Dropout(dropout)

        # スケーリング係数
        self.scale = 1.0 / math.sqrt(self.d_k)

    def _split_heads(self, x):
        """入力テンソルをヘッドごとに分割"""
        batch_size = x.size(0)
        x = x.view(batch_size, -1, self.num_heads, self.d_k)
        return x.transpose(1, 2)  # (batch_size, num_heads, seq_len, d_k)

    def _combine_heads(self, x):
        """ヘッドを結合"""
        batch_size = x.size(0)
        x = x.transpose(1, 2)  # (batch_size, seq_len, num_heads, d_k)
        return x.contiguous().view(batch_size, -1, self.d_model)

    def _relative_position_to_absolute_position(self, x):
        """相対位置を絶対位置に変換"""
        batch_size, num_heads, seq_len_q, seq_len_k = x.size()

        # パディングを追加
        col_pad = torch.zeros((batch_size, num_heads, seq_len_q, 1), device=x.device)
        x = torch.cat([x, col_pad], dim=-1)

        flat_x = x.view(batch_size, num_heads, seq_len_q, -1)
        flat_pad = torch.zeros((batch_size, num_heads, seq_len_q, seq_len_q-1), device=x.device)
        flat_x_padded = torch.cat([flat_x, flat_pad], dim=-1)

        final_x = flat_x_padded.view(batch_size, num_heads, seq_len_q + 1, seq_len_q)
        final_x = final_x[:, :, 1:]
        return final_x[:, :, :, :seq_len_k]

    def forward(self, q, k, v, mask=None, cached_k=None, cached_v=None, return_cache=False):
        batch_size = q.size(0)
        seq_len_q = q.size(1)
        seq_len_k = k.size(1)

        # キャッシュの使用
        if cached_k is not None:
            k = cached_k
        if cached_v is not None:
            v = cached_v

        # 線形変換とヘッドの分割
        q = self._split_heads(self.q_linear(q))  # (batch_size, num_heads, seq_len_q, d_k)
        k = self._split_heads(self.k_linear(k))  # (batch_size, num_heads, seq_len_k, d_k)
        v = self._split_heads(self.v_linear(v))  # (batch_size, num_heads, seq_len_k, d_k)

        # コンテンツベースのアテンション
        content_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        # 相対位置エンコーディングの適用
        q_with_bias_u = q  # コンテンツベースのクエリ
        q_with_bias_v = q  # 位置ベースのクエリ

        # 相対位置の計算
        position_bias = self._relative_position_to_absolute_position(
            torch.matmul(q_with_bias_v, self.rel_pos_enc.transpose(0, 1))
        )

        # スコアの組み合わせ
        attention_scores = content_scores + position_bias

        # マスクの適用
        if mask is not None:
            attention_scores = attention_scores.masked_fill(mask == 0, -1e9)

        # アテンション重みの計算
        attention_weights = torch.softmax(attention_scores, dim=-1)
        attention_weights = self.dropout(attention_weights)

        # 値との積
        context = torch.matmul(attention_weights, v)  # (batch_size, num_heads, seq_len_q, d_k)

        # ヘッドの結合
        context = self._combine_heads(context)  # (batch_size, seq_len_q, d_model)

        # 最終的な線形変換
        output = self.output_linear(context)

        if return_cache:
            return output, attention_weights, k, v
        return output


class EnhancedFeedForward(nn.Module):
    """
    強化版フィードフォワードネットワーク。
    GLUとGELUを使用して性能を向上させます。

    Attributes:
        glu (GatedLinearUnit): ゲート付き線形ユニット
        layer_norm (nn.LayerNorm): レイヤー正規化

    Args:
        d_model (int): モデルの次元数
        d_ff (int): 中間層の次元数
        dropout (float): ドロップアウト率
    """
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(EnhancedFeedForward, self).__init__()
        self.glu = GatedLinearUnit(d_model, d_ff, dropout)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, x):
        """
        強化版フィードフォワードネットワークの順伝播

        Args:
            x (torch.Tensor): 入力テンソル (batch_size, seq_len, d_model)

        Returns:
            torch.Tensor: 出力テンソル (batch_size, seq_len, d_model)
        """
        # 前置レイヤー正規化（Pre-LN方式）
        x_norm = self.layer_norm(x)

        # GLUの適用
        output = self.glu(x_norm)

        # 残差接続
        return x + output
