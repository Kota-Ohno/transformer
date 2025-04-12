import torch
import torch.nn as nn
import math
from config import CONFIG, MODEL_CONFIG

# キャッシュサイズの最大値
MAX_CACHE_ENTRIES = 100

class ScaledDotProductAttention(nn.Module):
    def __init__(self, d_k, dropout=0.1):
        super(ScaledDotProductAttention, self).__init__()
        self.d_k = d_k
        self.dropout = nn.Dropout(dropout)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, q, k, v, mask=None):
        """
        Args:
            q (torch.Tensor): (..., seq_len_q, d_k)
            k (torch.Tensor): (..., seq_len_k, d_k)
            v (torch.Tensor): (..., seq_len_v, d_v) seq_len_k == seq_len_v
            mask (torch.Tensor, optional): ブロードキャスト可能な形状 (..., seq_len_q, seq_len_k)
                                          マスクする箇所が False (0)
        Returns:
            output (torch.Tensor): (..., seq_len_q, d_v)
            attn_weights (torch.Tensor): (..., seq_len_q, seq_len_k)
        """
        # スコア計算 q * k^T / sqrt(d_k)
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        # attn_scores の形状: (..., seq_len_q, seq_len_k)

        if mask is not None:
            # mask が False (0) の位置を -65504.0 にする（混合精度トレーニングのため）
            attn_scores = attn_scores.masked_fill(mask == 0, -65504.0)

        attn_weights = self.softmax(attn_scores)
        attn_weights = self.dropout(attn_weights)

        output = torch.matmul(attn_weights, v)
        return output, attn_weights

class MultiHeadAttention(nn.Module):
    """
    マルチヘッドアテンション機構を実装するクラス。

    複数の注意ヘッドを用いて、入力の異なる表現を学習します。

    Attributes:
        num_heads (int): アテンションヘッドの数
        d_k (int): キーの次元数
        d_v (int): 値の次元数
        d_model (int): モデルの次元数
        q_linear (nn.Linear): クエリの線形変換層
        k_linear (nn.Linear): キーの線形変換層
        v_linear (nn.Linear): 値の線形変換層
        scaled_dot_product_attention (ScaledDotProductAttention): スケールドドットプロダクトアテンション
        final_linear (nn.Linear): 最終的な線形変換層

    Args:
        d_model (int): モデルの次元数
        d_k (int): キーの次元数
        d_v (int): 値の次元数
        num_heads (int): アテンションヘッドの数
    """

    def __init__(self, d_model: int, d_k: int, d_v: int, num_heads: int):
        super(MultiHeadAttention, self).__init__()
        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")

        self.num_heads = num_heads
        self.d_k = d_k
        self.d_v = d_v
        self.d_model = d_model

        self.q_linear = nn.Linear(d_model, num_heads * d_k)
        self.k_linear = nn.Linear(d_model, num_heads * d_k)
        self.v_linear = nn.Linear(d_model, num_heads * d_v)
        self.scaled_dot_product_attention = ScaledDotProductAttention(d_k=d_k)
        self.final_linear = nn.Linear(num_heads * d_v, d_model)

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor = None,
                cached_k: torch.Tensor = None, cached_v: torch.Tensor = None, return_cache: bool = False) -> torch.Tensor:
        """
        マルチヘッドアテンションの計算を行います。

        Args:
            q (torch.Tensor): クエリテンソル (batch_size, seq_len_q, d_model)
            k (torch.Tensor): キーテンソル (batch_size, seq_len_k, d_model)
            v (torch.Tensor): 値テンソル (batch_size, seq_len_v, d_model)
            mask (torch.Tensor): マスクテンソル (batch_size, 1, seq_len_q, seq_len_k)
            cached_k (torch.Tensor, optional): キャッシュされたキー
            cached_v (torch.Tensor, optional): キャッシュされた値
            return_cache (bool): キャッシュを返すかどうか

        Returns:
            torch.Tensor または tuple: アテンション出力とオプションでキャッシュ
        """
        batch_size = q.size(0)

        # 1. 線形変換
        q = self.q_linear(q)

        # キャッシュがない場合のみキーと値を計算
        k = self.k_linear(k) if cached_k is None else cached_k
        v = self.v_linear(v) if cached_v is None else cached_v

        # 形状変換: (batch_size, seq_len, num_heads * d_k) -> (batch_size, seq_len, num_heads, d_k)
        q = q.view(batch_size, -1, self.num_heads, self.d_k)

        # キャッシュがない場合のみ形状変換
        if cached_k is None:
            k = k.view(batch_size, -1, self.num_heads, self.d_k)
        if cached_v is None:
            v = v.view(batch_size, -1, self.num_heads, self.d_v)

        # (batch_size, seq_len, num_heads, d_k) -> (batch_size, num_heads, seq_len, d_k)
        q = q.transpose(1, 2)

        # キャッシュがない場合のみ転置
        if cached_k is None:
            k = k.transpose(1, 2)
        if cached_v is None:
            v = v.transpose(1, 2)

        # 2. スケールドドットプロダクトアテンション
        scaled_attention, attn_weights = self.scaled_dot_product_attention(q, k, v, mask)
        # scaled_attention: (batch_size, num_heads, seq_len_q, d_v)

        # 3. ヘッドを連結
        # (batch_size, num_heads, seq_len_q, d_v) -> (batch_size, seq_len_q, num_heads, d_v)
        scaled_attention = scaled_attention.transpose(1, 2).contiguous()

        # (batch_size, seq_len_q, num_heads, d_v) -> (batch_size, seq_len_q, num_heads * d_v)
        concat_attention = scaled_attention.view(batch_size, -1, self.num_heads * self.d_v)

        # 4. 最終的な線形変換
        output = self.final_linear(concat_attention)

        if return_cache:
            # キャッシュがない場合は新しく作る
            cached_k = k if cached_k is None else cached_k
            cached_v = v if cached_v is None else cached_v
            return output, attn_weights, cached_k, cached_v

        return output

# キャッシュ管理の補助関数
def prune_cache(cache, max_entries=MAX_CACHE_ENTRIES):
    """
    キャッシュサイズが指定された最大値を超えた場合に古いエントリを削除します。

    Args:
        cache (dict): 管理対象のキャッシュ辞書
        max_entries (int): 許容される最大エントリ数

    Returns:
        dict: サイズ調整後のキャッシュ
    """
    if len(cache) <= max_entries:
        return cache

    # 最も古いエントリを削除（ここではシンプルに最初のn個を削除）
    num_to_remove = len(cache) - max_entries
    if num_to_remove <= 0:
        return cache

    # キーのリストを取得
    keys = list(cache.keys())

    # 最初のnum_to_remove個のキーを削除
    for key in keys[:num_to_remove]:
        del cache[key]

    return cache
