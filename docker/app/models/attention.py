import torch
import torch.nn as nn
import math

# キャッシュサイズの最大値
MAX_CACHE_ENTRIES = 100

class ScaledDotProductAttention(nn.Module):
    """スケーリングされたドット積アテンション"""

    def __init__(self, dropout=0.1):
        super(ScaledDotProductAttention, self).__init__()
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, mask=None):
        """
        Args:
            q: クエリ [..., seq_len_q, d_k]
            k: キー [..., seq_len_k, d_k]
            v: 値 [..., seq_len_k, d_v]
            mask: マスク [..., seq_len_q, seq_len_k]
        Returns:
            output: アテンション適用後の出力 [..., seq_len_q, d_v]
            attention_weights: アテンションの重み [..., seq_len_q, seq_len_k]
        """
        d_k = q.size(-1)

        # Q・K^T / sqrt(d_k)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)

        # マスク適用（0のところに大きな負の値）
        if mask is not None:
            # dtypeに基づいて安全な負の値を計算
            if scores.dtype == torch.float16:
                # float16の場合は表現可能な範囲を考慮して-1e4を使用
                fill_value = -1e4
            else:
                # その他のdtypeの場合は適度に大きな負の定数を使用（数値安定性のため）
                fill_value = scores.new_tensor(-1e9)
            scores = scores.masked_fill(mask == 0, fill_value)

        # アテンションウェイト計算
        attention_weights = torch.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)

        # V との積
        output = torch.matmul(attention_weights, v)

        return output, attention_weights

class MultiHeadAttention(nn.Module):
    """マルチヘッドアテンション"""

    def __init__(self, d_model, num_heads, dropout=0.1):
        super(MultiHeadAttention, self).__init__()

        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        # 線形変換層
        self.wq = nn.Linear(d_model, d_model)
        self.wk = nn.Linear(d_model, d_model)
        self.wv = nn.Linear(d_model, d_model)
        self.wo = nn.Linear(d_model, d_model)

        self.attention = ScaledDotProductAttention(dropout)

    def split_heads(self, x):
        """入力テンソルをヘッドに分割"""
        batch_size = x.size(0)
        x = x.view(batch_size, -1, self.num_heads, self.d_k)
        return x.transpose(1, 2)

    def combine_heads(self, x):
        """ヘッドを結合"""
        batch_size = x.size(0)
        x = x.transpose(1, 2)
        return x.reshape(batch_size, -1, self.d_model)

    def forward(self, q, k, v, mask=None, cached_k=None, cached_v=None, return_cache=False):
        """
        Args:
            q: クエリ [batch_size, seq_len_q, d_model]
            k: キー [batch_size, seq_len_k, d_model]
            v: 値 [batch_size, seq_len_v, d_model]
            mask: マスク [batch_size, 1, seq_len_q, seq_len_k] or [batch_size, seq_len_q, seq_len_k]
            cached_k: キャッシュされたキー（推論時に使用）
                      Noneでない場合、形状は [batch_size, num_heads, seq_len_k, d_k] である必要があります。
                      キャッシュは線形変換（self.wk）とヘッド分割（split_heads）が既に適用された状態で提供される必要があります。
            cached_v: キャッシュされた値（推論時に使用）
                      Noneでない場合、形状は [batch_size, num_heads, seq_len_v, d_k] である必要があります。
                      キャッシュは線形変換（self.wv）とヘッド分割（split_heads）が既に適用された状態で提供される必要があります。
            return_cache: キャッシュを返すかどうか
        Returns:
            return_cache=Falseの場合:
                output: アテンション出力 [batch_size, seq_len_q, d_model]
            return_cache=Trueの場合:
                (output, attention_weights, k, v) の4タプル:
                - output: アテンション出力 [batch_size, seq_len_q, d_model]
                - attention_weights: アテンションの重み [batch_size, num_heads, seq_len_q, seq_len_k]
                - k: キャッシュされたキー [batch_size, num_heads, seq_len_k, d_k]
                - v: キャッシュされた値 [batch_size, num_heads, seq_len_v, d_k]
        """
        batch_size = q.size(0)

        # 線形変換
        q = self.wq(q)

        # キャッシュの整合性チェック
        if (cached_k is None) != (cached_v is None):
            raise ValueError("cached_k and cached_v must be provided together or both be None")

        # キャッシュを使用するか
        if cached_k is None:
            # キャッシュがない場合、通常の処理（両方の変換を適用）
            k = self.wk(k)
            v = self.wv(v)
        else:
            # キャッシュの形状を検証
            # キャッシュは線形変換とヘッド分割が既に適用された状態である必要がある
            if cached_k.dim() != 4:
                raise ValueError(f"cached_k must be 4-dimensional, got {cached_k.dim()} dimensions")
            if cached_k.size(0) != batch_size:
                raise ValueError(f"cached_k batch size mismatch. Expected {batch_size}, got {cached_k.size(0)}")
            if cached_k.size(1) != self.num_heads:
                raise ValueError(f"cached_k num_heads mismatch. Expected {self.num_heads}, got {cached_k.size(1)}")
            if cached_k.size(3) != self.d_k:
                raise ValueError(f"cached_k d_k mismatch. Expected {self.d_k}, got {cached_k.size(3)}")
            # cached_vの形状も検証
            if cached_v.dim() != 4:
                raise ValueError(f"cached_v must be 4-dimensional, got {cached_v.dim()} dimensions")
            if cached_v.size(0) != batch_size:
                raise ValueError(f"cached_v batch size mismatch. Expected {batch_size}, got {cached_v.size(0)}")
            if cached_v.size(1) != self.num_heads:
                raise ValueError(f"cached_v num_heads mismatch. Expected {self.num_heads}, got {cached_v.size(1)}")
            if cached_v.size(3) != self.d_k:
                raise ValueError(f"cached_v d_k mismatch. Expected {self.d_k}, got {cached_v.size(3)}")
            # シーケンス長の一致を確認
            if cached_k.size(2) != cached_v.size(2):
                raise ValueError(
                    f"cached_k and cached_v sequence length mismatch. "
                    f"cached_k.size(2)={cached_k.size(2)}, cached_v.size(2)={cached_v.size(2)}"
                )
            # 新しいk/vを計算
            new_k = self.wk(k)
            new_v = self.wv(v)
            # 新しいk/vをヘッド分割して4-D形状にする
            new_k = self.split_heads(new_k)  # [batch_size, num_heads, seq_len_new, d_k]
            new_v = self.split_heads(new_v)  # [batch_size, num_heads, seq_len_new, d_k]
            # cached_k/cached_vと新しいk/vを結合（seq_len次元で結合）
            k = torch.cat([cached_k, new_k], dim=2)  # [batch_size, num_heads, seq_len_cached + seq_len_new, d_k]
            v = torch.cat([cached_v, new_v], dim=2)  # [batch_size, num_heads, seq_len_cached + seq_len_new, d_k]

        # ヘッドに分割
        q = self.split_heads(q)  # [batch_size, num_heads, seq_len_q, d_k]

        if cached_k is None:
            k = self.split_heads(k)  # [batch_size, num_heads, seq_len_k, d_k]

        if cached_v is None:
            v = self.split_heads(v)  # [batch_size, num_heads, seq_len_v, d_k]

        # マスクの次元調整
        if mask is not None and mask.dim() == 3:
            # [batch_size, seq_len_q, seq_len_k] -> [batch_size, 1, seq_len_q, seq_len_k]
            mask = mask.unsqueeze(1)

        # スケールドドットプロダクトアテンション
        attn_output, attention_weights = self.attention(q, k, v, mask)

        # ヘッドの結合
        output = self.combine_heads(attn_output)  # [batch_size, seq_len_q, d_model]

        # 出力の線形変換
        output = self.wo(output)

        if return_cache:
            return output, attention_weights, k, v

        return output

# キャッシュ管理の補助関数
def prune_cache(cache, max_entries=MAX_CACHE_ENTRIES):
    """
    キャッシュサイズが指定された最大値を超えた場合に古いエントリを削除します。
    元のキャッシュは変更せず、新しい辞書を返します。

    Args:
        cache (dict): 管理対象のキャッシュ辞書
        max_entries (int): 許容される最大エントリ数（デフォルト: MAX_CACHE_ENTRIES）

    Returns:
        dict: サイズ調整後の新しいキャッシュ辞書（元のキャッシュは変更されません）
    """
    if len(cache) <= max_entries:
        # 元のキャッシュを変更しないため、新しい辞書を作成して返す
        return dict(cache)

    # 最も古いエントリを削除（ここではシンプルに最初のn個を削除）
    num_to_remove = len(cache) - max_entries

    # キーのリストを取得
    keys = list(cache.keys())

    # 新しい辞書を作成（最初のnum_to_remove個のキーを除く）
    new_cache = {key: cache[key] for key in keys[num_to_remove:]}

    return new_cache
