import torch
import torch.nn as nn
import math
from config import HIDDEN_SIZE

class ScaledDotProductAttention(nn.Module):
    def __init__(self, d_model, d_k, dropout=0.1):
        super(ScaledDotProductAttention, self).__init__()
        self.d_k = d_k
        self.dropout = nn.Dropout(dropout)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, q, k, v, mask=None):
        attn = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            attn = attn.masked_fill(mask == 0, float('-inf'))
        attn = self.softmax(attn)
        attn = self.dropout(attn)
        output = torch.matmul(attn, v)
        return output

class MultiHeadAttention(nn.Module):
    """
    マルチヘッドアテンション機構を実装するクラス。

    複数の注意ヘッドを用いて、入力の異なる表現を学習します。

    Attributes:
        num_heads (int): アテンションヘッドの数
        d_k (int): キーの次元数
        d_v (int): 値の次元数
        projection_dim (int): 各ヘッドの投影次元
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
        self.num_heads = num_heads
        self.d_k = d_k
        self.d_v = d_v
        self.projection_dim = d_model // num_heads
        self.d_model = d_model

        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.scaled_dot_product_attention = ScaledDotProductAttention(d_model, d_k)
        self.final_linear = nn.Linear(d_model, d_model)

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
        マルチヘッドアテンションの順伝播を行います。

        Args:
            q (torch.Tensor): クエリテンソル。形状は(batch_size, seq_len_q, d_model)
            k (torch.Tensor): キーテンソル。形状は(batch_size, seq_len_k, d_model)
            v (torch.Tensor): 値テンソル。形状は(batch_size, seq_len_v, d_model)
            mask (torch.Tensor, optional): マスクテンソル。デフォルトはNone

        Returns:
            torch.Tensor: アテンション適用後の出力テンソル。形状は(batch_size, seq_len_q, d_model)
        """
        batch_size = q.size(0)

        # 線形変換と分割を行う
        q = self.q_linear(q).view(batch_size, -1, self.num_heads, self.projection_dim)
        k = self.k_linear(k).view(batch_size, -1, self.num_heads, self.projection_dim)
        v = self.v_linear(v).view(batch_size, -1, self.num_heads, self.projection_dim)

        # スケールドドットプロダクト注意を適用
        scaled_attention = self.scaled_dot_product_attention(q, k, v, mask)

        # 出力を連結して射影する
        concat_attention = scaled_attention.view(batch_size, -1, self.d_model)
        output = self.final_linear(concat_attention)
        return output
