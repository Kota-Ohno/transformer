import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    """位置エンコーディング"""

    def __init__(self, d_model, max_seq_length=512):
        super(PositionalEncoding, self).__init__()

        # 位置エンコーディングの計算
        pe = torch.zeros(max_seq_length, d_model)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)

        # sin/cosペアで同じ周波数項を共有する標準的な実装
        # d_modelが奇数の場合、sinとcosのスライス長が異なるため、それぞれに合わせて計算
        num_sin = (d_model + 1) // 2  # pe[:, 0::2]の要素数
        num_cos = d_model // 2  # pe[:, 1::2]の要素数

        # sin用の周波数項（num_sin個）
        div_term_sin = torch.exp(torch.arange(0, num_sin, dtype=torch.float) * (-math.log(10000.0) / d_model))
        # cos用の周波数項（num_cos個）
        div_term_cos = torch.exp(torch.arange(0, num_cos, dtype=torch.float) * (-math.log(10000.0) / d_model))

        # sin と cos を使って位置エンコーディングを作成
        pe[:, 0::2] = torch.sin(position * div_term_sin)
        pe[:, 1::2] = torch.cos(position * div_term_cos)

        # バッチ次元を追加 [1, max_seq_length, d_model]
        pe = pe.unsqueeze(0)

        # モジュールのバッファとして登録 (パラメータではない)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        入力テンソルに位置エンコーディングを加算

        Args:
            x: 入力テンソル [batch_size, seq_len, d_model]

        Returns:
            位置情報が加算されたテンソル [batch_size, seq_len, d_model]
        """
        seq_len = x.size(1)
        if seq_len > self.pe.size(1):
            raise ValueError(
                f"入力シーケンス長 ({seq_len}) が最大シーケンス長 ({self.pe.size(1)}) を超えています"
            )
        # 入力シーケンス長に合わせて位置エンコーディングを加算
        x = x + self.pe[:, :seq_len]
        return x

class FeedForward(nn.Module):
    """フィードフォワードネットワーク"""

    def __init__(self, d_model, d_ff, dropout=0.1):
        super(FeedForward, self).__init__()

        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        Args:
            x: 入力テンソル [batch_size, seq_len, d_model]

        Returns:
            出力テンソル [batch_size, seq_len, d_model]
        """
        # 1つ目の線形層 + ReLU + ドロップアウト
        x = self.dropout(torch.relu(self.linear1(x)))

        # 2つ目の線形層
        x = self.linear2(x)

        return x
