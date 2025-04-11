import torch
import torch.nn as nn
import math
import logging

# 共通設定
MAX_CACHE_ENTRIES = 100

class RelativePositionalEncoding(nn.Module):
    """
    相対位置エンコーディングを実装するクラス。

    Shaw et al. (2018)とRaffer et al. (2019)の論文に基づく実装です。
    相対的な位置関係を明示的にモデル化し、長いシーケンスや未知の長さのシーケンスにも対応できます。

    Attributes:
        d_model (int): モデルの次元数
        max_dist (int): 考慮する最大相対距離
        pos_embeddings (nn.Parameter): 相対位置埋め込み

    Args:
        d_model (int): モデルの次元数
        max_dist (int): 考慮する最大相対距離（デフォルトは64）
    """
    def __init__(self, d_model, max_dist=64):
        super(RelativePositionalEncoding, self).__init__()
        self.d_model = d_model
        self.max_dist = max_dist

        # 相対位置埋め込みを初期化
        # max_dist * 2 + 1 の理由：負の相対位置、正の相対位置、0（同じ位置）をカバーするため
        self.pos_embeddings = nn.Parameter(torch.randn(max_dist * 2 + 1, d_model))

        # キャッシュを初期化
        self.cache = {}
        self.max_cache_size = MAX_CACHE_ENTRIES

        # 初期化
        self._reset_parameters()

    def _reset_parameters(self):
        """埋め込みの初期化"""
        nn.init.xavier_uniform_(self.pos_embeddings)

    def _get_relative_indices(self, seq_len_q, seq_len_k):
        """
        相対位置のインデックス行列を生成します。

        Args:
            seq_len_q (int): クエリのシーケンス長
            seq_len_k (int): キーのシーケンス長

        Returns:
            torch.Tensor: 相対位置のインデックス (seq_len_q, seq_len_k)
        """
        # 行列の各要素 (i, j) に対して、相対位置 j - i を計算
        i = torch.arange(seq_len_q).unsqueeze(1)
        j = torch.arange(seq_len_k).unsqueeze(0)
        rel_pos = j - i  # (seq_len_q, seq_len_k) の相対位置行列

        # 相対位置を [-max_dist, max_dist] の範囲にクリップ
        rel_pos = torch.clamp(rel_pos, -self.max_dist, self.max_dist)

        # [-max_dist, max_dist] の範囲を [0, 2*max_dist] の範囲にシフト
        rel_pos += self.max_dist

        return rel_pos

    def _prune_cache(self):
        """キャッシュサイズを制限する"""
        if len(self.cache) <= self.max_cache_size:
            return

        # キャッシュサイズが制限を超えた場合、古いエントリを削除
        num_to_remove = len(self.cache) - self.max_cache_size
        keys_to_remove = list(self.cache.keys())[:num_to_remove]

        for key in keys_to_remove:
            del self.cache[key]

        # 削除したことをログ出力
        logging.debug(f"相対位置エンコーディングのキャッシュから{num_to_remove}エントリを削除しました")

    def forward(self, seq_len_q, seq_len_k, device):
        """
        特定のシーケンス長に対応する相対位置エンコーディングを生成します。

        Args:
            seq_len_q (int): クエリのシーケンス長
            seq_len_k (int): キーのシーケンス長
            device (torch.device): 計算に使用するデバイス

        Returns:
            torch.Tensor: 相対位置エンコーディング (seq_len_q, seq_len_k, d_model)
        """
        # キャッシュキーを作成
        cache_key = (seq_len_q, seq_len_k, str(device))

        # キャッシュにある場合はキャッシュから取得
        if cache_key in self.cache:
            return self.cache[cache_key]

        # 相対位置インデックスを取得
        rel_pos_indices = self._get_relative_indices(seq_len_q, seq_len_k).to(device)

        # インデックスを使って位置埋め込みをルックアップ
        rel_pos_embeddings = self.pos_embeddings[rel_pos_indices]

        # キャッシュサイズをチェックして必要なら削除
        self._prune_cache()

        # 新しいエントリをキャッシュに追加
        self.cache[cache_key] = rel_pos_embeddings

        return rel_pos_embeddings


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
        num_heads (int): アテンションヘッドの数
        d_k (int): キーの次元数
        d_v (int): 値の次元数
        d_model (int): モデルの次元数
        q_linear (nn.Linear): クエリの線形変換層
        k_linear (nn.Linear): キーの線形変換層
        v_linear (nn.Linear): 値の線形変換層
        pos_proj (nn.Linear): 位置エンコーディングの射影
        dropout (nn.Dropout): ドロップアウト層
        final_linear (nn.Linear): 最終的な線形変換層
        rel_pos_enc (RelativePositionalEncoding): 相対位置エンコーディング

    Args:
        d_model (int): モデルの次元数
        num_heads (int): アテンションヘッドの数
        dropout (float): ドロップアウト率
        max_dist (int): 相対位置エンコーディングの最大距離
    """
    def __init__(self, d_model, num_heads, dropout=0.1, max_dist=64):
        super(RelativeMultiHeadAttention, self).__init__()

        # d_modelがnum_headsで割り切れる必要がある
        assert d_model % num_heads == 0

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads  # キーの次元数
        self.d_v = d_model // num_heads  # 値の次元数

        # 線形投影
        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)

        # 位置エンコーディングの射影
        self.pos_proj = nn.Linear(d_model, d_model)

        # ドロップアウト
        self.dropout = nn.Dropout(dropout)

        # 出力投影
        self.final_linear = nn.Linear(d_model, d_model)

        # 相対位置エンコーディング
        self.rel_pos_enc = RelativePositionalEncoding(d_model, max_dist)

        # スケーリング係数
        self.scale = 1.0 / math.sqrt(self.d_k)

        # Softmax
        self.softmax = nn.Softmax(dim=-1)

    def _rel_shift(self, x):
        """
        相対位置のシフト操作。
        これによりクエリとキーの相対位置を適切にアラインします。

        Args:
            x (torch.Tensor): 入力テンソル

        Returns:
            torch.Tensor: シフトされたテンソル
        """
        # 入力の形状: (batch, heads, seq_len, 2*seq_len-1)
        batch, heads, seq_len, _ = x.size()

        # パディングを追加してシフト操作を容易にする
        # 先頭に0の列を追加: (batch, heads, seq_len, 2*seq_len)
        zero_pad = torch.zeros((batch, heads, seq_len, 1), device=x.device, dtype=x.dtype)
        x_padded = torch.cat([zero_pad, x], dim=-1)

        # 形状を変換してシフト操作を実行: (batch, heads, seq_len+1, 2*seq_len-1)
        x_padded = x_padded.view(batch, heads, seq_len+1, 2*seq_len-1)

        # シフトした部分を抽出: (batch, heads, seq_len, seq_len)
        x = x_padded[:, :, 1:, :seq_len].contiguous()

        return x

    def forward(self, q, k, v, mask=None, cached_k=None, cached_v=None, return_cache=False):
        """
        相対位置情報を考慮したマルチヘッドアテンションの順伝播

        Args:
            q (torch.Tensor): クエリテンソル (batch_size, seq_len_q, d_model)
            k (torch.Tensor): キーテンソル (batch_size, seq_len_k, d_model)
            v (torch.Tensor): 値テンソル (batch_size, seq_len_v, d_model)
            mask (torch.Tensor, optional): マスク (batch_size, 1, seq_len_q, seq_len_k)
            cached_k (torch.Tensor, optional): キャッシュされたキー
            cached_v (torch.Tensor, optional): キャッシュされた値
            return_cache (bool): キャッシュを返すかどうか

        Returns:
            torch.Tensor または tuple: アテンション出力とオプションでキャッシュ
        """
        batch_size = q.size(0)
        seq_len_q = q.size(1)
        seq_len_k = k.size(1)

        # 線形変換
        q = self.q_linear(q)

        # キャッシュがない場合のみ計算
        k = self.k_linear(k) if cached_k is None else cached_k
        v = self.v_linear(v) if cached_v is None else cached_v

        # 多頭注意のための形状変換
        q = q.view(batch_size, seq_len_q, self.num_heads, self.d_k).transpose(1, 2)
        if cached_k is None:
            k = k.view(batch_size, seq_len_k, self.num_heads, self.d_k).transpose(1, 2)
        if cached_v is None:
            v = v.view(batch_size, -1, self.num_heads, self.d_v).transpose(1, 2)

        # 相対位置エンコーディングを計算
        # クエリとキーの長さが異なる場合でも対応できるように修正
        rel_pos_embedding = self.rel_pos_enc(seq_len_q, seq_len_k, q.device)
        rel_pos_embedding = self.pos_proj(rel_pos_embedding)
        rel_pos_embedding = rel_pos_embedding.view(seq_len_q, seq_len_k, self.num_heads, self.d_k)
        rel_pos_embedding = rel_pos_embedding.permute(2, 0, 1, 3)  # (heads, seq_len_q, seq_len_k, d_k)

        # コンテンツベースのアテンションスコア: (batch, heads, seq_len_q, seq_len_k)
        content_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        try:
            # 相対位置ベースのアテンションスコア（形状を合わせるために転置）
            # q: (batch, heads, seq_len_q, d_k)
            # rel_pos_embedding: (heads, seq_len_q, seq_len_k, d_k)
            q_expanded = q.unsqueeze(-2)  # (batch, heads, seq_len_q, 1, d_k)
            rel_pos_transposed = rel_pos_embedding.transpose(-2, -1)  # (heads, seq_len_q, d_k, seq_len_k)

            # バッチサイズ分展開
            rel_pos_transposed = rel_pos_transposed.unsqueeze(0).expand(batch_size, -1, -1, -1, -1)

            # 行列乗算
            rel_scores = torch.matmul(q_expanded, rel_pos_transposed)  # (batch, heads, seq_len_q, 1, seq_len_k)
            rel_scores = rel_scores.squeeze(-2)  # (batch, heads, seq_len_q, seq_len_k)

            # アテンションスコアの合計
            attn_scores = content_scores + rel_scores
        except RuntimeError as e:
            # テンソルサイズの不一致が発生した場合、相対位置スコアを省略
            logging.warning(f"相対位置スコア計算中にエラーが発生しました: {e}")
            logging.warning(f"コンテンツベースのスコアのみを使用します（q: {q.shape}, k: {k.shape}, rel_pos: {rel_pos_embedding.shape}）")
            attn_scores = content_scores  # 相対位置スコアを省略

        # マスキング
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, -65504.0)

        # アテンション重みの計算
        attn_weights = self.softmax(attn_scores)
        attn_weights = self.dropout(attn_weights)

        # 値とのアテンション適用
        context = torch.matmul(attn_weights, v)

        # 形状を元に戻す: (batch, seq_len_q, d_model)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len_q, self.d_model)
        output = self.final_linear(context)

        if return_cache:
            # キャッシュがない場合は新しく作る
            cached_k = k if cached_k is None else cached_k
            cached_v = v if cached_v is None else cached_v
            return output, attn_weights, cached_k, cached_v

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
