import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from config import CONFIG, LAYER_DROPOUT
from typing import List, Tuple, Dict, Optional
from torch.utils.checkpoint import checkpoint
import logging

# 高度なレイヤーモジュールをインポート
from advanced_layers import RelativeMultiHeadAttention, EnhancedFeedForward

# クロスアテンションマスクを適切に調整するユーティリティ関数
def adjust_mask_shape(mask, tgt_shape):
    """
    マスクの形状を目標形状に調整するユーティリティ関数

    Args:
        mask (torch.Tensor): 調整するマスク
        tgt_shape (tuple): 目標形状 (batch, heads, seq_q, seq_k)

    Returns:
        torch.Tensor: 調整されたマスク
    """
    if mask is None:
        return None

    # マスクの形状を取得
    if mask.dim() != 4 or mask.shape != tgt_shape:
        logging.debug(f"マスク形状を調整: {mask.shape} → {tgt_shape}")

        # マスクが4次元未満の場合は次元を追加
        while mask.dim() < 4:
            if mask.dim() == 1:
                mask = mask.unsqueeze(0)  # バッチ次元を追加
            elif mask.dim() == 2:
                mask = mask.unsqueeze(1)  # ヘッド次元を追加
            elif mask.dim() == 3:
                mask = mask.unsqueeze(1)  # ヘッド次元を追加

        # 目標形状に合わせて新しいマスクを作成
        new_mask = torch.ones(tgt_shape, device=mask.device)

        # 共通部分をコピー
        min_batch = min(mask.size(0), tgt_shape[0])
        min_heads = min(mask.size(1), tgt_shape[1])
        min_seq_q = min(mask.size(2), tgt_shape[2])
        min_seq_k = min(mask.size(3), tgt_shape[3])

        new_mask[:min_batch, :min_heads, :min_seq_q, :min_seq_k] = mask[:min_batch, :min_heads, :min_seq_q, :min_seq_k]

        return new_mask

    return mask

class EnhancedDecoderLayer(nn.Module):
    """
    相対位置エンコーディングとGLUを使用した改良版デコーダーレイヤー。

    Attributes:
        self_attention (RelativeMultiHeadAttention): 自己アテンション
        encoder_decoder_attention (RelativeMultiHeadAttention): エンコーダー出力に対するクロスアテンション
        feed_forward (EnhancedFeedForward): 強化版フィードフォワードネットワーク
        norm1 (nn.LayerNorm): 第1層のレイヤー正規化
        norm2 (nn.LayerNorm): 第2層のレイヤー正規化
        norm3 (nn.LayerNorm): 第3層のレイヤー正規化
        dropout (nn.Dropout): ドロップアウト

    Args:
        d_model (int): モデルの次元数
        num_heads (int): アテンションヘッドの数
        d_ff (int): フィードフォワード層の次元数
        dropout (float): ドロップアウト率
        max_dist (int): 相対位置エンコーディングの最大距離
    """
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float, max_dist: int = 64):
        super(EnhancedDecoderLayer, self).__init__()

        # 相対位置エンコーディングを使用したマルチヘッドアテンション
        self.self_attention = RelativeMultiHeadAttention(
            d_model=d_model,
            num_heads=num_heads,
            dropout=dropout,
            max_dist=max_dist
        )

        # エンコーダ-デコーダアテンション
        self.encoder_decoder_attention = RelativeMultiHeadAttention(
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
        self.dropout = nn.Dropout(dropout)

    def _adjust_mask(self, mask: Optional[torch.Tensor], batch_size: int, num_heads: int,
                     seq_len_q: int, seq_len_k: int) -> Optional[torch.Tensor]:
        """
        マスクの形状を調整する

        Args:
            mask (torch.Tensor): 調整するマスク
            batch_size (int): バッチサイズ
            num_heads (int): ヘッド数
            seq_len_q (int): クエリシーケンス長
            seq_len_k (int): キーシーケンス長

        Returns:
            torch.Tensor: 調整されたマスク
        """
        if mask is None:
            return None

        # マスクの形状確認と調整
        tgt_shape = (batch_size, num_heads, seq_len_q, seq_len_k)

        return adjust_mask_shape(mask, tgt_shape)

    def forward(self, x: torch.Tensor, encoder_output: torch.Tensor,
                self_attn_mask: Optional[torch.Tensor] = None,
                cross_attn_mask: Optional[torch.Tensor] = None,
                cache: Optional[Dict[str, torch.Tensor]] = None) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        デコーダーレイヤーの順伝播

        Args:
            x: デコーダー入力 (batch_size, tgt_len, d_model)
            encoder_output: エンコーダー出力 (batch_size, src_len, d_model)
            self_attn_mask: セルフアテンションマスク
            cross_attn_mask: クロスアテンションマスク
            cache: キャッシュ（推論時に使用）

        Returns:
            Tuple[torch.Tensor, Dict]: 出力とキャッシュ
        """
        # キャッシュの初期化
        cache = cache or {}
        new_cache = {}

        # サイズ情報の抽出
        batch_size = x.size(0)
        tgt_len = x.size(1)
        src_len = encoder_output.size(1)

        # 1. セルフアテンション（Pre-LN方式）
        residual = x
        x_norm = self.norm1(x)

        # マスクの調整
        if self_attn_mask is not None:
            self_attn_mask = self._adjust_mask(
                self_attn_mask,
                batch_size,
                self.self_attention.num_heads,
                tgt_len,
                tgt_len
            )

        # セルフアテンション実行
        self_attn_output, _, new_k, new_v = self.self_attention(
            q=x_norm, k=x_norm, v=x_norm,
            mask=self_attn_mask,
            cached_k=cache.get('self_k'),
            cached_v=cache.get('self_v'),
            return_cache=True
        )

        # キャッシュ更新
        new_cache['self_k'] = new_k
        new_cache['self_v'] = new_v

        # 残差接続
        x = residual + self.dropout(self_attn_output)

        # 2. エンコーダ-デコーダアテンション（Pre-LN方式）
        residual = x
        x_norm = self.norm2(x)

        # マスクの調整
        if cross_attn_mask is not None:
            cross_attn_mask = self._adjust_mask(
                cross_attn_mask,
                batch_size,
                self.encoder_decoder_attention.num_heads,
                tgt_len,
                src_len
            )

        # クロスアテンション実行
        cross_attn_output, _, new_k, new_v = self.encoder_decoder_attention(
            q=x_norm,
            k=encoder_output,
            v=encoder_output,
            mask=cross_attn_mask,
            cached_k=cache.get('memory_k'),
            cached_v=cache.get('memory_v'),
            return_cache=True
        )

        # キャッシュ更新
        new_cache['memory_k'] = new_k
        new_cache['memory_v'] = new_v

        # 残差接続
        x = residual + self.dropout(cross_attn_output)

        # 3. フィードフォワード（すでにPre-LN実装済み）
        x = self.feed_forward(x)

        return x, new_cache


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
        vocab_dim (int): 入力ボキャブラリーサイズ
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
        self.max_seq_length = CONFIG.get("MAX_SEQ_LENGTH", 512)
        self.max_context_length = 128  # メモリ効率化のため

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

        logging.info(f"EnhancedDecoder initialized: {num_layers} layers, {hidden_dim} hidden dim, {num_heads} heads")

    def ensure_tensor_dimensions(self, x: torch.Tensor, encoder_output: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        テンソル形状の互換性を確保するヘルパー関数

        Args:
            x (torch.Tensor): デコーダー入力
            encoder_output (torch.Tensor): エンコーダー出力

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: 調整されたテンソル
        """
        # 特徴量次元が一致しない場合の処理
        if x.size(-1) != encoder_output.size(-1):
            logging.warning(f"特徴量次元が一致しません: x={x.size(-1)}, encoder={encoder_output.size(-1)}")
            # 小さい方の次元に合わせる
            min_dim = min(x.size(-1), encoder_output.size(-1))
            x_adjusted = x[..., :min_dim]
            encoder_adjusted = encoder_output[..., :min_dim]
            return x_adjusted, encoder_adjusted

        # バッチサイズが一致しない場合の処理
        if x.size(0) != encoder_output.size(0):
            logging.warning(f"バッチサイズが一致しません: x={x.size(0)}, encoder={encoder_output.size(0)}")
            # 小さい方のバッチサイズに合わせる
            min_batch = min(x.size(0), encoder_output.size(0))
            x_adjusted = x[:min_batch]
            encoder_adjusted = encoder_output[:min_batch]
            return x_adjusted, encoder_adjusted

        # テンソルが既に互換性を持つ場合
        return x, encoder_output

    def forward(self, x: torch.Tensor, encoder_output: torch.Tensor,
                self_attn_mask: Optional[torch.Tensor] = None,
                cross_attn_mask: Optional[torch.Tensor] = None,
                cache: Optional[List[Dict[str, torch.Tensor]]] = None) -> Tuple[torch.Tensor, List[Dict[str, torch.Tensor]]]:
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
        # キャッシュの初期化
        if cache is None:
            cache = [{} for _ in range(len(self.layers))]
        elif len(cache) != len(self.layers):
            # キャッシュのサイズがレイヤー数と一致しない場合は調整
            if len(cache) < len(self.layers):
                cache.extend([{} for _ in range(len(self.layers) - len(cache))])
            else:
                cache = cache[:len(self.layers)]

        # 入力埋め込み
        x = self.embedding(x)
        x = self.dropout(x)

        # シーケンス長の制限
        if x.size(1) > self.max_seq_length:
            logging.warning(f"デコーダー入力シーケンス長を制限: {x.size(1)} → {self.max_seq_length}")
            x = x[:, :self.max_seq_length, :]

        # テンソル形状の互換性を確保
        x, encoder_output = self.ensure_tensor_dimensions(x, encoder_output)

        # エンコーダー出力のシーケンス長が長すぎる場合は制限
        if encoder_output.size(1) > self.max_context_length:
            logging.warning(f"エンコーダー出力シーケンス長を制限: {encoder_output.size(1)} → {self.max_context_length}")
            encoder_output = encoder_output[:, :self.max_context_length, :]

        # マスクの調整
        batch_size, tgt_len = x.size(0), x.size(1)
        src_len = encoder_output.size(1)

        # 推論時のマスク調整（因果的マスクを強制）
        if self_attn_mask is not None and not self.training:
            causal_mask = torch.tril(torch.ones(tgt_len, tgt_len), diagonal=0).unsqueeze(0).unsqueeze(0).to(x.device)
            self_attn_mask = adjust_mask_shape(self_attn_mask, (batch_size, 1, tgt_len, tgt_len)) * causal_mask

        # クロスアテンションマスクの調整
        if cross_attn_mask is not None:
            cross_attn_mask = adjust_mask_shape(cross_attn_mask, (batch_size, 1, tgt_len, src_len))

        # レイヤードロップアウトの準備
        layer_dropout = LAYER_DROPOUT if self.training else 0.0
        dropout_probs = torch.empty(len(self.layers)).uniform_()

        # 各レイヤーの適用
        for i, layer in enumerate(self.layers):
            # 訓練時にランダムにレイヤーをスキップ
            if self.training and dropout_probs[i] < layer_dropout:
                continue

            # チェックポイント使用時（メモリ効率化）
            if self.use_checkpointing and self.training:
                def custom_forward(module, x, enc_out, self_mask, cross_mask, layer_cache):
                    return module(x, enc_out, self_mask, cross_mask, layer_cache)

                try:
                    x, cache[i] = checkpoint(
                        custom_forward,
                        layer, x, encoder_output, self_attn_mask, cross_attn_mask, cache[i]
                    )
                except Exception as e:
                    logging.error(f"チェックポイント処理中にエラー発生（レイヤー{i}）: {e}")
                    # エラー発生時は通常処理にフォールバック
                    x, cache[i] = layer(x, encoder_output, self_attn_mask, cross_attn_mask, cache[i])
            else:
                # 通常の計算
                x, cache[i] = layer(x, encoder_output, self_attn_mask, cross_attn_mask, cache[i])

        # 最終的なレイヤー正規化と出力層
        output = self.output_layer(self.norm(x))

        return output, cache
