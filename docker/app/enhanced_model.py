import torch
import torch.nn as nn
from typing import List, Dict, Tuple
from utils import create_padding_mask, create_subsequent_mask
from advanced_encoder import EnhancedEncoder
from advanced_decoder import EnhancedDecoder
from config import MAX_SEQ_LENGTH, DEVICE

class EnhancedTranslationModel(nn.Module):
    """
    相対位置エンコーディングとGLUを使用した強化版翻訳モデル。
    標準の TranslationModel と同じインターフェースを持ちます。

    Attributes:
        encoder (EnhancedEncoder): 強化版エンコーダー
        decoder (EnhancedDecoder): 強化版デコーダー
        src_pad_idx (int): ソース言語のパディングインデックス
        tgt_pad_idx (int): ターゲット言語のパディングインデックス
        device (str): 使用するデバイス
        max_seq_length (int): 最大シーケンス長

    Args:
        encoder (EnhancedEncoder): 強化版エンコーダー
        decoder (EnhancedDecoder): 強化版デコーダー
        src_pad_idx (int): ソース言語のパディングインデックス
        tgt_pad_idx (int): ターゲット言語のパディングインデックス
        device (str): 使用するデバイス
        max_seq_length (int): 最大シーケンス長
    """
    def __init__(self, encoder: EnhancedEncoder, decoder: EnhancedDecoder,
                 src_pad_idx: int, tgt_pad_idx: int, device: str,
                 max_seq_length: int = MAX_SEQ_LENGTH):
        super(EnhancedTranslationModel, self).__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.src_pad_idx = src_pad_idx
        self.tgt_pad_idx = tgt_pad_idx
        self.device = device
        self.max_seq_length = max_seq_length

    def forward(self, src: torch.Tensor, tgt_input: torch.Tensor,
                cache: List[Dict[str, torch.Tensor]] = None) -> Tuple[torch.Tensor, List[Dict[str, torch.Tensor]]]:
        """
        順伝播処理

        Args:
            src (torch.Tensor): ソーステンソル (batch_size, src_len)
            tgt_input (torch.Tensor): ターゲット入力テンソル (batch_size, tgt_len)
            cache (List[Dict[str, torch.Tensor]], optional): キャッシュ

        Returns:
            Tuple[torch.Tensor, List[Dict[str, torch.Tensor]]]: デコーダー出力とキャッシュ
        """
        # マスクの作成
        src_mask = create_padding_mask(src, self.src_pad_idx).to(self.device)
        tgt_mask = create_subsequent_mask(tgt_input)
        tgt_pad_mask = create_padding_mask(tgt_input, self.tgt_pad_idx).to(self.device)

        # ターゲットマスクは、パディングマスクと後続マスクの論理積
        tgt_mask = torch.logical_and(tgt_pad_mask.expand(-1, -1, tgt_input.size(1), -1), tgt_mask)

        # エンコーダ出力に適用するマスク
        memory_mask = src_mask.expand(-1, -1, tgt_input.size(1), -1)

        # エンコーダ順伝播
        encoder_output = self.encoder(src, src_mask)

        # デコーダ順伝播
        decoder_output, new_cache = self.decoder(tgt_input, encoder_output,
                                               tgt_mask, memory_mask, cache=cache)

        return decoder_output, new_cache


def create_enhanced_model(input_dim: int, output_dim: int, hidden_dim: int,
                         num_heads: int, num_layers: int, ff_dim: int,
                         src_pad_idx: int, tgt_pad_idx: int, dropout: float,
                         device: str = DEVICE, max_dist: int = 64) -> EnhancedTranslationModel:
    """
    強化版翻訳モデルを作成するヘルパー関数

    Args:
        input_dim (int): 入力ボキャブラリーサイズ
        output_dim (int): 出力ボキャブラリーサイズ
        hidden_dim (int): 隠れ層の次元数
        num_heads (int): アテンションヘッドの数
        num_layers (int): エンコーダー/デコーダーレイヤーの数
        ff_dim (int): フィードフォワード層の次元数
        src_pad_idx (int): ソース言語のパディングインデックス
        tgt_pad_idx (int): ターゲット言語のパディングインデックス
        dropout (float): ドロップアウト率
        device (str): 使用するデバイス
        max_dist (int): 相対位置エンコーディングの最大距離

    Returns:
        EnhancedTranslationModel: 強化版翻訳モデル
    """
    # エンコーダーとデコーダーの作成
    encoder = EnhancedEncoder(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_heads=num_heads,
        num_layers=num_layers,
        ff_dim=ff_dim,
        dropout=dropout,
        device=device,
        max_dist=max_dist
    )

    decoder = EnhancedDecoder(
        vocab_dim=output_dim,
        hidden_dim=hidden_dim,
        num_heads=num_heads,
        num_layers=num_layers,
        ff_dim=ff_dim,
        output_dim=output_dim,
        dropout=dropout,
        device=device,
        max_dist=max_dist
    )

    # モデルを作成して返す
    model = EnhancedTranslationModel(
        encoder=encoder,
        decoder=decoder,
        src_pad_idx=src_pad_idx,
        tgt_pad_idx=tgt_pad_idx,
        device=device
    )

    return model
