import torch
import torch.nn as nn
import logging
from typing import List, Dict, Tuple, Optional
from utils import create_padding_mask, create_subsequent_mask
from advanced_encoder import EnhancedEncoder
from advanced_decoder import EnhancedDecoder
from config import CONFIG, DEVICE

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
                 max_seq_length: Optional[int] = None):
        super(EnhancedTranslationModel, self).__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.src_pad_idx = src_pad_idx
        self.tgt_pad_idx = tgt_pad_idx
        self.device = device
        self.max_seq_length = max_seq_length if max_seq_length is not None else CONFIG["MAX_SEQ_LENGTH"]

        logging.info(f"EnhancedTranslationModel initialized: max_seq_length={self.max_seq_length}")

    def _ensure_valid_tensors(self, src: torch.Tensor, tgt_input: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        入力テンソルが有効であることを確認し、必要に応じて調整する

        Args:
            src (torch.Tensor): ソーステンソル
            tgt_input (torch.Tensor): ターゲット入力テンソル

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: 調整されたテンソル
        """
        # シーケンス長の制限
        if src.size(1) > self.max_seq_length:
            logging.warning(f"ソースシーケンス長を制限: {src.size(1)} → {self.max_seq_length}")
            src = src[:, :self.max_seq_length]

        if tgt_input.size(1) > self.max_seq_length:
            logging.warning(f"ターゲットシーケンス長を制限: {tgt_input.size(1)} → {self.max_seq_length}")
            tgt_input = tgt_input[:, :self.max_seq_length]

        # バッチサイズの一致確認
        if src.size(0) != tgt_input.size(0):
            logging.warning(f"バッチサイズ不一致: src={src.size(0)}, tgt={tgt_input.size(0)}")
            min_batch = min(src.size(0), tgt_input.size(0))
            src = src[:min_batch]
            tgt_input = tgt_input[:min_batch]

        return src, tgt_input

    def forward(self, src: torch.Tensor, tgt_input: torch.Tensor,
                cache: Optional[List[Dict[str, torch.Tensor]]] = None) -> Tuple[torch.Tensor, List[Dict[str, torch.Tensor]]]:
        """
        順伝播処理

        Args:
            src (torch.Tensor): ソーステンソル (batch_size, src_len)
            tgt_input (torch.Tensor): ターゲット入力テンソル (batch_size, tgt_len)
            cache (List[Dict[str, torch.Tensor]], optional): キャッシュ

        Returns:
            Tuple[torch.Tensor, List[Dict[str, torch.Tensor]]]: デコーダー出力とキャッシュ
        """
        try:
            # 入力テンソルの調整
            src, tgt_input = self._ensure_valid_tensors(src, tgt_input)

            # マスクの作成
            src_mask = create_padding_mask(src, self.src_pad_idx).to(self.device)
            tgt_mask = create_subsequent_mask(tgt_input).to(self.device)
            tgt_pad_mask = create_padding_mask(tgt_input, self.tgt_pad_idx).to(self.device)

            # ターゲットマスクは、パディングマスクと後続マスクの論理積
            combined_tgt_mask = torch.logical_and(
                tgt_pad_mask.expand(-1, -1, tgt_input.size(1), -1),
                tgt_mask
            )

            # エンコーダ出力に適用するマスク
            memory_mask = src_mask.expand(-1, -1, tgt_input.size(1), -1)

            # エンコーダ順伝播
            encoder_output = self.encoder(src, src_mask)

            # デコーダ順伝播
            decoder_output, new_cache = self.decoder(
                tgt_input,
                encoder_output,
                combined_tgt_mask,
                memory_mask,
                cache=cache
            )

            return decoder_output, new_cache

        except Exception as e:
            logging.error(f"Forward処理中にエラー発生: {e}")
            # エラー発生時のフォールバック
            batch_size = src.size(0)
            tgt_len = tgt_input.size(1)
            hidden_dim = self.decoder.output_layer.out_features

            # 0埋めの出力を返す
            empty_output = torch.zeros(batch_size, tgt_len, hidden_dim, device=self.device)
            empty_cache = cache if cache is not None else [{} for _ in range(len(self.decoder.layers))]

            return empty_output, empty_cache

    def predict(self, src: torch.Tensor, max_length: int = 100, device: Optional[str] = None) -> torch.Tensor:
        """
        ソーステキストから翻訳を生成します。

        Args:
            src (torch.Tensor): ソーステキストのテンソル [batch_size, src_len]
            max_length (int): 生成する最大トークン数
            device (str, optional): 使用するデバイス（Noneの場合はself.deviceを使用）

        Returns:
            torch.Tensor: 生成された翻訳トークンIDのテンソル
        """
        if device is None:
            device = self.device

        batch_size = src.size(0)

        # シーケンス長の制限
        if src.size(1) > self.max_seq_length:
            src = src[:, :self.max_seq_length]

        # 初期トークンとして<s>を使用
        tgt_tokens = torch.ones(batch_size, 1).fill_(2).long().to(device)  # <s>トークンで初期化

        # エンコーダー出力のキャッシュを保持
        src_mask = create_padding_mask(src, self.src_pad_idx).to(device)

        try:
            # エンコーダー出力を計算
            encoder_output = self.encoder(src, src_mask)
            cache = None

            with torch.no_grad():
                for i in range(max_length):
                    # マスクを作成
                    tgt_mask = create_subsequent_mask(tgt_tokens).to(device)
                    tgt_pad_mask = create_padding_mask(tgt_tokens, self.tgt_pad_idx).to(device)
                    combined_tgt_mask = torch.logical_and(
                        tgt_pad_mask.expand(-1, -1, tgt_tokens.size(1), -1),
                        tgt_mask
                    )
                    memory_mask = src_mask.expand(-1, -1, tgt_tokens.size(1), -1)

                    # デコーダーの順伝播
                    decoder_output, cache = self.decoder(
                        tgt_tokens, encoder_output, combined_tgt_mask, memory_mask, cache=cache
                    )

                    # 次のトークンを予測
                    pred = decoder_output[:, -1, :]
                    next_token = pred.argmax(dim=1, keepdim=True)

                    # 予測トークンを追加
                    tgt_tokens = torch.cat([tgt_tokens, next_token], dim=1)

                    # EOSトークンが生成されたら終了
                    if (next_token == 3).all():  # </s>トークン
                        break

            return tgt_tokens

        except Exception as e:
            logging.error(f"predict処理中にエラー発生: {e}")
            # エラー発生時は空のテンソルを返す
            return torch.ones(batch_size, 1).fill_(3).long().to(device)  # </s>トークンのみ


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
    # ロギング設定
    logging.info(f"強化版モデルを作成します: hidden_dim={hidden_dim}, num_heads={num_heads}, num_layers={num_layers}")

    try:
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

    except Exception as e:
        logging.error(f"モデル作成中にエラー発生: {e}")
        raise
