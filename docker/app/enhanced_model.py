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

    def translate(self, src: torch.Tensor, max_len: int = None, beam_size: int = 5, alpha: float = 0.6) -> torch.Tensor:
        """
        ビームサーチを使用してソーステキストを翻訳します。

        Args:
            src (torch.Tensor): ソーステンソル (1, src_len)
            max_len (int, optional): 生成する最大トークン数
            beam_size (int): ビームサーチの幅
            alpha (float): 長さペナルティの係数

        Returns:
            torch.Tensor: 翻訳結果 (1, tgt_len)
        """
        if max_len is None:
            max_len = self.max_seq_length

        batch_size = src.size(0)
        if batch_size != 1:
            raise ValueError("翻訳は現在バッチサイズ1のみサポートしています")

        # デバイスを取得
        device = src.device

        # エンコーダ出力を計算
        src_mask = create_padding_mask(src, self.src_pad_idx).to(device)
        encoder_output = self.encoder(src, src_mask)

        # 開始トークンと終了トークンのID
        sos_idx = 2  # Start of sentence token
        eos_idx = 3  # End of sentence token

        # 最初のトークンはSOS
        input_seq = torch.tensor([[sos_idx]], dtype=torch.long, device=device)

        # 最初のビーム
        # [シーケンス, スコア, 完了フラグ, キャッシュ]
        beams = [
            [input_seq, 0.0, False, None]
        ]

        for _ in range(max_len):
            candidates = []

            # すべてのビームが完了したかチェック
            all_complete = True
            for _, _, is_complete, _ in beams:
                if not is_complete:
                    all_complete = False
                    break
            if all_complete:
                break

            # 各ビームを展開
            for seq, score, is_complete, beam_cache in beams:
                if is_complete:
                    # 完了したビームはそのまま候補に追加
                    candidates.append([seq, score, True, beam_cache])
                    continue

                # デコーダーで次のトークンを予測
                with torch.no_grad():
                    # メモリマスクを作成
                    memory_mask = src_mask.expand(-1, -1, seq.size(1), -1)

                    # 自己注意のマスクを作成
                    tgt_mask = create_subsequent_mask(seq)
                    tgt_pad_mask = create_padding_mask(seq, self.tgt_pad_idx).to(device)
                    tgt_mask = torch.logical_and(tgt_pad_mask.expand(-1, -1, seq.size(1), -1), tgt_mask)

                    # デコーダー出力を計算
                    decoder_output, new_cache = self.decoder(
                        seq, encoder_output, tgt_mask, memory_mask, cache=beam_cache
                    )

                    # 最後のトークンの確率分布
                    logits = decoder_output[:, -1, :]
                    log_probs = torch.log_softmax(logits, dim=-1)

                    # 上位beam_size個のトークンを取得
                    topk_log_probs, topk_indices = log_probs.topk(beam_size)

                    # 各候補を追加
                    for log_prob, token_idx in zip(topk_log_probs[0], topk_indices[0]):
                        # 新しいシーケンスを作成
                        new_seq = torch.cat([seq, token_idx.unsqueeze(0).unsqueeze(0)], dim=1)

                        # 新しいスコアを計算
                        new_score = score + log_prob.item()

                        # EOSに達したかどうか
                        is_eos = token_idx.item() == eos_idx

                        # 候補に追加
                        candidates.append([new_seq, new_score, is_eos, new_cache])

            # 長さペナルティを適用
            candidates_with_penalty = []
            for seq, score, is_complete, cache in candidates:
                # 長さペナルティ: (5 + len)^α / (5 + 1)^α
                penalty = ((5 + seq.size(1)) ** alpha) / (6 ** alpha)
                normalized_score = score / penalty
                candidates_with_penalty.append([seq, normalized_score, is_complete, cache])

            # スコア順にソート
            candidates_with_penalty.sort(key=lambda x: x[1], reverse=True)

            # 上位beam_size個を選択
            beams = []
            for seq, normalized_score, is_complete, cache in candidates_with_penalty[:beam_size]:
                # 元のスコアに戻す
                penalty = ((5 + seq.size(1)) ** alpha) / (6 ** alpha)
                original_score = normalized_score * penalty
                beams.append([seq, original_score, is_complete, cache])

        # 最良のビームを返す
        best_seq = max(beams, key=lambda x: x[1])[0]

        return best_seq


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
