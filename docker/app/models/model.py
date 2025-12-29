"""
Transformerモデルを作成するユーティリティ
"""

import torch
import torch.nn as nn
from models.encoder import Encoder
from models.decoder import Decoder
from utils.config import CONFIG

class TranslationModel(nn.Module):
    """
    翻訳モデル全体をカプセル化するクラス
    """
    def __init__(self, encoder, decoder, src_pad_idx, tgt_pad_idx):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_pad_idx = src_pad_idx
        self.tgt_pad_idx = tgt_pad_idx

    def make_src_mask(self, src):
        # src_mask: [batch_size, 1, 1, src_len]
        src_mask = (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2)
        return src_mask

    def make_tgt_mask(self, tgt):
        # tgt_mask: [batch_size, 1, tgt_len, tgt_len]
        tgt_pad_mask = (tgt != self.tgt_pad_idx).unsqueeze(1).unsqueeze(2)
        tgt_len = tgt.shape[1]
        tgt_sub_mask = torch.tril(torch.ones((tgt_len, tgt_len), device=CONFIG.device)).bool()
        tgt_mask = tgt_pad_mask & tgt_sub_mask
        return tgt_mask

    def forward(self, src, tgt):
        src_mask = self.make_src_mask(src)
        tgt_mask = self.make_tgt_mask(tgt)
        enc_src = self.encoder(src, src_mask)
        output, attention = self.decoder(tgt, enc_src, tgt_mask, src_mask)
        return output, attention

    def predict(self, src, max_length=None, start_token=2, end_token=3):
        """
        推論時に翻訳を生成する（greedy search）

        Args:
            src: ソース文のトークンID [batch_size, src_len]
            max_length: 最大生成長（デフォルトはCONFIGから取得）
            start_token: 開始トークンID（デフォルトは2 = <s>）
            end_token: 終了トークンID（デフォルトは3 = </s>）

        Returns:
            output_ids: 生成されたトークンIDのリスト [batch_size, tgt_len]
        """
        if max_length is None:
            max_length = CONFIG.model_hyperparameters.max_seq_length

        self.eval()
        device = next(self.parameters()).device
        batch_size = src.size(0)

        # エンコーダーでソースをエンコード
        src_mask = self.make_src_mask(src)
        enc_src = self.encoder(src, src_mask)

        # デコーダーの初期入力（<s>トークン）
        tgt = torch.full((batch_size, 1), start_token, dtype=torch.long, device=device)

        # キャッシュの初期化
        cache = None

        # 生成されたトークンIDを格納
        output_ids = []

        with torch.no_grad():
            for _ in range(max_length):
                # 現在のターゲットシーケンスのマスクを作成
                tgt_len = tgt.size(1)
                tgt_mask = self.make_tgt_mask(tgt)

                # デコーダーで次のトークンを予測
                decoder_output, cache = self.decoder(tgt, enc_src, tgt_mask, src_mask, cache)

                # 最後のトークンの予測を取得 [batch_size, vocab_size]
                next_token_logits = decoder_output[:, -1, :]

                # Greedy search: 最も確率の高いトークンを選択
                next_token = next_token_logits.argmax(dim=-1, keepdim=True)  # [batch_size, 1]

                # 生成されたトークンを追加
                output_ids.append(next_token)

                # 終了トークンが生成されたかチェック
                if (next_token == end_token).all():
                    break

                # 次のイテレーションのためにターゲットシーケンスに追加
                tgt = torch.cat([tgt, next_token], dim=1)

        # バッチごとにトークンIDを結合 [batch_size, tgt_len]
        output_ids = torch.cat(output_ids, dim=1)

        return output_ids

def create_transformer_model(input_vocab_size, output_vocab_size,
                          src_pad_idx=0, tgt_pad_idx=0,
                          hidden_size=None, num_heads=None, num_layers=None, d_ff=None,
                          dropout=None, max_seq_length=None):
    """
    基本的なTransformerモデルを作成する

    Args:
        input_vocab_size: 入力語彙サイズ
        output_vocab_size: 出力語彙サイズ
        src_pad_idx: ソースパディングインデックス
        tgt_pad_idx: ターゲットパディングインデックス
        hidden_size: 隠れ層の次元数（未指定時はCONFIGから取得）
        num_heads: アテンションヘッド数（未指定時はCONFIGから取得）
        num_layers: レイヤー数（未指定時はCONFIGから取得）
        d_ff: フィードフォワード層の次元数（未指定時はCONFIGから取得）
        dropout: ドロップアウト率（未指定時はCONFIGから取得）
        max_seq_length: 最大シーケンス長（未指定時はCONFIGから取得）

    Returns:
        TranslationModel: 作成したTransformerモデル
    """
    # デフォルト値をCONFIGから取得
    hidden_size = hidden_size or CONFIG.model_hyperparameters.hidden_size
    num_heads = num_heads or CONFIG.model_hyperparameters.num_heads
    num_layers = num_layers or CONFIG.model_hyperparameters.num_layers
    d_ff = d_ff or CONFIG.model_hyperparameters.d_ff
    dropout = dropout or CONFIG.model_hyperparameters.dropout_rate
    max_seq_length = max_seq_length or CONFIG.model_hyperparameters.max_seq_length

    # エンコーダーの作成
    encoder = Encoder(
        vocab_size=input_vocab_size,
        d_model=hidden_size,
        num_heads=num_heads,
        num_layers=num_layers,
        d_ff=d_ff,
        dropout=dropout,
        max_seq_length=max_seq_length
    )

    # デコーダーの作成
    decoder = Decoder(
        vocab_size=output_vocab_size,
        d_model=hidden_size,
        num_heads=num_heads,
        num_layers=num_layers,
        d_ff=d_ff,
        dropout=dropout,
        max_seq_length=max_seq_length
    )

    # モデルの作成
    model = TranslationModel(
        encoder=encoder,
        decoder=decoder,
        src_pad_idx=src_pad_idx,
        tgt_pad_idx=tgt_pad_idx
    )

    # デバイスに移動
    model = model.to(CONFIG.device)

    # パラメータの初期化 (Transformerでよく使われる方法)
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    return model
