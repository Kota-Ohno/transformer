"""
Transformerモデルを作成するユーティリティ
"""

import torch
import torch.nn as nn
from models.encoder import Encoder
from models.decoder import Decoder
from utils.config import CONFIG
from utils.utils import create_src_mask, create_tgt_mask

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

    def make_src_mask(self, src: torch.Tensor) -> torch.Tensor:
        """
        ソースシーケンス用のマスクを作成

        Args:
            src: ソースシーケンス [batch_size, src_len]

        Returns:
            ソースマスク [batch_size, 1, 1, src_len]
        """
        return create_src_mask(src, self.src_pad_idx)

    def make_tgt_mask(self, tgt: torch.Tensor) -> torch.Tensor:
        """
        ターゲットシーケンス用のマスクを作成

        Args:
            tgt: ターゲットシーケンス [batch_size, tgt_len]

        Returns:
            ターゲットマスク [batch_size, 1, tgt_len, tgt_len]
        """
        return create_tgt_mask(tgt, self.tgt_pad_idx)

    def forward(self, src: torch.Tensor, tgt: torch.Tensor, cache=None):
        """
        フォワードパスを実行します。

        Args:
            src: ソースシーケンス [batch_size, src_len]
            tgt: ターゲットシーケンス [batch_size, tgt_len]
            cache: キャッシュ情報（推論時に使用、オプション）

        Returns:
            output: 出力テンソル [batch_size, tgt_len, vocab_size]
            cache_or_attention: キャッシュが提供された場合はキャッシュ、そうでない場合はNone
        """
        src_mask = self.make_src_mask(src)
        tgt_mask = self.make_tgt_mask(tgt)
        enc_src = self.encoder(src, src_mask)
        output, new_cache = self.decoder(tgt, enc_src, tgt_mask, src_mask, cache=cache)
        # トレーニング時（cache=None）は None を返し、推論時（cacheが提供された場合）はキャッシュを返す
        return output, new_cache if cache is not None else None

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

        # 現在のトレーニングモードを保存
        was_training = self.training
        try:
            self.eval()
            device = next(self.parameters()).device
            batch_size = src.size(0)

            with torch.no_grad():
                # エンコーダーでソースをエンコード
                src_mask = self.make_src_mask(src)
                enc_src = self.encoder(src, src_mask)

                # デコーダーの初期入力（<s>トークン）
                tgt = torch.full((batch_size, 1), start_token, dtype=torch.long, device=device)

                # キャッシュの初期化
                cache = None

                # 生成されたトークンIDを格納
                output_ids = []

                # 各シーケンスの終了状態を追跡
                finished = torch.zeros(batch_size, dtype=torch.bool, device=device)

                for _ in range(max_length):
                    # キャッシュを使用する場合は最後のトークンのみをデコーダーに渡す
                    if cache is not None:
                        # キャッシュがある場合は、最後のトークンのみを使用
                        decoder_input = tgt[:, -1:]
                        tgt_mask = self.make_tgt_mask(decoder_input)
                        decoder_output, cache = self.decoder(decoder_input, enc_src, tgt_mask, src_mask, cache)
                    else:
                        # キャッシュがない場合は、全シーケンスを使用
                        tgt_mask = self.make_tgt_mask(tgt)
                        decoder_output, cache = self.decoder(tgt, enc_src, tgt_mask, src_mask, cache)

                    # 最後のトークンの予測を取得 [batch_size, vocab_size]
                    next_token_logits = decoder_output[:, -1, :]

                    # 終了したシーケンスの位置をマスクしてPADトークンを出力
                    # finishedがTrueの位置では、next_token_logitsをPADトークンに設定
                    next_token_logits = next_token_logits.clone()
                    next_token_logits[finished] = float('-inf')
                    next_token_logits[finished, self.tgt_pad_idx] = float('inf')

                    # Greedy search: 最も確率の高いトークンを選択
                    next_token = next_token_logits.argmax(dim=-1, keepdim=True)  # [batch_size, 1]

                    # 終了したシーケンスの位置ではPADトークンに置き換え
                    next_token[finished] = self.tgt_pad_idx

                    # 生成されたトークンを追加（終了していないシーケンスのみ）
                    output_ids.append(next_token)

                    # 終了トークンが生成されたかチェック
                    finished |= (next_token.squeeze(-1) == end_token)
                    if finished.all():
                        break

                    # 次のイテレーションのためにターゲットシーケンスに追加
                    if cache is not None:
                        # キャッシュを使用する場合は、最後のトークンのみを追加
                        tgt = next_token
                    else:
                        # キャッシュを使用しない場合は、全シーケンスに追加
                        tgt = torch.cat([tgt, next_token], dim=1)

                # バッチごとにトークンIDを結合 [batch_size, tgt_len]
                if len(output_ids) == 0:
                    # max_length == 0の場合、空のテンソルを作成
                    output_ids = torch.empty((batch_size, 0), dtype=torch.long, device=device)
                else:
                    output_ids = torch.cat(output_ids, dim=1)

            return output_ids
        finally:
            # 元のトレーニングモードを復元
            self.train(was_training)

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
    hidden_size = hidden_size if hidden_size is not None else CONFIG.model_hyperparameters.hidden_size
    num_heads = num_heads if num_heads is not None else CONFIG.model_hyperparameters.num_heads
    num_layers = num_layers if num_layers is not None else CONFIG.model_hyperparameters.num_layers
    d_ff = d_ff if d_ff is not None else CONFIG.model_hyperparameters.d_ff
    dropout = dropout if dropout is not None else CONFIG.model_hyperparameters.dropout_rate
    max_seq_length = max_seq_length if max_seq_length is not None else CONFIG.model_hyperparameters.max_seq_length

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

    # パラメータの選択的な初期化
    for name, param in model.named_parameters():
        if 'weight' in name:
            if 'embedding' in name:
                # 埋め込み層は正規分布で初期化
                # 埋め込み重みは2Dテンソルである必要がある [vocab_size, embedding_dim]
                if param.dim() != 2:
                    raise ValueError(
                        f"Expected embedding weight to be 2D [vocab_size, embedding_dim], "
                        f"got {param.dim()}-D tensor for parameter '{name}'"
                    )
                std = param.shape[1] ** -0.5
                nn.init.normal_(param, mean=0, std=std)
            elif 'norm' not in name:  # LayerNormは除外（デフォルトの初期化を使用）
                if param.dim() > 1:
                    # その他の重みはXavier uniform初期化
                    nn.init.xavier_uniform_(param)
        elif 'bias' in name:
            if 'norm' not in name:  # LayerNormのbiasは除外
                # バイアスは0で初期化
                nn.init.zeros_(param)

    return model
