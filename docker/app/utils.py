import torch
import torch.nn as nn
from config import CONFIG, MODEL_CONFIG
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction, corpus_bleu
import nltk
import os
import sacrebleu
import logging
import wandb
import numpy as np
import importlib
from importlib import reload

# 評価指標用のダウンロード
def download_nltk_resources():
    """必要なnltkリソースをダウンロードします"""
    try:
        import nltk
        # 必須のリソース
        resources = ['punkt']

        # リソースがまだダウンロードされていない場合にのみダウンロード
        for resource in resources:
            try:
                nltk.data.find(f'tokenizers/{resource}')
                logging.info(f"nltk resource {resource} はすでにダウンロード済みです")
            except LookupError:
                logging.info(f"nltk resource {resource} をダウンロードしています...")
                nltk.download(resource)
                logging.info(f"nltk resource {resource} のダウンロードが完了しました")

        # METEORで特に重要なリソースは削除

    except Exception as e:
        logging.error(f"nltkリソースのダウンロード中にエラーが発生しました: {e}")

def calculate_bleu(references, hypotheses):
    """
    複数の参照訳と仮説訳のBLEUスコアを計算します。

    Args:
        references (list): 参照訳のリスト（各参照はトークン文字列のリスト）
        hypotheses (list): 仮説訳のリスト（各仮説はトークン文字列のリスト）

    Returns:
        float: BLEUスコア（0.0〜1.0）
    """
    # 入力チェック
    if not references or not hypotheses:
        logging.warning("空の参照または仮説が与えられました。BLEU=0を返します。")
        return 0.0

    # 参照訳が1つしかない場合はそれをリストに変換
    if not all(isinstance(ref, list) for ref in references[0]):
        references = [[ref] for ref in references]

    # トークンが整数の場合は文字列に変換
    processed_hypotheses = []
    for hyp in hypotheses:
        if not hyp:
            processed_hypotheses.append([])
            continue

        if isinstance(hyp[0], int):
            processed_hypotheses.append([str(token) for token in hyp])
        else:
            processed_hypotheses.append(hyp)

    processed_references = []
    for refs in references:
        processed_refs = []
        for ref in refs:
            if not ref:
                processed_refs.append([])
                continue

            if isinstance(ref[0], int):
                processed_refs.append([str(token) for token in ref])
            else:
                processed_refs.append(ref)
        processed_references.append(processed_refs)

    # 平滑化関数
    smoothing = SmoothingFunction().method1

    # 各仮説のBLEUスコアを計算し平均をとる
    bleu_scores = []
    for hyp, refs in zip(processed_hypotheses, processed_references):
        # 空のhypothesisを避ける
        if not hyp:
            continue

        # 空の参照を避ける
        valid_refs = [ref for ref in refs if ref]
        if not valid_refs:
            continue

        try:
            score = sentence_bleu(valid_refs, hyp, smoothing_function=smoothing)
            bleu_scores.append(score)
        except Exception as e:
            logging.error(f"BLEU計算でエラーが発生: {e}, hyp={hyp}, refs={valid_refs}")
            continue

    # スコアがない場合は0を返す
    if not bleu_scores:
        return 0.0

    return sum(bleu_scores) / len(bleu_scores)

def calculate_sacrebleu(references, hypotheses):
    """
    SacreBLEUを使用して翻訳品質を評価します。
    SacreBLEUはBLEUのより標準化されたバージョンです。

    Args:
        references (list): 参照訳のリスト（文字列またはトークンのリスト）
        hypotheses (list): 仮説訳のリスト（文字列またはトークンのリスト）

    Returns:
        float: SacreBLEUスコア
    """
    # 入力チェック
    if not references or not hypotheses:
        logging.warning("空の参照または仮説が与えられました。SacreBLEU=0を返します。")
        return 0.0

    if len(references) != len(hypotheses):
        logging.warning(f"参照と仮説の数が一致しません。参照: {len(references)}, 仮説: {len(hypotheses)}")
        # 短い方に合わせる
        length = min(len(references), len(hypotheses))
        references = references[:length]
        hypotheses = hypotheses[:length]

    # トークンから文字列に変換
    # references_processedとhypotheses_processedは必ず文字列になる

    # 参照の処理
    references_processed = []
    if references and isinstance(references[0], list):
        if references[0] and isinstance(references[0][0], list):
            # 参照には複数の翻訳がある場合 [[ref1, ref2, ...], ...]
            for ref_list in references:
                if not ref_list:
                    references_processed.append("")
                    continue

                # 最初の参照のみ使用
                ref = ref_list[0]
                if not ref:
                    references_processed.append("")
                    continue

                if isinstance(ref[0], int):
                    # 整数IDから文字列に変換
                    references_processed.append(' '.join(map(str, ref)))
                else:
                    # 文字列の場合はそのまま結合
                    references_processed.append(' '.join(ref))
        else:
            # 参照が1つのトークンリスト [tokens, ...]
            for ref in references:
                if not ref:
                    references_processed.append("")
                    continue

                if isinstance(ref[0], int):
                    references_processed.append(' '.join(map(str, ref)))
                else:
                    references_processed.append(' '.join(ref))
    elif references and isinstance(references[0], str):
        # すでに文字列
        references_processed = references
    else:
        logging.warning("無効な参照形式です。SacreBLEU=0を返します。")
        return 0.0

    # 仮説の処理
    hypotheses_processed = []
    if hypotheses and isinstance(hypotheses[0], list):
        for hyp in hypotheses:
            if not hyp:
                hypotheses_processed.append("")
                continue

            if isinstance(hyp[0], int):
                hypotheses_processed.append(' '.join(map(str, hyp)))
            else:
                hypotheses_processed.append(' '.join(hyp))
    elif hypotheses and isinstance(hypotheses[0], str):
        # すでに文字列
        hypotheses_processed = hypotheses
    else:
        logging.warning("無効な仮説形式です。SacreBLEU=0を返します。")
        return 0.0

    # 空の文字列を除外
    valid_pairs = [(hyp, ref) for hyp, ref in zip(hypotheses_processed, references_processed) if hyp and ref]

    if not valid_pairs:
        logging.warning("有効な参照/仮説ペアがありません。SacreBLEU=0を返します。")
        return 0.0

    valid_hyps, valid_refs = zip(*valid_pairs)

    try:
        # sacrebleuが期待する形式に変換：
        # hypotheses: リスト[文字列]
        # references: リスト[リスト[文字列]]（複数の参照訳に対応）
        references_for_sacrebleu = [[ref] for ref in valid_refs]

        # SacreBLEUの入力形式に変換
        corpus_refs = list(zip(*references_for_sacrebleu))

        # SacreBLEUオブジェクトを作成して計算
        bleu = sacrebleu.corpus_bleu(
            valid_hyps,  # 仮説（リスト[文字列]）
            corpus_refs,  # 参照（リスト[リスト[文字列]]）
            tokenize='none'  # すでにトークン化済み
        )

        # スコアを0-1の範囲に正規化
        return bleu.score / 100.0
    except Exception as e:
        logging.error(f"SacreBLEU計算でエラーが発生: {e}")
        logging.error(f"仮説例: {valid_hyps[:1] if valid_hyps else []}")
        logging.error(f"参照例: {valid_refs[:1] if valid_refs else []}")
        return 0.0

# --- マスク生成関数 ---
def create_padding_mask(seq, pad_idx):
    """
    パディングトークンを無視するためのマスクを作成します。
    Args:
        seq (torch.Tensor): 入力シーケンス (batch_size, seq_len)
        pad_idx (int): パディングトークンのインデックス
    Returns:
        torch.Tensor: パディングマスク (batch_size, 1, 1, seq_len)
                      パディングされていない部分は True (1)、パディング部分は False (0)。
    """
    # パディングではない場所を1、パディングの場所を0にする
    mask = (seq != pad_idx).float().unsqueeze(1).unsqueeze(2)
    return mask

def create_subsequent_mask(seq):
    """
    後続のトークンを隠すためのマスクを作成します (デコーダーのセルフアテンション用)。
    Args:
        seq (torch.Tensor): ターゲットシーケンス (batch_size, seq_len)
    Returns:
        torch.Tensor: 後続マスク (batch_size, 1, seq_len, seq_len)
                      見ることができる位置は True (1)、マスクされる位置は False (0)。
    """
    # seq_lenを取得
    batch_size, seq_len = seq.size()

    # 下三角行列を作成（対角成分も含む）
    # 形状: (seq_len, seq_len)
    subsequent_mask = torch.tril(torch.ones((seq_len, seq_len), device=seq.device)).float()

    # バッチ次元とヘッド次元を追加して (batch_size, 1, seq_len, seq_len) の形状にする
    return subsequent_mask.unsqueeze(0).unsqueeze(1).expand(batch_size, 1, seq_len, seq_len)
# --- ここまでマスク生成関数 ---

class WarmupScheduler:
    """
    改良型Warmupスケジューラー。
    ウォームアップフェーズとその後の学習率調整を行います。

    基本的なVaswaniらのスケジュールに加えて、学習率の下限設定とlinear decayオプションを追加。

    Args:
        optimizer: 最適化器
        d_model: モデルの次元数（Vaswaniらの倍率計算に使用）
        warmup_steps: ウォームアップステップ数
        total_steps: 総ステップ数
        min_lr: 最小学習率
        initial_lr: 初期学習率
        decay_method: 減衰方法（'inverse_sqrt' または 'linear'）
    """
    def __init__(self, optimizer, d_model, warmup_steps, total_steps, min_lr=1e-5,
                 initial_lr=None, decay_method='inverse_sqrt'):
        self.optimizer = optimizer
        self.d_model = d_model
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.current_step = 0
        self.min_lr = min_lr
        self.initial_lr = initial_lr or optimizer.param_groups[0]['lr']
        self.decay_method = decay_method

        # Linear decayの場合の終了学習率を設定
        self.final_lr = min_lr if decay_method == 'linear' else None

        # Tensorboardでの可視化のために学習率の履歴を保存
        self.lr_history = []

    def step(self):
        """
        ステップごとに学習率を更新します。
        """
        self.current_step += 1

        if self.current_step < self.warmup_steps:
            # ウォームアップフェーズ: 学習率を徐々に上げる
            lr = self.initial_lr * (self.current_step / self.warmup_steps)
        else:
            # ウォームアップ後の減衰フェーズ
            if self.decay_method == 'inverse_sqrt':
                # Vaswaniらの論文による逆平方根減衰
                lr = self.initial_lr * (self.warmup_steps ** 0.5) / (self.current_step ** 0.5)
            elif self.decay_method == 'linear':
                # 線形減衰
                remaining_steps = self.total_steps - self.current_step
                total_decay_steps = self.total_steps - self.warmup_steps
                decay_factor = remaining_steps / total_decay_steps

                # 線形補間: lr = final_lr + (initial_lr - final_lr) * decay_factor
                lr = self.final_lr + (self.initial_lr - self.final_lr) * decay_factor
            else:
                raise ValueError(f"Unknown decay method: {self.decay_method}")

        # 最小学習率を下回らないようにする
        lr = max(self.min_lr, lr)

        # 学習率を設定
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

        # 履歴に追加
        self.lr_history.append(lr)

        return lr

    def state_dict(self):
        """
        スケジューラーの状態を保存するためのstate dictionaryを返します。
        """
        return {
            'current_step': self.current_step,
            'initial_lr': self.initial_lr,
            'lr_history': self.lr_history,
            'warmup_steps': self.warmup_steps,
            'total_steps': self.total_steps,
            'min_lr': self.min_lr,
            'decay_method': self.decay_method,
            'final_lr': self.final_lr
        }

    def load_state_dict(self, state_dict):
        """
        保存された状態をロードします。

        Args:
            state_dict: スケジューラーの状態を含む辞書
        """
        self.current_step = state_dict['current_step']
        self.initial_lr = state_dict['initial_lr']
        self.lr_history = state_dict['lr_history']
        self.warmup_steps = state_dict['warmup_steps']
        self.total_steps = state_dict['total_steps']
        self.min_lr = state_dict['min_lr']
        self.decay_method = state_dict['decay_method']
        self.final_lr = state_dict['final_lr']

def validate(model, val_loader, criterion, device, output_dim, output_vocab):
    model.eval()
    total_val_loss = 0
    references = []
    hypotheses = []

    # 逆引きボキャブラリを作成（IDからトークン文字列への変換用）
    id_to_token = {v: k for k, v in output_vocab.items()}

    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            # 長いシーケンスを切り詰め
            if X_batch.size(1) > CONFIG["MAX_SEQ_LENGTH"]:
                X_batch = X_batch[:, :CONFIG["MAX_SEQ_LENGTH"]]
            if y_batch.size(1) > CONFIG["MAX_SEQ_LENGTH"]:
                y_batch = y_batch[:, :CONFIG["MAX_SEQ_LENGTH"]]

            # デコーダーへの入力とターゲットを作成
            max_len = y_batch.size(1) - 1

            decoder_input = y_batch[:, :max_len]
            start_token_tensor = torch.full(
                (y_batch.size(0), 1),
                output_vocab['<s>'],
                dtype=torch.long,
                device=device
            )
            decoder_input = torch.cat((start_token_tensor, decoder_input[:, :-1]), dim=1)
            target_output = y_batch[:, 1:max_len+1]

            # 順伝播
            decoder_output, _ = model(X_batch, decoder_input)

            # 損失計算
            loss = criterion(
                decoder_output.view(-1, output_dim),
                target_output.reshape(-1)
            )
            total_val_loss += loss.item()

            # BLEU評価用のリファレンスと仮説を準備
            # 予測の生成
            pred_indices = decoder_output.argmax(dim=-1)

            # バッチ内の各サンプルに対して
            for i in range(X_batch.size(0)):
                # 参照訳（ゴールド）の準備 - パディングを除去
                ref = target_output[i].cpu().tolist()
                ref = [token for token in ref if token != output_vocab['<pad>']]

                # トークンIDを実際のトークン文字列に変換
                ref_tokens = []
                for token_id in ref:
                    if token_id in id_to_token:
                        ref_tokens.append(id_to_token[token_id])
                    else:
                        ref_tokens.append(f"<unknown_{token_id}>")

                references.append([ref_tokens])  # BLEUはリファレンスのリストを想定

                # 仮説訳（予測）の準備 - パディングを除去
                hyp = pred_indices[i].cpu().tolist()
                hyp = [token for token in hyp if token != output_vocab['<pad>']]

                # トークンIDを実際のトークン文字列に変換
                hyp_tokens = []
                for token_id in hyp:
                    if token_id in id_to_token:
                        hyp_tokens.append(id_to_token[token_id])
                    else:
                        hyp_tokens.append(f"<unknown_{token_id}>")

                hypotheses.append(hyp_tokens)

    # 評価指標の計算
    metrics = {}
    try:
        # BLEUスコア計算
        bleu_score = calculate_bleu(references, hypotheses)
        metrics["bleu"] = bleu_score
        logging.info(f"BLEUスコア: {bleu_score:.4f}")

        # SacreBLEU計算
        sacrebleu_score = calculate_sacrebleu(references, hypotheses)
        metrics["sacrebleu"] = sacrebleu_score
        logging.info(f"SacreBLEUスコア: {sacrebleu_score:.4f}")

    except Exception as e:
        logging.error(f"評価指標の計算中にエラーが発生しました: {e}")
        # エラーが発生した場合はゼロのスコアを返す
        metrics = {"bleu": 0, "sacrebleu": 0}

    # 平均検証損失を返す
    val_loss = total_val_loss / len(val_loader)
    return val_loss, metrics["bleu"]

class TranslationModel(nn.Module):
    def __init__(self, encoder, decoder, src_pad_idx, tgt_pad_idx, device):
        super(TranslationModel, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_pad_idx = src_pad_idx
        self.tgt_pad_idx = tgt_pad_idx
        self.device = device

    def forward(self, src, tgt_input, cache=None):
        src_mask = create_padding_mask(src, self.src_pad_idx).to(self.device)
        tgt_sub_mask = create_subsequent_mask(tgt_input)  # すでにバッチ次元を含む
        tgt_pad_mask = create_padding_mask(tgt_input, self.tgt_pad_idx).to(self.device)

        # ターゲットマスクは、パディングマスクと後続マスクの論理積
        # tgt_pad_mask: (batch_size, 1, 1, tgt_len) -> (batch_size, 1, tgt_len, tgt_len)に拡張
        tgt_mask = torch.logical_and(tgt_pad_mask.expand(-1, -1, tgt_input.size(1), -1), tgt_sub_mask)

        # エンコーダ出力に適用するマスク
        memory_mask = src_mask

        encoder_output = self.encoder(src, src_mask)

        # キャッシュ付きのデコーダー
        decoder_output, new_cache = self.decoder(tgt_input, encoder_output, tgt_mask, memory_mask, cache=cache)
        return decoder_output, new_cache
