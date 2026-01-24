"""
モデルの評価とBLEUスコアの計算を担当します。
"""
import torch
import logging
import time
import traceback
import sacrebleu
from typing import Optional, Dict, List, Tuple, Any, Union
from utils.config import CONFIG
from utils.constants import (
    LOG_INTERVAL_BATCHES, DEFAULT_MAX_SAMPLES_PER_BATCH,
    DEFAULT_BLEU_SAMPLE_BATCHES, DEFAULT_MAX_BLEU_SAMPLES,
    BLEU_SCORE_NORMALIZATION_FACTOR
)
from utils.logging_config import setup_logging

# ロギング設定
setup_logging()

def evaluate(
    model: torch.nn.Module,
    valid_loader: torch.utils.data.DataLoader,
    criterion: torch.nn.Module,
    device: torch.device,
    config: Any,
    tgt_vocab: Optional[Dict[str, int]] = None
) -> Tuple[float, float]:
    """
    モデルを評価する関数（性能最適化版）

    Args:
        model: 評価するモデル
        valid_loader: 検証データローダー
        criterion: 損失関数
        device: 評価を実行するデバイス
        config: 設定情報
        tgt_vocab: 対象言語の語彙辞書（BLEUスコア計算用）

    Returns:
        tuple: (平均損失, BLEUスコア)
    """
    model.eval()
    total_loss = 0
    batch_count = 0

    # BLEUスコア計算用のリストを初期化
    all_references = []
    all_hypotheses = []

    # 戻り値の初期化（例外発生時でも安全に返せるように）
    avg_loss = float('inf')
    bleu_score = 0.0

    # BLEUスコア計算用の最小サンプル数
    bleu_sample_batches = getattr(config, 'bleu_sample_batches', DEFAULT_BLEU_SAMPLE_BATCHES)
    max_bleu_samples = min(bleu_sample_batches, len(valid_loader))
    bleu_sample_count = 0

    # 性能計測用変数
    total_batch_time = 0
    total_inference_time = 0

    # 最大シーケンス長（メモリ節約のため必要に応じて切り捨て）
    model_hyperparameters = getattr(config, 'model_hyperparameters', None)
    max_length = getattr(model_hyperparameters, 'max_seq_length', 512) if model_hyperparameters is not None else 512
    if model_hyperparameters is None or not hasattr(model_hyperparameters, 'max_seq_length'):
        logging.warning(
            "config.model_hyperparametersが存在しないか、max_seq_length属性がありません。"
            "デフォルト値512を使用します。"
        )

    # 評価する最大バッチ数（性能向上のため削減）
    training_cfg = getattr(config, 'training_config', None)
    requested_max = getattr(training_cfg, 'max_eval_batches', None) if training_cfg is not None else None
    max_batches = min(requested_max, len(valid_loader)) if requested_max is not None else len(valid_loader)

    logging.info(f"評価開始: {max_batches}バッチを評価、BLEUスコア用に最大{max_bleu_samples}バッチを使用")

    # BLEU計算用の特殊トークンIDを事前に取得
    pad_id = None
    eos_id = None
    if tgt_vocab is not None:
        if '<pad>' in tgt_vocab:
            pad_id = tgt_vocab['<pad>']
        if '<eos>' in tgt_vocab:
            eos_id = tgt_vocab['<eos>']

    try:
        with torch.no_grad():
            for batch_idx, batch in enumerate(valid_loader):
                if batch_idx >= max_batches:
                    logging.info(f"最大バッチ数({max_batches})に達したため評価を終了")
                    break

                batch_start_time = time.time()

                # データをGPUに転送
                src = batch[0].to(device, non_blocking=True)
                tgt = batch[1].to(device, non_blocking=True)

                # メモリを節約するためにシーケンスを切り捨て
                if src.size(1) > max_length:
                    src = src[:, :max_length]
                if tgt.size(1) > max_length:
                    tgt = tgt[:, :max_length]

                # 入力と出力を分離
                tgt_input = tgt[:, :-1]
                tgt_output = tgt[:, 1:]

                # 推論時間計測
                inference_start = time.time()
                # 混合精度を使用（FP16/BF16）
                with torch.amp.autocast(device_type='cuda', enabled=device.type=='cuda'):
                    output = model(src, tgt_input)
                inference_time = time.time() - inference_start
                total_inference_time += inference_time

                # 損失計算
                output_dim = output.shape[-1]
                output_flat = output.contiguous().view(-1, output_dim)
                tgt_output_flat = tgt_output.contiguous().view(-1)

                loss = criterion(output_flat, tgt_output_flat)
                total_loss += loss.item()

                # BLEU計算用サンプル収集（最初のいくつかのバッチのみ、かつ効率化）
                if batch_idx < max_bleu_samples and tgt_vocab is not None and bleu_sample_count < DEFAULT_MAX_BLEU_SAMPLES:
                    try:
                        # 少数のサンプルだけをデコード（バッチの最初の数個のみ）
                        max_samples_per_batch = min(src.size(0), DEFAULT_MAX_SAMPLES_PER_BATCH)

                        # BLEUスコア計算のためのデコード（バッチの一部のみ）
                        batch_references, batch_hypotheses = decode_for_bleu(
                            output[:max_samples_per_batch],
                            tgt_output[:max_samples_per_batch],
                            tgt_vocab,
                            pad_id=pad_id,
                            eos_id=eos_id
                        )

                        all_references.extend(batch_references)
                        all_hypotheses.extend(batch_hypotheses)
                        bleu_sample_count += len(batch_references)

                        if bleu_sample_count >= DEFAULT_MAX_BLEU_SAMPLES:
                            logging.info(f"BLEUスコア計算用に十分なサンプル({bleu_sample_count}個)を収集しました")
                    except Exception as e:
                        logging.error(f"BLEUデコードエラー: {e}")

                # バッチ時間計測
                batch_time = time.time() - batch_start_time
                total_batch_time += batch_time

                # ログ出力（定期的）- ログ記録頻度を減らして高速化
                if (batch_idx + 1) % LOG_INTERVAL_BATCHES == 0 or batch_idx == max_batches - 1:
                    avg_batch_time = total_batch_time / (batch_idx + 1)
                    avg_inference_time = total_inference_time / (batch_idx + 1)
                    logging.info(f"評価進捗: [{batch_idx+1}/{max_batches}], "
                                f"バッチ時間: {avg_batch_time:.4f}秒, "
                                f"推論時間: {avg_inference_time:.4f}秒")

                # 中間変数の削除（メモリ効率化）
                del src, tgt, tgt_input, tgt_output, output, output_flat, tgt_output_flat

                batch_count += 1

            # 評価結果
            avg_loss = total_loss / batch_count if batch_count > 0 else float('inf')

            # BLEUスコア計算
            bleu_score = 0.0
            if all_hypotheses and all_references:
                bleu_score = calculate_bleu_score(all_hypotheses, all_references)
                logging.info(f"BLEU: {bleu_score:.4f} (サンプル数: {len(all_hypotheses)})")

            # 性能統計
            if batch_count > 0:
                avg_batch_time = total_batch_time / batch_count
                logging.info(f"評価完了: 平均バッチ時間: {avg_batch_time:.4f}秒, 平均損失: {avg_loss:.4f}")

    except Exception as e:
        logging.error(f"評価中にエラーが発生: {e}")
        logging.error(traceback.format_exc())
        # エラーを再発生させて呼び出し元で処理できるようにする
        raise

    return avg_loss, bleu_score

def decode_for_bleu(
    output: torch.Tensor,
    tgt_output: torch.Tensor,
    tgt_vocab: Dict[str, int],
    pad_id: Optional[int] = None,
    eos_id: Optional[int] = None
) -> Tuple[List[List[List[str]]], List[List[str]]]:
    """
    BLEUスコア計算のためにモデル出力と正解データをデコードする

    Args:
        output: モデルの出力（ロジット）
        tgt_output: 正解データ（トークンID）
        tgt_vocab: 対象言語の語彙辞書
        pad_id: パディングトークンのID（オプション、提供されない場合はtgt_vocabから取得）
        eos_id: 終了トークンのID（オプション、提供されない場合はtgt_vocabから取得）

    Returns:
        tuple: (参照文のリスト（List[List[List[str]]]）、予測文のリスト（List[List[str]]）)

    Raises:
        ValueError: pad_idまたはeos_idが提供されず、tgt_vocabにも存在しない場合
    """
    # 逆引き辞書（ID→単語）を作成
    id2word = {v: k for k, v in tgt_vocab.items()}

    # 出力を予測クラスに変換
    output_dim = output.shape[-1]
    output_reshaped = output.view(-1, output_dim)
    pred_tokens = output_reshaped.argmax(dim=1)

    # 正解と予測を元の形状に戻す（堅牢な形状処理）
    # tgt_outputの次元数に基づいてbatch_sizeとseq_lenを計算
    if tgt_output.dim() > 1:
        batch_size = tgt_output.size(0)
        seq_len = tgt_output.size(1)
    else:
        # 1次元の場合
        batch_size = 1
        seq_len = tgt_output.numel()

    # pred_tokensとtgt_outputの要素数が一致することを確認
    if pred_tokens.numel() != tgt_output.numel():
        raise ValueError(
            f"pred_tokensの要素数 ({pred_tokens.numel()}) と "
            f"tgt_outputの要素数 ({tgt_output.numel()}) が一致しません。"
        )

    # 形状を再構築
    pred_tokens = pred_tokens.view(batch_size, -1)
    tgt_reshaped = tgt_output.view(batch_size, -1)

    # CPUに移動してリスト化
    pred_tokens_cpu = pred_tokens.detach().cpu().tolist()
    tgt_tokens_cpu = tgt_reshaped.detach().cpu().tolist()

    # 特殊トークンのIDを取得
    if pad_id is None:
        if '<pad>' not in tgt_vocab:
            raise ValueError(
                "pad_idが提供されておらず、tgt_vocabに'<pad>'キーが存在しません。"
                "pad_idパラメータを明示的に指定するか、tgt_vocabに'<pad>'キーを追加してください。"
            )
        pad_id = tgt_vocab['<pad>']

    if eos_id is None:
        if '<eos>' not in tgt_vocab:
            raise ValueError(
                "eos_idが提供されておらず、tgt_vocabに'<eos>'キーが存在しません。"
                "eos_idパラメータを明示的に指定するか、tgt_vocabに'<eos>'キーを追加してください。"
            )
        eos_id = tgt_vocab['<eos>']

    # 変換結果格納用リスト
    target_sentences: List[List[List[str]]] = []
    predicted_sentences: List[List[str]] = []

    # 除外されたサンプル数をカウント
    excluded_empty_pairs = 0
    total_samples = len(pred_tokens_cpu)

    # 各サンプルについて処理
    for pred, target in zip(pred_tokens_cpu, tgt_tokens_cpu):
        # パディングとEOSトークン以降を除去
        pred_clean = []
        for token_id in pred:
            if token_id == pad_id or token_id == eos_id:
                break
            pred_clean.append(id2word.get(token_id, '<unk>'))

        target_clean = []
        for token_id in target:
            if token_id == pad_id or token_id == eos_id:
                break
            target_clean.append(id2word.get(token_id, '<unk>'))

        # 両方が非空の場合のみ追加（BLEUスコアの計算に影響するため）
        if pred_clean and target_clean:
            predicted_sentences.append(pred_clean)  # List[List[str]]: 各予測文はトークンのリスト
            target_sentences.append([target_clean])  # List[List[List[str]]]: BLEUの形式に合わせて参照訳をリストのリストに
        else:
            excluded_empty_pairs += 1

    # 除外されたサンプル数をログに記録
    if excluded_empty_pairs > 0:
        logging.info(
            f"BLEU計算用サンプルフィルタリング: "
            f"総サンプル数={total_samples}, "
            f"除外されたサンプル数={excluded_empty_pairs} "
            f"(pred_tokensまたはtarget_tokensが空のため)"
        )

    return target_sentences, predicted_sentences

def calculate_sacrebleu(
    references: Union[List[List[str]], List[List[List[str]]]],
    hypotheses: List[List[str]],
    strict: bool = False
) -> float:
    """
    SacreBLEUを使用して翻訳品質を評価します。
    SacreBLEUはBLEUのより標準化されたバージョンです。

    Args:
        references: 参照訳のリスト。以下のいずれかの形式:
            - List[List[str]]: 各参照訳がトークンのリスト
            - List[List[List[str]]]: 各参照訳が複数の翻訳候補を含む（最初の候補のみ使用）
        hypotheses: 仮説訳のリスト（List[List[str]]: 各仮説訳はトークンのリスト）
        strict: Trueの場合、referencesとhypothesesの長さが異なるとValueErrorを発生させる。
                Falseの場合、警告を出して短い方に合わせて切り詰める（デフォルト: False）

    Returns:
        float: SacreBLEUスコア

    Raises:
        ValueError: strict=Trueかつreferencesとhypothesesの長さが異なる場合
    """
    # 入力チェック
    if not references or not hypotheses:
        logging.warning("空の参照または仮説が与えられました。SacreBLEU=0を返します。")
        return 0.0

    if len(references) != len(hypotheses):
        if strict:
            raise ValueError(
                f"references ({len(references)}) と hypotheses ({len(hypotheses)}) の長さが異なります。"
                f"strict=Trueのため、処理を中断します。"
            )
        else:
            logging.warning(
                f"参照と仮説の数が一致しません。参照: {len(references)}, 仮説: {len(hypotheses)}。"
                f"短い方に合わせて切り詰めます。"
            )
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
        references_processed = list(references)
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
        hypotheses_processed = list(hypotheses)
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
        return bleu.score / BLEU_SCORE_NORMALIZATION_FACTOR
    except Exception as e:
        logging.error(f"SacreBLEU計算でエラーが発生: {e}")
        logging.error(f"仮説例: {valid_hyps[:1] if valid_hyps else []}")
        logging.error(f"参照例: {valid_refs[:1] if valid_refs else []}")
        return 0.0

def calculate_bleu_score(
    hypotheses: List[List[str]],
    references: List[List[List[str]]]
) -> float:
    """
    BLEUスコアを計算する関数（SacreBLEUを使用）

    Args:
        hypotheses: 予測文のリスト（List[List[str]]: 各予測文はトークンのリスト）
        references: 参照文のリスト（List[List[List[str]]]: BLEUの形式に合わせたリストのリスト）

    Returns:
        float: BLEUスコア
    """
    return calculate_sacrebleu(references=references, hypotheses=hypotheses)
