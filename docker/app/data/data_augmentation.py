import torch
import random
from tqdm import tqdm
import os
import logging
from typing import List, Tuple, Dict

from utils.config import CONFIG
from utils.constants import (
    DEFAULT_MASK_PROB, DEFAULT_DELETION_PROB, DEFAULT_REPLACEMENT_PROB,
    DEFAULT_PERMUTATION_PROB, DEFAULT_WINDOW_SIZE, DEFAULT_MIN_SEQUENCE_LENGTH,
    MAX_DATA_AUGMENTATION_ERRORS, DEFAULT_BATCH_SIZE_SMALL_VRAM
)
from utils.logging_config import setup_logging
from data.tokenizer_utils import normalize_text, tokenize_with_sentencepiece
from data.tokenizer_utils import train_and_load_sp_models

# ロギング設定
setup_logging()


def normalize_lang_code(lang_code: str) -> str:
    """言語コードを正規化する。

    Args:
        lang_code: 正規化する言語コード（例: "ja", "jpn", "ja_JP", "JA"）

    Returns:
        正規化された言語コード（小文字、アンダースコア区切り）
    """
    if not lang_code:
        return lang_code

    # 小文字に変換してトリム
    normalized = lang_code.strip().lower()

    # 言語コードのエイリアスマッピング
    lang_aliases = {
        # 日本語のエイリアス
        "ja": "ja",
        "jpn": "ja",
        "japanese": "ja",
        "ja_jp": "ja",
        "ja-jp": "ja",
        # 英語のエイリアス
        "en": "en",
        "eng": "en",
        "english": "en",
        "en_us": "en",
        "en-us": "en",
        "en_gb": "en",
        "en-gb": "en",
    }

    # エイリアスをチェック
    if normalized in lang_aliases:
        return lang_aliases[normalized]

    # エイリアスにない場合は、アンダースコアやハイフンで分割して最初の部分を返す
    # 例: "ja_JP" -> "ja", "en-US" -> "en"
    parts = normalized.replace("-", "_").split("_")
    base_lang = parts[0] if parts else normalized

    # ベース言語がエイリアスにある場合はそれを使用
    if base_lang in lang_aliases:
        return lang_aliases[base_lang]

    # それでも見つからない場合は正規化された値を返す
    return normalized


def resolve_lang_alias(lang_code: str, canonical_source: str) -> bool:
    """言語コードが正規化されたソース言語と一致するかチェックする。

    Args:
        lang_code: チェックする言語コード
        canonical_source: 正規化されたソース言語コード（例: "ja"）

    Returns:
        言語コードがソース言語と一致する場合True
    """
    normalized_lang = normalize_lang_code(lang_code)
    normalized_source = normalize_lang_code(canonical_source)
    return normalized_lang == normalized_source


class DataAugmentor:
    """データ拡張を行うクラス"""

    def __init__(self, sp_src, sp_tgt, translation_model=None):
        """
        Args:
            sp_src: ソース言語のSentencePieceモデル
            sp_tgt: ターゲット言語のSentencePieceモデル
            translation_model: 逆翻訳に使用するモデル（オプション）
        """
        self.sp_src = sp_src
        self.sp_tgt = sp_tgt
        self.translation_model = translation_model
        # モデルをデバイスに移動（一度だけ）
        if self.translation_model is not None:
            self.translation_model = self.translation_model.to(CONFIG.device)
            self.translation_model.eval()
            # パラメータからデバイスを取得、パラメータがない場合はフォールバック
            first_param = next(self.translation_model.parameters(), None)
            if first_param is not None:
                self._model_device = first_param.device
            elif hasattr(self.translation_model, 'device'):
                self._model_device = self.translation_model.device
            else:
                self._model_device = torch.device('cpu')
        else:
            self._model_device = None

    def token_masking(self, token_ids: List[int], mask_prob: float = DEFAULT_MASK_PROB, rng: random.Random = None,
                      tokenizer=None) -> List[int]:
        """
        トークンの一部をマスクする拡張手法

        Args:
            token_ids: 入力トークンID列
            mask_prob: マスクする確率
            rng: ランダム数生成器（Noneの場合はグローバルrandomを使用）
            tokenizer: 使用するSentencePieceトークナイザー（Noneの場合はsp_srcを使用）

        Returns:
            拡張されたトークンID列
        """
        if rng is None:
            rng = random

        if not token_ids or not isinstance(token_ids, list):
            return token_ids if isinstance(token_ids, list) else []

        # 各要素が整数であることを確認
        valid_tokens = []
        for token in token_ids:
            try:
                valid_tokens.append(int(token))
            except (ValueError, TypeError):
                # 整数に変換できない場合はスキップ
                logging.warning(f"token_ids内に整数でない要素があります: {token}")
                continue

        result = valid_tokens.copy()
        if not result:
            return result

        # 使用するトークナイザーを決定
        if tokenizer is None:
            tokenizer = self.sp_src

        # トークナイザーから<unk>トークンIDを動的に取得
        mask_id = 1  # デフォルト値（フォールバック用）
        try:
            if hasattr(tokenizer, 'unk_id') and callable(tokenizer.unk_id):
                unk_id = tokenizer.unk_id()
                if unk_id >= 0:
                    mask_id = unk_id
        except Exception as e:
            logging.warning(f"<unk>トークンIDの取得中にエラーが発生しました: {e}。デフォルト値({mask_id})を使用します。")

        # マスク対象の位置をランダムに選択
        for i in range(len(result)):
            if rng.random() < mask_prob:
                result[i] = mask_id

        return result

    def token_deletion(self, token_ids: List[int], del_prob: float = DEFAULT_DELETION_PROB, rng: random.Random = None) -> List[int]:
        """
        トークンの一部を削除する拡張手法

        Args:
            token_ids: 入力トークンID列
            del_prob: 削除する確率
            rng: ランダム数生成器（Noneの場合はグローバルrandomを使用）

        Returns:
            拡張されたトークンID列
        """
        if rng is None:
            rng = random

        if not token_ids or not isinstance(token_ids, list):
            return token_ids if isinstance(token_ids, list) else []

        # 各要素が整数であることを確認
        valid_tokens = []
        for token in token_ids:
            try:
                valid_tokens.append(int(token))
            except (ValueError, TypeError):
                # 整数に変換できない場合はスキップ
                logging.warning(f"token_ids内に整数でない要素があります: {token}")
                continue

        if len(valid_tokens) <= DEFAULT_MIN_SEQUENCE_LENGTH:
            return valid_tokens

        result = []

        # 削除対象の位置をランダムに選択
        for i in range(len(valid_tokens)):
            if rng.random() >= del_prob:  # 削除しない場合
                result.append(valid_tokens[i])

        return result if result else valid_tokens  # 空になった場合は元に戻す

    def token_replacement(self, token_ids: List[int], replace_prob: float = DEFAULT_REPLACEMENT_PROB, rng: random.Random = None,
                          tokenizer=None) -> List[int]:
        """
        トークンの一部をランダムに置換する拡張手法

        Args:
            token_ids: 入力トークンID列
            replace_prob: 置換する確率
            rng: ランダム数生成器（Noneの場合はグローバルrandomを使用）
            tokenizer: 使用するSentencePieceトークナイザー（Noneの場合はsp_srcを使用）

        Returns:
            拡張されたトークンID列
        """
        if rng is None:
            rng = random

        if not token_ids or not isinstance(token_ids, list):
            return token_ids if isinstance(token_ids, list) else []

        # 各要素が整数であることを確認
        valid_tokens = []
        for token in token_ids:
            try:
                valid_tokens.append(int(token))
            except (ValueError, TypeError):
                # 整数に変換できない場合はスキップ
                logging.warning(f"token_ids内に整数でない要素があります: {token}")
                continue

        result = valid_tokens.copy()
        if not result:
            return result

        # 使用するトークナイザーを決定
        if tokenizer is None:
            tokenizer = self.sp_src

        # トークナイザーから特殊トークンIDを動的に取得
        special_ids = set()
        try:
            # SentencePieceの特殊トークンIDを取得
            if hasattr(tokenizer, 'pad_id') and callable(tokenizer.pad_id):
                pad_id = tokenizer.pad_id()
                if pad_id >= 0:
                    special_ids.add(pad_id)
            if hasattr(tokenizer, 'unk_id') and callable(tokenizer.unk_id):
                unk_id = tokenizer.unk_id()
                if unk_id >= 0:
                    special_ids.add(unk_id)
            if hasattr(tokenizer, 'bos_id') and callable(tokenizer.bos_id):
                bos_id = tokenizer.bos_id()
                if bos_id >= 0:
                    special_ids.add(bos_id)
            if hasattr(tokenizer, 'eos_id') and callable(tokenizer.eos_id):
                eos_id = tokenizer.eos_id()
                if eos_id >= 0:
                    special_ids.add(eos_id)
        except Exception as e:
            logging.warning(f"特殊トークンIDの取得中にエラーが発生しました: {e}。デフォルト値を使用します。")
            # フォールバック: 一般的な特殊トークンID（0-3）を仮定
            special_ids = {0, 1, 2, 3}

        # 特殊トークンIDを除外したIDのリストを作成
        vocab_size = len(tokenizer)
        non_special_ids = [i for i in range(vocab_size) if i not in special_ids]
        if not non_special_ids:
            # すべてが特殊トークンの場合（異常なケース）
            logging.error(f"すべてのIDが特殊トークンです。置換をスキップします。")
            return result

        # 置換対象の位置をランダムに選択して置換
        for i in range(len(result)):
            if rng.random() < replace_prob:
                # 特殊トークンを除外したIDからランダムに選択
                result[i] = rng.choice(non_special_ids)

        return result

    def token_permutation(self, token_ids: List[int], window_size: int = DEFAULT_WINDOW_SIZE, perm_prob: float = DEFAULT_PERMUTATION_PROB, rng: random.Random = None) -> List[int]:
        """
        トークンの順序を局所的に入れ替える拡張手法

        Args:
            token_ids: 入力トークンID列
            window_size: 入れ替えを行う窓サイズ
            perm_prob: 入れ替えを行う確率
            rng: ランダム数生成器（Noneの場合はグローバルrandomを使用）

        Returns:
            拡張されたトークンID列
        """
        if rng is None:
            rng = random

        # window_sizeの早期検証
        if not isinstance(window_size, int) or window_size <= 0:
            logging.warning(f"window_sizeは正の整数である必要があります。現在の値: {window_size}。元のデータを返します。")
            return token_ids if isinstance(token_ids, list) else []

        if not token_ids or not isinstance(token_ids, list):
            return token_ids if isinstance(token_ids, list) else []

        # 各要素が整数であることを確認
        valid_tokens = []
        for token in token_ids:
            try:
                valid_tokens.append(int(token))
            except (ValueError, TypeError):
                # 整数に変換できない場合はスキップ
                logging.warning(f"token_ids内に整数でない要素があります: {token}")
                continue

        if len(valid_tokens) <= window_size:
            return valid_tokens

        result = valid_tokens.copy()

        # 窓内でのトークン入れ替え
        for i in range(0, len(result) - window_size + 1, window_size):
            if rng.random() < perm_prob:
                window = result[i:i+window_size]
                rng.shuffle(window)
                result[i:i+window_size] = window

        return result

    def back_translation(self, src_texts: List[str], src_lang: str, tgt_lang: str, batch_size: int = DEFAULT_BATCH_SIZE_SMALL_VRAM) -> List[str]:
        """
        逆翻訳によるデータ拡張

        Args:
            src_texts: 元の言語テキストのリスト
            src_lang: 元の言語コード
            tgt_lang: 翻訳先言語コード
            batch_size: バッチサイズ（デフォルト: DEFAULT_BATCH_SIZE_SMALL_VRAM）
                       メモリ安全のため、デフォルトは保守的な値に設定されています。
                       より大きなバッチサイズが必要な場合は明示的に指定してください。

        Returns:
            拡張されたテキストのリスト
        """
        if self.translation_model is None:
            logging.warning("逆翻訳機能はトレーニング済みモデルが必要です。スキップします。")
            return src_texts

        augmented_texts = []

        # バッチ処理のための準備
        batches = [src_texts[i:i+batch_size] for i in range(0, len(src_texts), batch_size)]

        # 元言語 -> 目標言語へ翻訳
        forward_translations = []
        for batch in tqdm(batches, desc="逆翻訳（前方）"):
            try:
                # 元言語から目標言語への翻訳
                translations = self._translate_batch(batch, src_lang, tgt_lang)
                forward_translations.extend(translations)
            except Exception as e:
                logging.error(f"翻訳中にエラーが発生しました: {e}")
                # エラー発生時は元のテキストをそのまま使用
                forward_translations.extend(batch)

        # 目標言語 -> 元言語へ逆翻訳
        batches = [forward_translations[i:i+batch_size] for i in range(0, len(forward_translations), batch_size)]
        for batch_idx, batch in enumerate(tqdm(batches, desc="逆翻訳（後方）")):
            try:
                # 目標言語から元言語への翻訳
                translations = self._translate_batch(batch, tgt_lang, src_lang)
                augmented_texts.extend(translations)
            except Exception as e:
                logging.error(f"逆翻訳中にエラーが発生しました: {e}")
                # エラー発生時は元のテキストをそのまま使用
                # バッチの開始インデックスを計算して、対応するsrc_textsの要素を取得
                batch_start_idx = batch_idx * batch_size
                fallback_texts = []
                for i in range(len(batch)):
                    original_idx = batch_start_idx + i
                    if original_idx < len(src_texts):
                        fallback_texts.append(src_texts[original_idx])
                    else:
                        logging.warning(f"インデックス {original_idx} が範囲外です。元のバッチ要素を使用します。")
                        fallback_texts.append(batch[i])
                augmented_texts.extend(fallback_texts)

        return augmented_texts

    def _translate_batch(self, texts: List[str], src_lang: str, tgt_lang: str) -> List[str]:
        """
        テキストのバッチを翻訳する内部メソッド（真のバッチ処理を実装）
        """
        if self.translation_model is None or self._model_device is None:
            return texts

        if not texts:
            return []

        model = self.translation_model
        model.eval()

        # translate()メソッドの存在を確認
        has_translate = hasattr(model, "translate") and callable(getattr(model, "translate", None))
        has_predict = hasattr(model, "predict") and callable(getattr(model, "predict", None))

        if not has_translate and not has_predict:
            error_msg = (
                f"モデル {type(model).__name__} には 'translate()' メソッドも 'predict()' メソッドもありません。"
                f"翻訳を実行できません。"
            )
            logging.error(error_msg)
            raise AttributeError(error_msg)

        if not has_translate:
            logging.warning(
                f"モデル {type(model).__name__} には 'translate()' メソッドがありません。"
                f"'predict()' メソッドを使用してフォールバックします。"
            )

        # 使用するトークナイザーを決定（言語コードを正規化して比較）
        src_tokenizer = self.sp_src if resolve_lang_alias(src_lang, CONFIG.data_config.translation_source) else self.sp_tgt
        tgt_tokenizer = self.sp_src if resolve_lang_alias(tgt_lang, CONFIG.data_config.translation_source) else self.sp_tgt

        # パディングIDを取得
        try:
            pad_id = src_tokenizer.pad_id() if hasattr(src_tokenizer, 'pad_id') and callable(src_tokenizer.pad_id) else 0
        except Exception:
            pad_id = 0

        # 最大シーケンス長を取得
        max_seq_length = CONFIG.model_hyperparameters.max_seq_length

        # CPU上で全テキストをトークナイズ
        tokenized_texts = []
        for text in texts:
            try:
                tokens = tokenize_with_sentencepiece(text, src_tokenizer, src_lang)
                # 最大長で切り詰め
                if len(tokens) > max_seq_length:
                    tokens = tokens[:max_seq_length]
                tokenized_texts.append(tokens)
            except Exception as e:
                logging.error(f"トークナイズ中にエラーが発生しました: {e}")
                tokenized_texts.append([])  # エラー時は空リスト

        # バッチ内の最大長を取得
        if not tokenized_texts:
            return texts

        # 非空のトークンリストの長さを収集
        lengths = [len(tokens) for tokens in tokenized_texts if tokens]
        if not lengths:
            return texts

        max_len = max(lengths)

        # CPU上でパディングしてバッチテンソルを作成
        batch_tensors = []
        valid_indices = []  # 有効なテキストのインデックス
        for idx, tokens in enumerate(tokenized_texts):
            if not tokens:
                continue
            # パディング
            padded_tokens = tokens + [pad_id] * (max_len - len(tokens))
            batch_tensors.append(padded_tokens)
            valid_indices.append(idx)

        if not batch_tensors:
            return texts

        # バッチテンソルを作成（CPU上）
        batch_tensor = torch.tensor(batch_tensors, dtype=torch.long)

        # 一度だけGPUに移動
        batch_tensor = batch_tensor.to(self._model_device)

        # バッチ全体で翻訳を実行
        result = [None] * len(texts)
        with torch.no_grad():
            try:
                if has_translate:
                    output_tensor = model.translate(batch_tensor)
                elif has_predict:
                    output_tensor = model.predict(batch_tensor)
                else:
                    # この時点で到達することはないはず（上でチェック済み）が、念のため
                    error_msg = (
                        f"モデル {type(model).__name__} には 'translate()' メソッドも 'predict()' メソッドもありません。"
                        f"翻訳を実行できません。"
                    )
                    logging.error(error_msg)
                    raise AttributeError(error_msg)

                # 特殊トークンIDを取得（フィルタリング用）
                special_token_ids = set()
                try:
                    if hasattr(tgt_tokenizer, 'pad_id') and callable(tgt_tokenizer.pad_id):
                        pad_token_id = tgt_tokenizer.pad_id()
                        if pad_token_id is not None and pad_token_id >= 0:
                            special_token_ids.add(pad_token_id)
                    if hasattr(tgt_tokenizer, 'eos_id') and callable(tgt_tokenizer.eos_id):
                        eos_token_id = tgt_tokenizer.eos_id()
                        if eos_token_id is not None and eos_token_id >= 0:
                            special_token_ids.add(eos_token_id)
                    if hasattr(tgt_tokenizer, 'unk_id') and callable(tgt_tokenizer.unk_id):
                        unk_token_id = tgt_tokenizer.unk_id()
                        if unk_token_id is not None and unk_token_id >= 0:
                            special_token_ids.add(unk_token_id)
                    if hasattr(tgt_tokenizer, 'bos_id') and callable(tgt_tokenizer.bos_id):
                        bos_token_id = tgt_tokenizer.bos_id()
                        if bos_token_id is not None and bos_token_id >= 0:
                            special_token_ids.add(bos_token_id)
                except Exception as e:
                    logging.warning(f"特殊トークンIDの取得中にエラーが発生しました: {e}")

                # 各アイテムごとに後処理
                for batch_idx, orig_idx in enumerate(valid_indices):
                    try:
                        # 出力をトークンIDに変換（CPUに移動してから）
                        output_ids_raw = output_tensor[batch_idx].cpu().numpy().tolist()

                        # パディングと特殊トークンを除去し、eos_token_idで停止
                        output_ids = []
                        eos_token_id = None
                        if hasattr(tgt_tokenizer, 'eos_id') and callable(tgt_tokenizer.eos_id):
                            try:
                                eos_token_id = tgt_tokenizer.eos_id()
                            except Exception:
                                pass

                        for token_id in output_ids_raw:
                            # eos_token_idが見つかったら停止
                            if eos_token_id is not None and token_id == eos_token_id:
                                break
                            # 特殊トークンをスキップ
                            if token_id not in special_token_ids:
                                output_ids.append(token_id)

                        # IDをトークンに変換
                        try:
                            if hasattr(tgt_tokenizer, 'decode_ids') and callable(tgt_tokenizer.decode_ids):
                                output_tokens = tgt_tokenizer.decode_ids(output_ids)
                            elif hasattr(tgt_tokenizer, 'decode') and callable(tgt_tokenizer.decode):
                                # decode_idsが存在しない場合はdecodeを使用
                                output_tokens = tgt_tokenizer.decode(output_ids)
                            else:
                                # どちらも存在しない場合はトークンを結合
                                output_tokens = ' '.join(str(token_id) for token_id in output_ids)
                            result[orig_idx] = output_tokens
                        except (AttributeError, TypeError) as e:
                            logging.error(f"出力処理中にエラーが発生しました（デコードメソッドの問題）: {e}")
                            result[orig_idx] = texts[orig_idx]  # エラー時は元のテキストを使用
                    except Exception as e:
                        logging.error(f"出力処理中にエラーが発生しました: {e}")
                        result[orig_idx] = texts[orig_idx]  # エラー時は元のテキストを使用
            except Exception as e:
                logging.error(f"バッチ翻訳中にエラーが発生しました: {e}")
                # エラー時は元のテキストを返す
                return texts

        # エラーで処理されなかったテキストは元のテキストを使用
        for idx, output in enumerate(result):
            if output is None:
                result[idx] = texts[idx]

        return result

    def apply_augmentations(self, token_ids: List[int], techniques: List[str] = None,
                            probs: Dict[str, float] = None, rng: random.Random = None,
                            use_source_tokenizer: bool = True) -> List[int]:
        """
        指定された拡張手法を組み合わせて適用する

        Args:
            token_ids: 入力トークンID列
            techniques: 適用する拡張手法のリスト
            probs: 各手法の適用確率
            rng: ランダム数生成器（Noneの場合はグローバルrandomを使用）
            use_source_tokenizer: Trueの場合はsp_src、Falseの場合はsp_tgtを使用

        Returns:
            拡張されたトークンID列
        """
        if rng is None:
            rng = random

        # token_idsの早期検証と簡素化
        if not isinstance(token_ids, list):
            logging.warning(f"token_idsがリストではありません: {type(token_ids)}。空のリストを返します。")
            return []

        if not token_ids:
            return []

        if techniques is None:
            techniques = ["masking", "deletion", "replacement", "permutation"]

        if probs is None:
            probs = {
                "masking": DEFAULT_MASK_PROB,
                "deletion": DEFAULT_DELETION_PROB,
                "replacement": DEFAULT_REPLACEMENT_PROB,
                "permutation": DEFAULT_PERMUTATION_PROB
            }

        # 使用するトークナイザーを決定
        tokenizer = self.sp_src if use_source_tokenizer else self.sp_tgt

        augmented_ids = token_ids.copy()

        # 拡張処理時のエラーカウント
        error_count = 0

        for technique in techniques:
            try:
                if technique == "masking":
                    augmented_ids = self.token_masking(augmented_ids, mask_prob=probs.get("masking", DEFAULT_MASK_PROB), rng=rng,
                                                      tokenizer=tokenizer)
                elif technique == "deletion":
                    augmented_ids = self.token_deletion(augmented_ids, del_prob=probs.get("deletion", DEFAULT_DELETION_PROB), rng=rng)
                elif technique == "replacement":
                    augmented_ids = self.token_replacement(augmented_ids, replace_prob=probs.get("replacement", DEFAULT_REPLACEMENT_PROB),
                                                           rng=rng, tokenizer=tokenizer)
                elif technique == "permutation":
                    augmented_ids = self.token_permutation(augmented_ids, perm_prob=probs.get("permutation", DEFAULT_PERMUTATION_PROB), rng=rng)
            except Exception as e:
                error_count += 1
                # スタックトレースも含めて詳細なエラー情報を出力
                logging.error(f"{technique}拡張適用中にエラーが発生しました: {e}", exc_info=True)

                # エラーが多すぎる場合は警告
                if error_count > MAX_DATA_AUGMENTATION_ERRORS:
                    logging.warning(f"エラーが{MAX_DATA_AUGMENTATION_ERRORS}回以上発生しました。データ拡張プロセスに問題がある可能性があります。")
                    break

        return augmented_ids

    def apply_pair_augmentations(self, src_tokens: List[int], tgt_tokens: List[int],
                                  techniques: List[str] = None, probs: Dict[str, float] = None,
                                  seed: int = None, parent_rng: random.Random = None) -> Tuple[List[int], List[int]]:
        """
        ソースとターゲットのペアに対して同期された拡張を適用する

        Args:
            src_tokens: ソース言語のトークンID列
            tgt_tokens: ターゲット言語のトークンID列
            techniques: 適用する拡張手法のリスト
            probs: 各手法の適用確率
            seed: ランダムシード（Noneの場合は親RNGまたはローカルRNGから生成）
            parent_rng: 親ランダム数生成器（seedがNoneの場合に使用、再現性を保つため）

        Returns:
            拡張された(src_tokens, tgt_tokens)のタプル
        """
        # ペアごとに決定論的なRNGを生成
        if seed is None:
            if parent_rng is not None:
                # 親RNGからシードを生成（再現性を保つため）
                seed = parent_rng.randint(0, 2**31 - 1)
            else:
                # 決定論的なシードを生成（timeベース）
                import time
                seed = int(time.time() * 1e6) % (2**31)
                logging.warning(f"seedとparent_rngが両方Noneのため、時間ベースのシード({seed})を使用します。再現性が保証されません。")
        # 同じシードで2つの独立したRNGを作成して、srcとtgtが同じ決定論的な拡張を受け取るようにする
        src_rng = random.Random(seed)
        tgt_rng = random.Random(seed)

        # 独立したRNGを使用してsrcとtgtの両方を拡張
        # ソースにはsp_src、ターゲットにはsp_tgtを使用
        aug_src_tokens = self.apply_augmentations(src_tokens, techniques, probs, rng=src_rng, use_source_tokenizer=True)
        aug_tgt_tokens = self.apply_augmentations(tgt_tokens, techniques, probs, rng=tgt_rng, use_source_tokenizer=False)

        return aug_src_tokens, aug_tgt_tokens

def augment_dataset(train_data: List[Tuple[List[int], List[int]]],
                   sp_src, sp_tgt,
                   augmentation_factor: float = 0.3,
                   techniques: List[str] = None,
                   seed: int = None) -> List[Tuple[List[int], List[int]]]:
    """
    トレーニングデータセットを拡張する関数

    Args:
        train_data: オリジナルのトレーニングデータ [(src_tokens, tgt_tokens), ...]
        sp_src: ソース言語のSentencePieceモデル
        sp_tgt: ターゲット言語のSentencePieceモデル
        augmentation_factor: 元のデータセットに対する拡張データの割合
        techniques: 適用する拡張手法のリスト
        seed: ランダムシード（再現性のため、Noneの場合は非決定論的）

    Returns:
        拡張されたトレーニングデータ
    """
    if techniques is None:
        techniques = ["masking", "deletion", "replacement", "permutation"]

    # 決定論的なRNGを作成
    rng = random.Random(seed) if seed is not None else random

    # 拡張するサンプル数を計算
    num_samples = len(train_data)
    num_augmented = int(num_samples * augmentation_factor)

    # サンプル数より多く拡張しようとしていないか確認
    if num_augmented > num_samples:
        num_augmented = num_samples
        logging.warning(f"拡張サンプル数を調整しました: {num_augmented}")

    # データ拡張器の初期化
    augmentor = DataAugmentor(sp_src, sp_tgt)

    # 拡張データを格納するリスト
    augmented_data = []

    # サンプルをランダムに選択して拡張
    indices = rng.sample(range(num_samples), num_augmented)

    for idx in tqdm(indices, desc="データ拡張中"):
        try:
            if idx >= len(train_data):
                logging.warning(f"インデックス {idx} がデータサイズ {len(train_data)} を超えています。スキップします。")
                continue

            sample = train_data[idx]
            if not isinstance(sample, tuple) or len(sample) != 2:
                logging.warning(f"サンプル {idx} が予期された形式ではありません: {sample}。スキップします。")
                continue

            src_tokens, tgt_tokens = sample

            if not isinstance(src_tokens, list) or not isinstance(tgt_tokens, list):
                logging.warning(f"サンプル {idx} のトークンがリスト形式ではありません: {type(src_tokens)}, {type(tgt_tokens)}。スキップします。")
                continue

            # ソースとターゲットのペアに対して同期された拡張を適用
            # ペアごとに決定論的なシードを生成して、srcとtgtで同じランダム状態を使用
            pair_seed = rng.randint(0, 2**31 - 1)
            aug_src_tokens, aug_tgt_tokens = augmentor.apply_pair_augmentations(
                src_tokens, tgt_tokens, techniques, seed=pair_seed, parent_rng=rng
            )

            # 拡張データを追加
            augmented_data.append((aug_src_tokens, aug_tgt_tokens))
        except Exception as e:
            # スタックトレース情報を含めた詳細なログ
            logging.error(f"サンプル {idx} の拡張中にエラーが発生しました: {e}", exc_info=True)
            # 重大なエラーの場合は処理を中断するオプションも考慮
            if isinstance(e, (MemoryError, KeyboardInterrupt)):
                logging.critical("重大なエラーが発生したため処理を中断します")
                raise
            continue

    # 元のデータと拡張データを結合
    logging.info(f"データ拡張完了: 元のデータ {len(train_data)} サンプル + 拡張データ {len(augmented_data)} サンプル")
    return train_data + augmented_data

# 注: setup_back_translation関数は現在使用されていないため削除しました
# この関数は逆翻訳のためのセットアップを行うもので、将来必要になった場合は再実装することができます

# メインの処理（単体テスト用）
if __name__ == "__main__":
    # SentencePieceモデルをロードまたはトレーニング
    try:
        # テスト用のダミーデータ
        test_data = [
            ("これはテストです。", "This is a test."),
            ("私は猫が好きです。", "I like cats."),
            ("明日は晴れるでしょう。", "It will be sunny tomorrow.")
        ]

        # SentencePieceモデルをトレーニング
        sp_src, sp_tgt = train_and_load_sp_models(
            [text[0] for text in test_data],
            [text[1] for text in test_data]
        )

        # データ拡張器の初期化
        augmentor = DataAugmentor(sp_src, sp_tgt)

        # テストデータをトークン化
        tokenized_data = []
        for src_text, tgt_text in test_data:
            src_tokens = tokenize_with_sentencepiece(src_text, sp_src, CONFIG.data_config.translation_source)
            tgt_tokens = tokenize_with_sentencepiece(tgt_text, sp_tgt, CONFIG.data_config.translation_destination)
            tokenized_data.append((src_tokens, tgt_tokens))

        # データ拡張を適用
        augmented_data = augment_dataset(tokenized_data, sp_src, sp_tgt, augmentation_factor=1.0)

        print(f"元のデータ: {len(tokenized_data)} サンプル")
        print(f"拡張後のデータ: {len(augmented_data)} サンプル")

    except Exception as e:
        logging.error(f"データ拡張テスト中にエラーが発生しました: {e}")
