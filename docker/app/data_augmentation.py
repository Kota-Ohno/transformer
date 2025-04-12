import torch
import random
from tqdm import tqdm
import os
import logging
from typing import List, Tuple, Dict

from config import CONFIG
from data import normalize_text, tokenize_with_sentencepiece
# ロギング設定
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def train_and_load_sp_models(train_texts_src, train_texts_tgt):
    """
    ソースとターゲットのSentencePieceモデルをトレーニングして読み込みます。

    Args:
        train_texts_src (list): ソース言語のトレーニングテキスト
        train_texts_tgt (list): ターゲット言語のトレーニングテキスト

    Returns:
        tuple: (ソースモデル, ターゲットモデル)
    """
    # 追加でsentencepieceをインポート
    import sentencepiece as spm
    from data import train_sentencepiece

    # モデルパス
    src_model_prefix = os.path.join("models", "sp_src")
    tgt_model_prefix = os.path.join("models", "sp_tgt")

    # models ディレクトリがない場合は作成
    os.makedirs("models", exist_ok=True)

    # モデルをトレーニング
    print("ソース言語のSentencePieceモデルをトレーニング中...")
    train_sentencepiece(train_texts_src, src_model_prefix)

    print("ターゲット言語のSentencePieceモデルをトレーニング中...")
    train_sentencepiece(train_texts_tgt, tgt_model_prefix)

    # モデルをロード
    sp_src = spm.SentencePieceProcessor()
    sp_tgt = spm.SentencePieceProcessor()

    try:
        sp_src.load(f"{src_model_prefix}.model")
        sp_tgt.load(f"{tgt_model_prefix}.model")
    except Exception as e:
        logging.error(f"SentencePieceモデルのロード中にエラーが発生しました: {e}")
        raise

    return sp_src, sp_tgt

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

    def token_masking(self, token_ids: List[int], mask_prob: float = 0.15) -> List[int]:
        """
        トークンの一部をマスクする拡張手法

        Args:
            token_ids: 入力トークンID列
            mask_prob: マスクする確率

        Returns:
            拡張されたトークンID列
        """
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

        # マスク用のIDを取得（<unk>トークンを使用）
        mask_id = 1  # <unk>のIDを使用

        # マスク対象の位置をランダムに選択
        for i in range(len(result)):
            if random.random() < mask_prob:
                result[i] = mask_id

        return result

    def token_deletion(self, token_ids: List[int], del_prob: float = 0.1) -> List[int]:
        """
        トークンの一部を削除する拡張手法

        Args:
            token_ids: 入力トークンID列
            del_prob: 削除する確率

        Returns:
            拡張されたトークンID列
        """
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

        if len(valid_tokens) <= 3:  # 短すぎる場合は削除しない
            return valid_tokens

        result = []

        # 削除対象の位置をランダムに選択
        for i in range(len(valid_tokens)):
            if random.random() >= del_prob:  # 削除しない場合
                result.append(valid_tokens[i])

        return result if result else valid_tokens  # 空になった場合は元に戻す

    def token_replacement(self, token_ids: List[int], replace_prob: float = 0.1) -> List[int]:
        """
        トークンの一部をランダムに置換する拡張手法

        Args:
            token_ids: 入力トークンID列
            replace_prob: 置換する確率

        Returns:
            拡張されたトークンID列
        """
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

        # 置換用の語彙サイズを取得
        vocab_size_src = len(self.sp_src)

        # 置換対象の位置をランダムに選択して置換
        for i in range(len(result)):
            if random.random() < replace_prob:
                # ソース言語の語彙からランダムに選択（特殊トークンを避ける）
                result[i] = random.randint(4, vocab_size_src - 1)  # 特殊トークンを避ける

        return result

    def token_permutation(self, token_ids: List[int], window_size: int = 3, perm_prob: float = 0.1) -> List[int]:
        """
        トークンの順序を局所的に入れ替える拡張手法

        Args:
            token_ids: 入力トークンID列
            window_size: 入れ替えを行う窓サイズ
            perm_prob: 入れ替えを行う確率

        Returns:
            拡張されたトークンID列
        """
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
            if random.random() < perm_prob:
                window = result[i:i+window_size]
                random.shuffle(window)
                result[i:i+window_size] = window

        return result

    def back_translation(self, src_texts: List[str], src_lang: str, tgt_lang: str, batch_size: int = 32) -> List[str]:
        """
        逆翻訳によるデータ拡張

        Args:
            src_texts: 元の言語テキストのリスト
            src_lang: 元の言語コード
            tgt_lang: 翻訳先言語コード
            batch_size: バッチサイズ

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
        for batch in tqdm(batches, desc="逆翻訳（後方）"):
            try:
                # 目標言語から元言語への翻訳
                translations = self._translate_batch(batch, tgt_lang, src_lang)
                augmented_texts.extend(translations)
            except Exception as e:
                logging.error(f"逆翻訳中にエラーが発生しました: {e}")
                # エラー発生時は元のテキストをそのまま使用
                augmented_texts.extend([src_texts[len(augmented_texts) + i] for i in range(len(batch))])

        return augmented_texts

    def _translate_batch(self, texts: List[str], src_lang: str, tgt_lang: str) -> List[str]:
        """
        テキストのバッチを翻訳する内部メソッド
        """
        if self.translation_model is None:
            return texts

        # ここでモデルを使って実際の翻訳を行う
        # 実装はトレーニング済みモデルに依存
        result = []
        model = self.translation_model
        model.eval()

        with torch.no_grad():
            for text in texts:
                # テキスト正規化
                normalized_text = normalize_text(text, src_lang)

                # トークン化
                if src_lang == CONFIG["TRANSLATION_SOURCE"]:
                    tokens = tokenize_with_sentencepiece(normalized_text, self.sp_src)
                else:
                    tokens = tokenize_with_sentencepiece(normalized_text, self.sp_tgt)

                # トークンをテンソルに変換
                input_tensor = torch.tensor([tokens], dtype=torch.long).to(CONFIG["DEVICE"])

                # 翻訳
                try:
                    output_tensor = model.translate(input_tensor)

                    # 出力をトークンIDに変換
                    output_ids = output_tensor[0].cpu().numpy().tolist()

                    # IDをトークンに変換
                    if tgt_lang == CONFIG["TRANSLATION_SOURCE"]:
                        output_tokens = self.sp_src.decode_ids(output_ids)
                    else:
                        output_tokens = self.sp_tgt.decode_ids(output_ids)

                    result.append(output_tokens)
                except Exception as e:
                    logging.error(f"翻訳処理中にエラーが発生しました: {e}")
                    result.append(text)  # エラー時は元のテキストを使用

        return result

    def apply_augmentations(self, token_ids: List[int], techniques: List[str] = None,
                            probs: Dict[str, float] = None) -> List[int]:
        """
        指定された拡張手法を組み合わせて適用する

        Args:
            token_ids: 入力トークンID列
            techniques: 適用する拡張手法のリスト
            probs: 各手法の適用確率

        Returns:
            拡張されたトークンID列
        """
        if not isinstance(token_ids, list):
            logging.warning(f"token_idsがリストではありません: {type(token_ids)}。元のデータを返します。")
            return [] if token_ids is None else [token_ids] if not isinstance(token_ids, list) else token_ids

        if techniques is None:
            techniques = ["masking", "deletion", "replacement", "permutation"]

        if probs is None:
            probs = {
                "masking": 0.15,
                "deletion": 0.1,
                "replacement": 0.1,
                "permutation": 0.1
            }

        augmented_ids = token_ids.copy()

        # 拡張処理時のエラーカウント
        error_count = 0
        max_errors = 3  # 許容する最大エラー数

        for technique in techniques:
            try:
                if technique == "masking":
                    if random.random() < probs.get("masking", 0.15):
                        augmented_ids = self.token_masking(augmented_ids, mask_prob=probs.get("masking", 0.15))
                elif technique == "deletion":
                    if random.random() < probs.get("deletion", 0.1):
                        augmented_ids = self.token_deletion(augmented_ids, del_prob=probs.get("deletion", 0.1))
                elif technique == "replacement":
                    if random.random() < probs.get("replacement", 0.1):
                        augmented_ids = self.token_replacement(augmented_ids, replace_prob=probs.get("replacement", 0.1))
                elif technique == "permutation":
                    if random.random() < probs.get("permutation", 0.1):
                        augmented_ids = self.token_permutation(augmented_ids, perm_prob=probs.get("permutation", 0.1))
            except Exception as e:
                error_count += 1
                # スタックトレースも含めて詳細なエラー情報を出力
                logging.error(f"{technique}拡張適用中にエラーが発生しました: {e}", exc_info=True)

                # エラーが多すぎる場合は警告
                if error_count > max_errors:
                    logging.warning(f"エラーが{max_errors}回以上発生しました。データ拡張プロセスに問題がある可能性があります。")
                    break

        return augmented_ids

def augment_dataset(train_data: List[Tuple[List[int], List[int]]],
                   sp_src, sp_tgt,
                   augmentation_factor: float = 0.3,
                   techniques: List[str] = None) -> List[Tuple[List[int], List[int]]]:
    """
    トレーニングデータセットを拡張する関数

    Args:
        train_data: オリジナルのトレーニングデータ [(src_tokens, tgt_tokens), ...]
        sp_src: ソース言語のSentencePieceモデル
        sp_tgt: ターゲット言語のSentencePieceモデル
        augmentation_factor: 元のデータセットに対する拡張データの割合
        techniques: 適用する拡張手法のリスト

    Returns:
        拡張されたトレーニングデータ
    """
    if techniques is None:
        techniques = ["masking", "deletion", "replacement", "permutation"]

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
    indices = random.sample(range(num_samples), num_augmented)

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

            # ソースとターゲットの両方を拡張
            aug_src_tokens = augmentor.apply_augmentations(src_tokens, techniques)
            aug_tgt_tokens = augmentor.apply_augmentations(tgt_tokens, techniques)

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
    import torch

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
            src_tokens = tokenize_with_sentencepiece(src_text, sp_src, CONFIG["TRANSLATION_SOURCE"])
            tgt_tokens = tokenize_with_sentencepiece(tgt_text, sp_tgt, CONFIG["TRANSLATION_DESTINATION"])
            tokenized_data.append((src_tokens, tgt_tokens))

        # データ拡張を適用
        augmented_data = augment_dataset(tokenized_data, sp_src, sp_tgt, augmentation_factor=1.0)

        print(f"元のデータ: {len(tokenized_data)} サンプル")
        print(f"拡張後のデータ: {len(augmented_data)} サンプル")

    except Exception as e:
        logging.error(f"データ拡張テスト中にエラーが発生しました: {e}")
