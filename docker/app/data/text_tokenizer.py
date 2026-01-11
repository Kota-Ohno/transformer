import os
import shutil
import torch
from typing import List, Union, Tuple, Optional
from data.tokenizer_utils import normalize_text
from data.tokenizer_utils import train_and_load_sp_models
from utils.config import CONFIG  # Import CONFIG dictionary
from tqdm import tqdm
import sentencepiece as spm
import logging
import sys

# モジュールスコープのロガーを作成
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# ルートロガーにメッセージを伝播させる
logger.propagate = True

# ルートロガーにハンドラーがなく、このロガーにもハンドラーがない場合のみ、
# ローカルにハンドラーを追加（単独インポート時のフォールバック）
root_logger = logging.getLogger()
if not root_logger.handlers and not logger.handlers:
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)


class TextTokenizer:
    """SentencePieceモデルを使用したテキストトークナイザー。

    このクラスは、ソース言語とターゲット言語のSentencePieceモデルを訓練・読み込み、
    テキストのトークン化を提供します。
    """

    def __init__(
        self,
        sp_src: Optional[spm.SentencePieceProcessor] = None,
        sp_tgt: Optional[spm.SentencePieceProcessor] = None,
        src_lang: Optional[str] = None,
        tgt_lang: Optional[str] = None,
    ) -> None:
        """TextTokenizerを初期化します。

        Args:
            sp_src: ソース言語のSentencePieceモデル。Noneの場合は後で読み込む必要があります。
            sp_tgt: ターゲット言語のSentencePieceモデル。Noneの場合は後で読み込む必要があります。
            src_lang: ソース言語コード（例: 'ja_JP'）。Noneの場合はCONFIGから取得します。
            tgt_lang: ターゲット言語コード（例: 'en_US'）。Noneの場合はCONFIGから取得します。
        """
        self.sp_src = sp_src
        self.sp_tgt = sp_tgt
        self.src_lang = src_lang or CONFIG.data_config.translation_source
        self.tgt_lang = tgt_lang or CONFIG.data_config.translation_destination
        self.device = CONFIG.get_device()
        # モデルファイルのパスを追跡
        self.sp_src_path: Optional[str] = None
        self.sp_tgt_path: Optional[str] = None

    @classmethod
    def train(
        cls,
        train_texts_src: Union[str, List[str]],
        train_texts_tgt: Union[str, List[str]],
        save_path_src: Optional[str] = None,
        save_path_tgt: Optional[str] = None,
    ) -> "TextTokenizer":
        """SentencePieceモデルを訓練してTextTokenizerインスタンスを作成します。

        Args:
            train_texts_src: ソース言語のトレーニングテキスト（文字列またはリスト）
            train_texts_tgt: ターゲット言語のトレーニングテキスト（文字列またはリスト）
            save_path_src: ソースモデルの保存パス（.model拡張子なし）。Noneの場合はデフォルトパスを使用。
            save_path_tgt: ターゲットモデルの保存パス（.model拡張子なし）。Noneの場合はデフォルトパスを使用。

        Returns:
            TextTokenizer: 訓練されたモデルを持つTextTokenizerインスタンス
        """
        logger.info("SentencePieceモデルを訓練中...")
        # save_path_src/save_path_tgtが指定されている場合は、訓練時に直接そのパスに保存
        sp_src, sp_tgt = train_and_load_sp_models(
            train_texts_src, train_texts_tgt, save_path_src, save_path_tgt
        )

        # save_path_src/save_path_tgtが指定されている場合は、明示的に保存を実行
        tokenizer = cls(sp_src=sp_src, sp_tgt=sp_tgt)
        if save_path_src:
            # 拡張子付きパスを計算
            save_path_src_with_ext = (
                save_path_src if save_path_src.endswith(".model") else f"{save_path_src}.model"
            )
            # 拡張子付きパスを save_model に渡す（save_model 内で存在チェックが行われる）
            # save_model は既に拡張子がある場合は追加しないので、拡張子付きパスを渡す
            tokenizer.save_model(save_path_src_with_ext, is_source=True)
            # 使用されたパス（拡張子付き）を設定
            tokenizer.sp_src_path = save_path_src_with_ext
        if save_path_tgt:
            # 拡張子付きパスを計算
            save_path_tgt_with_ext = (
                save_path_tgt if save_path_tgt.endswith(".model") else f"{save_path_tgt}.model"
            )
            # 拡張子付きパスを save_model に渡す（save_model 内で存在チェックが行われる）
            tokenizer.save_model(save_path_tgt_with_ext, is_source=False)
            # 使用されたパス（拡張子付き）を設定
            tokenizer.sp_tgt_path = save_path_tgt_with_ext

        return tokenizer

    @classmethod
    def load(
        cls,
        model_path_src: Optional[str] = None,
        model_path_tgt: Optional[str] = None,
    ) -> "TextTokenizer":
        """保存されたSentencePieceモデルを読み込んでTextTokenizerインスタンスを作成します。

        Args:
            model_path_src: ソースモデルのパス（.model拡張子付き）。Noneの場合はデフォルトパスを使用。
            model_path_tgt: ターゲットモデルのパス（.model拡張子付き）。Noneの場合はデフォルトパスを使用。

        Returns:
            TextTokenizer: 読み込まれたモデルを持つTextTokenizerインスタンス

        Raises:
            FileNotFoundError: モデルファイルが見つからない場合
        """
        # デフォルトパス
        if model_path_src is None:
            model_path_src = os.path.join("models", "sp_src.model")
        if model_path_tgt is None:
            model_path_tgt = os.path.join("models", "sp_tgt.model")

        # ファイルの存在確認
        if not os.path.exists(model_path_src):
            raise FileNotFoundError(
                f"ソースモデルファイルが見つかりません: {model_path_src}"
            )
        if not os.path.exists(model_path_tgt):
            raise FileNotFoundError(
                f"ターゲットモデルファイルが見つかりません: {model_path_tgt}"
            )

        # モデルを読み込み
        sp_src = spm.SentencePieceProcessor()
        sp_tgt = spm.SentencePieceProcessor()

        try:
            sp_src.load(model_path_src)
            sp_tgt.load(model_path_tgt)
            logger.info(f"モデルを読み込みました: {model_path_src}, {model_path_tgt}")
        except Exception as e:
            logger.error(f"モデルの読み込み中にエラーが発生しました: {e}")
            raise

        # インスタンスを作成してパスを設定
        instance = cls(sp_src=sp_src, sp_tgt=sp_tgt)
        instance.sp_src_path = model_path_src
        instance.sp_tgt_path = model_path_tgt
        return instance

    def save_model(self, model_path: str, is_source: bool = True) -> None:
        """SentencePieceモデルを指定されたパスに保存します。

        まず追跡されているパスからコピーを試み、それが存在しない場合は
        メモリ内のモデルをシリアライズして保存します。

        Args:
            model_path: 保存先のパス（.model拡張子付きまたはなし、どちらでも可）。
                       拡張子がない場合は自動的に.modelが追加されます。
                       trainメソッドから呼び出される場合、.model拡張子付きのパスが渡されることがあります。
            is_source: Trueの場合はソースモデル、Falseの場合はターゲットモデルを保存

        Raises:
            ValueError: モデルが初期化されていない場合
            FileNotFoundError: 追跡されているパスが存在せず、モデルもシリアライズできない場合
        """
        model = self.sp_src if is_source else self.sp_tgt
        if model is None:
            raise ValueError(
                f"保存するモデルが初期化されていません（is_source={is_source}）"
            )

        # 追跡されているパスを取得
        tracked_path = self.sp_src_path if is_source else self.sp_tgt_path

        # 保存先のディレクトリを作成
        dir_name = os.path.dirname(model_path)
        save_dir = dir_name if dir_name else "."
        if save_dir and save_dir != ".":
            os.makedirs(save_dir, exist_ok=True)

        # .model拡張子を追加
        if not model_path.endswith(".model"):
            model_path = f"{model_path}.model"

        # 指定されたパスにファイルが既に存在する場合はコピーをスキップ
        if os.path.exists(model_path):
            logger.info(
                f"モデルファイルは既に存在します: {model_path} "
                f"(is_source={is_source})。コピーをスキップします。"
            )
            return

        # 追跡されているパスからコピーを試みる
        if tracked_path and os.path.exists(tracked_path):
            try:
                shutil.copy2(tracked_path, model_path)
                logger.info(
                    f"モデルを保存しました: {tracked_path} -> {model_path} "
                    f"(is_source={is_source})"
                )
                return
            except Exception as e:
                logger.warning(
                    f"追跡されているパスからのコピーに失敗しました: {e}。"
                    f"メモリ内のモデルをシリアライズして保存します。"
                )

        # フォールバック: メモリ内のモデルをシリアライズ
        try:
            serialized_model = model.serialized_model_proto()
            with open(model_path, 'wb') as f:
                f.write(serialized_model)
            logger.info(
                f"モデルをシリアライズして保存しました: {model_path} "
                f"(is_source={is_source})"
            )
        except Exception as e:
            logger.error(f"モデルの保存中にエラーが発生しました: {e}")
            raise

    def tokenize(
        self,
        texts: Union[str, List[str]],
        lang: Optional[str] = None,
        is_source: bool = True,
        normalize_numeric: Union[str, None, bool] = "<NUM>",
        return_tensors: bool = True,
        batch_size: Optional[int] = None,
    ) -> Union[List[List[int]], torch.Tensor]:
        """テキストをトークン化します。

        Args:
            texts: トークン化するテキスト（文字列またはリスト）
            lang: 言語コード（'ja_JP' または 'en_US'）。Noneの場合はis_sourceに基づいて決定。
            is_source: Trueの場合はソース言語、Falseの場合はターゲット言語として処理
            normalize_numeric: 数字の正規化方法。デフォルトは'<NUM>'。
            return_tensors: Trueの場合はtorch.Tensorを返す、Falseの場合はリストを返す
            batch_size: バッチサイズ。Noneの場合はCONFIGから取得。

        Returns:
            トークン化されたテキスト。return_tensorsがTrueの場合はtorch.Tensor、
            Falseの場合はList[List[int]]。

        Raises:
            ValueError: モデルが初期化されていない場合
        """
        # モデルの確認
        model = self.sp_src if is_source else self.sp_tgt
        if model is None:
            raise ValueError(
                f"トークン化に使用するモデルが初期化されていません（is_source={is_source}）"
            )

        # 言語の決定
        if lang is None:
            lang = self.src_lang if is_source else self.tgt_lang

        # テキストをリストに変換
        if isinstance(texts, str):
            texts = [texts]
        texts = list(texts)

        if not texts:
            if return_tensors:
                # 空テンソルを[0, max_seq_length]の形状で返す
                max_sequence_length = CONFIG.model_hyperparameters.max_seq_length
                return torch.empty((0, max_sequence_length), dtype=torch.long, device=self.device)
            return []

        # バッチサイズの決定
        if batch_size is None:
            batch_size = CONFIG.data_config.tokenize_batch_size

        # バッチ処理でトークン化
        tokenized_texts = self._tokenize_texts_batch(
            texts, model, lang, normalize_numeric, batch_size
        )

        # テンソルに変換（必要な場合）
        if return_tensors:
            # パディングを追加してテンソルに変換
            # SentencePieceモデルからpad_idを取得（pad_id() -> eos_id() -> 専用padトークン追加の順で試行）
            pad_id = None
            if hasattr(model, 'pad_id') and callable(model.pad_id):
                pad_id_value = model.pad_id()
                if pad_id_value is not None and pad_id_value >= 0:
                    pad_id = pad_id_value

            # pad_idが取得できなかった場合はeos_idを試す
            if pad_id is None:
                if hasattr(model, 'eos_id') and callable(model.eos_id):
                    eos_id_value = model.eos_id()
                    if eos_id_value is not None and eos_id_value >= 0:
                        pad_id = eos_id_value
                        logger.warning(
                            "pad_idが取得できなかったため、eos_idをパディングトークンとして使用します。"
                        )

            # それでも取得できなかった場合は専用のpadトークンを追加する必要がある
            # この場合はエラーを出すか、デフォルト値を使う（SentencePieceモデルの設定を確認）
            if pad_id is None:
                error_msg = (
                    "pad_idとeos_idの両方が取得できませんでした。"
                    "SentencePieceモデルに<pad>トークンが含まれているか確認してください。"
                    "SentencePieceモデルを訓練する際に、明示的に<pad>トークンを追加してください。"
                )
                logger.error(error_msg)
                raise RuntimeError(error_msg)

            # 固定の最大シーケンス長を使用
            max_sequence_length = CONFIG.model_hyperparameters.max_seq_length

            padded_tokens = []
            for tokens in tokenized_texts:
                # すべてのシーケンスをmax_sequence_lengthに切り詰め
                if len(tokens) > max_sequence_length:
                    # シーケンスがmax_sequence_lengthより長い場合は切り詰め、警告をログに記録
                    logger.warning(
                        f"トークンシーケンスがmax_sequence_length ({max_sequence_length}) を超えています "
                        f"(長さ: {len(tokens)})。切り詰めます。"
                    )
                    tokens = tokens[:max_sequence_length]

                # すべてのシーケンスをmax_sequence_lengthにパディング
                if len(tokens) < max_sequence_length:
                    tokens = tokens + [pad_id] * (max_sequence_length - len(tokens))
                # len(tokens) == max_sequence_length の場合はそのまま使用
                padded_tokens.append(tokens)

            return torch.tensor(padded_tokens, dtype=torch.long, device=self.device)

        return tokenized_texts

    def _tokenize_texts_batch(
        self,
        texts: List[str],
        model: spm.SentencePieceProcessor,
        lang: str,
        normalize_numeric: Union[str, None, bool],
        batch_size: int,
    ) -> List[List[int]]:
        """バッチ処理でテキストをトークン化します（tqdmを使用）。

        Args:
            texts: トークン化するテキストのリスト
            model: SentencePieceモデル
            lang: 言語コード
            normalize_numeric: 数字の正規化方法
            batch_size: バッチサイズ

        Returns:
            トークン化されたテキストのリスト（各要素はトークンIDのリスト）
        """
        # バッチに分割
        batches = [
            texts[i : i + batch_size] for i in range(0, len(texts), batch_size)
        ]

        # バッチ処理でトークン化（tqdmで進捗表示）
        tokenized_texts: List[List[int]] = []
        for batch in tqdm(batches, desc="トークン化中"):
            batch_tokens = []
            for text in batch:
                normalized = normalize_text(text, lang, normalize_numeric=normalize_numeric)
                tokens = model.encode_as_ids(normalized)
                batch_tokens.append(tokens)
            tokenized_texts.extend(batch_tokens)

        return tokenized_texts

    def tokenize_texts(
        self,
        texts: Union[str, List[str]],
        lang: Optional[str] = None,
        is_source: bool = True,
        normalize_numeric: Union[str, None, bool] = "<NUM>",
    ) -> List[List[int]]:
        """テキストをトークン化します（tokenize()のエイリアス、リストを返す）。

        Args:
            texts: トークン化するテキスト（文字列またはリスト）
            lang: 言語コード。Noneの場合はis_sourceに基づいて決定。
            is_source: Trueの場合はソース言語、Falseの場合はターゲット言語として処理
            normalize_numeric: 数字の正規化方法。デフォルトは'<NUM>'。

        Returns:
            トークン化されたテキストのリスト（各要素はトークンIDのリスト）
        """
        return self.tokenize(
            texts=texts,
            lang=lang,
            is_source=is_source,
            normalize_numeric=normalize_numeric,
            return_tensors=False,
        )
