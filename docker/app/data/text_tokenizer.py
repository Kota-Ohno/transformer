import os
import shutil
import torch
from typing import List, Union, Tuple, Optional
from data.data import set_data
from data.tokenizer_utils import normalize_text
from data.tokenizer_utils import train_and_load_sp_models
from utils.config import CONFIG  # Import CONFIG dictionary
from tqdm import tqdm
import sentencepiece as spm
import logging

# モジュールスコープのロガーを作成
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


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

        # save_path_src/save_path_tgtが指定されているが、訓練時に直接保存されなかった場合のフォールバック
        # （train_and_load_sp_modelsがデフォルトパスに保存した場合、指定されたパスにコピー）
        if save_path_src or save_path_tgt:
            tokenizer = cls(sp_src=sp_src, sp_tgt=sp_tgt)
            # 訓練時に指定されたパスに保存されなかった場合（デフォルトパスに保存された場合）のみコピー
            if save_path_src:
                # 指定されたパスにファイルが存在しない場合のみコピー
                save_path_src_with_ext = (
                    save_path_src if save_path_src.endswith(".model") else f"{save_path_src}.model"
                )
                if not os.path.exists(save_path_src_with_ext):
                    tokenizer.save_model(save_path_src, is_source=True)
            if save_path_tgt:
                # 指定されたパスにファイルが存在しない場合のみコピー
                save_path_tgt_with_ext = (
                    save_path_tgt if save_path_tgt.endswith(".model") else f"{save_path_tgt}.model"
                )
                if not os.path.exists(save_path_tgt_with_ext):
                    tokenizer.save_model(save_path_tgt, is_source=False)

        return cls(sp_src=sp_src, sp_tgt=sp_tgt)

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

        return cls(sp_src=sp_src, sp_tgt=sp_tgt)

    def save_model(self, model_path: str, is_source: bool = True) -> None:
        """SentencePieceモデルを保存します。

        Args:
            model_path: 保存先のパス（.model拡張子なし）
            is_source: Trueの場合はソースモデル、Falseの場合はターゲットモデルを保存

        Raises:
            ValueError: モデルが初期化されていない場合
            FileNotFoundError: 訓練時に保存されたモデルファイルが見つからない場合
        """
        model = self.sp_src if is_source else self.sp_tgt
        if model is None:
            raise ValueError(
                f"保存するモデルが初期化されていません（is_source={is_source}）"
            )

        # 訓練時に保存されたデフォルトのモデルファイルパス
        default_model_path = (
            os.path.join("models", "sp_src.model")
            if is_source
            else os.path.join("models", "sp_tgt.model")
        )

        # デフォルトのモデルファイルが存在するか確認
        if not os.path.exists(default_model_path):
            raise FileNotFoundError(
                f"訓練時に保存されたモデルファイルが見つかりません: {default_model_path}。"
                f"モデルを訓練してから保存してください。"
            )

        # 保存先のディレクトリを作成
        save_dir = os.path.dirname(model_path) if os.path.dirname(model_path) else "."
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

        # モデルファイルをコピー
        try:
            shutil.copy2(default_model_path, model_path)
            logger.info(
                f"モデルを保存しました: {default_model_path} -> {model_path} "
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
            return torch.tensor([], dtype=torch.long, device=self.device) if return_tensors else []

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
            max_len = max(len(tokens) for tokens in tokenized_texts) if tokenized_texts else 0
            padded_tokens = [
                tokens + [0] * (max_len - len(tokens)) if len(tokens) < max_len else tokens[:max_len]
                for tokens in tokenized_texts
            ]
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
