import os
import torch
from data import set_data, normalize_text, train_sentencepiece
from config import TRANSLATION_SOURCE, TRANSLATION_DESTINATION, TRANSLATION_SOURCE2, TRANSLATION_DESTINATION2, TOKENIZE_BATCH_SIZE, DATA_AUGMENTATION_FACTOR, DATA_AUGMENTATION_TECHNIQUES
from tqdm import tqdm
import sentencepiece as spm
import multiprocessing

from sklearn.model_selection import train_test_split
from datasets import load_dataset

from concurrent.futures import ProcessPoolExecutor
import argparse
import logging

# データ拡張モジュールをインポート
from data_augmentation import augment_dataset

# ロギング設定
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def save_tokenized_data(data, path):
    torch.save(data, path)

def load_tokenized_data(path):
    if os.path.exists(path):
        return torch.load(path)
    else:
        return None

def train_and_load_sp_models(train_texts_src, train_texts_tgt):
    """
    ソースとターゲットのSentencePieceモデルをトレーニングして読み込みます。

    Args:
        train_texts_src (list): ソース言語のトレーニングテキスト
        train_texts_tgt (list): ターゲット言語のトレーニングテキスト

    Returns:
        tuple: (ソースモデル, ターゲットモデル)
    """
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

def tokenize_wrapper(args):
    text, lang, sp_model = args
    try:
        # テキスト正規化して SentencePiece でトークナイズ
        normalized_text = normalize_text(text, lang)
        return sp_model.encode_as_ids(normalized_text)
    except Exception as e:
        logging.error(f"トークン化中にエラーが発生しました: {text}: {e}")
        # エラー時はUNKトークンを返す
        return [1]  # <unk>のIDを返す

# 新しいトップレベル関数：ソーステキストのトークン化
def tokenize_source_text(text, lang_src, sp_src):
    return tokenize_wrapper((text, lang_src, sp_src))

# 新しいトップレベル関数：ターゲットテキストのトークン化
def tokenize_target_text(text, lang_tgt, sp_tgt):
    return tokenize_wrapper((text, lang_tgt, sp_tgt))

# バッチジェネレータ
def batch_generator(data, batch_size):
    for i in range(0, len(data), batch_size):
        yield data[i:i + batch_size]

def process_batch_data(data_src, data_tgt, base_vocab, dest_vocab, batch_size, data_type="train"):
    token_ids = []
    total_batches = len(data_src) // batch_size + 1

    for batch_idx, (batch_src, batch_tgt) in enumerate(tqdm(zip(
        batch_generator(data_src, batch_size),
        batch_generator(data_tgt, batch_size)
    ), total=total_batches, desc=f"{data_type}データの処理中")):
        try:
            batch_ids = [(src, tgt) for src, tgt in zip(batch_src, batch_tgt)]
            token_ids.extend(batch_ids)
        except Exception as e:
            print(f"バッチ{batch_idx}の処理中にエラーが発生: {str(e)}")
            # エラーが起きたバッチは飛ばして続行する
            continue
    return token_ids

def process_and_save_data(train_dataset, val_dataset, lang_src, lang_tgt, path_train, path_val, batch_size=TOKENIZE_BATCH_SIZE, enable_augmentation=False, augmentation_factor=DATA_AUGMENTATION_FACTOR):
    """
    データをトークン化し、オプションでデータ拡張も行います。

    Args:
        train_dataset: トレーニングデータセット
        val_dataset: 検証データセット
        lang_src: ソース言語
        lang_tgt: ターゲット言語
        path_train: トレーニングデータの保存先パス
        path_val: 検証データの保存先パス
        batch_size: バッチサイズ
        enable_augmentation: データ拡張を有効にするかどうか
        augmentation_factor: データ拡張の倍率

    Returns:
        tuple: (トレーニングトークンID, 検証トークンID)
    """
    # 前処理とSentencePieceモデルのトレーニング
    print("======= 前処理とSentencePieceモデルのトレーニング ======")

    # トレーニングデータからテキストを抽出
    train_texts_src = [item[0] for item in train_dataset]
    train_texts_tgt = [item[1] for item in train_dataset]

    # SentencePieceモデルをトレーニングして読み込み
    try:
        sp_src, sp_tgt = train_and_load_sp_models(train_texts_src, train_texts_tgt)
    except Exception as e:
        logging.error(f"SentencePieceモデルのトレーニング中にエラーが発生しました: {e}")
        raise

    # トークナイズ処理
    print("======= tokenize now ======")

    # 利用可能なCPUコア数に基づいてワーカー数を決定
    num_workers = min(multiprocessing.cpu_count(), 8)  # 最大8プロセスまで
    print(f"並列処理に {num_workers} ワーカーを使用します")

    # 進捗バーを使用して処理状況を表示
    print("学習データのトークナイズ中...")
    tokenized_train_src = []
    tokenized_train_tgt = []

    # 大きなデータセットを分割して処理する
    chunk_size = 10000  # 一度に処理する最大データ数

    for i in range(0, len(train_dataset), chunk_size):
        chunk_end = min(i + chunk_size, len(train_dataset))
        chunk_dataset = train_dataset[i:chunk_end]

        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            # ソーステキストのトークナイズ（ラムダをトップレベル関数に置き換え）
            src_texts = [x[0] for x in chunk_dataset]
            tokenized_src_chunk = list(tqdm(
                executor.map(tokenize_source_text, src_texts,
                            [lang_src] * len(src_texts),
                            [sp_src] * len(src_texts)),
                total=len(src_texts),
                desc=f"ソーステキスト {i+1}-{chunk_end}/{len(train_dataset)}"
            ))
            tokenized_train_src.extend(tokenized_src_chunk)

            # ターゲットテキストのトークナイズ（ラムダをトップレベル関数に置き換え）
            tgt_texts = [x[1] for x in chunk_dataset]
            tokenized_tgt_chunk = list(tqdm(
                executor.map(tokenize_target_text, tgt_texts,
                           [lang_tgt] * len(tgt_texts),
                           [sp_tgt] * len(tgt_texts)),
                total=len(tgt_texts),
                desc=f"ターゲットテキスト {i+1}-{chunk_end}/{len(train_dataset)}"
            ))
            tokenized_train_tgt.extend(tokenized_tgt_chunk)

    # 検証データも同様に処理
    print("検証データのトークナイズ中...")
    tokenized_val_src = []
    tokenized_val_tgt = []

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        try:
            # ソーステキストのトークナイズ（ラムダをトップレベル関数に置き換え）
            src_texts = [x[0] for x in val_dataset]
            tokenized_val_src = list(tqdm(
                executor.map(tokenize_source_text, src_texts,
                           [lang_src] * len(src_texts),
                           [sp_src] * len(src_texts)),
                total=len(src_texts),
                desc="検証ソーステキスト"
            ))

            # ターゲットテキストのトークナイズ（ラムダをトップレベル関数に置き換え）
            tgt_texts = [x[1] for x in val_dataset]
            tokenized_val_tgt = list(tqdm(
                executor.map(tokenize_target_text, tgt_texts,
                           [lang_tgt] * len(tgt_texts),
                           [sp_tgt] * len(tgt_texts)),
                total=len(tgt_texts),
                desc="検証ターゲットテキスト"
            ))
        except Exception as e:
            logging.error(f"検証データのトークン化中にエラーが発生しました: {e}")
            # 最低限のデータを確保
            if not tokenized_val_src or not tokenized_val_tgt:
                # トレーニングデータの一部を検証に使用
                train_size = len(tokenized_train_src)
                val_size = min(1000, train_size // 10)
                tokenized_val_src = tokenized_train_src[-val_size:]
                tokenized_val_tgt = tokenized_train_tgt[-val_size:]
                tokenized_train_src = tokenized_train_src[:-val_size]
                tokenized_train_tgt = tokenized_train_tgt[:-val_size]

    # SentencePieceモデルを保存
    print("======= save SentencePiece models ======")
    torch.save(sp_src, os.path.join("models", "sp_src.pth"))
    torch.save(sp_tgt, os.path.join("models", "sp_tgt.pth"))

    # ボキャブラリも保存（.vocabファイルから直接読み込む）
    # train.pyはmodels/vocab_input.pthとmodels/vocab_output.pthを参照するため
    print("======= ボキャブラリを保存しています ======")

    # .vocabファイルのパス
    src_vocab_path = os.path.join("models", "sp_src.vocab")
    tgt_vocab_path = os.path.join("models", "sp_tgt.vocab")

    # .vocabファイルからボキャブラリを読み込む
    src_vocab = {}
    tgt_vocab = {}

    # ソース言語のボキャブラリを読み込む
    try:
        with open(src_vocab_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if line.strip():
                    token = line.split('\t')[0]
                    src_vocab[token] = i
        print(f"ソース言語のボキャブラリを読み込みました: {len(src_vocab)}個のトークン")
    except Exception as e:
        print(f"ソース言語のボキャブラリファイル読み込みに失敗しました: {e}")
        # 失敗した場合はモデルから直接抽出
        for i in range(sp_src.get_piece_size()):
            piece = sp_src.id_to_piece(i)
            src_vocab[piece] = i

    # ターゲット言語のボキャブラリを読み込む
    try:
        with open(tgt_vocab_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if line.strip():
                    token = line.split('\t')[0]
                    tgt_vocab[token] = i
        print(f"ターゲット言語のボキャブラリを読み込みました: {len(tgt_vocab)}個のトークン")
    except Exception as e:
        print(f"ターゲット言語のボキャブラリファイル読み込みに失敗しました: {e}")
        # 失敗した場合はモデルから直接抽出
        for i in range(sp_tgt.get_piece_size()):
            piece = sp_tgt.id_to_piece(i)
            tgt_vocab[piece] = i

    # 特殊トークンが辞書にあることを確認
    special_tokens = ['<pad>', '<unk>', '<s>', '</s>']
    for token in special_tokens:
        if token not in src_vocab:
            src_vocab[token] = len(src_vocab)
        if token not in tgt_vocab:
            tgt_vocab[token] = len(tgt_vocab)

    # ボキャブラリをPTHファイルとして保存
    torch.save(src_vocab, os.path.join("models", "vocab_input.pth"))
    torch.save(tgt_vocab, os.path.join("models", "vocab_output.pth"))

    # トークナイズしたデータをペアにする
    train_token_ids = [(src, tgt) for src, tgt in zip(tokenized_train_src, tokenized_train_tgt)]
    val_token_ids = [(src, tgt) for src, tgt in zip(tokenized_val_src, tokenized_val_tgt)]

    # データ拡張（オプション）
    if enable_augmentation and augmentation_factor > 0:
        print(f"======= データ拡張を適用しています (倍率: {augmentation_factor}) ======")
        try:
            # データ拡張技術を適用
            augmented_train_token_ids = augment_dataset(
                train_token_ids,
                sp_src,
                sp_tgt,
                augmentation_factor=augmentation_factor,
                techniques=DATA_AUGMENTATION_TECHNIQUES
            )
            logging.info(f"データ拡張が完了しました: 元のデータ {len(train_token_ids)} → 拡張後 {len(augmented_train_token_ids)} サンプル")
            train_token_ids = augmented_train_token_ids
        except Exception as e:
            logging.error(f"データ拡張中にエラーが発生しました: {e}")
            logging.info("元のデータセットをそのまま使用します")

    # データを保存
    save_tokenized_data(train_token_ids, path_train)
    save_tokenized_data(val_token_ids, path_val)

    return train_token_ids, val_token_ids

def main():
    parser = argparse.ArgumentParser(description='テキストデータのトークン化と前処理を行います')
    parser.add_argument('--sample-size', type=int, default=1000, help='使用するサンプル数（0で全データ使用）')
    parser.add_argument('--augment', action='store_true', help='データ拡張を有効にする')
    parser.add_argument('--augment-factor', type=float, default=DATA_AUGMENTATION_FACTOR, help='データ拡張の倍率')
    args = parser.parse_args()

    try:
        # データセットを読み込む
        train_data = load_dataset("Verah/JParaCrawl-Filtered-English-Japanese-Parallel-Corpus", split="train")

        # model2_acceptedカラムが1のデータのみを抽出
        filtered_train_data = train_data.filter(lambda filter: filter['model2_accepted'] == 1)

        # サンプルサイズの処理
        if args.sample_size > 0:
            # 開発環境では指定されたサンプル数のデータを使用
            sample_size = args.sample_size
            filtered_train_data = filtered_train_data.select(range(min(sample_size, len(filtered_train_data))))
            print(f"学習に{sample_size}件のデータを使用します。本番環境では全データの使用を推奨。")
        else:
            print(f"すべてのデータ({len(filtered_train_data)}件)を使用します。")

        # データセットを設定
        train_dataset = set_data(filtered_train_data[TRANSLATION_SOURCE2], filtered_train_data[TRANSLATION_DESTINATION2])

        # train_dataを学習用データとテストデータに分割
        train_dataset, val_dataset = train_test_split(train_dataset, test_size=0.1, random_state=42)

        # データのトークナイズと保存
        train_data_path = "tokenized_train_data.pth"
        val_data_path = "tokenized_val_data.pth"

        # データ拡張オプションを指定
        train_token_ids, val_token_ids = process_and_save_data(
            train_dataset,
            val_dataset,
            TRANSLATION_SOURCE,
            TRANSLATION_DESTINATION,
            train_data_path,
            val_data_path,
            enable_augmentation=args.augment,
            augmentation_factor=args.augment_factor
        )
        print("処理が完了しました")
        print(f"トレーニングデータ: {len(train_token_ids)} サンプル")
        print(f"検証データ: {len(val_token_ids)} サンプル")

    except Exception as e:
        logging.error(f"処理中にエラーが発生しました: {e}")
        raise

if __name__ == "__main__":
    main()
