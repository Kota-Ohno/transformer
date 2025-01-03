import os
import torch
from data import tokenize, tokens_to_ids, set_data, build_vocabulary
from config import TRANSLATION_SOURCE, TRANSLATION_DESTINATION, TRANSLATION_SOURCE2, TRANSLATION_DESTINATION2, TOKENIZE_BATCH_SIZE

from sklearn.model_selection import train_test_split
from datasets import load_dataset

from concurrent.futures import ProcessPoolExecutor

def save_tokenized_data(data, path):
    torch.save(data, path)

def load_tokenized_data(path):
    if os.path.exists(path):
        return torch.load(path)
    else:
        return None

def tokenize_wrapper(args):
    text, lang = args
    return tokenize(text, lang)

# バッチジェネレータ
def batch_generator(data, batch_size):
    for i in range(0, len(data), batch_size):
        yield data[i:i + batch_size]

def process_batch_data(data_src, data_tgt, base_vocab, dest_vocab, batch_size, data_type="train"):
    token_ids = []
    total_batches = len(data_src) // batch_size + 1
    
    for batch_idx, (batch_src, batch_tgt) in enumerate(zip(
        batch_generator(data_src, batch_size),
        batch_generator(data_tgt, batch_size)
    )):
        try:
            batch_ids = [(tokens_to_ids(src, base_vocab), tokens_to_ids(tgt, dest_vocab))
                        for src, tgt in zip(batch_src, batch_tgt)]
            token_ids.extend(batch_ids)
            if (batch_idx + 1) % 10 == 0:
                print(f"{data_type}データの処理進捗: {batch_idx + 1}/{total_batches}バッチ完了")
        except Exception as e:
            print(f"バッチ{batch_idx}の処理中にエラーが発生: {str(e)}")
            raise
    return token_ids

def process_and_save_data(train_dataset, val_dataset, lang_src, lang_tgt, path_train, path_val, batch_size=TOKENIZE_BATCH_SIZE):
    # バッチ処理の実装
    print("======= tokenize now ======")
    tokenized_train_src = []  # トークナイズされたソースデータを格納するリスト
    tokenized_train_tgt = []  # トークナイズされたターゲットデータを格納するリスト
    tokenized_val_src = []    # 検証用ソースデータ
    tokenized_val_tgt = []    # 検証用ターゲットデータ
    with ProcessPoolExecutor(max_workers=6) as executor:
        for train_batch in batch_generator(train_dataset, batch_size):
            tokenized_pairs_train = list(executor.map(tokenize_wrapper, [(x[0], lang_src) for x in train_batch] + [(x[1], lang_tgt) for x in train_batch]))
            tokenized_train_src.extend(tokenized_pairs_train[:len(tokenized_pairs_train)//2])
            tokenized_train_tgt.extend(tokenized_pairs_train[len(tokenized_pairs_train)//2:])
        # 検証データのバッチ処理
        for val_batch in batch_generator(val_dataset, batch_size):
            # srcとtgtを同時に処理
            tokenized_pairs_val = list(executor.map(tokenize_wrapper, [(x[0], lang_src) for x in val_batch] + [(x[1], lang_tgt) for x in val_batch]))
            tokenized_val_src.extend(tokenized_pairs_val[:len(tokenized_pairs_val)//2])
            tokenized_val_tgt.extend(tokenized_pairs_val[len(tokenized_pairs_val)//2:])

    # ボキャブラリの構築
    print("======= build vocabulary now ======")
    base_vocabulary = build_vocabulary(tokenized_train_src)
    destination_vocabulary = build_vocabulary(tokenized_train_tgt)
    print("======= build vocabulary finished ======")

    # ボキャブラリの保存
    print("======= save vocabulary now ======")
    vocab_input_path = os.path.join("models", "vocab_input.pth")
    torch.save(base_vocabulary, vocab_input_path)
    vocab_output_path = os.path.join("models", "vocab_output.pth")
    torch.save(destination_vocabulary, vocab_output_path)
    print("======= save vocabulary finished ======")

    token_ids_train = process_batch_data(
        tokenized_train_src,
        tokenized_train_tgt,
        base_vocabulary,
        destination_vocabulary,
        TOKENIZE_BATCH_SIZE,
        "train"
    )

    save_tokenized_data(token_ids_train, path_train)

    token_ids_val = process_batch_data(
        tokenized_val_src,
        tokenized_val_tgt,
        base_vocabulary,
        destination_vocabulary,
        TOKENIZE_BATCH_SIZE,
        "validation"
    )

    save_tokenized_data(token_ids_val, path_val)

    return token_ids_train, token_ids_val

def main():

    # データセットを読み込む
    train_data = load_dataset("Verah/JParaCrawl-Filtered-English-Japanese-Parallel-Corpus", split="train")

    # model2_acceptedカラムが1のデータのみを抽出
    filtered_train_data = train_data.filter(lambda filter: filter['model2_accepted'] == 1)

    # データセットを設定
    train_dataset = set_data(filtered_train_data[TRANSLATION_SOURCE2], filtered_train_data[TRANSLATION_DESTINATION2])

    # train_dataを学習用データとテストデータに分割
    train_dataset, val_dataset = train_test_split(train_dataset, test_size=0.1, random_state=42)

    # データのトークナイズと保存
    train_data_path = "tokenized_train_data.pth"
    val_data_path = "tokenized_val_data.pth"
    train_token_ids, val_token_ids = process_and_save_data(train_dataset, val_dataset, TRANSLATION_SOURCE, TRANSLATION_DESTINATION, train_data_path, val_data_path)
    print("finished")

if __name__ == "__main__":
    main()
