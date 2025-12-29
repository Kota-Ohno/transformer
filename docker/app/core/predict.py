import torch
import sys
import os
import glob
import logging
import argparse
from utils.config import CONFIG, INPUT_VOCAB_PATH, OUTPUT_VOCAB_PATH
from data.data import tokenize, tokens_to_ids, ids_to_tokens
from models.model import create_transformer_model

# ロギング設定
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_model(input_vocab, output_vocab, model_path=None):
    """
    モデルをロードする

    Args:
        input_vocab: 入力語彙
        output_vocab: 出力語彙
        model_path: モデルファイルパス（指定しない場合は最新のモデルを使用）

    Returns:
        モデルインスタンス
    """
    # 語彙サイズを取得
    input_dim = len(input_vocab)
    output_dim = len(output_vocab)

    # パディングインデックスを取得
    src_pad_idx = input_vocab['<pad>']
    tgt_pad_idx = output_vocab['<pad>']

    # モデルパスを決定
    if model_path:
        # 指定されたモデルパスを使用
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"指定されたモデルファイルが見つかりません: {model_path}")
        logging.info(f"指定されたモデルを使用します: {model_path}")
    else:
        # modelsディレクトリから最新のモデルファイル名を取得
        model_files = glob.glob('models/*.pth')
        # 語彙ファイルを除外
        model_files = [f for f in model_files if not f.endswith(('vocab_input.pth', 'vocab_output.pth'))]
        if not model_files:
            raise FileNotFoundError("学習済みモデルファイルが見つかりません。'models/' ディレクトリを確認してください。")
        model_path = max(model_files, key=os.path.getctime)  # 最新のファイルを選択
        logging.info(f"最新のモデルを使用します: {model_path}")

    # モデルの読み込み
    logging.info(f"モデルをロード中: {model_path}")
    checkpoint = torch.load(model_path, map_location=CONFIG.device)

    # 保存されたモデル設定を読み込む
    saved_config = None

    # デフォルト値を設定
    hidden_size = CONFIG.model_hyperparameters.hidden_size
    num_heads = CONFIG.model_hyperparameters.num_heads
    num_layers = CONFIG.model_hyperparameters.num_layers
    d_ff = CONFIG.model_hyperparameters.d_ff
    dropout_rate = CONFIG.model_hyperparameters.dropout_rate

    if isinstance(checkpoint, dict) and 'model_config' in checkpoint:
        saved_config = checkpoint['model_config']
        logging.info(f"保存された設定を使用します: {saved_config}")

        # 必要な設定値を取得
        hidden_size = saved_config.get('HIDDEN_SIZE', hidden_size)
        num_heads = saved_config.get('NUM_HEADS', num_heads)
        num_layers = saved_config.get('NUM_LAYERS', num_layers)
        d_ff = saved_config.get('D_FF', d_ff)
        dropout_rate = saved_config.get('DROPOUT_RATE', dropout_rate)

    # モデルを作成
    model = create_transformer_model(
        input_vocab_size=input_dim,
        output_vocab_size=output_dim,
        src_pad_idx=src_pad_idx,
        tgt_pad_idx=tgt_pad_idx,
        hidden_size=hidden_size,
        num_heads=num_heads,
        num_layers=num_layers,
        d_ff=d_ff,
        dropout=dropout_rate
    )

    # チェックポイントがdict形式で'model_state_dict'キーを持っている場合は取り出す
    try:
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            # 従来の形式（モデルの状態辞書が直接保存されている場合）
            model.load_state_dict(checkpoint)
    except Exception as e:
        logging.error(f"モデルの読み込みに失敗しました: {e}")
        logging.error("モデルのアーキテクチャと保存されたモデルの設定が一致していない可能性があります。")
        raise

    model.eval()
    return model

def load_vocab(vocab_path):
    return torch.load(vocab_path)

def preprocess_input(sentence, input_vocab):
    """
    入力文をトークン化してテンソルに変換

    Args:
        sentence: 入力文
        input_vocab: 入力語彙

    Returns:
        入力テンソル
    """
    try:
        tokens = tokenize(sentence, CONFIG.data_config.translation_source)
        token_ids = tokens_to_ids(tokens, input_vocab)
        return torch.tensor([token_ids], dtype=torch.long)
    except KeyError as e:
        logging.error(f"エラー: 未知の単語が含まれています: {e}")
        # 未知の単語をUNKトークンに置き換える
        return handle_unknown_tokens(sentence, input_vocab)
    except Exception as e:
        logging.error(f"予期せぬエラーが発生しました: {e}")
        return None

def handle_unknown_tokens(sentence, input_vocab):
    """未知トークンを<unk>に置き換えて処理する"""
    try:
        tokens = tokenize(sentence, CONFIG.data_config.translation_source)
        token_ids = []
        for token in tokens:
            if token in input_vocab:
                token_ids.append(input_vocab[token])
            else:
                logging.warning(f"未知トークン '{token}' を <unk> に置き換えます")
                token_ids.append(input_vocab['<unk>'])
        return torch.tensor([token_ids], dtype=torch.long)
    except Exception as e:
        logging.error(f"未知トークン処理中にエラーが発生しました: {e}")
        return None

def translate(model, input_tensor, max_length=None):
    """
    モデルを使って翻訳を実行

    Args:
        model: 翻訳モデル
        input_tensor: 入力テンソル
        max_length: 最大生成長

    Returns:
        翻訳結果のトークンID
    """
    if max_length is None:
        max_length = CONFIG.model_hyperparameters.max_seq_length
    if input_tensor is None:
        return None

    # 入力テンソルをモデルと同じデバイスに移動
    input_tensor = input_tensor.to(next(model.parameters()).device)

    # 翻訳を実行（勾配計算なし）
    with torch.no_grad():
        try:
            output_ids = model.predict(input_tensor, max_length=max_length)
            return output_ids
        except RuntimeError as e:
            # メモリ不足エラーの場合
            if "out of memory" in str(e).lower():
                logging.warning("メモリ不足のため、入力を分割して処理します")
                max_safe_length = 100
                if input_tensor.size(1) > max_safe_length:
                    input_tensor = input_tensor[:, :max_safe_length]
                torch.cuda.empty_cache()
                output_ids = model.predict(input_tensor, max_length=max_length)
                return output_ids
            else:
                logging.error(f"翻訳中にエラーが発生しました: {e}")
                raise

def main():
    """メイン関数"""
    # コマンドライン引数のパース
    parser = argparse.ArgumentParser(description='Transformer翻訳モデルによる推論')
    parser.add_argument('--model', type=str, help='使用するモデルファイルのパス')
    args = parser.parse_args()

    # 語彙のロード
    input_vocab = load_vocab(INPUT_VOCAB_PATH)
    output_vocab = load_vocab(OUTPUT_VOCAB_PATH)

    try:
        # モデルのロード
        model = load_model(input_vocab, output_vocab, model_path=args.model)
    except FileNotFoundError as e:
        logging.error(f"エラー: {e}")
        sys.exit(1)

    # 英語なら単語間にスペースを入れる
    spacer = " " if CONFIG.data_config.translation_destination == 'en_US' else ""

    logging.info("モデルをロードしました。入力を待っています...")
    logging.info("exitと入力すると終了します...")

    # 対話式で翻訳を行う
    for line in sys.stdin:
        stripped_line = line.strip()
        if stripped_line.lower() == "exit":
            break

        # 入力処理
        input_tensor = preprocess_input(stripped_line, input_vocab)
        if input_tensor is None:
            continue

        # 翻訳実行
        output_ids = translate(model, input_tensor)
        if output_ids is None:
            continue

        # 特殊トークンを除去
        output_text = []
        for ids in output_ids:
            # 最初の<s>と最後の</s>を除去
            token_ids = [t.item() for t in ids]
            if token_ids[0] == 2:  # <s>
                token_ids = token_ids[1:]
            if token_ids and token_ids[-1] == 3:  # </s>
                token_ids = token_ids[:-1]

            # トークンをテキストに変換
            tokens = ids_to_tokens(token_ids, output_vocab)
            output_text.append(spacer.join(tokens))

        # 出力
        print("翻訳結果:", output_text[0])

    logging.info("終了しました")

if __name__ == "__main__":
    main()
