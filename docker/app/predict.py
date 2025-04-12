import torch
import sys
import os
import glob
import logging
import argparse
import gc  # ガベージコレクション用
from encoder import Encoder
from decoder import Decoder
from utils import TranslationModel, create_padding_mask, create_subsequent_mask
from config import CONFIG, MODEL_CONFIG, DEVICE, INPUT_VOCAB_PATH, OUTPUT_VOCAB_PATH
from data import tokenize, tokens_to_ids, ids_to_tokens

# 拡張モデルをインポート
from enhanced_model import create_enhanced_model

# ロギング設定
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_model(input_vocab, output_vocab, enhanced=False, model_path=None):
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
        model_filename = max(model_files, key=os.path.getctime)  # 最新のファイルを選択
        model_path = os.path.join("models", os.path.basename(model_filename))
        logging.info(f"最新のモデルを使用します: {model_path}")

    # モデルの読み込み
    logging.info(f"モデルをロード中: {model_path}")
    checkpoint = torch.load(model_path, map_location=DEVICE)

    # 保存されたモデル設定を読み込む
    saved_config = None
    # Use MODEL_CONFIG for defaults
    hidden_size = MODEL_CONFIG.hidden_size
    num_heads = MODEL_CONFIG.num_heads
    num_layers = MODEL_CONFIG.num_layers
    d_ff = MODEL_CONFIG.d_ff
    dropout_rate = MODEL_CONFIG.dropout
    rel_pos_max_distance = MODEL_CONFIG.rel_pos_max_distance

    if isinstance(checkpoint, dict):
        if 'model_config' in checkpoint:
            saved_config = checkpoint['model_config']
            logging.info(f"保存された設定を使用します: {saved_config}")

            # 必要な設定値を取得 (チェックポイントの設定を優先)
            hidden_size = saved_config.get('HIDDEN_SIZE', hidden_size)
            num_heads = saved_config.get('NUM_HEADS', num_heads)
            num_layers = saved_config.get('NUM_LAYERS', num_layers)
            d_ff = saved_config.get('D_FF', d_ff)
            dropout_rate = saved_config.get('DROPOUT_RATE', dropout_rate)
            rel_pos_max_distance = saved_config.get('REL_POS_MAX_DISTANCE', rel_pos_max_distance)
        elif 'model_state_dict' in checkpoint:
            # 古い形式の場合、レイヤー数を推定
            encoder_layers = 0
            decoder_layers = 0
            for key in checkpoint['model_state_dict'].keys():
                if '.encoder.layers.' in key:
                    layer_num = int(key.split('.encoder.layers.')[1].split('.')[0])
                    encoder_layers = max(encoder_layers, layer_num + 1)
                if '.decoder.layers.' in key:
                    layer_num = int(key.split('.decoder.layers.')[1].split('.')[0])
                    decoder_layers = max(decoder_layers, layer_num + 1)

            if encoder_layers > 0:
                logging.info(f"モデル状態辞書から推定したレイヤー数: {encoder_layers}")
                num_layers = encoder_layers

    if enhanced:
        # 拡張モデルを作成
        logging.info("拡張モデルを使用します")
        model = create_enhanced_model(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_dim=hidden_size,
            num_heads=num_heads,
            num_layers=num_layers,
            ff_dim=d_ff,
            src_pad_idx=src_pad_idx,
            tgt_pad_idx=tgt_pad_idx,
            dropout=dropout_rate,
            device=DEVICE,
            max_dist=rel_pos_max_distance
        )
    else:
        # 標準モデルを作成
        logging.info("標準モデルを使用します")
        encoder = Encoder(input_dim, hidden_size, num_heads, num_layers, d_ff, dropout_rate, DEVICE).to(DEVICE)
        decoder = Decoder(output_dim, hidden_size, num_heads, num_layers, d_ff, output_dim, dropout_rate, DEVICE).to(DEVICE)
        model = TranslationModel(encoder, decoder, src_pad_idx, tgt_pad_idx, DEVICE).to(DEVICE)

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
        logging.error(f"使用した設定: hidden_size={hidden_size}, num_heads={num_heads}, num_layers={num_layers}")
        raise

    model.eval()
    return model

def load_vocab(vocab_path):
    return torch.load(vocab_path)

def preprocess_input(sentence, input_vocab):
    try:
        tokens = tokenize(sentence, CONFIG["TRANSLATION_SOURCE"])
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
        tokens = tokenize(sentence, CONFIG["TRANSLATION_SOURCE"])
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

def predict(model, input_tensor, input_vocab, output_vocab, max_len=CONFIG["MAX_SEQ_LENGTH"], beam_size=5, alpha=0.7):
    """
    ビームサーチを使用して翻訳を生成する関数。

    Args:
        model: 翻訳モデル
        input_tensor: 入力テンソル
        input_vocab: 入力語彙
        output_vocab: 出力語彙
        max_len: 最大生成長
        beam_size: ビームサーチの幅
        alpha: 長さ正規化パラメータ (0.0-1.0)
    """
    if input_tensor is None:
        return None

    # 入力テンソルをモデルと同じデバイスに移動
    device = next(model.parameters()).device
    input_tensor = input_tensor.to(device)

    # 評価モードに設定し、勾配計算を無効化
    model.eval()
    torch.set_grad_enabled(False)

    # メモリを節約するためにキャッシュを削除
    if device == 'cuda':
        torch.cuda.empty_cache()

    # 特殊トークンのIDを取得
    start_token = output_vocab['<s>']
    end_token = output_vocab.get('</s>', output_vocab.get('</s>', -1))
    src_pad_idx = input_vocab['<pad>']
    tgt_pad_idx = output_vocab['<pad>']

    # エンコーダー処理とソースマスク
    with torch.no_grad():
        src_mask = create_padding_mask(input_tensor, src_pad_idx).to(device)
        try:
            encoder_output = model.encoder(input_tensor, src_mask)
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                logging.warning("メモリ不足のため、入力を分割して処理します")
                max_safe_length = min(100, input_tensor.size(1) // 2)
                input_tensor = input_tensor[:, :max_safe_length]
                src_mask = create_padding_mask(input_tensor, src_pad_idx).to(device)
                torch.cuda.empty_cache()
                encoder_output = model.encoder(input_tensor, src_mask)
            else:
                raise

    # ビームの初期化
    beams = [(
        [start_token],  # トークンシーケンス
        0.0,  # スコア
        False  # 完了フラグ
    )]

    # ビームサーチループ
    for _ in range(max_len):
        if all(beam[2] for beam in beams):
            break  # すべてのビームが完了していれば終了

        candidates = []
        for beam_tokens, beam_score, is_finished in beams:
            if is_finished:
                candidates.append((beam_tokens, beam_score, True))
                continue

            # デコーダー入力を準備
            decoder_input = torch.tensor([beam_tokens], dtype=torch.long, device=device)

            # マスクを生成
            tgt_mask = create_subsequent_mask(decoder_input).to(device)
            tgt_pad_mask = create_padding_mask(decoder_input, tgt_pad_idx).to(device)
            combined_mask = torch.logical_and(
                tgt_pad_mask.expand(-1, -1, decoder_input.size(1), -1),
                tgt_mask
            )

            # デコーダー出力を取得
            with torch.no_grad():
                decoder_output, _ = model.decoder(
                    decoder_input,
                    encoder_output,
                    combined_mask,
                    src_mask
                )

            # 最後の位置の出力に対して確率を計算
            next_token_logits = decoder_output[:, -1, :]
            next_token_log_probs = torch.log_softmax(next_token_logits, dim=-1)

            # 上位k個の次トークンを取得
            topk_log_probs, topk_indices = next_token_log_probs[0].topk(beam_size)

            # 各候補を追加
            for log_prob, token_idx in zip(topk_log_probs, topk_indices):
                new_tokens = beam_tokens + [token_idx.item()]
                new_score = beam_score + log_prob.item()
                is_end = token_idx.item() == end_token
                candidates.append((new_tokens, new_score, is_end))

        # 長さ正規化を適用してスコアを計算
        normalized_candidates = []
        for tokens, score, is_end in candidates:
            length_penalty = ((5 + len(tokens)) / 6) ** alpha
            normalized_score = score / length_penalty
            normalized_candidates.append((tokens, normalized_score, is_end))

        # 上位beam_size個を選択
        normalized_candidates.sort(key=lambda x: x[1], reverse=True)
        beams = []
        for tokens, normalized_score, is_end in normalized_candidates[:beam_size]:
            # 元のスコアを復元
            length_penalty = ((5 + len(tokens)) / 6) ** alpha
            original_score = normalized_score * length_penalty
            beams.append((tokens, original_score, is_end))

    # 最良の結果を選択
    best_tokens = max(beams, key=lambda x: x[1])[0]

    # 開始トークンと終了トークンを除去
    if best_tokens[0] == start_token:
        best_tokens = best_tokens[1:]
    if end_token != -1 and best_tokens and best_tokens[-1] == end_token:
        best_tokens = best_tokens[:-1]

    return best_tokens

def main():
    # コマンドライン引数のパース
    parser = argparse.ArgumentParser(description='Transformer翻訳モデルによる推論')
    parser.add_argument('--model', type=str, help='使用するモデルファイルのパス', default='models/best_model_20250411.pth')
    parser.add_argument('--enhanced', action='store_true', help='拡張モデルを使用する')
    args = parser.parse_args()

    # モデルパスの修正（READMEの指示が古い場合の対応）
    if args.model == 'models/best_model.pth':
        logging.info("READMEで指定されたモデルパスが古いため、利用可能なモデルに置き換えます")
        # 最新の日付付きモデルファイルを探す
        model_files = [f for f in glob.glob('models/best_model_*.pth')]
        if model_files:
            latest_model = max(model_files, key=os.path.getctime)
            args.model = latest_model
            logging.info(f"最新のモデルを使用します: {args.model}")
        else:
            args.model = 'models/best_model_20250411.pth'  # フォールバックとして特定のバージョンを指定

    input_vocab = load_vocab(INPUT_VOCAB_PATH)
    output_vocab = load_vocab(OUTPUT_VOCAB_PATH)
    try:
        model = load_model(input_vocab, output_vocab, enhanced=args.enhanced, model_path=args.model)
    except FileNotFoundError as e:
        logging.error(f"エラー: {e}")
        sys.exit(1)

    spacer = " " if CONFIG["TRANSLATION_DESTINATION"] == 'en_US' else ""

    logging.info("モデルをロードしました。入力を待っています...")
    logging.info("exitと入力すると終了します...")
    for line in sys.stdin:
        stripped_line = line.strip()
        if stripped_line.lower() == "exit":
            break

        input_tensor = preprocess_input(stripped_line, input_vocab)
        if input_tensor is None:
            continue

        output_ids = predict(model, input_tensor, input_vocab, output_vocab, CONFIG["MAX_SEQ_LENGTH"], beam_size=5, alpha=0.7)
        if output_ids is None:
            continue

        output_tokens = ids_to_tokens(output_ids, output_vocab)
        print("Output:", spacer.join(output_tokens))
    logging.info("終了しました")

if __name__ == "__main__":
    main()
