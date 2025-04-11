import torch
import sys
import os
import glob
import logging
from encoder import Encoder
from decoder import Decoder
from utils import TranslationModel, create_padding_mask, create_subsequent_mask
from config import (
    DEVICE, HIDDEN_SIZE, NUM_HEADS, NUM_LAYERS, D_FF, DROPOUT_RATE,
    TRANSLATION_SOURCE, TRANSLATION_DESTINATION, INPUT_VOCAB_PATH,
    OUTPUT_VOCAB_PATH, MAX_SEQ_LENGTH
)
from data import tokenize, tokens_to_ids, ids_to_tokens

# ロギング設定
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_model(input_vocab, output_vocab):
    input_dim = len(input_vocab)
    output_dim = len(output_vocab)

    # パディングインデックスを取得
    src_pad_idx = input_vocab['<pad>']
    tgt_pad_idx = output_vocab['<pad>']

    encoder = Encoder(input_dim, HIDDEN_SIZE, NUM_HEADS, NUM_LAYERS, D_FF, DROPOUT_RATE, DEVICE).to(DEVICE)
    decoder = Decoder(output_dim, HIDDEN_SIZE, NUM_HEADS, NUM_LAYERS, D_FF, output_dim, DROPOUT_RATE, DEVICE).to(DEVICE)
    model = TranslationModel(encoder, decoder, src_pad_idx, tgt_pad_idx, DEVICE).to(DEVICE)

    # modelsディレクトリから最新のモデルファイル名を取得
    model_files = glob.glob('models/*.pth')
    # 語彙ファイルを除外
    model_files = [f for f in model_files if not f.endswith(('vocab_input.pth', 'vocab_output.pth'))]
    if not model_files:
        raise FileNotFoundError("学習済みモデルファイルが見つかりません。'models/' ディレクトリを確認してください。")
    model_filename = max(model_files, key=os.path.getctime)  # 最新のファイルを選択

    # パス結合
    model_path = os.path.join("models", os.path.basename(model_filename))
    logging.info(f"Loading model from: {model_path}")
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))  # DEVICEにマップ
    model.eval()
    return model

def load_vocab(vocab_path):
    return torch.load(vocab_path)

def preprocess_input(sentence, input_vocab):
    try:
        tokens = tokenize(sentence, TRANSLATION_SOURCE)
        token_ids = tokens_to_ids(tokens, input_vocab)
        return torch.tensor([token_ids], dtype=torch.long).to(DEVICE)
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
        tokens = tokenize(sentence, TRANSLATION_SOURCE)
        token_ids = []
        for token in tokens:
            if token in input_vocab.token2id:
                token_ids.append(input_vocab.token2id[token])
            else:
                logging.warning(f"未知トークン '{token}' を <unk> に置き換えます")
                token_ids.append(input_vocab.token2id['<unk>'])
        return torch.tensor([token_ids], dtype=torch.long).to(DEVICE)
    except Exception as e:
        logging.error(f"未知トークン処理中にエラーが発生しました: {e}")
        return None

def predict(model, input_tensor, input_vocab, output_vocab, max_len=MAX_SEQ_LENGTH, beam_size=5, alpha=0.7):
    """
    改良版ビームサーチで高品質な翻訳を生成する関数。
    キャッシュを活用して計算を効率化し、長さ正規化とN-gramペナルティを導入。

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

    model.eval()
    start_token = output_vocab['<s>']
    end_token = output_vocab.token2id.get('</s>', -1)
    src_pad_idx = input_vocab['<pad>']
    tgt_pad_idx = output_vocab['<pad>']

    # シーケンス生成中に重複するN-gramに対するペナルティを設定
    ngram_size = 3
    beta = 0.5  # N-gramペナルティの強さ

    # エンコーダー処理とソースマスク (一度だけ計算)
    with torch.no_grad():
        # ソースパディングマスク生成
        src_mask = create_padding_mask(input_tensor, src_pad_idx).to(DEVICE)
        encoder_output = model.encoder(input_tensor, src_mask)

    # 初期ビーム
    # 各ビームは [トークンIDのリスト, 累積対数確率, 完了フラグ, エンコーダキャッシュ, N-gramカウント]
    beams = [[
        [start_token],  # トークンシーケンス
        0.0,            # 累積スコア
        False,          # 完了フラグ
        [None] * len(model.decoder.layers),  # デコーダーキャッシュ
        {}              # N-gramカウント（重複防止用）
    ]]

    # ビームサーチループ
    for i in range(max_len):
        if i > 0 and all(beam[2] for beam in beams):
            break  # すべてのビームが完了していれば終了

        candidates = []

        for beam_tokens, beam_score, is_finished, beam_cache, ngram_counts in beams:
            if is_finished:
                candidates.append([beam_tokens, beam_score, True, beam_cache, ngram_counts])
                continue

            # デコーダー入力を準備
            decoder_input = torch.tensor([beam_tokens[-1:]], dtype=torch.long, device=DEVICE)  # 新しいトークンだけを入力

            # もし特殊なトークンがあれば直接候補に追加
            if beam_tokens[-1] == end_token:
                candidates.append([beam_tokens, beam_score, True, beam_cache, ngram_counts])
                continue

            # マスク生成
            full_sequence_len = len(beam_tokens)
            dummy_sequence = torch.ones((1, full_sequence_len), dtype=torch.long, device=DEVICE)

            tgt_sub_mask = create_subsequent_mask(dummy_sequence).to(DEVICE)
            tgt_pad_mask = create_padding_mask(dummy_sequence, tgt_pad_idx).to(DEVICE)
            tgt_mask = tgt_pad_mask.expand(-1, -1, dummy_sequence.size(1), -1) & tgt_sub_mask

            with torch.no_grad():
                # デコーダー実行（キャッシュ利用）
                decoder_output, new_cache = model.decoder(
                    decoder_input, encoder_output,
                    tgt_mask[:, :, -1:, :full_sequence_len],
                    src_mask,
                    cache=beam_cache
                )

            # 次のトークンの確率を取得
            next_token_logits = decoder_output[:, -1, :]  # (1, vocab_size)
            next_token_log_probs = torch.log_softmax(next_token_logits, dim=-1)

            # 上位beam_size個の次トークンを取得
            topk_log_probs, topk_indices = next_token_log_probs[0].topk(beam_size * 2)  # 2倍のトークンを候補に

            # 各候補を候補リストに追加
            for log_prob, token_idx in zip(topk_log_probs, topk_indices):
                token_idx = token_idx.item()
                new_tokens = beam_tokens + [token_idx]

                # N-gramペナルティの計算
                # 新しいN-gramを作成し、重複をチェック
                new_ngram_counts = ngram_counts.copy()
                ngram_penalty = 0.0

                # 現在のN-gramを計算し、ペナルティを適用
                for n in range(1, min(ngram_size + 1, len(new_tokens))):
                    ngram = tuple(new_tokens[-n:])
                    if ngram in new_ngram_counts:
                        # 重複するN-gramに対してペナルティを適用
                        ngram_penalty += beta * new_ngram_counts[ngram]
                        new_ngram_counts[ngram] += 1
                    else:
                        new_ngram_counts[ngram] = 1

                # スコア計算（対数確率 - N-gramペナルティ）
                new_score = beam_score + log_prob.item() - ngram_penalty

                # 終了トークンの場合はフラグを立てる
                is_end = token_idx == end_token if end_token != -1 else False

                candidates.append([new_tokens, new_score, is_end, new_cache, new_ngram_counts])

        # スコア計算に長さ正規化を適用
        for i, (beam_tokens, beam_score, _, _, _) in enumerate(candidates):
            # 長さ正規化 (length penalty)
            lp = ((5 + len(beam_tokens)) / 6) ** alpha
            normalized_score = beam_score / lp
            candidates[i][1] = normalized_score  # 正規化スコアで一時的に置き換え

        # すべての候補から正規化スコアで上位beam_size個を選択
        candidates.sort(key=lambda x: x[1], reverse=True)
        candidates = candidates[:beam_size]

        # 元のスコアを復元
        for i, (beam_tokens, beam_score, _, _, _) in enumerate(candidates):
            lp = ((5 + len(beam_tokens)) / 6) ** alpha
            original_score = beam_score * lp
            candidates[i][1] = original_score

        beams = candidates

    # 最高スコアのビームを選択
    best_beam = max(beams, key=lambda x: x[1])
    best_tokens = best_beam[0]

    # 開始トークンと終了トークンを除去
    if best_tokens[0] == start_token:
        best_tokens = best_tokens[1:]
    if end_token != -1 and best_tokens and best_tokens[-1] == end_token:
        best_tokens = best_tokens[:-1]

    return best_tokens

def main():
    input_vocab = load_vocab(INPUT_VOCAB_PATH)
    output_vocab = load_vocab(OUTPUT_VOCAB_PATH)
    try:
        model = load_model(input_vocab, output_vocab)
    except FileNotFoundError as e:
        logging.error(f"エラー: {e}")
        sys.exit(1)

    spacer = " " if TRANSLATION_DESTINATION == 'en_US' else ""

    logging.info("モデルをロードしました。入力を待っています...")
    logging.info("exitと入力すると終了します...")
    for line in sys.stdin:
        stripped_line = line.strip()
        if stripped_line.lower() == "exit":
            break

        input_tensor = preprocess_input(stripped_line, input_vocab)
        if input_tensor is None:
            continue

        output_ids = predict(model, input_tensor, input_vocab, output_vocab, MAX_SEQ_LENGTH, beam_size=5, alpha=0.7)
        if output_ids is None:
            continue

        output_tokens = ids_to_tokens(output_ids, output_vocab)
        print("Output:", spacer.join(output_tokens))
    logging.info("終了しました")

if __name__ == "__main__":
    main()
