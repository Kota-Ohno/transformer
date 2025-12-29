import torch
import sys
import os
import glob
import logging
import argparse
from typing import Dict, Optional, Any, Callable
from utils.config import CONFIG, INPUT_VOCAB_PATH, OUTPUT_VOCAB_PATH
from data.data import tokenize, tokens_to_ids, ids_to_tokens
from models.model import create_transformer_model, TranslationModel
from utils.constants import (
    DEFAULT_INITIAL_CHUNK_SIZE, DEFAULT_MIN_CHUNK_SIZE, DEFAULT_MAX_RETRIES,
    DEFAULT_START_TOKEN_ID, DEFAULT_END_TOKEN_ID
)
from utils.logging_config import setup_logging

# ロギング設定
setup_logging()

def load_model(input_vocab: Dict[str, int], output_vocab: Dict[str, int], model_path: Optional[str] = None) -> TranslationModel:
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

    # パディングインデックスを取得（'<pad>' の存在を明示的にチェック）
    def get_pad_idx(vocab, vocab_name):
        pad_idx = vocab.get('<pad>')
        if pad_idx is None:
            error_msg = f"語彙 '{vocab_name}' に '<pad>' トークンが存在しません。モデルのパディングインデックスを設定できません。"
            logging.error(error_msg)
            raise ValueError(error_msg)
        return pad_idx

    src_pad_idx = get_pad_idx(input_vocab, "input_vocab")
    tgt_pad_idx = get_pad_idx(output_vocab, "output_vocab")

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
    except Exception:
        logging.error("モデルのアーキテクチャと保存されたモデルの設定が一致していない可能性があります。")
        raise

    model.eval()
    return model

def load_vocab(vocab_path: str) -> Dict[str, int]:
    if not os.path.exists(vocab_path):
        raise FileNotFoundError(f"語彙ファイルが見つかりません: {vocab_path}")
    return torch.load(vocab_path, weights_only=True)

def preprocess_input(sentence: str, input_vocab: Dict[str, int]) -> Optional[torch.Tensor]:
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

def handle_unknown_tokens(sentence: str, input_vocab: Dict[str, int]) -> Optional[torch.Tensor]:
    """未知トークンを<unk>に置き換えて処理する"""
    try:
        # <unk>トークンの存在をチェック
        unk_id = input_vocab.get('<unk>')
        if unk_id is None:
            error_msg = "入力語彙に '<unk>' トークンが存在しません。未知トークンを処理できません。"
            logging.error(error_msg)
            raise ValueError(error_msg)

        tokens = tokenize(sentence, CONFIG.data_config.translation_source)
        token_ids = []
        for token in tokens:
            if token in input_vocab:
                token_ids.append(input_vocab[token])
            else:
                logging.warning(f"未知トークン '{token}' を <unk> に置き換えます")
                token_ids.append(unk_id)
        return torch.tensor([token_ids], dtype=torch.long)
    except ValueError:
        # <unk>が存在しない場合のエラーは再発生
        raise
    except Exception as e:
        logging.error(f"未知トークン処理中にエラーが発生しました: {e}")
        return None

def _chunked_predict(
    tensor: torch.Tensor,
    _run_predict: Any,
    is_cuda: bool,
    start_token_id: int,
    end_token_id: int,
    initial_chunk_size: int = DEFAULT_INITIAL_CHUNK_SIZE,
    min_chunk_size: int = DEFAULT_MIN_CHUNK_SIZE,
    max_retries: int = DEFAULT_MAX_RETRIES
) -> Optional[torch.Tensor]:
    """
    OOM 時に入力をチャンク分割して推論をリトライする

    Args:
        tensor: 入力テンソル
        _run_predict: 単一チャンクで predict を実行する関数
        is_cuda: CUDA が利用可能かどうか
        start_token_id: 開始トークンID
        end_token_id: 終了トークンID
        initial_chunk_size: 初期チャンクサイズ
        min_chunk_size: 最小チャンクサイズ
        max_retries: 最大リトライ回数

    Returns:
        推論結果のテンソル

    - チャンクはオーバーラップさせて文脈を多少保持
    - OOM が出た場合はチャンクサイズを徐々に縮小
    """
    seq_len = tensor.size(1)
    # 入力が十分短い場合はそのまま実行
    if seq_len <= initial_chunk_size:
        return _run_predict(tensor)

    chunk_size = max(min(initial_chunk_size, seq_len), min_chunk_size)
    original_error = None
    last_error = None

    for attempt in range(1, max_retries + 1):
        overlap = max(chunk_size // 4, 8) if chunk_size > min_chunk_size else 0
        logging.info(
            f"チャンク推論を実行します: attempt={attempt}, chunk_size={chunk_size}, overlap={overlap}, seq_len={seq_len}"
        )

        if is_cuda:
            try:
                torch.cuda.empty_cache()
            except Exception as ce:
                logging.warning(f"CUDA キャッシュ解放中にエラーが発生しました: {ce}")

        try:
            chunks = []
            start = 0
            while start < seq_len:
                end = min(start + chunk_size, seq_len)
                chunk = tensor[:, start:end]
                chunks.append(chunk)
                if end >= seq_len:
                    break
                # オーバーラップさせて次の開始位置を決定
                start = max(end - overlap, 0)

            # 各チャンクを推論して結合
            chunk_outputs = []
            for idx, c in enumerate(chunks):
                logging.debug(f"チャンク {idx+1}/{len(chunks)} を推論中 (len={c.size(1)})")
                out = _run_predict(c)
                chunk_outputs.append(out)

            # 出力をマージ（<s>, </s> をある程度意識して結合）
            if not chunk_outputs:
                return None

            merged_tokens = []
            num_chunks = len(chunk_outputs)

            for i, out in enumerate(chunk_outputs):
                # [batch_size, tgt_len] を前提に 1 文バッチで扱う
                tokens = out[0].tolist()

                if i == 0:
                    # 最初のチャンク: 終了トークンは最後のチャンク以外では除去
                    if num_chunks > 1 and tokens and tokens[-1] == end_token_id:
                        tokens = tokens[:-1]
                    merged_tokens.extend(tokens)
                else:
                    # 2 個目以降: 先頭の開始トークンを削除
                    while tokens and tokens[0] == start_token_id:
                        tokens = tokens[1:]
                    # 最後のチャンク以外では末尾の終了トークンを削除
                    if i < num_chunks - 1 and tokens and tokens[-1] == end_token_id:
                        tokens = tokens[:-1]
                    merged_tokens.extend(tokens)

            if not merged_tokens:
                return None

            merged_tensor = torch.tensor(merged_tokens, dtype=torch.long, device=tensor.device).unsqueeze(0)
            return merged_tensor

        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                if original_error is None:
                    original_error = e
                last_error = e
                logging.warning(
                    f"チャンク推論中にメモリ不足が発生しました (attempt={attempt}, chunk_size={chunk_size}): {e}"
                )
                # チャンクサイズを縮小してリトライ
                next_chunk_size = max(chunk_size // 2, min_chunk_size)
                if next_chunk_size == chunk_size:
                    logging.error(
                        "チャンクサイズをこれ以上縮小できません。OOM リトライを中止します。"
                    )
                    break
                chunk_size = next_chunk_size
                continue
            else:
                logging.error(f"チャンク推論中に予期しないエラーが発生しました: {e}")
                raise

    # ここまで到達した場合、すべてのリトライが失敗
    logging.error(
        f"OOM リトライ (max_retries={max_retries}) がすべて失敗しました。"
        f" original_error={original_error}, last_error={last_error}"
    )
    if original_error is not None:
        raise original_error
    raise RuntimeError("チャンク推論のリトライがすべて失敗しましたが、詳細な OOM エラーは取得できませんでした。")

def translate(model: TranslationModel, input_tensor: torch.Tensor, output_vocab: Dict[str, int], max_length: Optional[int] = None) -> Optional[torch.Tensor]:
    """
    モデルを使って翻訳を実行

    Args:
        model: 翻訳モデル
        input_tensor: 入力テンソル
        output_vocab: 出力語彙（開始/終了トークンIDを取得するために使用）
        max_length: 最大生成長

    Returns:
        翻訳結果のトークンID
    """
    if max_length is None:
        max_length = CONFIG.model_hyperparameters.max_seq_length
    if input_tensor is None:
        return None

    # 出力語彙から開始/終了トークンIDを取得
    start_token_id = output_vocab.get('<s>')
    end_token_id = output_vocab.get('</s>')

    if start_token_id is None:
        logging.warning(f"出力語彙に '<s>' トークンが見つかりません。デフォルト値{DEFAULT_START_TOKEN_ID}を使用します。")
        start_token_id = DEFAULT_START_TOKEN_ID
    if end_token_id is None:
        logging.warning(f"出力語彙に '</s>' トークンが見つかりません。デフォルト値{DEFAULT_END_TOKEN_ID}を使用します。")
        end_token_id = DEFAULT_END_TOKEN_ID

    # 入力テンソルをモデルと同じデバイスに移動
    device = next(model.parameters()).device
    input_tensor = input_tensor.to(device)

    # CUDA 利用可否を判定
    is_cuda = torch.cuda.is_available() and getattr(device, "type", None) == "cuda"

    def _run_predict(tensor):
        """単一チャンクで predict を実行するヘルパー"""
        return model.predict(tensor, max_length=max_length, start_token=start_token_id, end_token=end_token_id)

    # 翻訳を実行（勾配計算なし）
    with torch.no_grad():
        try:
            output_ids = _run_predict(input_tensor)
            return output_ids
        except RuntimeError as e:
            # メモリ不足エラーの場合
            if "out of memory" in str(e).lower():
                logging.warning("メモリ不足のため、チャンク推論による再試行を行います")
                # 初期チャンクサイズは max_seq_length か入力長の小さい方
                seq_len = input_tensor.size(1)
                initial_chunk_size = min(
                    seq_len,
                    getattr(CONFIG.model_hyperparameters, "max_seq_length", seq_len),
                )
                try:
                    return _chunked_predict(
                        input_tensor,
                        _run_predict=_run_predict,
                        is_cuda=is_cuda,
                        start_token_id=start_token_id,
                        end_token_id=end_token_id,
                        initial_chunk_size=initial_chunk_size
                    )
                except Exception as retry_err:
                    logging.error(
                        f"チャンク推論のリトライにも失敗しました。"
                        f" original_error={e}, retry_error={retry_err}"
                    )
                    raise
            else:
                logging.error(f"翻訳中にエラーが発生しました: {e}")
                raise

def main() -> None:
    """メイン関数"""
    # コマンドライン引数のパース
    parser = argparse.ArgumentParser(description='Transformer翻訳モデルによる推論')
    parser.add_argument('--model', type=str, help='使用するモデルファイルのパス')
    args = parser.parse_args()

    # 語彙のロード
    try:
        input_vocab = load_vocab(INPUT_VOCAB_PATH)
    except FileNotFoundError as e:
        logging.error(f"エラー: {e}")
        sys.exit(1)
    except Exception as e:
        logging.error(f"語彙ファイルのロード中にエラーが発生しました: {e}")
        sys.exit(1)

    try:
        output_vocab = load_vocab(OUTPUT_VOCAB_PATH)
    except FileNotFoundError as e:
        logging.error(f"エラー: {e}")
        sys.exit(1)
    except Exception as e:
        logging.error(f"語彙ファイルのロード中にエラーが発生しました: {e}")
        sys.exit(1)

    try:
        # モデルのロード
        model = load_model(input_vocab, output_vocab, model_path=args.model)
    except FileNotFoundError as e:
        logging.error(f"エラー: {e}")
        sys.exit(1)
    except (ValueError, RuntimeError) as e:
        logging.error(f"モデルのロード中にエラーが発生しました: {e}")
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

        # 前処理
        input_tensor = preprocess_input(stripped_line, input_vocab)
        if input_tensor is None:
            logging.warning(f"入力の前処理に失敗しました: {stripped_line}")
            continue

        # 翻訳実行
        output_ids = translate(model, input_tensor, output_vocab)
        if output_ids is None:
            logging.warning(f"翻訳に失敗しました: {stripped_line}")
            continue

        # 出力処理
        # 特殊トークンのIDを取得
        start_token_id = output_vocab.get('<s>', None)
        end_token_id = output_vocab.get('</s>', None)

        # 特殊トークンを除去
        output_text = []
        for ids in output_ids:
            token_ids = [t.item() for t in ids]
            # 最初の<s>を除去
            if start_token_id is not None and token_ids and token_ids[0] == start_token_id:
                token_ids = token_ids[1:]
            # 最後の</s>を除去
            if end_token_id is not None and token_ids and token_ids[-1] == end_token_id:
                token_ids = token_ids[:-1]

            # トークンをテキストに変換
            tokens = ids_to_tokens(token_ids, output_vocab)
            output_text.append(spacer.join(tokens))

        # 出力
        if not output_text:
            logging.warning(f"翻訳結果が生成されませんでした: {stripped_line}")
            continue
        print("翻訳結果:", output_text[0])

    logging.info("終了しました")

if __name__ == "__main__":
    main()
