import torch
import torch.nn as nn
import torch.optim as optim
import time
import os
import argparse
import subprocess
from datetime import datetime
from data import create_data_loader
from config import CONFIG, MODEL_CONFIG, DEVICE, INPUT_VOCAB_PATH, OUTPUT_VOCAB_PATH
from utils import validate, TranslationModel, WarmupScheduler, download_nltk_resources
from text_tokenizer import load_tokenized_data
import logging
from typing import Tuple, List
import sys
import wandb
# GradScalerのインポートエラーを修正
try:
    from torch.amp import autocast, GradScaler
except ImportError:
    from torch.cuda.amp import autocast, GradScaler
from torch.nn.parallel import DistributedDataParallel as DDP
import traceback

# データ拡張モジュールをインポート
from data_augmentation import augment_dataset

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# BLEU改善の最小閾値を設定
BLEU_IMPROVEMENT_THRESHOLD = 0.01

# 損失改善の最小閾値
LOSS_IMPROVEMENT_THRESHOLD = 0.02

def load_data(train_data_path: str, val_data_path: str, retry_count: int = CONFIG["MAX_RETRY_COUNT"]) -> Tuple[List[List[int]], List[List[int]]]:
    """
    トークナイズ済みデータを読み込む関数（リトライ機能付き）

    Args:
        train_data_path: トレーニングデータのパス
        val_data_path: 検証データのパス
        retry_count: 読み込み失敗時の最大再試行回数

    Returns:
        トレーニングデータと検証データのタプル
    """
    for attempt in range(retry_count):
        try:
            train_token_ids = load_tokenized_data(train_data_path)
            val_token_ids = load_tokenized_data(val_data_path)

            if train_token_ids is None or val_token_ids is None:
                if attempt < retry_count - 1:
                    logging.warning(f"データの読み込みに失敗しました。再試行します... ({attempt+1}/{retry_count})")
                    time.sleep(1)  # 少し待機してから再試行
                    continue
                else:
                    logging.error("先にtext_tokenizer.pyを実行してください")
                    raise ValueError("トレーニングデータまたは検証データがロードできません")

            return train_token_ids, val_token_ids

        except Exception as e:
            if attempt < retry_count - 1:
                logging.warning(f"データの読み込み中にエラーが発生しました: {e}. 再試行します... ({attempt+1}/{retry_count})")
                time.sleep(1)  # 少し待機してから再試行
            else:
                logging.error(f"データの読み込みに失敗しました: {e}")
                raise

    # ここには到達しないはずだが、念のため
    raise ValueError("データの読み込みに失敗しました")

def determine_batch_size():
    """GPUメモリに基づいて適切なバッチサイズを決定する"""
    if not torch.cuda.is_available():
        return CONFIG["BATCH_SIZE"]

    # GPUメモリ情報を取得
    try:
        gpu_props = torch.cuda.get_device_properties(0)
        total_memory = gpu_props.total_memory / 1024**2  # MB単位

        # モデルサイズを推定（HIDDEN_SIZEに基づく単純な見積もり）
        est_model_size = MODEL_CONFIG.hidden_size * MODEL_CONFIG.hidden_size * MODEL_CONFIG.num_layers * 2 * 4 / 1024**2  # MB単位

        # 利用可能なバッチサイズを推定
        available_memory = total_memory * 0.8  # 80%をモデル用に
        estimated_batch_size = int(available_memory / est_model_size)

        # 範囲内に収める
        return max(CONFIG["MIN_BATCH_SIZE"], min(CONFIG["MAX_BATCH_SIZE"], estimated_batch_size))
    except Exception as e:
        logging.warning(f"バッチサイズの自動決定に失敗しました: {e}")
        return CONFIG["BATCH_SIZE"]

def truncate_long_sequences(X_batch, y_batch, max_seq_length):
    """
    長すぎるシーケンスを切り詰めて、メモリ使用量を最適化します

    Args:
        X_batch (torch.Tensor): ソース入力バッチ
        y_batch (torch.Tensor): ターゲット入力バッチ
        max_seq_length (int): 最大シーケンス長

    Returns:
        tuple: (切り詰められたX_batch, 切り詰められたy_batch)
    """
    # テンソルのサイズと次元を検証
    if X_batch.dim() < 2 or y_batch.dim() < 2:
        logging.warning(f"入力テンソルの次元が小さすぎます: X_batch: {X_batch.dim()}, y_batch: {y_batch.dim()}")
        return X_batch, y_batch

    # 入力情報をログに記録（デバッグ用）
    if logging.getLogger().isEnabledFor(logging.DEBUG):
        logging.debug(f"シーケンス長: X={X_batch.size(1)}, y={y_batch.size(1)}, max={max_seq_length}")
        try:
            logging.debug(f"X_batch範囲: min={X_batch.min().item()}, max={X_batch.max().item()}")
            logging.debug(f"y_batch範囲: min={y_batch.min().item()}, max={y_batch.max().item()}")
        except Exception as e:
            logging.debug(f"バッチ統計計算中のエラー: {e}")

    # ソースシーケンスの切り詰め
    if X_batch.size(1) > max_seq_length:
        logging.info(f"ソースシーケンスを切り詰めます: {X_batch.size(1)} → {max_seq_length}")
        X_batch = X_batch[:, :max_seq_length]

    # ターゲットシーケンスの切り詰め
    if y_batch.size(1) > max_seq_length:
        logging.info(f"ターゲットシーケンスを切り詰めます: {y_batch.size(1)} → {max_seq_length}")
        y_batch = y_batch[:, :max_seq_length]

    return X_batch, y_batch

def determine_model_size():
    """
    利用可能なGPUメモリに基づいてモデルのサイズを調整します。

    Returns:
        tuple: (hidden_size, num_heads, num_layers)
    """
    try:
        # config.pyのModelConfigクラスを使用して適切な設定を取得
        from config import ModelConfig

        # GPUメモリに基づいた設定を自動的に取得
        model_config = ModelConfig.from_gpu_memory()

        # 設定値をログに出力
        logging.info(f"GPUメモリに基づくモデル設定を適用します: hidden_size={model_config.hidden_size}, "
                    f"num_heads={model_config.num_heads}, num_layers={model_config.num_layers}")

        return model_config.hidden_size, model_config.num_heads, model_config.num_layers
    except Exception as e:
        logging.warning(f"モデルサイズの自動調整に失敗しました: {e}")
        # エラー時はデフォルト値を使用
        return MODEL_CONFIG.hidden_size, MODEL_CONFIG.num_heads, MODEL_CONFIG.num_layers

def apply_data_augmentation(train_token_ids, sp_src, sp_tgt, augmentation_factor=CONFIG["DATA_AUGMENTATION_FACTOR"]):
    """
    トレーニングデータに対してデータ拡張を適用します

    Args:
        train_token_ids: トレーニングデータのトークンID
        sp_src: ソース言語のSentencePieceモデル
        sp_tgt: ターゲット言語のSentencePieceモデル
        augmentation_factor: 拡張データの割合

    Returns:
        拡張後のトレーニングデータ
    """
    try:
        logging.info(f"データ拡張を適用します (拡張率: {augmentation_factor})")
        augmented_data = augment_dataset(
            train_token_ids,
            sp_src,
            sp_tgt,
            augmentation_factor=augmentation_factor,
            techniques=CONFIG["DATA_AUGMENTATION_TECHNIQUES"]
        )
        logging.info(f"データ拡張が完了しました: {len(train_token_ids)} サンプル → {len(augmented_data)} サンプル")
        return augmented_data
    except Exception as e:
        logging.error(f"データ拡張中にエラーが発生しました: {e}")
        logging.warning("元のデータをそのまま使用します")
        return train_token_ids

def setup_checkpointing_directory():
    """チェックポイント保存用のディレクトリを設定します"""
    checkpoint_dir = os.path.join("models", "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    return checkpoint_dir

def save_checkpoint(model, optimizer, scheduler, epoch, val_loss, bleu_score, is_best=False,
                  model_hidden_size=None, model_num_heads=None, model_num_layers=None):
    """
    モデルのチェックポイントを保存します

    Args:
        model: モデル
        optimizer: オプティマイザ
        scheduler: スケジューラ
        epoch: 現在のエポック
        val_loss: 検証損失
        bleu_score: BLEUスコア
        is_best: 最良モデルかどうか
        model_hidden_size: 実際に使用した隠れ層のサイズ
        model_num_heads: 実際に使用したアテンションヘッドの数
        model_num_layers: 実際に使用したレイヤー数
    """
    checkpoint_dir = setup_checkpointing_directory()

    # 現在の日付を取得
    current_date = datetime.now().strftime("%Y%m%d")

    # モデル設定情報を取得
    # モデルからhidden_size, num_heads, num_layersなどを取得
    if hasattr(model, 'encoder') and hasattr(model.encoder, 'layers'):
        num_layers = len(model.encoder.layers)
    else:
        num_layers = MODEL_CONFIG.num_layers

    # 引数で渡された値があれば優先して使用
    hidden_size = model_hidden_size or MODEL_CONFIG.hidden_size
    num_heads = model_num_heads or MODEL_CONFIG.num_heads
    num_layers = model_num_layers or num_layers

    # モデル設定を辞書に保存
    model_config = {
        'HIDDEN_SIZE': hidden_size,
        'NUM_HEADS': num_heads,
        'NUM_LAYERS': num_layers,
        'D_FF': MODEL_CONFIG.d_ff,
        'DROPOUT_RATE': MODEL_CONFIG.dropout,
        'MAX_SEQ_LENGTH': MODEL_CONFIG.max_seq_length,
        'REL_POS_MAX_DISTANCE': MODEL_CONFIG.rel_pos_max_distance
    }

    # チェックポイント情報を準備
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'val_loss': val_loss,
        'bleu_score': bleu_score,
        'date': current_date,
        'model_config': model_config  # モデル設定情報を追加
    }

    # 定期的なチェックポイントを保存
    checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch}_{current_date}.pth")
    torch.save(checkpoint, checkpoint_path)
    logging.info(f"チェックポイントを保存しました: {checkpoint_path}")

    # 最良モデルの場合は別名で保存
    if is_best:
        best_model_path = os.path.join("models", f"best_model_{current_date}.pth")
        torch.save(checkpoint, best_model_path)
        logging.info(f"最良モデルを保存しました: {best_model_path}")

def load_checkpoint(checkpoint_path, model, optimizer=None, scheduler=None, update_globals=False):
    """
    チェックポイントからモデルを読み込みます

    Args:
        checkpoint_path: チェックポイントファイルのパス
        model: モデル
        optimizer: オプティマイザ（オプション）
        scheduler: スケジューラ（オプション）
        update_globals: グローバル変数を更新するかどうか（デフォルトはFalse、configから読むため）

    Returns:
        モデル、エポック、検証損失、BLEUスコア、モデル設定（辞書）
    """
    try:
        logging.info(f"チェックポイントを読み込んでいます: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=DEVICE)

        model.load_state_dict(checkpoint['model_state_dict'])

        if optimizer is not None and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        if scheduler is not None and 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        epoch = checkpoint.get('epoch', 0)
        val_loss = checkpoint.get('val_loss', float('inf'))
        bleu_score = checkpoint.get('bleu_score', 0.0)

        # モデル設定情報を取得
        model_config = checkpoint.get('model_config', {})

        logging.info(f"チェックポイントを読み込みました (エポック {epoch}, 検証損失 {val_loss:.4f}, BLEU {bleu_score:.4f})")
        return model, epoch, val_loss, bleu_score, model_config

    except Exception as e:
        logging.error(f"チェックポイントの読み込みに失敗しました: {e}")
        traceback.print_exc()
        return model, 0, float('inf'), 0.0, {}

def find_latest_checkpoint():
    """
    最新のチェックポイントを探します

    Returns:
        最新のチェックポイントパス、存在しない場合はNone
    """
    checkpoint_dir = os.path.join("models", "checkpoints")
    if not os.path.exists(checkpoint_dir):
        return None

    checkpoints = [os.path.join(checkpoint_dir, f) for f in os.listdir(checkpoint_dir) if f.startswith("checkpoint")]
    if not checkpoints:
        return None

    # 最新のファイルを見つける
    latest_checkpoint = max(checkpoints, key=os.path.getctime)
    return latest_checkpoint

def train(model, train_loader, optimizer, criterion, scheduler, scaler, device):
    """
    1エポックのトレーニングを実行する関数

    Args:
        model: トレーニングするモデル
        train_loader: トレーニングデータのデータローダー
        optimizer: オプティマイザ
        criterion: 損失関数
        scheduler: 学習率スケジューラ
        scaler: 混合精度トレーニング用のスケーラー
        device: 使用するデバイス

    Returns:
        平均トレーニング損失
    """
    model.train()
    epoch_loss = 0
    total_batches = len(train_loader)
    last_log_time = time.time()
    valid_batch_count = 0
    max_seq_length = CONFIG["MAX_SEQ_LENGTH"]

    for i, (src, tgt) in enumerate(train_loader):
        try:
            # シーケンスが長すぎる場合は切り詰める
            src, tgt = truncate_long_sequences(src, tgt, max_seq_length)

            # 入力と出力をデバイスに移動
            src = src.to(device)
            tgt = tgt.to(device)

            # サイズ情報をログに記録
            if i == 0:
                logging.info(f"バッチサイズ: {src.size(0)}, ソースシーケンス長: {src.size(1)}, ターゲットシーケンス長: {tgt.size(1)}")

            # テンソルサイズの検証
            if src.size(0) < 1 or tgt.size(0) < 1:
                logging.warning(f"バッチサイズが小さすぎます: src={src.size(0)}, tgt={tgt.size(0)}")
                continue

            # ターゲットの入力と出力を準備
            tgt_input = tgt[:, :-1]
            tgt_output = tgt[:, 1:]

            # グラデーションをゼロにリセット
            optimizer.zero_grad()

            # 混合精度トレーニングのコンテキスト
            with autocast():
                try:
                    # モデルの順伝播
                    output, _ = model(src, tgt_input)

                    # 予測と実際の出力のサイズを揃える
                    output_dim = output.shape[-1]
                    output = output.contiguous().view(-1, output_dim)
                    tgt_output = tgt_output.contiguous().view(-1)

                    # 損失の計算
                    loss = criterion(output, tgt_output)

                except RuntimeError as e:
                    # テンソルサイズエラー等のランタイムエラーをキャッチ
                    if "size mismatch" in str(e) or "shape mismatch" in str(e) or "out of memory" in str(e):
                        logging.warning(f"バッチ処理中にエラーが発生しました（バッチ {i+1}/{total_batches}）: {e}")
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()  # GPUメモリをクリア
                        continue
                    else:
                        raise  # その他のエラーは再び投げる

            # 損失のスケーリングと逆伝播
            scaler.scale(loss).backward()

            # グラデーションのクリッピング
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), CONFIG["GRAD_CLIP_NORM"])

            # オプティマイザとスケジューラの更新
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            # 損失の累積
            epoch_loss += loss.item()
            valid_batch_count += 1

            # 定期的に進捗を表示
            current_time = time.time()
            if current_time - last_log_time > 10:  # 10秒ごとに表示
                logging.info(f"バッチ {i+1}/{total_batches} 完了 ({((i+1)/total_batches*100):.1f}%)")
                last_log_time = current_time

        except Exception as e:
            logging.error(f"バッチ処理中に予期せぬエラーが発生しました（バッチ {i+1}/{total_batches}）: {e}")
            # スタックトレースを出力（デバッグ用）
            import traceback
            logging.error(traceback.format_exc())
            # バッチをスキップして次へ
            continue

    # 平均損失を計算して返す（有効なバッチがある場合のみ）
    return epoch_loss / max(1, valid_batch_count)

def main():
    """
    翻訳モデルのトレーニングを実行する主要な関数。

    この関数は以下の手順を実行します：
    1. データの読み込みと前処理
    2. モデル、損失関数、オプティマイザの初期化
    3. トレーニングループの実行
    4. 検証と早期停止の処理
    5. モデルの保存

    Raises:
        ValueError: データの読み込みに失敗した場合
        Exception: その他の予期せぬエラーが発生した場合
    """
    parser = argparse.ArgumentParser(description='Transformerモデルのトレーニング')
    parser.add_argument('--resume', action='store_true', help='最新のチェックポイントから再開する')
    parser.add_argument('--checkpoint', type=str, help='特定のチェックポイントから再開する')
    parser.add_argument('--augment', action='store_true', help='データ拡張を有効にする')
    parser.add_argument('--augment-factor', type=float, default=CONFIG["DATA_AUGMENTATION_FACTOR"],
                        help='データ拡張の割合')
    parser.add_argument('--epochs', type=int, default=CONFIG["NUM_EPOCHS"], help='トレーニングのエポック数')
    parser.add_argument('--batch-size', type=int, default=CONFIG["BATCH_SIZE"], help='バッチサイズ')
    parser.add_argument('--no-wandb', action='store_true', help='Weights & Biasesのログを無効にする')
    parser.add_argument('--enhanced', action='store_true', help='相対位置エンコーディングとGLUを使用した強化版モデルを使用する')
    parser.add_argument('--rel-pos-max-dist', type=int, default=CONFIG["REL_POS_MAX_DISTANCE"],
                       help='相対位置エンコーディングの最大距離')
    parser.add_argument('--warmup-steps', type=int, default=CONFIG["WARMUP_STEPS"],
                       help='ウォームアップステップ数')
    args = parser.parse_args()

    # Dockerコンテナ内で実行されているかを確認して対応する
    docker_detected = os.path.exists('/.dockerenv') or os.environ.get('DOCKER_CONTAINER') == 'true'
    if docker_detected:
        logging.info("Dockerコンテナ内での実行を検出しました")

    # 強化版モデルの使用フラグ（コマンドラインまたは設定ファイル）
    use_enhanced_model = args.enhanced or CONFIG["USE_ENHANCED_MODEL"]

    try:
        # NLTK リソースのダウンロード
        download_nltk_resources()

        # トークナイズ済みデータのパス
        train_data_path = "tokenized_train_data.pth"
        val_data_path = "tokenized_val_data.pth"

        # データの読み込み
        train_token_ids, val_token_ids = load_data(train_data_path, val_data_path)

        # ボキャブラリの読み込み
        input_vocab = torch.load(INPUT_VOCAB_PATH)
        output_vocab = torch.load(OUTPUT_VOCAB_PATH)

        # データ拡張（オプション）
        if args.augment:
            # SentencePieceモデルをロードして拡張に使用
            sp_src_path = os.path.join("models", "sp_src.pth")
            sp_tgt_path = os.path.join("models", "sp_tgt.pth")

            sp_src = torch.load(sp_src_path)
            sp_tgt = torch.load(sp_tgt_path)

            # データ拡張を適用
            train_token_ids = apply_data_augmentation(
                train_token_ids,
                sp_src,
                sp_tgt,
                augmentation_factor=args.augment_factor
            )

        # 入力と出力の次元を設定
        input_dim = len(input_vocab)
        output_dim = len(output_vocab)

        # ボキャブラリーサイズをログに記録
        logging.info(f"入力ボキャブラリーサイズ: {input_dim}, 出力ボキャブラリーサイズ: {output_dim}")

        # パディングインデックスを取得
        src_pad_idx = input_vocab['<pad>']
        tgt_pad_idx = output_vocab['<pad>']

        # GPUメモリに基づいてモデルサイズを決定
        model_hidden_size, model_num_heads, model_num_layers = determine_model_size()
        logging.info(f"モデル設定: hidden_size={model_hidden_size}, heads={model_num_heads}, layers={model_num_layers}")

        # 使用するモデルタイプのログ
        if use_enhanced_model:
            logging.info(f"強化版モデルを使用します（相対位置エンコーディング、最大距離: {args.rel_pos_max_dist}、GLU使用: {CONFIG['USE_GLU']}）")
        else:
            logging.info("標準のTransformerモデルを使用します")

        # チェックポイントからの復元（前の処理と順序を入れ替え）
        start_epoch = 0
        best_val_loss = float('inf')
        best_bleu_score = 0.0

        # モデルの作成
        if use_enhanced_model:
            # 強化版モデルを作成
            from enhanced_model import create_enhanced_model
            model = create_enhanced_model(
                input_dim=input_dim,
                output_dim=output_dim,
                hidden_dim=model_hidden_size,
                num_heads=model_num_heads,
                num_layers=model_num_layers,
                ff_dim=MODEL_CONFIG.d_ff, # Use MODEL_CONFIG
                src_pad_idx=src_pad_idx,
                tgt_pad_idx=tgt_pad_idx,
                dropout=MODEL_CONFIG.dropout, # Use MODEL_CONFIG
                device=DEVICE,
                max_dist=args.rel_pos_max_dist
            )
        else:
            # 標準のモデルを作成
            from encoder import Encoder
            from decoder import Decoder

            encoder = Encoder(input_dim, model_hidden_size, model_num_heads, model_num_layers, MODEL_CONFIG.d_ff, MODEL_CONFIG.dropout, DEVICE).to(DEVICE)
            decoder = Decoder(output_dim, model_hidden_size, model_num_heads, model_num_layers, MODEL_CONFIG.d_ff, output_dim, MODEL_CONFIG.dropout, DEVICE).to(DEVICE)
            model = TranslationModel(encoder, decoder, src_pad_idx, tgt_pad_idx, DEVICE).to(DEVICE)

        # GPUメモリに基づいてバッチサイズを決定
        batch_size = determine_batch_size() if args.batch_size == CONFIG["BATCH_SIZE"] else args.batch_size
        logging.info(f"使用するバッチサイズ: {batch_size}")

        # デバイスにモデルを配置
        model = model.to(DEVICE)

        # Weights & Biasesの初期化
        if not args.no_wandb:
            run_name = f"train_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            wandb.init(
                project="translation_project",
                name=run_name,
                config={
                    "model_type": "enhanced" if use_enhanced_model else "standard",
                    "hidden_size": model_hidden_size,
                    "num_heads": model_num_heads,
                    "num_layers": model_num_layers,
                    "batch_size": batch_size,
                    "learning_rate": CONFIG["LEARNING_RATE"],
                    "epochs": args.epochs,
                    "dropout": MODEL_CONFIG.dropout,
                    "warmup_steps": args.warmup_steps,
                    "data_augmentation": args.augment,
                    "augment_factor": args.augment_factor if args.augment else 0,
                    "rel_pos_max_dist": args.rel_pos_max_dist if use_enhanced_model else None,
                }
            )

        # 損失関数を定義
        criterion = nn.CrossEntropyLoss(ignore_index=output_vocab['<pad>']).to(DEVICE)

        # オプティマイザとスケジューラを定義
        optimizer = optim.AdamW(
            model.parameters(),
            lr=CONFIG["LEARNING_RATE"],
            weight_decay=CONFIG["WEIGHT_DECAY"]  # 重み減衰を追加して過学習を抑制
        )

        # 改良されたスケジューラを使用（線形減衰を選択）
        scheduler = WarmupScheduler(
            optimizer,
            model_hidden_size,
            args.warmup_steps,
            args.epochs * (len(train_token_ids) // args.batch_size + 1),
            min_lr=1e-6,
            initial_lr=CONFIG["LEARNING_RATE"],
            decay_method='linear'  # 線形減衰を使用
        )

        # チェックポイントからモデルを復元（オプション）
        if args.resume or args.checkpoint:
            checkpoint_path = args.checkpoint if args.checkpoint else find_latest_checkpoint()
            if checkpoint_path:
                logging.info(f"チェックポイントから復元を試みます: {checkpoint_path}")
                model, start_epoch, best_val_loss, best_bleu_score, loaded_model_config = load_checkpoint(
                    checkpoint_path, model, optimizer, scheduler
                )
                # 必要であればロードした設定でモデルサイズを更新
                model_hidden_size = loaded_model_config.get('HIDDEN_SIZE', model_hidden_size)
                model_num_heads = loaded_model_config.get('NUM_HEADS', model_num_heads)
                model_num_layers = loaded_model_config.get('NUM_LAYERS', model_num_layers)
                start_epoch += 1  # 次のエポックから開始
                logging.info(f"チェックポイントから復元完了: エポック {start_epoch-1}, 検証損失 {best_val_loss:.4f}, BLEU {best_bleu_score:.4f}")
            else:
                logging.warning("チェックポイントが見つかりませんでした。トレーニングを最初から開始します。")

        # データローダーを作成
        train_loader = create_data_loader(train_token_ids, batch_size)
        val_loader = create_data_loader(val_token_ids, batch_size)

        patience_counter = 0
        total_steps = len(train_loader)

        # 混合精度トレーニング用のスケーラー
        scaler = GradScaler() if torch.cuda.is_available() else None

        # トレーニングループ
        for epoch in range(start_epoch, args.epochs):
            try:
                # エポックごとのトレーニング
                logging.info(f"エポック {epoch+1}/{args.epochs} 開始")
                start_time = time.time()

                # トレーニング
                try:
                    train_loss = train(model, train_loader, optimizer, criterion, scheduler, scaler, DEVICE)
                    logging.info(f"トレーニング損失: {train_loss:.4f}")
                except Exception as e:
                    logging.error(f"トレーニング中にエラーが発生しました: {e}")
                    logging.error(traceback.format_exc())
                    # バッチサイズを小さくして再試行するか適切なフォールバック処理を行う
                    if batch_size > CONFIG["MIN_BATCH_SIZE"]:
                        new_batch_size = max(CONFIG["MIN_BATCH_SIZE"], batch_size // 2)
                        logging.warning(f"バッチサイズを{batch_size}から{new_batch_size}に減らして再試行します")
                        batch_size = new_batch_size
                        train_loader = create_data_loader(train_token_ids, batch_size)
                        val_loader = create_data_loader(val_token_ids, batch_size)
                        continue  # 同じエポックを再試行
                    else:
                        logging.error("最小バッチサイズでもエラーが発生しました。トレーニングを中断します。")
                        break

                # 検証
                try:
                    logging.info("検証を開始...")
                    val_loss, bleu_score, translations = validate(
                        model, val_loader, criterion,
                        output_vocab, DEVICE, return_translations=True
                    )

                    # サンプル翻訳をログに記録
                    if len(translations) > 0:
                        logging.info("翻訳サンプル:")
                        for i, sample in enumerate(translations[:3]):  # 最初の3つのみ表示
                            logging.info(f"サンプル {i+1}:")
                            logging.info(f"  ソーステキスト: {sample['source']}")
                            logging.info(f"  目標訳: {sample['target']}")
                            logging.info(f"  モデル訳: {sample['prediction']}")

                    logging.info(f"Epoch {epoch+1}/{args.epochs} - 検証損失: {val_loss:.4f}, BLEUスコア: {bleu_score:.4f}")

                    # 検証指標をログに記録
                    if not args.no_wandb:
                        wandb.log({
                            "epoch": epoch,
                            "val_loss": val_loss,
                            "bleu_score": bleu_score,
                            "learning_rate": optimizer.param_groups[0]['lr']
                        })

                    # 最良モデルの保存とEarly Stopping
                    if val_loss < best_val_loss:
                        logging.info(f"検証損失が改善しました ({best_val_loss:.4f} -> {val_loss:.4f})。モデルを保存します...")
                        best_val_loss = val_loss
                        best_model_path = os.path.join("models", f"best_model_loss_{epoch+1}.pt")
                        torch.save({
                            'epoch': epoch,
                            'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
                            'val_loss': val_loss,
                            'batch_size': batch_size,
                            'bleu_score': bleu_score
                        }, best_model_path)
                        patience_counter = 0
                    elif bleu_score > best_bleu_score:
                        logging.info(f"BLEUスコアが改善しました ({best_bleu_score:.4f} -> {bleu_score:.4f})。モデルを保存します...")
                        best_bleu_score = bleu_score
                        best_model_bleu_path = os.path.join("models", f"best_model_bleu_{epoch+1}.pt")
                        torch.save({
                            'epoch': epoch,
                            'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
                            'val_loss': val_loss,
                            'batch_size': batch_size,
                            'bleu_score': bleu_score
                        }, best_model_bleu_path)
                        patience_counter = 0
                    else:
                        patience_counter += 1
                        logging.info(f"検証指標が改善していません。Early Stoppingカウンター: {patience_counter}/{CONFIG['PATIENCE']}")

                    # Early Stopping
                    if patience_counter >= CONFIG['PATIENCE']:
                        logging.info(f"{CONFIG['PATIENCE']}エポック連続で改善がないため、トレーニングを早期終了します。")
                        break

                except Exception as e:
                    logging.error(f"検証ステップでエラーが発生しました: {e}")
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

                # エポックごとの要約を出力
                epoch_time = time.time() - start_time
                logging.info(f"Epoch {epoch+1} took {epoch_time:.2f} seconds.")

            except Exception as e:
                logging.error(f"エポック {epoch+1} 処理中に予期せぬエラーが発生しました: {e}")
                logging.error(traceback.format_exc())
                # 次のエポックに進む
                continue

        # トレーニング完了
        logging.info("トレーニングが完了しました")

        # 最良モデルパスを検索
        best_model_path = None
        models_dir = "models"
        for file in os.listdir(models_dir):
            if file.startswith("best_model_"):
                best_model_path = os.path.join(models_dir, file)
                break

        if best_model_path:
            logging.info(f"最良モデルパス: {best_model_path}")

        if not args.no_wandb:
            wandb.finish()

    except ValueError as ve:
        logging.error(f"データの読み込みに失敗しました: {ve}")
        sys.exit(1)
    except Exception as e:
        logging.error(f"予期せぬエラーが発生しました: {e}")
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
