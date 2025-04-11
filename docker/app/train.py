import torch
import torch.nn as nn
import torch.optim as optim
import time
import os
import argparse
import subprocess
from datetime import datetime
from data import create_data_loader
from config import (
    WARMUP_STEPS, PATIENCE, BATCH_SIZE, MIN_BATCH_SIZE, MAX_BATCH_SIZE,
    ACCUMULATED_BATCHES, LEARNING_RATE, INPUT_VOCAB_PATH, GRAD_CLIP_NORM,
    OUTPUT_VOCAB_PATH, HIDDEN_SIZE, NUM_HEADS, NUM_LAYERS, D_FF,
    DROPOUT_RATE, DEVICE, NUM_EPOCHS, MAX_SEQ_LENGTH, LAYER_DROPOUT, WEIGHT_DECAY,
    DATA_AUGMENTATION_FACTOR, DATA_AUGMENTATION_TECHNIQUES, MAX_RETRY_COUNT,
    USE_ENHANCED_MODEL, REL_POS_MAX_DISTANCE, USE_GLU
)
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

def load_data(train_data_path: str, val_data_path: str, retry_count: int = MAX_RETRY_COUNT) -> Tuple[List[List[int]], List[List[int]]]:
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
        return BATCH_SIZE

    # GPUメモリ情報を取得
    try:
        gpu_props = torch.cuda.get_device_properties(0)
        total_memory = gpu_props.total_memory / 1024**2  # MB単位

        # モデルサイズを推定（HIDDEN_SIZEに基づく単純な見積もり）
        est_model_size = HIDDEN_SIZE * HIDDEN_SIZE * NUM_LAYERS * 2 * 4 / 1024**2  # MB単位

        # 利用可能なバッチサイズを推定
        available_memory = total_memory * 0.8  # 80%をモデル用に
        estimated_batch_size = int(available_memory / est_model_size)

        # 範囲内に収める
        return max(MIN_BATCH_SIZE, min(MAX_BATCH_SIZE, estimated_batch_size))
    except Exception as e:
        logging.warning(f"バッチサイズの自動決定に失敗しました: {e}")
        return BATCH_SIZE

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
    # ソースシーケンスの切り詰め
    if X_batch.size(1) > max_seq_length:
        X_batch = X_batch[:, :max_seq_length]

    # ターゲットシーケンスの切り詰め
    if y_batch.size(1) > max_seq_length:
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
        from config import HIDDEN_SIZE, NUM_HEADS, NUM_LAYERS
        return HIDDEN_SIZE, NUM_HEADS, NUM_LAYERS

def apply_data_augmentation(train_token_ids, sp_src, sp_tgt, augmentation_factor=DATA_AUGMENTATION_FACTOR):
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
            techniques=DATA_AUGMENTATION_TECHNIQUES
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

def save_checkpoint(model, optimizer, scheduler, epoch, val_loss, bleu_score, is_best=False):
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
    """
    checkpoint_dir = setup_checkpointing_directory()

    # 現在の日付を取得
    current_date = datetime.now().strftime("%Y%m%d")

    # チェックポイント情報を準備
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'val_loss': val_loss,
        'bleu_score': bleu_score,
        'date': current_date
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

def load_checkpoint(checkpoint_path, model, optimizer=None, scheduler=None):
    """
    チェックポイントからモデルを読み込みます

    Args:
        checkpoint_path: チェックポイントファイルのパス
        model: モデル
        optimizer: オプティマイザ（オプション）
        scheduler: スケジューラ（オプション）

    Returns:
        モデル、エポック、検証損失、BLEUスコア
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

        logging.info(f"チェックポイントを読み込みました (エポック {epoch}, 検証損失 {val_loss:.4f}, BLEU {bleu_score:.4f})")
        return model, epoch, val_loss, bleu_score

    except Exception as e:
        logging.error(f"チェックポイントの読み込みに失敗しました: {e}")
        traceback.print_exc()
        return model, 0, float('inf'), 0.0

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
    parser.add_argument('--augment-factor', type=float, default=DATA_AUGMENTATION_FACTOR,
                        help='データ拡張の割合')
    parser.add_argument('--epochs', type=int, default=NUM_EPOCHS, help='トレーニングのエポック数')
    parser.add_argument('--batch-size', type=int, default=BATCH_SIZE, help='バッチサイズ')
    parser.add_argument('--no-wandb', action='store_true', help='Weights & Biasesのログを無効にする')
    parser.add_argument('--enhanced', action='store_true', help='相対位置エンコーディングとGLUを使用した強化版モデルを使用する')
    parser.add_argument('--rel-pos-max-dist', type=int, default=REL_POS_MAX_DISTANCE,
                       help='相対位置エンコーディングの最大距離')
    parser.add_argument('--warmup-steps', type=int, default=WARMUP_STEPS,
                       help='ウォームアップステップ数')
    args = parser.parse_args()

    # Dockerコンテナ内で実行されているかを確認して対応する
    docker_detected = os.path.exists('/.dockerenv') or os.environ.get('DOCKER_CONTAINER') == 'true'
    if docker_detected:
        logging.info("Dockerコンテナ内での実行を検出しました")

    # 強化版モデルの使用フラグ（コマンドラインまたは設定ファイル）
    use_enhanced_model = args.enhanced or USE_ENHANCED_MODEL

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

        # パディングインデックスを取得
        src_pad_idx = input_vocab['<pad>']
        tgt_pad_idx = output_vocab['<pad>']

        # GPUメモリに基づいてモデルサイズを決定
        model_hidden_size, model_num_heads, model_num_layers = determine_model_size()
        logging.info(f"モデル設定: hidden_size={model_hidden_size}, heads={model_num_heads}, layers={model_num_layers}")

        # 使用するモデルタイプのログ
        if use_enhanced_model:
            logging.info(f"強化版モデルを使用します（相対位置エンコーディング、最大距離: {args.rel_pos_max_dist}、GLU使用: {USE_GLU}）")
        else:
            logging.info("標準のTransformerモデルを使用します")

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
                ff_dim=D_FF,
                src_pad_idx=src_pad_idx,
                tgt_pad_idx=tgt_pad_idx,
                dropout=DROPOUT_RATE,
                device=DEVICE,
                max_dist=args.rel_pos_max_dist
            )
        else:
            # 標準のモデルを作成
            from encoder import Encoder
            from decoder import Decoder

            encoder = Encoder(input_dim, model_hidden_size, model_num_heads, model_num_layers, D_FF, DROPOUT_RATE, DEVICE).to(DEVICE)
            decoder = Decoder(output_dim, model_hidden_size, model_num_heads, model_num_layers, D_FF, output_dim, DROPOUT_RATE, DEVICE).to(DEVICE)
            model = TranslationModel(encoder, decoder, src_pad_idx, tgt_pad_idx, DEVICE).to(DEVICE)

        # GPUメモリに基づいてバッチサイズを決定
        batch_size = determine_batch_size() if args.batch_size == BATCH_SIZE else args.batch_size
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
                    "learning_rate": LEARNING_RATE,
                    "epochs": args.epochs,
                    "dropout": DROPOUT_RATE,
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
            lr=LEARNING_RATE,
            weight_decay=WEIGHT_DECAY  # 重み減衰を追加して過学習を抑制
        )

        # 改良されたスケジューラを使用（線形減衰を選択）
        scheduler = WarmupScheduler(
            optimizer,
            model_hidden_size,
            args.warmup_steps,
            args.epochs * (len(train_token_ids) // args.batch_size + 1),
            min_lr=1e-6,
            initial_lr=LEARNING_RATE,
            decay_method='linear'  # 線形減衰を使用
        )

        # チェックポイントからの復元（オプション）
        start_epoch = 0
        best_val_loss = float('inf')
        best_bleu_score = 0.0

        if args.resume or args.checkpoint:
            checkpoint_path = args.checkpoint if args.checkpoint else find_latest_checkpoint()
            if checkpoint_path:
                model, start_epoch, best_val_loss, best_bleu_score = load_checkpoint(
                    checkpoint_path, model, optimizer, scheduler
                )
                start_epoch += 1  # 次のエポックから開始
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
            start_time = time.time()
            model.train()  # 訓練モードに設定

            # バッチ開始時間を初期化
            batch_start_time = time.time()
            total_epoch_loss = 0

            # 勾配累積のためのカウンター
            accumulation_count = 0

            for i, (X_batch, y_batch) in enumerate(train_loader):
                try:
                    X_batch = X_batch.to(DEVICE)
                    y_batch = y_batch.to(DEVICE)

                    # 長すぎるシーケンスを切り詰め
                    X_batch, y_batch = truncate_long_sequences(X_batch, y_batch, MAX_SEQ_LENGTH)

                    # デコーダーへの入力を作成 (Teacher Forcing)
                    # まずターゲット出力サイズを決定
                    max_len = y_batch.size(1) - 1

                    # デコーダー入力とターゲット出力を同じサイズに保つ
                    decoder_input = y_batch[:, :max_len]
                    start_token_tensor = torch.full(
                        (y_batch.size(0), 1),
                        output_vocab['<s>'],
                        dtype=torch.long,
                        device=DEVICE
                    )
                    decoder_input = torch.cat((start_token_tensor, decoder_input[:, :-1]), dim=1)

                    # 損失計算用のターゲットを作成
                    target_output = y_batch[:, 1:max_len+1]

                    # 最初のバッチのみでグラデーションをゼロ初期化
                    if accumulation_count == 0:
                        optimizer.zero_grad()

                    # 混合精度で順伝播（CUDA利用可能時のみ）
                    if scaler:
                        with autocast():
                            # 順伝播 (デコーダーには decoder_input を渡す)
                            decoder_output, _ = model(X_batch, decoder_input)

                            # 損失計算 (decoder_output と target_output で計算)
                            loss = criterion(
                                decoder_output.view(-1, output_dim),
                                target_output.reshape(-1)
                            )

                            # バッチサイズで正規化して勾配累積を行う
                            loss = loss / ACCUMULATED_BATCHES

                        # スケーラーで逆伝播
                        scaler.scale(loss).backward()
                    else:
                        # CPUでの通常のトレーニング
                        decoder_output, _ = model(X_batch, decoder_input)
                        loss = criterion(
                            decoder_output.view(-1, output_dim),
                            target_output.reshape(-1)
                        )
                        loss = loss / ACCUMULATED_BATCHES
                        loss.backward()

                    # バッチ毎の損失を記録
                    batch_loss = loss.item() * ACCUMULATED_BATCHES
                    total_epoch_loss += batch_loss

                    # 勾配累積カウンターを更新
                    accumulation_count += 1

                    # 指定のバッチ数たまったら勾配を適用
                    if accumulation_count == ACCUMULATED_BATCHES or i == len(train_loader) - 1:
                        # 勾配クリッピング
                        if scaler:
                            scaler.unscale_(optimizer)

                        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=GRAD_CLIP_NORM)

                        # 最適化ステップ
                        if scaler:
                            scaler.step(optimizer)
                            scaler.update()
                        else:
                            optimizer.step()

                        scheduler.step()

                        # カウンターリセット
                        accumulation_count = 0

                        # より詳細なメトリクスをログに記録
                        if not args.no_wandb:
                            wandb.log({
                                "batch_loss": batch_loss,
                                "learning_rate": optimizer.param_groups[0]["lr"],
                                "grad_norm": grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm,
                                "batch_time": time.time() - batch_start_time  # バッチ処理時間を追加
                            })

                    # 進捗状況を出力
                    if i % 10 == 0 or i == len(train_loader) - 1:  # 10バッチごとに出力
                        logging.info(f"Epoch [{epoch+1}/{args.epochs}] Step [{i+1}/{total_steps}], "
                                    f"Loss: {batch_loss:.4f}, LR: {optimizer.param_groups[0]['lr']:.6f}")

                    # 次のバッチの開始時間を記録
                    batch_start_time = time.time()

                except Exception as e:
                    logging.error(f"バッチ処理中にエラーが発生しました: {e}")
                    traceback.print_exc()
                    # トレーニングを続行

            # エポック終了時の平均損失
            avg_epoch_loss = total_epoch_loss / len(train_loader)
            if not args.no_wandb:
                wandb.log({"epoch_loss": avg_epoch_loss})

            # 検証部分
            model.eval()  # 評価モードに設定
            try:
                val_loss, bleu_result = validate(model, val_loader, criterion, DEVICE, output_dim, output_vocab)

                # 返り値がmetricsの辞書型かbleu_scoreの値かをチェック
                if isinstance(bleu_result, dict):
                    bleu_score = bleu_result.get("bleu", 0.0)
                else:
                    bleu_score = bleu_result

                logging.info(f"Validation Loss: {val_loss:.4f}, BLEU Score: {bleu_score:.4f}")
                if not args.no_wandb:
                    wandb.log({"val_loss": val_loss, "bleu_score": bleu_score})
            except Exception as e:
                logging.error(f"検証中にエラーが発生しました: {e}")
                val_loss = float('inf')
                bleu_score = 0.0

            # チェックポイントの保存
            save_checkpoint(
                model, optimizer, scheduler, epoch, val_loss, bleu_score,
                is_best=(val_loss < best_val_loss or bleu_score > best_bleu_score)
            )

            # Early Stoppingのチェック (BLEUスコアも考慮)
            improved = False
            if val_loss < best_val_loss * (1.0 - LOSS_IMPROVEMENT_THRESHOLD):  # 2%以上の改善を要求
                best_val_loss = val_loss
                improved = True
                logging.info(f"検証損失が改善しました: {val_loss:.4f}")

            if bleu_score > best_bleu_score + BLEU_IMPROVEMENT_THRESHOLD:  # 閾値以上の改善
                best_bleu_score = bleu_score
                improved = True
                logging.info(f"BLEUスコアが改善しました: {bleu_score:.4f}")

            if improved:
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= PATIENCE:
                    logging.info("Early stopping triggered")
                    break

            # エポックごとの要約を出力
            epoch_time = time.time() - start_time
            logging.info(f"Epoch {epoch+1} took {epoch_time:.2f} seconds.")

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
