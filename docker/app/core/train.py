import torch
import torch.nn as nn
import torch.optim as optim
import os
import argparse
import logging
import sys
from typing import Optional, List, Tuple, Dict, Any
from packaging import version
from data.data import create_data_loader, set_data, collate_fn
from utils.config import CONFIG, INPUT_VOCAB_PATH, OUTPUT_VOCAB_PATH
from utils.scheduler import WarmupScheduler
from data.text_tokenizer import load_tokenized_data
from data.data_augmentation import augment_dataset
from utils.trainer import Trainer
from utils.utils import download_nltk_resources
from utils.constants import (
    VRAM_THRESHOLD_4GB, VRAM_THRESHOLD_8GB,
    DEFAULT_BATCH_SIZE_SMALL_VRAM, DEFAULT_BATCH_SIZE_MEDIUM_VRAM,
    FAST_MODE_MAX_EPOCHS, FAST_MODE_DEFAULT_SAMPLES,
    FAST_MODE_MAX_EVAL_BATCHES, FAST_MODE_BLEU_SAMPLE_BATCHES,
    FAST_MODE_MIN_GRAD_ACCUM_STEPS,
    DEFAULT_PREFETCH_FACTOR, DEFAULT_NUM_WORKERS, MAX_NUM_WORKERS_CUDA,
    BYTES_PER_MB, BYTES_PER_GB
)
from utils.logging_config import setup_logging

# Set up logging
setup_logging()


def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    """
    コマンドライン引数を解析します。

    Args:
        argv: コマンドライン引数のリスト（Noneの場合はsys.argvを使用）

    Returns:
        解析された引数オブジェクト
    """
    parser = argparse.ArgumentParser(description='Transformerモデルのトレーニング')
    parser.add_argument('--resume', action='store_true', help='最新のチェックポイントから再開')
    parser.add_argument('--checkpoint', type=str, help='指定したチェックポイントから再開')
    parser.add_argument('--augment', action='store_true', help='データ拡張を適用')
    parser.add_argument('--augment-factor', type=float, default=CONFIG.data_config.data_augmentation_factor,
                      help='データ拡張の割合 (0.0〜1.0)')
    parser.add_argument('--epochs', type=int, default=CONFIG.training_config.num_epochs, help='エポック数')
    parser.add_argument('--batch-size', type=int, default=CONFIG.training_config.batch_size, help='バッチサイズ')
    parser.add_argument('--learning-rate', type=float, default=CONFIG.training_config.learning_rate, help='学習率')
    parser.add_argument('--patience', type=int, default=CONFIG.training_config.patience, help='早期停止のペイシェンス')
    parser.add_argument('--warmup-steps', type=int, default=CONFIG.training_config.warmup_steps, help='Warmupステップ数')
    parser.add_argument('--fast', action='store_true', help='高速モード (少ないエポック数での実験)')
    parser.add_argument('--no-wandb', action='store_true', help='Weights & Biasesを無効化')
    parser.add_argument('--grad-accum-steps', type=int, default=CONFIG.training_config.gradient_accumulation_steps,
                      help='勾配蓄積ステップ数')
    parser.add_argument('--verbose-mask', action='store_true', help='マスクの詳細ログを表示')
    parser.add_argument('--limit-samples', type=int, default=0, help='使用するサンプル数を制限')
    parser.add_argument('--jit', action='store_true', help='JITコンパイルを使用')
    parser.add_argument('--num-workers', type=int, default=DEFAULT_NUM_WORKERS,
                       help='データロードに使用するワーカー数')
    parser.add_argument('--no-nltk-download', action='store_true',
                       help='NLTKリソースのダウンロードをスキップする')
    return parser.parse_args(args=argv)


def _check_and_setup_gpu(args: argparse.Namespace) -> torch.device:
    """
    GPU環境をチェックし、必要に応じて高速モードを自動有効化します。

    Args:
        args: コマンドライン引数オブジェクト

    Returns:
        使用するデバイス
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f"Using device: {device}")

    # GPUメモリチェック（fast-mode設定の前に実行）
    if device.type == 'cuda':
        gpu_props = torch.cuda.get_device_properties(0)
        # メモリ制約がある場合は高速トレーニングモードを自動的に有効化
        if gpu_props.total_memory < VRAM_THRESHOLD_8GB * BYTES_PER_MB:
            logging.info("GPUメモリが限られているため、高速トレーニングモードを自動的に有効化します")
            args.fast = True

    return device


def _apply_fast_mode_settings(args: argparse.Namespace) -> None:
    """
    高速トレーニングモードの設定を適用します。

    Args:
        args: コマンドライン引数オブジェクト
    """
    if args.fast:
        logging.info("高速トレーニングモードが有効です - 精度よりも速度を優先します")
        # データサンプル数の制限
        if args.limit_samples == 0:
            args.limit_samples = FAST_MODE_DEFAULT_SAMPLES
        # エポック数の制限
        if args.epochs > FAST_MODE_MAX_EPOCHS:
            args.epochs = FAST_MODE_MAX_EPOCHS
        # 評価頻度の削減
        CONFIG.training_config.max_eval_batches = FAST_MODE_MAX_EVAL_BATCHES
        # 勾配蓄積ステップ数の増加
        args.grad_accum_steps = max(args.grad_accum_steps, FAST_MODE_MIN_GRAD_ACCUM_STEPS)
        # BLEUスコア計算用サンプル数の削減
        CONFIG.training_config.bleu_sample_batches = FAST_MODE_BLEU_SAMPLE_BATCHES


def _load_and_prepare_data(
    args: argparse.Namespace
) -> Tuple[List, List, Dict[str, int], Dict[str, int]]:
    """
    データを読み込み、前処理を実行します。

    Args:
        args: コマンドライン引数オブジェクト

    Returns:
        (train_token_ids, val_token_ids, input_vocab, output_vocab)のタプル

    Raises:
        SystemExit: データファイルが見つからない場合
        RuntimeError: SentencePieceモデルファイルが見つからない場合
    """
    # NLTK リソースのダウンロード
    download_nltk_resources()

    # トークナイズ済みデータのパス
    train_data_path = "tokenized_train_data.pth"
    val_data_path = "tokenized_val_data.pth"

    # データの読み込み
    train_token_ids, val_token_ids = load_tokenized_data(train_data_path), load_tokenized_data(val_data_path)
    if train_token_ids is None or val_token_ids is None:
        logging.error("トークナイズ済みデータファイルが見つかりません。先にtext_tokenizer.pyを実行してください。")
        sys.exit(1)

    # データサンプル数の制限（高速実験用）
    if args.limit_samples > 0 and len(train_token_ids) > args.limit_samples:
        logging.info(f"トレーニングサンプル数を制限します: {len(train_token_ids)} → {args.limit_samples}")
        import random
        random.shuffle(train_token_ids)
        train_token_ids = train_token_ids[:args.limit_samples]

    # ボキャブラリの読み込み
    input_vocab = torch.load(INPUT_VOCAB_PATH, weights_only=True)
    output_vocab = torch.load(OUTPUT_VOCAB_PATH, weights_only=True)

    # データ拡張（オプション）
    if args.augment:
        # SentencePieceモデルをロードして拡張に使用
        sp_src_path = os.path.join("models", "sp_src.pth")
        sp_tgt_path = os.path.join("models", "sp_tgt.pth")

        # ファイルの存在チェック
        if not os.path.exists(sp_src_path):
            error_msg = (
                f"エラー: SentencePieceモデルファイルが見つかりません: {sp_src_path}\n"
                f"--augmentオプションを使用するには、事前に構築されたSentencePieceモデルファイルが必要です。"
            )
            logging.error(error_msg)
            raise RuntimeError(error_msg)

        if not os.path.exists(sp_tgt_path):
            error_msg = (
                f"エラー: SentencePieceモデルファイルが見つかりません: {sp_tgt_path}\n"
                f"--augmentオプションを使用するには、事前に構築されたSentencePieceモデルファイルが必要です。"
            )
            logging.error(error_msg)
            raise RuntimeError(error_msg)

        sp_src = torch.load(sp_src_path, weights_only=True)
        sp_tgt = torch.load(sp_tgt_path, weights_only=True)

        # データ拡張を適用
        train_token_ids = augment_dataset(
            train_token_ids,
            sp_src,
            sp_tgt,
            augmentation_factor=args.augment_factor
        )

    return train_token_ids, val_token_ids, input_vocab, output_vocab


def _create_data_loaders(
    train_token_ids: List,
    val_token_ids: List,
    args: argparse.Namespace,
    device: torch.device
) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader, int]:
    """
    データローダーを作成します。

    Args:
        train_token_ids: トレーニングデータのトークンIDリスト
        val_token_ids: 検証データのトークンIDリスト
        args: コマンドライン引数オブジェクト
        device: 使用するデバイス

    Returns:
        (train_loader, val_loader, adjusted_batch_size)のタプル
    """
    # GPU情報のログ記録とメモリキャッシュのクリア
    if device.type == 'cuda':
        gpu_props = torch.cuda.get_device_properties(0)
        logging.info(f"GPU: {gpu_props.name}, Memory: {gpu_props.total_memory / BYTES_PER_MB:.0f}MB")
        torch.cuda.empty_cache()

    # データセットとデータローダーの作成
    train_dataset = set_data(train_token_ids, train_token_ids)
    val_dataset = set_data(val_token_ids, val_token_ids)

    # バッチサイズの調整
    batch_size = args.batch_size
    adjusted_batch_size = batch_size

    # 実際のバッチサイズはGPUメモリによって調整可能
    if device.type == 'cuda':
        vram_mb = torch.cuda.get_device_properties(0).total_memory / BYTES_PER_MB
        if vram_mb < VRAM_THRESHOLD_4GB:
            adjusted_batch_size = min(batch_size, DEFAULT_BATCH_SIZE_SMALL_VRAM)
        elif vram_mb < VRAM_THRESHOLD_8GB:
            adjusted_batch_size = min(batch_size, DEFAULT_BATCH_SIZE_MEDIUM_VRAM)

    logging.info(f"バッチサイズ: {adjusted_batch_size} (元の設定: {batch_size})")

    # 勾配蓄積を使用する場合は実効バッチサイズを表示
    effective_batch_size = adjusted_batch_size * CONFIG.training_config.gradient_accumulation_steps
    if CONFIG.training_config.gradient_accumulation_steps > 1:
        logging.info(f"勾配蓄積ステップ数: {CONFIG.training_config.gradient_accumulation_steps}, 実効バッチサイズ: {effective_batch_size}")

    # 高速モードの場合はデータローダーのオプションを最適化
    if args.fast:
        num_workers = 0
        pin_memory = False
        persistent_workers = False
    else:
        num_workers = args.num_workers if not torch.cuda.is_available() else min(args.num_workers, MAX_NUM_WORKERS_CUDA)
        pin_memory = torch.cuda.is_available()
        persistent_workers = num_workers > 0

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=adjusted_batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=pin_memory,
        prefetch_factor=DEFAULT_PREFETCH_FACTOR if num_workers > 0 else None,
        persistent_workers=persistent_workers
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=adjusted_batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    return train_loader, val_loader, adjusted_batch_size


def _initialize_model(
    input_vocab: Dict[str, int],
    output_vocab: Dict[str, int],
    device: torch.device
) -> nn.Module:
    """
    モデルを初期化します。

    Args:
        input_vocab: 入力語彙
        output_vocab: 出力語彙
        device: 使用するデバイス

    Returns:
        初期化されたモデル
    """
    input_dim = len(input_vocab)
    output_dim = len(output_vocab)

    logging.info(f"入力ボキャブラリーサイズ: {input_dim}, 出力ボキャブラリーサイズ: {output_dim}")

    # パディングインデックスを取得
    src_pad_idx = input_vocab['<pad>']
    tgt_pad_idx = output_vocab['<pad>']

    # GPUメモリに基づいてモデルサイズを決定
    model_hidden_size = CONFIG.model_hyperparameters.hidden_size
    model_num_heads = CONFIG.model_hyperparameters.num_heads
    model_num_layers = CONFIG.model_hyperparameters.num_layers

    logging.info(f"モデルタイプ: 標準Transformer")
    logging.info(f"モデル設定: hidden_size={model_hidden_size}, heads={model_num_heads}, layers={model_num_layers}")

    # モデルの初期化
    from models.model import create_transformer_model
    model = create_transformer_model(
        input_vocab_size=input_dim,
        output_vocab_size=output_dim,
        src_pad_idx=src_pad_idx,
        tgt_pad_idx=tgt_pad_idx,
        hidden_size=model_hidden_size,
        num_heads=model_num_heads,
        num_layers=model_num_layers,
        d_ff=CONFIG.model_hyperparameters.d_ff,
        dropout=CONFIG.model_hyperparameters.dropout_rate,
        max_seq_length=CONFIG.model_hyperparameters.max_seq_length
    )

    # モデルをデバイスに移動（JITウォームアップ前に実行）
    model = model.to(device)
    # 注意: DataParallel/DistributedDataParallelを使用する場合は、デバイス移動後にラップする

    # JITコンパイルをオプションで適用
    if CONFIG.training_config.use_jit_compile and version.parse(torch.__version__) >= version.parse("2.0.0") and device.type == 'cuda':
        try:
            logging.info("PyTorch JITコンパイルを適用します")
            import torch._dynamo as dynamo
            torch._dynamo.config.suppress_errors = True
            torch._dynamo.config.cache_size_limit = 64

            if CONFIG.training_config.debug_mode:
                dynamo.config.debug = True
                dynamo.config.output_code = True

            optimized_model = torch.compile(
                model,
                backend="inductor",
                mode="max-autotune",
                fullgraph=False
            )

            try:
                logging.info("JITコンパイルのウォームアップ実行...")
                dummy_batch_size = 2
                dummy_seq_len = 16
                dummy_src = torch.randint(0, int(input_dim), (dummy_batch_size, dummy_seq_len), device=device)
                dummy_tgt = torch.randint(0, int(output_dim), (dummy_batch_size, dummy_seq_len), device=device)

                with torch.no_grad():
                    optimized_model(dummy_src, dummy_tgt[:, :-1])
                    torch.cuda.synchronize()

                model = optimized_model
                logging.info("JITコンパイルのウォームアップ完了")
            except Exception as warmup_err:
                logging.warning(f"JITコンパイルのウォームアップに失敗しました: {warmup_err}。通常のモデルを使用します。")
        except Exception as e:
            logging.warning(f"JITコンパイルの適用に失敗しました: {e}")

    return model


def _setup_training_components(
    model: nn.Module,
    args: argparse.Namespace,
    train_loader: torch.utils.data.DataLoader
) -> Tuple[optim.Optimizer, nn.Module, WarmupScheduler, torch.cuda.amp.GradScaler]:
    """
    トレーニングに必要なコンポーネントを設定します。

    Args:
        model: モデル
        args: コマンドライン引数オブジェクト
        train_loader: トレーニングデータローダー

    Returns:
        (optimizer, criterion, scheduler, scaler)のタプル
    """
    # パディングインデックスを取得（モデルから取得）
    tgt_pad_idx = model.tgt_pad_idx if hasattr(model, 'tgt_pad_idx') else 0

    # 損失関数とオプティマイザの設定
    criterion = nn.CrossEntropyLoss(ignore_index=tgt_pad_idx)
    optimizer = optim.AdamW(
        model.parameters(),
        lr=CONFIG.training_config.learning_rate,
        weight_decay=CONFIG.training_config.weight_decay if hasattr(CONFIG.training_config, 'weight_decay') else 0.01,
        eps=1e-8
    )

    # 勾配スケーラーの設定
    scaler = torch.cuda.amp.GradScaler()

    # 総ステップ数を計算
    total_steps = len(train_loader) * args.epochs // CONFIG.training_config.gradient_accumulation_steps

    # 学習率スケジューラの設定
    model_hidden_size = CONFIG.model_hyperparameters.hidden_size
    scheduler = WarmupScheduler(
        optimizer,
        d_model=model_hidden_size,
        warmup_steps=args.warmup_steps,
        total_steps=total_steps,
        min_lr=1e-6
    )

    return optimizer, criterion, scheduler, scaler


def main(argv: Optional[List[str]] = None) -> None:
    """
    翻訳モデルのトレーニングを実行する主要な関数。

    この関数は以下の手順を実行します：
    1. 引数の解析
    2. GPU環境のチェックと設定
    3. データの読み込みと前処理
    4. データローダーの作成
    5. モデルの初期化
    6. トレーニングコンポーネントの設定
    7. トレーニングループの実行

    Raises:
        ValueError: データの読み込みに失敗した場合
        RuntimeError: SentencePieceモデルファイルが見つからない場合
    """
    # 引数の解析
    args = _parse_args(argv)

    # GPU環境のチェックと設定
    device = _check_and_setup_gpu(args)

    # 高速モード設定の適用
    _apply_fast_mode_settings(args)

    # 設定の更新
    if args.no_nltk_download:
        os.environ['SKIP_NLTK_DOWNLOAD'] = '1'
        logging.info("NLTKリソースのダウンロードをスキップします")

    docker_detected = os.path.exists('/.dockerenv') or os.environ.get('DOCKER_CONTAINER') == 'true'
    if docker_detected:
        logging.info("Dockerコンテナ内での実行を検出しました")

    CONFIG.verbose_mask_logs = args.verbose_mask
    CONFIG.training_config.gradient_accumulation_steps = args.grad_accum_steps
    CONFIG.training_config.use_jit_compile = args.jit

    # データの読み込みと前処理
    train_token_ids, val_token_ids, input_vocab, output_vocab = _load_and_prepare_data(args)

    # データローダーの作成
    train_loader, val_loader, adjusted_batch_size = _create_data_loaders(
        train_token_ids, val_token_ids, args, device
    )

    # モデルの初期化
    model = _initialize_model(input_vocab, output_vocab, device)

    # トレーニングコンポーネントの設定
    optimizer, criterion, scheduler, scaler = _setup_training_components(model, args, train_loader)

    # Trainerクラスのインスタンスを作成し、トレーニングを開始
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        criterion=criterion,
        scheduler=scheduler,
        scaler=scaler,
        device=device,
        args=args,
        input_vocab=input_vocab,
        output_vocab=output_vocab
    )
    trainer.train_model()


if __name__ == "__main__":
    main()
