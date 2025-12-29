import torch
import torch.nn as nn
import torch.optim as optim
import os
import argparse
import subprocess
import logging
import sys
from packaging import version
from data.data import create_data_loader, set_data, collate_fn
from utils.config import CONFIG, INPUT_VOCAB_PATH, OUTPUT_VOCAB_PATH
from utils.scheduler import WarmupScheduler
from data.text_tokenizer import load_tokenized_data
from data.data_augmentation import augment_dataset
from utils.trainer import Trainer
from utils.utils import download_nltk_resources

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')



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
    parser.add_argument('--num-workers', type=int, default=4,
                       help='データロードに使用するワーカー数')
    parser.add_argument('--no-nltk-download', action='store_true',
                       help='NLTKリソースのダウンロードをスキップする')
    args = parser.parse_args()

    # 高速トレーニングモードの設定を適用
    if args.fast:
        logging.info("高速トレーニングモードが有効です - 精度よりも速度を優先します")
        # データサンプル数の制限
        if args.limit_samples == 0:
            args.limit_samples = 10000  # デフォルトで1万サンプルに制限
        # エポック数の制限
        if args.epochs > 5:
            args.epochs = 5
        # 評価頻度の削減
        CONFIG.training_config.max_eval_batches = 50
        # 勾配蓄積ステップ数の増加
        args.grad_accum_steps = max(args.grad_accum_steps, 4)
        # BLEUスコア計算用サンプル数の削減
        CONFIG.training_config.bleu_sample_batches = 1

    # NLTKリソースダウンロードのスキップ設定
    if args.no_nltk_download:
        os.environ['SKIP_NLTK_DOWNLOAD'] = '1'
        logging.info("NLTKリソースのダウンロードをスキップします")

    # Dockerコンテナ内で実行されているかを確認して対応する
    docker_detected = os.path.exists('/.dockerenv') or os.environ.get('DOCKER_CONTAINER') == 'true'
    if docker_detected:
        logging.info("Dockerコンテナ内での実行を検出しました")

    # マスクログの設定を更新
    CONFIG.verbose_mask_logs = args.verbose_mask

    # 勾配蓄積ステップ数を更新
    CONFIG.training_config.gradient_accumulation_steps = args.grad_accum_steps

    # JITコンパイル設定を更新
    CONFIG.training_config.use_jit_compile = args.jit

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

        sp_src = torch.load(sp_src_path)
        sp_tgt = torch.load(sp_tgt_path)

        # データ拡張を適用
        train_token_ids = augment_dataset(
            train_token_ids,
            sp_src,
            sp_tgt,
            augmentation_factor=args.augment_factor
        )

    # デバイスの設定
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f"Using device: {device}")

    if device.type == 'cuda':
        # GPU情報をログに記録
        gpu_props = torch.cuda.get_device_properties(0)
        logging.info(f"GPU: {gpu_props.name}, Memory: {gpu_props.total_memory / 1024**2:.0f}MB")

        # CUDA確保メモリのキャッシュをクリア
        torch.cuda.empty_cache()

        # メモリ制約がある場合はバッチサイズを自動調整するためのフラグ
        if gpu_props.total_memory < 8 * 1024 * 1024 * 1024:  # 8GB未満
            logging.info("GPUメモリが限られているため、高速トレーニングモードを自動的に有効化します")
            args.fast = True

    # データセットとデータローダーの作成
    train_dataset = set_data(train_token_ids, train_token_ids)
    val_dataset = set_data(val_token_ids, val_token_ids)

    # バッチサイズの調整（オプション）
    batch_size = args.batch_size
    adjusted_batch_size = batch_size

    # 実際のバッチサイズはGPUメモリによって調整可能
    if device.type == 'cuda':
        vram_mb = torch.cuda.get_device_properties(0).total_memory / (1024 * 1024)
        if vram_mb < 4000:  # 4GB未満
            adjusted_batch_size = min(batch_size, 8)
        elif vram_mb < 8000:  # 8GB未満
            adjusted_batch_size = min(batch_size, 16)

    logging.info(f"バッチサイズ: {adjusted_batch_size} (元の設定: {batch_size})")

    # 勾配蓄積を使用する場合は実効バッチサイズを表示
    effective_batch_size = adjusted_batch_size * CONFIG.training_config.gradient_accumulation_steps
    if CONFIG.training_config.gradient_accumulation_steps > 1:
        logging.info(f"勾配蓄積ステップ数: {CONFIG.training_config.gradient_accumulation_steps}, 実効バッチサイズ: {effective_batch_size}")

    # 高速モードの場合はデータローダーのオプションを最適化
    if args.fast:
        # ワーカー数を削減し、メモリ使用を最適化
        num_workers = 0
        pin_memory = False
        persistent_workers = False
    else:
        # 通常モード
        num_workers = args.num_workers if not torch.cuda.is_available() else min(args.num_workers, 2)
        pin_memory = torch.cuda.is_available()
        persistent_workers = num_workers > 0

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=adjusted_batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=pin_memory,
        prefetch_factor=2 if num_workers > 0 else None,
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

    # 入力と出力の次元を設定
    input_dim = len(input_vocab)
    output_dim = len(output_vocab)

    # ボキャブラリーサイズをログに記録
    logging.info(f"入力ボキャブラリーサイズ: {input_dim}, 出力ボキャブラリーサイズ: {output_dim}")

    # パディングインデックスを取得
    src_pad_idx = input_vocab['<pad>']
    tgt_pad_idx = output_vocab['<pad>']

    # GPUメモリに基づいてモデルサイズを決定
    model_hidden_size = CONFIG.model_hyperparameters.hidden_size
    model_num_heads = CONFIG.model_hyperparameters.num_heads
    model_num_layers = CONFIG.model_hyperparameters.num_layers

    # モデルのタイプとサイズをログに記録
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

    # JITコンパイルをオプションで適用
    if CONFIG.training_config.use_jit_compile and version.parse(torch.__version__) >= version.parse("2.0.0") and device.type == 'cuda':
        try:
            logging.info("PyTorch JITコンパイルを適用します")
            import torch._dynamo as dynamo
            # エラー抑制と最適化レベル設定
            torch._dynamo.config.suppress_errors = True
            torch._dynamo.config.cache_size_limit = 64  # キャッシュサイズを増やす

            # デバッグモードの場合は生成されたコードを保存
            if CONFIG.training_config.debug_mode:
                dynamo.config.debug = True
                dynamo.config.output_code = True

            # Inductor バックエンドを使用（最速の選択肢）
            optimized_model = torch.compile(
                model,
                backend="inductor",
                mode="max-autotune",
                fullgraph=False  # 部分コンパイルを許可（失敗時に通常実行にフォールバック）
            )

            # 最適化モデルを試す小さなバッチを実行
            try:
                logging.info("JITコンパイルのウォームアップ実行...")
                dummy_batch_size = 2
                dummy_seq_len = 16
                dummy_src = torch.randint(0, input_dim-1, (dummy_batch_size, dummy_seq_len), device=device)
                dummy_tgt = torch.randint(0, output_dim-1, (dummy_batch_size, dummy_seq_len), device=device)

                with torch.no_grad():
                    # ウォームアップ実行
                    optimized_model(dummy_src, dummy_tgt[:, :-1])
                    torch.cuda.synchronize()

                # 成功したら置き換え
                model = optimized_model
                logging.info("JITコンパイルのウォームアップ完了")
            except Exception as warmup_err:
                logging.warning(f"JITコンパイルのウォームアップに失敗しました: {warmup_err}。通常のモデルを使用します。")
        except Exception as e:
            logging.warning(f"JITコンパイルの適用に失敗しました: {e}")

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
    scheduler = WarmupScheduler(
        optimizer,
        d_model=model_hidden_size,
        warmup_steps=args.warmup_steps,
        total_steps=total_steps,
        min_lr=1e-6
    )

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
