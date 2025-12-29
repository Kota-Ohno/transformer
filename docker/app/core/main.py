#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Transformer翻訳モデルトレーニングのメインスクリプト
最適化設定を自動的に適用してトレーニングを開始します。
"""

import os
import sys
import argparse
import logging
import torch
from datetime import datetime
import colorama

# カラー出力の初期化
colorama.init()

# プロジェクトのルートディレクトリをPythonパスに追加
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ローカルモジュールのインポート
from core.train import main as train_main
from utils.config import CONFIG

# カラー定義
class Colors:
    INFO = colorama.Fore.CYAN
    SUCCESS = colorama.Fore.GREEN
    WARNING = colorama.Fore.YELLOW
    ERROR = colorama.Fore.RED
    RESET = colorama.Fore.RESET

def setup_logging():
    """ロギングの設定"""
    log_format = f'{Colors.INFO}%(asctime)s - %(levelname)s - %(message)s{Colors.RESET}'
    logging.basicConfig(level=logging.INFO, format=log_format)

    # 標準出力へのハンドラーを追加し、カラーフォーマットを適用
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(logging.INFO)
    formatter = logging.Formatter(log_format)
    console.setFormatter(formatter)

    # 既存のハンドラーをクリアして新しいハンドラーを追加
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    root_logger.addHandler(console)

def check_gpu_environment():
    """GPU環境の情報を収集してログに記録"""
    logging.info(f"{Colors.INFO}環境チェック中...{Colors.RESET}")

    if torch.cuda.is_available():
        device_count = torch.cuda.device_count()
        logging.info(f"{Colors.SUCCESS}GPU検出: {device_count}台のGPUが利用可能{Colors.RESET}")

        for i in range(device_count):
            device_props = torch.cuda.get_device_properties(i)
            total_memory_gb = device_props.total_memory / (1024 ** 3)
            logging.info(f"{Colors.INFO}GPU {i}: {device_props.name}, メモリ: {total_memory_gb:.2f} GB{Colors.RESET}")

        # PyTorchバージョンチェック
        cuda_version = torch.version.cuda
        logging.info(f"{Colors.INFO}PyTorchバージョン: {torch.__version__}, CUDA: {cuda_version}{Colors.RESET}")
    else:
        logging.warning(f"{Colors.WARNING}利用可能なGPUがありません - CPUで実行します{Colors.RESET}")

def show_config_summary():
    """設定の概要を表示"""
    print("\n" + "="*50)
    print(f"{Colors.INFO}モデル設定の概要:{Colors.RESET}")
    print(f"  • モデルサイズ: hidden_dim={CONFIG.model_hyperparameters.hidden_size}, heads={CONFIG.model_hyperparameters.num_heads}, layers={CONFIG.model_hyperparameters.num_layers}")
    print(f"  • トレーニング: batch_size={CONFIG.training_config.batch_size}, epochs={CONFIG.training_config.num_epochs}")
    print(f"  • 最適化: learning_rate={CONFIG.training_config.learning_rate}, dropout={CONFIG.model_hyperparameters.dropout_rate}")
    print("="*50 + "\n")

def main():
    """メイン関数"""
    # コマンドライン引数の解析
    parser = argparse.ArgumentParser(description='Transformerモデルのトレーニングを実行')
    parser.add_argument('--no-nltk-download', action='store_true',
                        help='NLTKリソースのダウンロードをスキップします')
    parser.add_argument('--fast', action='store_true',
                        help='高速モード: トレーニング時間を短縮するための最適化設定を適用します')
    parser.add_argument('--small-model', action='store_true',
                        help='小さいモデルを使用: メモリ使用量を削減し、トレーニング速度を向上させます')
    parser.add_argument('--limit-samples', type=int, default=0,
                        help='トレーニングに使用するサンプル数を制限します（開発用）')
    args, unknown_args = parser.parse_known_args()

    # 環境変数を設定
    if args.no_nltk_download:
        os.environ['SKIP_NLTK_DOWNLOAD'] = '1'
        print("NLTKリソースのダウンロードをスキップします")

    # 高速モードの設定
    if args.fast:
        os.environ['TRANSFORMER_TRAINING_BATCH_SIZE'] = '16'
        os.environ['TRANSFORMER_TRAINING_NUM_EPOCHS'] = '3'
        os.environ['TRANSFORMER_TRAINING_PATIENCE'] = '1'
        if '--fast' not in unknown_args:
            unknown_args.append('--fast')
        print(f"{Colors.SUCCESS}高速モードが有効です: 少ないエポック数でトレーニングを高速化します{Colors.RESET}")

    # 小さいモデルの設定
    if args.small_model:
        os.environ['TRANSFORMER_MODEL_HIDDEN_SIZE'] = '256'
        os.environ['TRANSFORMER_MODEL_NUM_HEADS'] = '4'
        os.environ['TRANSFORMER_MODEL_NUM_LAYERS'] = '4'
        os.environ['TRANSFORMER_MODEL_D_FF'] = '1024'
        print(f"{Colors.SUCCESS}小さいモデルを使用: hidden_size=256, heads=4, layers=4{Colors.RESET}")

    # データサンプル数制限の設定
    if args.limit_samples > 0:
        if '--limit-samples' not in unknown_args:
            unknown_args.extend(['--limit-samples', str(args.limit_samples)])
        print(f"{Colors.WARNING}トレーニングデータを{args.limit_samples}サンプルに制限します{Colors.RESET}")

    # ロギングの設定
    setup_logging()

    # 環境チェック
    check_gpu_environment()

    # 設定の概要表示
    show_config_summary()

    # トレーニングを開始
    logging.info(f"{Colors.SUCCESS}トレーニングを開始します...{Colors.RESET}")

    # 他の引数を付けてトレーニングメイン関数を呼び出す
    sys.argv = [sys.argv[0]] + unknown_args
    train_main()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logging.info(f"{Colors.WARNING}ユーザーによって中断されました{Colors.RESET}")
    except Exception as e:
        logging.error(f"{Colors.ERROR}エラーが発生しました: {e}{Colors.RESET}")
        import traceback
        logging.error(traceback.format_exc())
        sys.exit(1)
