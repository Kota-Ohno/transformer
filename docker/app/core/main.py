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
import traceback
import torch
import copy
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

# カスタムフォーマッターでログレベルに応じた色分けを実装
class ColoredFormatter(logging.Formatter):
    LEVEL_COLORS = {
        logging.DEBUG: Colors.INFO,
        logging.INFO: Colors.INFO,
        logging.WARNING: Colors.WARNING,
        logging.ERROR: Colors.ERROR,
        logging.CRITICAL: Colors.ERROR,
    }

    def format(self, record):
        color = self.LEVEL_COLORS.get(record.levelno, Colors.RESET)
        record_copy = copy.copy(record)
        # 完全にフォーマットされたメッセージを取得
        formatted_message = record.getMessage()
        # 色付きメッセージを設定
        record_copy.msg = f"{color}{formatted_message}{Colors.RESET}"
        # さらなるフォーマットを避けるためにargsをクリア
        record_copy.args = ()
        # レベル名も色付け
        record_copy.levelname = f"{color}{record.levelname}{Colors.RESET}"
        return super().format(record_copy)

def setup_logging():
    """ロギングの設定（root loggerを設定）"""

    # root loggerを取得（logging.getLogger()またはlogging.getLogger(None)）
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)

    # 既存のコンソールハンドラーをチェック（同じタイプとストリームのハンドラーが存在するか）
    formatter = ColoredFormatter('%(asctime)s - %(levelname)s - %(message)s')
    has_console_handler = False

    for handler in root_logger.handlers[:]:  # コピーを作成してイテレート
        # StreamHandlerでsys.stdoutを使用しているハンドラーを検出
        if isinstance(handler, logging.StreamHandler):
            if handler.stream is sys.stdout:
                # 同じフォーマッタータイプかチェック（ColoredFormatterかどうか）
                if isinstance(handler.formatter, ColoredFormatter):
                    has_console_handler = True
                    break

    # 既存のコンソールハンドラーがない場合のみ追加
    if not has_console_handler:
        console = logging.StreamHandler(sys.stdout)
        console.setLevel(logging.INFO)
        console.setFormatter(formatter)
        root_logger.addHandler(console)

def check_gpu_environment():
    """GPU環境の情報を収集してログに記録"""
    logging.info("環境チェック中...")

    if torch.cuda.is_available():
        try:
            device_count = torch.cuda.device_count()
            logging.info(f"GPU検出: {device_count}台のGPUが利用可能")

            for i in range(device_count):
                try:
                    device_props = torch.cuda.get_device_properties(i)
                    total_memory_gb = device_props.total_memory / (1024 ** 3)
                    logging.info(f"GPU {i}: {device_props.name}, メモリ: {total_memory_gb:.2f} GB")
                except (RuntimeError, AssertionError) as e:
                    logging.warning(f"GPU {i} の情報取得に失敗しました: {e}")

            # PyTorchバージョンチェック
            cuda_version = torch.version.cuda
            cuda_version_str = cuda_version if cuda_version is not None else "N/A (PyTorch built without CUDA)"
            logging.info(f"PyTorchバージョン: {torch.__version__}, CUDA: {cuda_version_str}")
        except (RuntimeError, AssertionError) as e:
            logging.warning(f"CUDA環境の確認中にエラーが発生しました: {e}。CPUで実行します。")
    else:
        logging.warning("利用可能なGPUがありません - CPUで実行します")

def show_config_summary():
    """設定の概要を表示"""
    try:
        # 安全にCONFIG属性にアクセス（デフォルト値付き）
        model_hp = getattr(CONFIG, 'model_hyperparameters', None)
        training_cfg = getattr(CONFIG, 'training_config', None)

        # モデルハイパーパラメータの取得（デフォルト値付き）
        hidden_size = getattr(model_hp, 'hidden_size', None) if model_hp else None
        num_heads = getattr(model_hp, 'num_heads', None) if model_hp else None
        num_layers = getattr(model_hp, 'num_layers', None) if model_hp else None
        dropout_rate = getattr(model_hp, 'dropout_rate', None) if model_hp else None

        # トレーニング設定の取得（デフォルト値付き）
        batch_size = getattr(training_cfg, 'batch_size', None) if training_cfg else None
        num_epochs = getattr(training_cfg, 'num_epochs', None) if training_cfg else None
        learning_rate = getattr(training_cfg, 'learning_rate', None) if training_cfg else None

        # 欠けている属性をチェック
        missing_attrs = []
        if model_hp is None:
            missing_attrs.append('CONFIG.model_hyperparameters')
        if training_cfg is None:
            missing_attrs.append('CONFIG.training_config')
        if hidden_size is None:
            missing_attrs.append('CONFIG.model_hyperparameters.hidden_size')
        if num_heads is None:
            missing_attrs.append('CONFIG.model_hyperparameters.num_heads')
        if num_layers is None:
            missing_attrs.append('CONFIG.model_hyperparameters.num_layers')
        if dropout_rate is None:
            missing_attrs.append('CONFIG.model_hyperparameters.dropout_rate')
        if batch_size is None:
            missing_attrs.append('CONFIG.training_config.batch_size')
        if num_epochs is None:
            missing_attrs.append('CONFIG.training_config.num_epochs')
        if learning_rate is None:
            missing_attrs.append('CONFIG.training_config.learning_rate')

        # 欠けている属性がある場合はエラーログを出力してプロセスを停止
        if missing_attrs:
            logging.error(f"CONFIG構造が不完全です。欠けている属性: {', '.join(missing_attrs)}")
            print(f"\n{Colors.ERROR}警告: 設定情報の一部が取得できませんでした。{Colors.RESET}")
            print(f"{Colors.ERROR}欠けている属性: {', '.join(missing_attrs)}{Colors.RESET}\n")
            raise RuntimeError(f"CONFIG構造が不完全です。欠けている属性: {', '.join(missing_attrs)}")

        # 設定の概要を表示
        print("\n" + "="*50)
        print(f"{Colors.INFO}モデル設定の概要:{Colors.RESET}")
        print(f"  • モデルサイズ: hidden_dim={hidden_size}, heads={num_heads}, layers={num_layers}")
        print(f"  • トレーニング: batch_size={batch_size}, epochs={num_epochs}")
        print(f"  • 最適化: learning_rate={learning_rate}, dropout={dropout_rate}")
        print("="*50 + "\n")

    except AttributeError as e:
        # 予期しないAttributeErrorをキャッチ
        error_msg = f"CONFIG属性へのアクセス中にエラーが発生しました: {e}"
        logging.error(error_msg)
        print(f"\n{Colors.ERROR}警告: 設定情報の取得中にエラーが発生しました: {e}{Colors.RESET}\n")
        raise RuntimeError(error_msg) from e

def non_negative_int(value: str) -> int:
    """非負の整数を検証する関数。

    Args:
        value: 文字列形式の整数値

    Returns:
        非負の整数値

    Raises:
        argparse.ArgumentTypeError: 値が整数でない場合、または負の整数の場合
    """
    try:
        int_value = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"--limit-samples は整数である必要があります。指定された値: {value}")
    if int_value < 0:
        raise argparse.ArgumentTypeError(f"--limit-samples は0以上の整数である必要があります。指定された値: {int_value}")
    return int_value


def main() -> int:
    """メイン関数"""
    # コマンドライン引数の解析
    parser = argparse.ArgumentParser(description='Transformerモデルのトレーニングを実行')
    parser.add_argument('--no-nltk-download', action='store_true',
                        help='NLTKリソースのダウンロードをスキップします')
    parser.add_argument('--fast', action='store_true',
                        help='高速モード: トレーニング時間を短縮するための最適化設定を適用します')
    parser.add_argument('--small-model', action='store_true',
                        help='小さいモデルを使用: メモリ使用量を削減し、トレーニング速度を向上させます')
    parser.add_argument('--limit-samples', type=non_negative_int, default=0,
                        help='トレーニングに使用するサンプル数を制限します（開発用、0以上）')
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
        os.environ['TRANSFORMER_FAST_MODE'] = '1'
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
        os.environ['TRANSFORMER_LIMIT_SAMPLES'] = str(args.limit_samples)
        print(f"{Colors.WARNING}トレーニングデータを{args.limit_samples}サンプルに制限します{Colors.RESET}")

    # 環境変数設定後にCONFIGを再読み込み
    CONFIG.reload_from_env()

    # ロギングの設定
    setup_logging()

    # 環境チェック
    check_gpu_environment()

    # 設定の概要表示
    show_config_summary()

    # トレーニングを開始
    logging.info("トレーニングを開始します...")

    # 他の引数を付けてトレーニングメイン関数を呼び出す
    # unknown_argsが空の場合はNoneを渡す（train_mainはOptional[List[str]]を受け取る）
    result = train_main(unknown_args if unknown_args else None)
    if isinstance(result, int):
        return result
    return 0

if __name__ == "__main__":
    try:
        rc = main()
        sys.exit(rc)
    except KeyboardInterrupt:
        logging.info("ユーザーによって中断されました")
        sys.exit(130)  # SIGINT の標準的な終了コード
    except Exception as e:
        logging.error(f"エラーが発生しました: {e}")
        logging.error(traceback.format_exc())
        sys.exit(1)
