"""
ロギング設定の統一管理
"""
import logging
import sys
import copy
from typing import Optional

# coloramaのインポートを試行（オプショナル）
try:
    import colorama
    COLORAMA_AVAILABLE = True
except ImportError:
    COLORAMA_AVAILABLE = False
    colorama = None


# カスタムフォーマッターでログレベルに応じた色分けを実装
class ColoredFormatter(logging.Formatter):
    """カラー出力対応のログフォーマッター"""
    LEVEL_COLORS = {
        logging.DEBUG: None,  # coloramaが利用可能な場合に設定される
        logging.INFO: None,
        logging.WARNING: None,
        logging.ERROR: None,
        logging.CRITICAL: None,
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # coloramaが利用可能な場合に色を設定
        if COLORAMA_AVAILABLE and colorama:
            self.LEVEL_COLORS = {
                logging.DEBUG: colorama.Fore.CYAN,
                logging.INFO: colorama.Fore.CYAN,
                logging.WARNING: colorama.Fore.YELLOW,
                logging.ERROR: colorama.Fore.RED,
                logging.CRITICAL: colorama.Fore.RED,
            }

    def format(self, record):
        if COLORAMA_AVAILABLE and colorama:
            color = self.LEVEL_COLORS.get(record.levelno, colorama.Fore.RESET)
            record_copy = copy.copy(record)
            # levelnameのみを色付きに設定（msgとargsは変更しない）
            record_copy.levelname = f"{color}{record.levelname}{colorama.Fore.RESET}"
            # 完全にフォーマットされたメッセージを取得
            formatted_message = super().format(record_copy)
            # フォーマット済みの文字列全体に色を適用
            return f"{color}{formatted_message}{colorama.Fore.RESET}"
        else:
            return super().format(record)


def setup_logging(
    level: int = logging.INFO,
    format_string: Optional[str] = None,
    use_colored_output: bool = True
) -> None:
    """
    アプリケーション全体で使用するロギング設定を行います。

    Args:
        level: ログレベル（デフォルト: logging.INFO）
        format_string: カスタムフォーマット文字列（Noneの場合はデフォルトを使用）
        use_colored_output: カラー出力を使用するかどうか（デフォルト: True）
    """
    if format_string is None:
        format_string = '%(asctime)s - %(levelname)s - %(message)s'

    # ルートロガーを取得
    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    # 既存のコンソールハンドラーをチェック・削除
    has_console_handler = False
    if use_colored_output:
        if COLORAMA_AVAILABLE:
            colorama.init()
            formatter = ColoredFormatter(format_string)

            # 既存のstdoutハンドラーを削除し、ColoredFormatterの検出を行う
            for handler in root_logger.handlers[:]:
                if isinstance(handler, logging.StreamHandler):
                    if handler.stream is sys.stdout:
                        if isinstance(handler.formatter, ColoredFormatter):
                            has_console_handler = True
                        root_logger.removeHandler(handler)

            # 新しいハンドラーを作成して追加
            console = logging.StreamHandler(sys.stdout)
            console.setLevel(level)
            console.setFormatter(formatter)
            root_logger.addHandler(console)
        else:
            # coloramaがインストールされていない場合は通常のフォーマッターを使用
            use_colored_output = False

    if not use_colored_output:
        formatter = logging.Formatter(format_string)

        # 既存のstdoutハンドラーを削除し、ColoredFormatterの検出を行う
        for handler in root_logger.handlers[:]:
            if isinstance(handler, logging.StreamHandler):
                if handler.stream is sys.stdout:
                    if isinstance(handler.formatter, ColoredFormatter):
                        has_console_handler = True
                    root_logger.removeHandler(handler)

        # 新しいハンドラーを作成して追加
        console = logging.StreamHandler(sys.stdout)
        console.setLevel(level)
        console.setFormatter(formatter)
        root_logger.addHandler(console)


def get_logger(name: str) -> logging.Logger:
    """
    指定された名前のロガーを取得します。

    Args:
        name: ロガー名（通常は__name__）

    Returns:
        logging.Logger: 設定済みのロガーインスタンス
    """
    return logging.getLogger(name)
