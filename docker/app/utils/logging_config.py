"""
ロギング設定の統一管理
"""
import logging
import sys
import copy
from typing import Optional


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

    # 既存のコンソールハンドラーをチェック
    has_console_handler = False
    if use_colored_output:
        try:
            import colorama
            colorama.init()

            # カスタムフォーマッターでログレベルに応じた色分けを実装
            class ColoredFormatter(logging.Formatter):
                LEVEL_COLORS = {
                    logging.DEBUG: colorama.Fore.CYAN,
                    logging.INFO: colorama.Fore.CYAN,
                    logging.WARNING: colorama.Fore.YELLOW,
                    logging.ERROR: colorama.Fore.RED,
                    logging.CRITICAL: colorama.Fore.RED,
                }

                def format(self, record):
                    color = self.LEVEL_COLORS.get(record.levelno, colorama.Fore.RESET)
                    record_copy = copy.copy(record)
                    record_copy.levelname = f"{color}{record.levelname}{colorama.Fore.RESET}"
                    record_copy.msg = f"{color}{record.msg}{colorama.Fore.RESET}"
                    return super().format(record_copy)

            formatter = ColoredFormatter(format_string)

            for handler in root_logger.handlers[:]:
                if isinstance(handler, logging.StreamHandler):
                    if handler.stream is sys.stdout:
                        if isinstance(handler.formatter, ColoredFormatter):
                            has_console_handler = True
                            break
                        root_logger.removeHandler(handler)

            if not has_console_handler:
                console = logging.StreamHandler(sys.stdout)
                console.setLevel(level)
                console.setFormatter(formatter)
                root_logger.addHandler(console)
        except ImportError:
            # coloramaがインストールされていない場合は通常のフォーマッターを使用
            use_colored_output = False

    if not use_colored_output:
        formatter = logging.Formatter(format_string)

        # 既存のハンドラーをチェック
        for handler in root_logger.handlers[:]:
            if isinstance(handler, logging.StreamHandler):
                if handler.stream is sys.stdout:
                    has_console_handler = True
                    break

        if not has_console_handler:
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
