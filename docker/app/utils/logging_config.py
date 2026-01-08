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
            # 完全にフォーマットされたメッセージを取得
            formatted_message = super().format(record)
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

    # coloramaが利用可能で、カラー出力が有効な場合はColoredFormatterを使用
    should_use_colored = use_colored_output and COLORAMA_AVAILABLE
    if should_use_colored:
        colorama.init()
        desired_formatter = ColoredFormatter(format_string)
    else:
        desired_formatter = logging.Formatter(format_string)

    # 既存のstdoutハンドラーを検索（ColoredFormatterを持つもののみ）
    has_console_handler = False
    for handler in root_logger.handlers:
        # StreamHandlerでsys.stdoutを使用し、ColoredFormatterを持つハンドラーを検出
        if isinstance(handler, logging.StreamHandler):
            if handler.stream is sys.stdout and isinstance(handler.formatter, ColoredFormatter):
                has_console_handler = True
                # 既存のハンドラーのフォーマット文字列を確認
                current_fmt = getattr(handler.formatter, '_fmt', None)
                desired_fmt = getattr(desired_formatter, '_fmt', None)
                if current_fmt == desired_fmt:
                    # 既に適切なフォーマッターが設定されている場合はスキップ
                    return
                break

    # 既存のColoredFormatterを持つハンドラーが存在しない場合のみ、新しいハンドラーを追加
    if not has_console_handler:
        console = logging.StreamHandler(sys.stdout)
        console.setLevel(level)
        console.setFormatter(desired_formatter)
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
