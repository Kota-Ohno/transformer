"""
アプリケーション全体で使用する定数定義
"""
from typing import Final

# VRAM閾値（MB）
VRAM_THRESHOLD_4GB: Final[int] = 4096
VRAM_THRESHOLD_8GB: Final[int] = 8192
VRAM_THRESHOLD_16GB: Final[int] = 16384

# バッチサイズ関連
DEFAULT_BATCH_SIZE_SMALL_VRAM: Final[int] = 8  # 4GB未満のVRAM用
DEFAULT_BATCH_SIZE_MEDIUM_VRAM: Final[int] = 16  # 8GB未満のVRAM用

# チャンク推論関連
DEFAULT_INITIAL_CHUNK_SIZE: Final[int] = 512
DEFAULT_MIN_CHUNK_SIZE: Final[int] = 32
DEFAULT_MAX_RETRIES: Final[int] = 5

# データ拡張確率（デフォルト値）
DEFAULT_MASK_PROB: Final[float] = 0.15
DEFAULT_DELETION_PROB: Final[float] = 0.1
DEFAULT_REPLACEMENT_PROB: Final[float] = 0.1
DEFAULT_PERMUTATION_PROB: Final[float] = 0.1
DEFAULT_WINDOW_SIZE: Final[int] = 3

# ログ出力頻度
LOG_INTERVAL_BATCHES: Final[int] = 100  # 100バッチごとにログ出力

# 評価関連
DEFAULT_MAX_SAMPLES_PER_BATCH: Final[int] = 5  # BLEU計算用のバッチあたり最大サンプル数
DEFAULT_MIN_SEQUENCE_LENGTH: Final[int] = 3  # データ拡張で削除しない最小シーケンス長

# チェックポイント保存頻度
CHECKPOINT_FREQUENCY_SHORT: Final[int] = 1  # 5エポック以下の場合
CHECKPOINT_FREQUENCY_MEDIUM: Final[int] = 2  # 20エポック以下の場合
CHECKPOINT_FREQUENCY_LONG: Final[int] = 5  # 20エポック超の場合

# エポック関連
FAST_MODE_MAX_EPOCHS: Final[int] = 5
FAST_MODE_DEFAULT_SAMPLES: Final[int] = 10000
FAST_MODE_MAX_EVAL_BATCHES: Final[int] = 50
FAST_MODE_BLEU_SAMPLE_BATCHES: Final[int] = 1
FAST_MODE_MIN_GRAD_ACCUM_STEPS: Final[int] = 4

# トレーニング関連
MAX_EPOCH_RETRIES: Final[int] = 3  # エポックあたりの最大リトライ回数
MAX_DATA_AUGMENTATION_ERRORS: Final[int] = 3  # データ拡張で許容する最大エラー数

# トークンID関連
DEFAULT_START_TOKEN_ID: Final[int] = 2  # <s>
DEFAULT_END_TOKEN_ID: Final[int] = 3  # </s>
DEFAULT_PAD_TOKEN_ID: Final[int] = 0  # <pad>
DEFAULT_UNK_TOKEN_ID: Final[int] = 1  # <unk>

# 特殊トークンセット
SPECIAL_TOKENS: Final[frozenset[str]] = frozenset({'<pad>', '<unk>', '<s>', '</s>', '<bos>', '<eos>'})

# データローダー関連
DEFAULT_PREFETCH_FACTOR: Final[int] = 2
DEFAULT_NUM_WORKERS: Final[int] = 4
MAX_NUM_WORKERS_CUDA: Final[int] = 2  # CUDA使用時の最大ワーカー数

# BLEUスコア計算関連
DEFAULT_BLEU_SAMPLE_BATCHES: Final[int] = 10
DEFAULT_MAX_BLEU_SAMPLES: Final[int] = 50
BLEU_SCORE_NORMALIZATION_FACTOR: Final[float] = 100.0  # SacreBLEUスコアを0-1に正規化

# メモリ関連
BYTES_PER_MB: Final[int] = 1024 * 1024
BYTES_PER_GB: Final[int] = 1024 * 1024 * 1024
