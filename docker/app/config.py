import torch
import os
import json

# デフォルト設定
DEFAULT_CONFIG = {
    "MAX_SEQ_LENGTH": 512,
    "HIDDEN_SIZE": 512,
    "NUM_HEADS": 8,
    "NUM_LAYERS": 6,
    "D_FF": 2048,
    "DROPOUT_RATE": 0.1,
    "NUM_EPOCHS": 10000,
    "PATIENCE": 2,
    "WARMUP_STEPS": 4000,
    "BATCH_SIZE": 16,
    "LEARNING_RATE": 0.001,
    "GRAD_CLIP_NORM": 5.0,
    "MAX_BATCH_SIZE": 32,
    "MIN_BATCH_SIZE": 4,
    "ACCUMULATED_BATCHES": 4,
    "TRANSLATION_SOURCE": "ja_JP",
    "TRANSLATION_DESTINATION": "en_US",
    "TRANSLATION_SOURCE2": "japanese",
    "TRANSLATION_DESTINATION2": "english",
    "TOKENIZE_BATCH_SIZE": 128,
    "LAYER_DROPOUT": 0.1,  # 過学習を防ぐための層ドロップアウト率
    "WEIGHT_DECAY": 0.0001,  # 正則化係数

    # データ拡張関連の設定
    "DATA_AUGMENTATION_FACTOR": 0.3,  # 元のデータセットに対する拡張データの割合
    "DATA_AUGMENTATION_TECHNIQUES": ["masking", "deletion", "replacement", "permutation"],  # 使用する拡張テクニック
    "TOKEN_MASK_PROB": 0.15,  # トークンをマスクする確率
    "TOKEN_DELETE_PROB": 0.1,  # トークンを削除する確率
    "TOKEN_REPLACE_PROB": 0.1,  # トークンを置換する確率
    "TOKEN_PERMUTE_PROB": 0.1,  # トークンの順序を入れ替える確率
    "BACK_TRANSLATION_BATCH_SIZE": 32,  # 逆翻訳のバッチサイズ

    # エラーハンドリング関連設定
    "MAX_RETRY_COUNT": 3,  # 処理失敗時の最大再試行回数
    "ERROR_SKIP_THRESHOLD": 0.1,  # エラー発生時にスキップせずに中断する閾値（データの何%でエラーが出たら中断するか）

    # モデルアーキテクチャ関連設定
    "USE_ENHANCED_MODEL": False,  # 強化版モデルを使用するかどうか
    "REL_POS_MAX_DISTANCE": 64,  # 相対位置エンコーディングの最大距離
    "USE_GLU": True,  # Gated Linear Unitを使用するかどうか
}

# モデル設定用クラス
class ModelConfig:
    """モデルアーキテクチャに関連する設定を管理するクラス"""

    def __init__(self, hidden_size=512, num_heads=8, num_layers=6, d_ff=2048,
                 dropout=0.1, max_seq_length=512, layer_dropout=0.1,
                 use_enhanced=False, rel_pos_max_distance=64, use_glu=True):
        """
        モデル設定を初期化します。

        Args:
            hidden_size (int): 隠れ層の次元数
            num_heads (int): アテンションヘッドの数
            num_layers (int): エンコーダー/デコーダーレイヤーの数
            d_ff (int): フィードフォワード層の次元数
            dropout (float): ドロップアウト率
            max_seq_length (int): 最大シーケンス長
            layer_dropout (float): レイヤードロップアウト率
            use_enhanced (bool): 強化版モデルを使用するかどうか
            rel_pos_max_distance (int): 相対位置エンコーディングの最大距離
            use_glu (bool): Gated Linear Unitを使用するかどうか
        """
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.d_ff = d_ff
        self.dropout = dropout
        self.max_seq_length = max_seq_length
        self.layer_dropout = layer_dropout
        self.use_enhanced = use_enhanced
        self.rel_pos_max_distance = rel_pos_max_distance
        self.use_glu = use_glu

    @classmethod
    def from_config(cls, config_dict):
        """設定辞書からインスタンスを作成します"""
        return cls(
            hidden_size=config_dict.get("HIDDEN_SIZE", 512),
            num_heads=config_dict.get("NUM_HEADS", 8),
            num_layers=config_dict.get("NUM_LAYERS", 6),
            d_ff=config_dict.get("D_FF", 2048),
            dropout=config_dict.get("DROPOUT_RATE", 0.1),
            max_seq_length=config_dict.get("MAX_SEQ_LENGTH", 512),
            layer_dropout=config_dict.get("LAYER_DROPOUT", 0.1),
            use_enhanced=config_dict.get("USE_ENHANCED_MODEL", False),
            rel_pos_max_distance=config_dict.get("REL_POS_MAX_DISTANCE", 64),
            use_glu=config_dict.get("USE_GLU", True)
        )

    @classmethod
    def from_gpu_memory(cls, gpu_memory_mb=None):
        """
        GPUメモリサイズに基づいて適切な設定を返します。

        Args:
            gpu_memory_mb (float, optional): GPUメモリサイズ（MB）。
                                         Noneの場合は自動検出を試みます。

        Returns:
            ModelConfig: GPUメモリに適した設定
        """
        if gpu_memory_mb is None:
            if torch.cuda.is_available():
                try:
                    gpu_props = torch.cuda.get_device_properties(0)
                    gpu_memory_mb = gpu_props.total_memory / 1024**2
                except Exception:
                    # GPU情報の取得に失敗した場合はデフォルト設定を返す
                    return cls()
            else:
                # GPUが利用できない場合はデフォルト設定を返す
                return cls()

        # メモリサイズに基づいて設定を調整
        if gpu_memory_mb > 16000:  # 16GB以上
            return cls(hidden_size=512, num_heads=8, num_layers=6)
        elif gpu_memory_mb > 8000:  # 8GB以上
            return cls(hidden_size=512, num_heads=8, num_layers=4)
        elif gpu_memory_mb > 4000:  # 4GB以上
            return cls(hidden_size=384, num_heads=6, num_layers=4, d_ff=1536)
        else:  # 4GB未満
            return cls(hidden_size=256, num_heads=4, num_layers=3, d_ff=1024)

    def to_dict(self):
        """設定を辞書として返します"""
        return {
            "HIDDEN_SIZE": self.hidden_size,
            "NUM_HEADS": self.num_heads,
            "NUM_LAYERS": self.num_layers,
            "D_FF": self.d_ff,
            "DROPOUT_RATE": self.dropout,
            "MAX_SEQ_LENGTH": self.max_seq_length,
            "LAYER_DROPOUT": self.layer_dropout,
            "USE_ENHANCED_MODEL": self.use_enhanced,
            "REL_POS_MAX_DISTANCE": self.rel_pos_max_distance,
            "USE_GLU": self.use_glu
        }

# 設定ファイルから読み込み
def load_config(config_path="config.json"):
    config = DEFAULT_CONFIG.copy()

    # 設定ファイルが存在する場合は読み込む
    if os.path.exists(config_path):
        try:
            with open(config_path, "r") as f:
                user_config = json.load(f)
                # デフォルト設定をユーザー設定で上書き
                for key, value in user_config.items():
                    if key in config:
                        config[key] = value
        except Exception as e:
            print(f"設定ファイルの読み込みに失敗しました: {e}")

    # 環境変数での上書き
    for key in config:
        env_key = f"TRANSFORMER_{key}"
        env_value = os.getenv(env_key)
        if env_value is not None:
            # 型に応じて変換
            if isinstance(config[key], int):
                config[key] = int(env_value)
            elif isinstance(config[key], float):
                config[key] = float(env_value)
            elif isinstance(config[key], bool):
                config[key] = env_value.lower() in ("true", "yes", "1")
            elif isinstance(config[key], list):
                # カンマ区切りの文字列をリストに変換
                config[key] = [item.strip() for item in env_value.split(",")]
            else:
                config[key] = env_value

    return config

# 設定を読み込む
CONFIG = load_config()

# CUDA利用可能性
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# モデル設定オブジェクトを作成
MODEL_CONFIG = ModelConfig.from_config(CONFIG)

# ファイルパス
INPUT_VOCAB_PATH = "models/vocab_input.pth"
OUTPUT_VOCAB_PATH = "models/vocab_output.pth"
