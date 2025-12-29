import torch
import os
import json
from dataclasses import dataclass, field
from typing import List, Any, get_origin

# GPUメモリに基づくモデル設定の自動調整
class ModelConfig:
    def __init__(self, hidden_size: int, num_heads: int, num_layers: int, d_ff: int, dropout: float, max_seq_length: int, rel_pos_max_distance: int):
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.d_ff = d_ff
        self.dropout = dropout
        self.max_seq_length = max_seq_length
        self.rel_pos_max_distance = rel_pos_max_distance

    @classmethod
    def from_gpu_memory(cls):
        if not torch.cuda.is_available():
            # GPUがない場合はデフォルト設定
            return cls(hidden_size=512, num_heads=8, num_layers=6, d_ff=2048, dropout=0.1, max_seq_length=512, rel_pos_max_distance=128)

        total_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3) # GB

        if total_memory >= 16:
            # 16GB以上: 6レイヤー (hidden=512, heads=8)
            return cls(hidden_size=512, num_heads=8, num_layers=6, d_ff=2048, dropout=0.1, max_seq_length=512, rel_pos_max_distance=128)
        elif total_memory >= 8:
            # 8GB以上16GB未満: 4レイヤー (hidden=512, heads=8)
            return cls(hidden_size=512, num_heads=8, num_layers=4, d_ff=2048, dropout=0.1, max_seq_length=512, rel_pos_max_distance=128)
        elif total_memory >= 4:
            # 4GB以上8GB未満: 4レイヤー (hidden=384, heads=6)
            return cls(hidden_size=384, num_heads=6, num_layers=4, d_ff=1536, dropout=0.1, max_seq_length=512, rel_pos_max_distance=128)
        else:
            # 4GB未満: 3レイヤー (hidden=256, heads=4)
            return cls(hidden_size=256, num_heads=4, num_layers=3, d_ff=1024, dropout=0.1, max_seq_length=512, rel_pos_max_distance=128)

@dataclass
class ModelHyperparameters:
    hidden_size: int = 512
    num_heads: int = 8
    num_layers: int = 6
    d_ff: int = 2048
    dropout_rate: float = 0.1
    max_seq_length: int = 512
    rel_pos_max_distance: int = 128

@dataclass
class TrainingConfig:
    num_epochs: int = 10
    patience: int = 2
    warmup_steps: int = 4000
    batch_size: int = 64
    learning_rate: float = 0.0001
    weight_decay: float = 0.01
    grad_clip_norm: float = 1.0
    gradient_accumulation_steps: int = 1
    min_batch_size: int = 16 # 自動バッチサイズ調整の最小値
    max_batch_size: int = 256 # 自動バッチサイズ調整の最大値
    use_jit_compile: bool = False
    debug_mode: bool = False
    bleu_sample_batches: int = 2 # BLEUスコア計算に使用するバッチ数
    max_eval_batches: int = 200 # 評価に使用する最大バッチ数

@dataclass
class DataConfig:
    translation_source: str = "ja_JP"
    translation_destination: str = "en_US"
    translation_source2: str = "japanese" # データセットのキー名
    translation_destination2: str = "english" # データセットのキー名
    tokenize_batch_size: int = 128
    data_augmentation_factor: float = 0.3
    data_augmentation_techniques: List[str] = field(default_factory=lambda: ["masking", "deletion", "replacement", "permutation"])
    input_vocab_path: str = "models/vocab_input.pth"
    output_vocab_path: str = "models/vocab_output.pth"

@dataclass
class GlobalConfig:
    model_hyperparameters: ModelHyperparameters = field(default_factory=ModelHyperparameters)
    training_config: TrainingConfig = field(default_factory=TrainingConfig)
    data_config: DataConfig = field(default_factory=DataConfig)
    device: str = field(default_factory=lambda: 'cuda' if torch.cuda.is_available() else 'cpu')
    verbose_mask_logs: bool = False

    def __post_init__(self):
        # GPUメモリに基づいてモデルのハイパーパラメータを調整
        adjusted_model_config = ModelConfig.from_gpu_memory()
        self.model_hyperparameters.hidden_size = adjusted_model_config.hidden_size
        self.model_hyperparameters.num_heads = adjusted_model_config.num_heads
        self.model_hyperparameters.num_layers = adjusted_model_config.num_layers
        self.model_hyperparameters.d_ff = adjusted_model_config.d_ff
        self.model_hyperparameters.dropout_rate = adjusted_model_config.dropout
        self.model_hyperparameters.max_seq_length = adjusted_model_config.max_seq_length
        self.model_hyperparameters.rel_pos_max_distance = adjusted_model_config.rel_pos_max_distance

        # config.jsonからのオーバーライド（最初に読み込む）
        self._load_from_json("config.json")

        # 環境変数からのオーバーライド（最高優先度、ネストされたフィールドも正しくオーバーライド）
        self._override_from_env(self.model_hyperparameters, "TRANSFORMER_MODEL_")
        self._override_from_env(self.training_config, "TRANSFORMER_TRAINING_")
        self._override_from_env(self.data_config, "TRANSFORMER_DATA_")
        self._override_from_env(self, "TRANSFORMER_GLOBAL_")

    def _override_from_env(self, obj: Any, prefix: str):
        for field_name in obj.__dataclass_fields__:
            env_var_name = f"{prefix}{field_name.upper()}"
            env_value = os.getenv(env_var_name)
            if env_value is not None:
                original_type = obj.__dataclass_fields__[field_name].type
                try:
                    if original_type is int:
                        setattr(obj, field_name, int(env_value))
                    elif original_type is float:
                        setattr(obj, field_name, float(env_value))
                    elif original_type is bool:
                        setattr(obj, field_name, env_value.lower() in ("true", "yes", "1"))
                    elif get_origin(original_type) is list:
                        setattr(obj, field_name, [item.strip() for item in env_value.split(",")])
                    else:
                        setattr(obj, field_name, env_value)
                except ValueError:
                    print(f"Warning: Could not convert environment variable {env_var_name}='{env_value}' to type {original_type}.")

    def _load_from_json(self, config_path: str):
        if os.path.exists(config_path):
            try:
                with open(config_path, "r") as f:
                    user_config = json.load(f)
                    for section, values in user_config.items():
                        if hasattr(self, section):
                            target_obj = getattr(self, section)
                            if isinstance(values, dict):
                                for key, value in values.items():
                                    if hasattr(target_obj, key):
                                        setattr(target_obj, key, value)
            except Exception as e:
                print(f"Warning: Failed to load config from {config_path}: {e}")

# グローバル設定インスタンス
CONFIG = GlobalConfig()


INPUT_VOCAB_PATH = CONFIG.data_config.input_vocab_path
OUTPUT_VOCAB_PATH = CONFIG.data_config.output_vocab_path
