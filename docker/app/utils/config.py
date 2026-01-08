import torch
import os
import json
from dataclasses import dataclass, field, is_dataclass
from typing import List, Any, get_origin, get_args, Tuple
import logging

class ModelConfig:
    """
    GPUメモリに基づいてモデル設定を自動調整するためのユーティリティクラス。

    このクラスは、利用可能なGPUメモリに応じて最適なモデルサイズを決定します。
    GlobalConfigの初期化時に使用され、ModelHyperparametersのデフォルト値を設定します。

    注意: このクラスは直接インスタンス化せず、from_gpu_memory()クラスメソッドを使用してください。
    """
    def __init__(self, hidden_size: int, num_heads: int, num_layers: int, d_ff: int, dropout: float, max_seq_length: int, rel_pos_max_distance: int):
        """
        Args:
            hidden_size: 隠れ層の次元数
            num_heads: アテンションヘッド数
            num_layers: エンコーダー/デコーダーのレイヤー数
            d_ff: フィードフォワード層の次元数
            dropout: ドロップアウト率
            max_seq_length: 最大シーケンス長
            rel_pos_max_distance: 相対位置エンコーディングの最大距離
        """
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.d_ff = d_ff
        self.dropout = dropout
        self.max_seq_length = max_seq_length
        self.rel_pos_max_distance = rel_pos_max_distance

    @classmethod
    def from_gpu_memory(cls):
        """
        GPUメモリに基づいて最適なモデル設定を生成します。

        Returns:
            ModelConfig: GPUメモリに応じたモデル設定インスタンス

        メモリ別の設定:
            - 16GB以上: hidden=768, heads=12, layers=10
            - 8GB以上16GB未満: hidden=512, heads=8, layers=4
            - 4GB以上8GB未満: hidden=384, heads=6, layers=4
            - 4GB未満: hidden=256, heads=4, layers=3
            - GPUなし: hidden=512, heads=8, layers=6 (デフォルト)
        """
        if not torch.cuda.is_available():
            # GPUがない場合はデフォルト設定
            return cls(hidden_size=512, num_heads=8, num_layers=6, d_ff=2048, dropout=0.1, max_seq_length=512, rel_pos_max_distance=128)

        try:
            total_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3) # GB
        except (RuntimeError, AssertionError) as e:
            # GPU プロパティへのアクセスに失敗した場合はデフォルト設定にフォールバック
            print(f"Warning: Failed to access GPU properties: {e}. Using default configuration.")
            return cls(hidden_size=512, num_heads=8, num_layers=6, d_ff=2048, dropout=0.1, max_seq_length=512, rel_pos_max_distance=128)

        if total_memory >= 16:
            # 16GB以上: より大きなモデル (hidden=768, heads=12, layers=10)
            return cls(hidden_size=768, num_heads=12, num_layers=10, d_ff=3072, dropout=0.1, max_seq_length=1024, rel_pos_max_distance=256)
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
    """
    Transformerモデルのハイパーパラメータ設定。

    GlobalConfigの初期化時に、GPUメモリに基づいて自動的に調整されます。
    環境変数やconfig.jsonファイルでオーバーライド可能です。
    """
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
    """
    アプリケーション全体の設定を管理するグローバル設定クラス。

    設定の優先順位（高い順）:
    1. 環境変数
    2. config.jsonファイル
    3. GPUメモリに基づく自動調整
    4. デフォルト値
    """
    model_hyperparameters: ModelHyperparameters = field(default_factory=ModelHyperparameters)
    training_config: TrainingConfig = field(default_factory=TrainingConfig)
    data_config: DataConfig = field(default_factory=DataConfig)
    device: str = field(default_factory=lambda: 'cuda' if torch.cuda.is_available() else 'cpu')
    verbose_mask_logs: bool = False

    def __post_init__(self):
        """
        設定の初期化処理。

        処理順序:
        1. GPUメモリに基づいてモデルのハイパーパラメータを自動調整
        2. config.jsonファイルから設定を読み込み（存在する場合）
        3. 環境変数から設定を読み込み（最高優先度）
        """
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

        # TRANSFORMER_DEVICE環境変数もサポート（後方互換性のため）
        device_env = os.getenv("TRANSFORMER_DEVICE")
        if device_env is not None:
            self.device = device_env.lower()

        # デバイスの検証とフォールバック処理
        self._validate_and_fallback_device()

    def _get_expected_type(self, obj: Any, key: str):
        """
        属性の期待される型を取得します。
        まず型アノテーションから取得を試み、失敗した場合は現在の値の型を使用します。

        Args:
            obj: 対象オブジェクト
            key: 属性名

        Returns:
            期待される型、またはNone（型が特定できない場合）
        """
        # dataclassフィールドの型アノテーションから取得
        if is_dataclass(obj) and hasattr(obj, '__dataclass_fields__'):
            if key in obj.__dataclass_fields__:
                return obj.__dataclass_fields__[key].type

        # 現在の値の型から推測
        current_value = getattr(obj, key, None)
        if current_value is not None:
            return type(current_value)

        return None

    def _coerce_value(self, value: Any, expected_type: Any, key: str, section: str) -> Tuple[Any, bool]:
        """
        JSON値を期待される型に安全に変換します。

        Args:
            value: 変換する値
            expected_type: 期待される型
            key: 属性名（ログ用）
            section: セクション名（ログ用）

        Returns:
            (変換された値, 成功フラグ)のタプル
        """
        # Noneの場合はスキップ
        if value is None:
            return None, False

        # 型アノテーションが複雑な型（List, Dict等）の場合を先にチェック
        origin = get_origin(expected_type)
        if origin is None:
            # ジェネリクスでない場合のみ、isinstanceチェックを実行
            try:
                if isinstance(value, expected_type):
                    return value, True
            except TypeError:
                # isinstanceがジェネリクス型に対してTypeErrorを発生させた場合
                # 後続の処理に進む（このケースは通常発生しないが、念のため）
                pass

        # 型アノテーションが複雑な型（List, Dict等）の場合
        if origin is not None:
            if origin is list:
                # List型の場合
                if isinstance(value, list):
                    # 要素の型を取得
                    args = get_args(expected_type)
                    if args:
                        element_type = args[0]
                        try:
                            coerced_list = [self._coerce_value(item, element_type, key, section)[0]
                                          for item in value]
                            return coerced_list, True
                        except (ValueError, TypeError):
                            logging.warning(
                                f"Type mismatch in config.json: {section}.{key} expects List[{element_type.__name__}], "
                                f"but got {type(value).__name__}. Skipping assignment."
                            )
                            return None, False
                    return value, True
                else:
                    logging.warning(
                        f"Type mismatch in config.json: {section}.{key} expects list, "
                        f"but got {type(value).__name__}. Skipping assignment."
                    )
                    return None, False
            # 他の複雑な型（Dict等）は将来の拡張用
            return value, isinstance(value, origin)

        # 基本型への変換を試みる
        try:
            if expected_type is int:
                # 文字列の数字からintへの変換（符号付き整数をサポート）
                if isinstance(value, str):
                    try:
                        return int(value.strip()), True
                    except ValueError:
                        raise ValueError(f"Cannot convert string '{value}' to int")
                elif isinstance(value, (int, float)):
                    return int(value), True
                else:
                    raise ValueError(f"Cannot convert {type(value).__name__} to int")

            elif expected_type is float:
                # 文字列の数字からfloatへの変換
                if isinstance(value, str):
                    return float(value), True
                elif isinstance(value, (int, float)):
                    return float(value), True
                else:
                    raise ValueError(f"Cannot convert {type(value).__name__} to float")

            elif expected_type is bool:
                # 文字列からboolへの変換
                if isinstance(value, str):
                    return value.lower() in ("true", "yes", "1", "on"), True
                elif isinstance(value, bool):
                    return value, True
                elif isinstance(value, (int, float)):
                    return bool(value), True
                else:
                    raise ValueError(f"Cannot convert {type(value).__name__} to bool")

            elif expected_type is str:
                # 任意の型から文字列への変換
                return str(value), True

            else:
                # その他の型は型チェックのみ
                if isinstance(value, expected_type):
                    return value, True
                else:
                    raise ValueError(f"Cannot convert {type(value).__name__} to {expected_type.__name__}")

        except (ValueError, TypeError) as e:
            logging.warning(
                f"Type mismatch in config.json: {section}.{key} expects {expected_type.__name__}, "
                f"but got {type(value).__name__} (value: {value}). Error: {e}. Skipping assignment."
            )
            return None, False

    def _override_from_env(self, obj: Any, prefix: str):
        for field_name in obj.__dataclass_fields__:
            # フィールドの現在の値を取得
            current_value = getattr(obj, field_name, None)
            # dataclassインスタンスの場合はスキップ（ネストされたdataclassを保護）
            if is_dataclass(current_value):
                continue

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
                        # 空の環境変数値は空のリストとして扱う
                        if not env_value.strip():
                            setattr(obj, field_name, [])
                        else:
                            # 要素型を取得
                            args = get_args(original_type)
                            if args:
                                element_type = args[0]
                                try:
                                    # 各要素を要素型に変換
                                    converted_items = []
                                    for item in env_value.split(","):
                                        stripped_item = item.strip()
                                        if element_type is int:
                                            converted_items.append(int(stripped_item))
                                        elif element_type is float:
                                            converted_items.append(float(stripped_item))
                                        elif element_type is bool:
                                            converted_items.append(stripped_item.lower() in ("true", "yes", "1"))
                                        elif element_type is str:
                                            converted_items.append(stripped_item)
                                        else:
                                            # サポートされていない型の場合は文字列として扱う
                                            converted_items.append(stripped_item)
                                    setattr(obj, field_name, converted_items)
                                except (ValueError, TypeError):
                                    # 変換に失敗した場合は文字列の動作にフォールバック
                                    setattr(obj, field_name, [item.strip() for item in env_value.split(",")])
                            else:
                                # 型引数がない場合は文字列の動作にフォールバック
                                setattr(obj, field_name, [item.strip() for item in env_value.split(",")])
                    else:
                        setattr(obj, field_name, env_value)
                except ValueError:
                    print(f"Warning: Could not convert environment variable {env_var_name}='{env_value}' to type {original_type}.")

    def _load_from_json(self, config_path: str):
        """
        JSONファイルから設定を読み込み、型バリデーションと変換を行います。

        各属性について:
        1. 期待される型を型アノテーションまたは現在の値から決定
        2. JSON値が期待される型と一致するか、安全に変換可能かチェック
        3. バリデーション/変換が成功した場合のみsetattrを呼び出す
        4. 不一致の場合は警告をログに出力してスキップ
        """
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
                                        # 期待される型を取得
                                        expected_type = self._get_expected_type(target_obj, key)

                                        if expected_type is None:
                                            # 型が特定できない場合は警告を出してスキップ
                                            logging.warning(
                                                f"Could not determine expected type for {section}.{key} in config.json. "
                                                f"Skipping assignment."
                                            )
                                            continue

                                        # 型変換とバリデーション
                                        coerced_value, success = self._coerce_value(value, expected_type, key, section)

                                        if success:
                                            setattr(target_obj, key, coerced_value)
                                        # 失敗時は既に警告がログに出力されている
            except Exception as e:
                logging.warning(f"Failed to load config from {config_path}: {e}")

    def _validate_and_fallback_device(self):
        """
        デバイス設定を検証し、CUDAが利用できない場合はCPUにフォールバックします。

        環境変数で'cuda'が指定されていても、実際にCUDAが利用できない場合は
        警告を出して'cpu'にフォールバックします。
        """
        if self.device.lower() == 'cuda':
            if not torch.cuda.is_available():
                logging.warning(
                    "環境変数でCUDAが指定されましたが、CUDAが利用できません。"
                    "CPUにフォールバックします。"
                )
                self.device = 'cpu'
            else:
                # CUDAが利用可能な場合でも、実際にデバイスにアクセスできるか確認
                try:
                    # デバイス0にアクセスして確認
                    _ = torch.cuda.get_device_properties(0)
                    logging.info(f"CUDAデバイスが利用可能です: {torch.cuda.get_device_name(0)}")
                except (RuntimeError, AssertionError) as e:
                    logging.warning(
                        f"CUDAデバイスへのアクセスに失敗しました: {e}。"
                        "CPUにフォールバックします。"
                    )
                    self.device = 'cpu'
        elif self.device.lower() not in ('cpu', 'cuda'):
            logging.warning(
                f"無効なデバイス設定 '{self.device}' が指定されました。"
                "有効な値は 'cpu' または 'cuda' です。CPUにフォールバックします。"
            )
            self.device = 'cpu'

    def get_device(self) -> torch.device:
        """
        デバイス設定をtorch.deviceオブジェクトとして取得します。

        Returns:
            torch.device: 使用するデバイス
        """
        return torch.device(self.device)

# グローバル設定インスタンス
CONFIG = GlobalConfig()


INPUT_VOCAB_PATH = CONFIG.data_config.input_vocab_path
OUTPUT_VOCAB_PATH = CONFIG.data_config.output_vocab_path
