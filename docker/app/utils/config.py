import torch
import os
import json
import threading
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
    def __init__(self, hidden_size: int, num_heads: int, num_layers: int, d_ff: int, dropout_rate: float, max_seq_length: int, rel_pos_max_distance: int):
        """
        Args:
            hidden_size: 隠れ層の次元数
            num_heads: アテンションヘッド数
            num_layers: エンコーダー/デコーダーのレイヤー数
            d_ff: フィードフォワード層の次元数
            dropout_rate: ドロップアウト率
            max_seq_length: 最大シーケンス長
            rel_pos_max_distance: 相対位置エンコーディングの最大距離
        """
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.d_ff = d_ff
        self.dropout_rate = dropout_rate
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
            return cls(hidden_size=512, num_heads=8, num_layers=6, d_ff=2048, dropout_rate=0.1, max_seq_length=512, rel_pos_max_distance=128)

        try:
            # 動的にGPUデバイスインデックスを取得
            device_idx = torch.cuda.current_device()
            total_memory = torch.cuda.get_device_properties(device_idx).total_memory / (1024**3) # GB
        except (RuntimeError, AssertionError) as e:
            # GPU プロパティへのアクセスに失敗した場合はデフォルト設定にフォールバック
            logger = logging.getLogger(__name__)
            logger.warning(f"Failed to access GPU properties: {e}. Using default configuration.")
            return cls(hidden_size=512, num_heads=8, num_layers=6, d_ff=2048, dropout_rate=0.1, max_seq_length=512, rel_pos_max_distance=128)

        if total_memory >= 16:
            # 16GB以上: より大きなモデル (hidden=768, heads=12, layers=10)
            return cls(hidden_size=768, num_heads=12, num_layers=10, d_ff=3072, dropout_rate=0.1, max_seq_length=1024, rel_pos_max_distance=256)
        elif total_memory >= 8:
            # 8GB以上16GB未満: 4レイヤー (hidden=512, heads=8)
            return cls(hidden_size=512, num_heads=8, num_layers=4, d_ff=2048, dropout_rate=0.1, max_seq_length=512, rel_pos_max_distance=128)
        elif total_memory >= 4:
            # 4GB以上8GB未満: 4レイヤー (hidden=384, heads=6)
            return cls(hidden_size=384, num_heads=6, num_layers=4, d_ff=1536, dropout_rate=0.1, max_seq_length=512, rel_pos_max_distance=128)
        else:
            # 4GB未満: 3レイヤー (hidden=256, heads=4)
            return cls(hidden_size=256, num_heads=4, num_layers=3, d_ff=1024, dropout_rate=0.1, max_seq_length=512, rel_pos_max_distance=128)

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

    def __post_init__(self) -> None:
        """
        ハイパーパラメータのバリデーションを実行します。

        Transformerモデルの制約として、hidden_sizeはnum_headsで割り切れる必要があります。
        この制約を満たさない場合、モデル初期化時にエラーが発生するため、
        設定段階で検証を行います。

        Raises:
            ValueError: hidden_sizeがnum_headsで割り切れない場合。
                エラーメッセージにはhidden_sizeとnum_headsの値が含まれます。
        """
        if self.hidden_size % self.num_heads != 0:
            raise ValueError(
                f"Invalid ModelHyperparameters: hidden_size ({self.hidden_size}) must be divisible by "
                f"num_heads ({self.num_heads}). This constraint is required for Transformer model initialization."
            )

    @classmethod
    def from_model_config(cls, model_config: ModelConfig) -> "ModelHyperparameters":
        """ModelConfigからModelHyperparametersインスタンスを作成します。

        Args:
            model_config: ModelConfigインスタンス

        Returns:
            ModelHyperparameters: 変換されたModelHyperparametersインスタンス
        """
        return cls(
            hidden_size=model_config.hidden_size,
            num_heads=model_config.num_heads,
            num_layers=model_config.num_layers,
            d_ff=model_config.d_ff,
            dropout_rate=model_config.dropout_rate,
            max_seq_length=model_config.max_seq_length,
            rel_pos_max_distance=model_config.rel_pos_max_distance
        )

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

    def _apply_env_overrides(self):
        """
        環境変数から設定をオーバーライドし、デバイスを検証します。

        このメソッドは、環境変数からのオーバーライドとデバイス検証を
        一括で実行します。__post_init__とreload_from_envの両方で使用されます。
        """
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

    def __post_init__(self):
        """
        設定の初期化処理。

        処理順序（優先度の低い順）:
        1. GPUメモリに基づく自動調整（デフォルト値の設定）
        2. config.jsonファイルから設定を読み込み（存在する場合）
        3. 環境変数から設定を読み込み（最高優先度）
        """
        # GPUメモリに基づく設定調整を最初に実行（デフォルト値の設定）
        self.initialize_gpu_aware_defaults()

        # config.jsonからのオーバーライド
        self._load_from_json("config.json")

        # 環境変数からのオーバーライドとデバイス検証（最高優先度）
        self._apply_env_overrides()

    def initialize_gpu_aware_defaults(self):
        """
        GPUメモリに基づいてモデルのハイパーパラメータを自動調整します。

        このメソッドは、未設定の値のみを設定する（idempotent）ように動作します。
        __post_init__の最初で呼び出され、その後JSONと環境変数でオーバーライドされます。
        """
        # GPUメモリに基づいてモデルのハイパーパラメータを調整
        # 既に設定されている値は上書きしない（idempotent）
        adjusted_model_config = ModelConfig.from_gpu_memory()
        adjusted_hyperparams = ModelHyperparameters.from_model_config(adjusted_model_config)

        # 現在の値がデフォルト値の場合のみ更新（未設定の値のみ設定）
        if self.model_hyperparameters.hidden_size == ModelHyperparameters.hidden_size:
            self.model_hyperparameters.hidden_size = adjusted_hyperparams.hidden_size
        if self.model_hyperparameters.num_heads == ModelHyperparameters.num_heads:
            self.model_hyperparameters.num_heads = adjusted_hyperparams.num_heads
        if self.model_hyperparameters.num_layers == ModelHyperparameters.num_layers:
            self.model_hyperparameters.num_layers = adjusted_hyperparams.num_layers
        if self.model_hyperparameters.d_ff == ModelHyperparameters.d_ff:
            self.model_hyperparameters.d_ff = adjusted_hyperparams.d_ff
        if self.model_hyperparameters.dropout_rate == ModelHyperparameters.dropout_rate:
            self.model_hyperparameters.dropout_rate = adjusted_hyperparams.dropout_rate
        if self.model_hyperparameters.max_seq_length == ModelHyperparameters.max_seq_length:
            self.model_hyperparameters.max_seq_length = adjusted_hyperparams.max_seq_length
        if self.model_hyperparameters.rel_pos_max_distance == ModelHyperparameters.rel_pos_max_distance:
            self.model_hyperparameters.rel_pos_max_distance = adjusted_hyperparams.rel_pos_max_distance

    def reload_from_env(self):
        """
        環境変数から設定を再読み込みします。

        このメソッドは、環境変数が動的に変更された後に呼び出すことで、
        設定を最新の環境変数の値に更新できます。
        """
        # 環境変数からのオーバーライドとデバイス検証
        self._apply_env_overrides()

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
            Tuple[Any, bool]: 変換された値と成功フラグのタプル。
                変換に成功した場合は (value, True)、失敗した場合は (None, False)。
        """
        # 型アノテーションが複雑な型（List, Dict等）の場合
        origin = get_origin(expected_type)
        args = get_args(expected_type)

        if origin is not None:
            if origin is list:
                # List型の場合（typing.List または list）
                # 文字列の場合はカンマ区切りとして扱う
                if isinstance(value, str):
                    # 空の文字列は空のリストとして扱う
                    if not value.strip():
                        return [], True
                    # カンマ区切りで分割
                    value = [item.strip() for item in value.split(",")]

                if isinstance(value, list):
                    # 要素の型を取得
                    if args:
                        element_type = args[0]
                        try:
                            coerced_list = []
                            # 型名を安全に取得
                            element_type_name = getattr(element_type, "__name__", None)
                            if element_type_name is None:
                                if isinstance(element_type, str):
                                    element_type_name = element_type
                                else:
                                    element_type_name = str(element_type) or repr(element_type)
                            for item in value:
                                coerced_item, success = self._coerce_value(item, element_type, key, section)
                                if not success:
                                    logging.warning(
                                        f"Type mismatch in _coerce_value: Failed to coerce list element in {section}.{key}. "
                                        f"Expected {element_type_name}, but got {type(item).__name__}. Skipping entire list."
                                    )
                                    return None, False
                                coerced_list.append(coerced_item)
                            return coerced_list, True
                        except (ValueError, TypeError) as e:
                            # element_type_name は既に安全に取得済み
                            element_type_name_safe = getattr(element_type, "__name__", None)
                            if element_type_name_safe is None:
                                element_type_name_safe = str(element_type) or repr(element_type)
                            logging.warning(
                                f"Type mismatch in _coerce_value: {section}.{key} expects List[{element_type_name_safe}], "
                                f"but got {type(value).__name__}. Error: {e}. Skipping assignment."
                            )
                            return None, False
                    else:
                        # 型引数がない場合はそのまま返す
                        return value, True
                else:
                    logging.warning(
                        f"Type mismatch in _coerce_value: {section}.{key} expects list, "
                        f"but got {type(value).__name__}. Skipping assignment."
                    )
                    return None, False
            # 他の複雑な型（Dict等）は将来の拡張用
            if isinstance(value, origin):
                return value, True
            else:
                logging.warning(
                    f"Type mismatch in _coerce_value: {section}.{key} expects {origin.__name__}, "
                    f"but got {type(value).__name__}. Skipping assignment."
                )
                return None, False

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
                # 文字列からboolへの変換（明示的なtrue/false値のみ受け入れる）
                if isinstance(value, str):
                    value_lower = value.strip().lower()
                    if value_lower in ("true", "yes", "1", "on"):
                        return True, True
                    elif value_lower in ("false", "no", "0", "off"):
                        return False, True
                    else:
                        raise ValueError(
                            f"Invalid boolean string value '{value}'. "
                            f"Expected one of: 'true', 'yes', '1', 'on', 'false', 'no', '0', 'off' (case-insensitive)"
                        )
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
                    # expected_type の名前を安全に取得
                    expected_type_name = getattr(expected_type, "__name__", None)
                    if expected_type_name is None:
                        # typing の Union などの場合は str() を使用
                        expected_type_name = str(expected_type)
                    raise ValueError(f"Cannot convert {type(value).__name__} to {expected_type_name}")

        except (ValueError, TypeError) as e:
            # expected_type の名前を安全に取得
            expected_type_name = getattr(expected_type, "__name__", None)
            if expected_type_name is None:
                # typing の Union などの場合は str() を使用
                expected_type_name = str(expected_type)
            logging.warning(
                f"Type mismatch in config.json: {section}.{key} expects {expected_type_name}, "
                f"but got {type(value).__name__} (value: {value}). Error: {e}. Skipping assignment."
            )
            return None, False

    def _override_from_env(self, obj: Any, prefix: str):
        # dataclassでない場合はスキップ
        if not is_dataclass(obj):
            return

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
                # _coerce_valueを使用して型変換（カンマ区切りリストもサポート）
                coerced_value, success = self._coerce_value(env_value, original_type, field_name, prefix.rstrip("_"))
                if success:
                    setattr(obj, field_name, coerced_value)
                else:
                    logging.warning(
                        f"Could not convert environment variable {env_var_name}='{env_value}' to type {original_type}."
                    )

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
                with open(config_path, "r", encoding="utf-8") as f:
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
        # 正規化されたデバイス名を取得（一度だけ計算）
        normalized = self.device.lower()

        if normalized == 'cuda':
            if not torch.cuda.is_available():
                logging.warning(
                    "環境変数でCUDAが指定されましたが、CUDAが利用できません。"
                    "CPUにフォールバックします。"
                )
                self.device = 'cpu'
            else:
                # CUDAが利用可能な場合でも、実際にデバイスにアクセスできるか確認
                try:
                    # 動的にGPUデバイスインデックスを取得
                    device_idx = torch.cuda.current_device()
                    device_name = torch.cuda.get_device_name(device_idx)
                    logging.info(f"CUDAデバイスが利用可能です: {device_name}")
                    # 成功時に正規化された値を設定
                    self.device = 'cuda'
                except (RuntimeError, AssertionError) as e:
                    logging.warning(
                        f"CUDAデバイスへのアクセスに失敗しました: {e}。"
                        "CPUにフォールバックします。"
                    )
                    self.device = 'cpu'
        elif normalized == 'cpu':
            # CPUの場合は正規化された値を設定
            self.device = 'cpu'
        else:
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

# グローバル設定インスタンス（遅延初期化）
_CONFIG: GlobalConfig | None = None
# 初期化用のロック（スレッドセーフティのため）
_CONFIG_LOCK: threading.Lock = threading.Lock()


def get_config() -> GlobalConfig:
    """
    グローバル設定インスタンスを取得します（遅延初期化、スレッドセーフ）。

    最初の呼び出し時に GlobalConfig を初期化します。
    GPUメモリに基づく設定調整は __post_init__ 内で自動的に実行されます。
    以降の呼び出しでは、同じインスタンスを返します。

    Returns:
        GlobalConfig: グローバル設定インスタンス
    """
    global _CONFIG
    # Double-checked lockingパターンでスレッドセーフに初期化
    if _CONFIG is None:
        with _CONFIG_LOCK:
            # ロック取得後に再度チェック（別スレッドが既に初期化した可能性があるため）
            if _CONFIG is None:
                _CONFIG = GlobalConfig()
                # initialize_gpu_aware_defaults() は __post_init__ 内で既に呼び出されている
    return _CONFIG


def __getattr__(name: str) -> GlobalConfig:
    """
    モジュールレベルの属性アクセスを処理します。

    CONFIG にアクセスした際に、遅延初期化を行います。
    これにより、モジュールインポート時には GPU プローブが発生しません。

    Args:
        name: アクセスする属性名

    Returns:
        GlobalConfig: グローバル設定インスタンス（name が 'CONFIG' の場合）
    """
    if name == "CONFIG":
        return get_config()
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


def get_input_vocab_path() -> str:
    """入力語彙パスを取得します。

    Returns:
        入力語彙ファイルのパス
    """
    return CONFIG.data_config.input_vocab_path


def get_output_vocab_path() -> str:
    """出力語彙パスを取得します。

    Returns:
        出力語彙ファイルのパス
    """
    return CONFIG.data_config.output_vocab_path
