"""
カスタム例外クラス定義
"""
from typing import Optional


class TransformerError(Exception):
    """Transformerモデル関連の基底例外クラス"""
    pass


class ModelConfigurationError(TransformerError):
    """モデル設定に関するエラー"""
    pass


class TokenValidationError(TransformerError):
    """トークン検証に関するエラー"""
    pass


class CheckpointError(TransformerError):
    """チェックポイント処理に関するエラー"""
    pass


class DataProcessingError(TransformerError):
    """データ処理に関するエラー"""
    pass


class VocabularyError(TransformerError):
    """語彙に関するエラー"""
    pass


class TrainingError(TransformerError):
    """トレーニング処理に関するエラー"""
    pass


class InferenceError(TransformerError):
    """推論処理に関するエラー"""
    pass
