"""
tokenizer_utils モジュールのテスト
"""
from data.tokenizer_utils import normalize_text


def test_normalize_text_with_numeric_replacement():
    """数字が '<NUM>' に置き換えられることをテスト"""
    text = "I have 123 apples and 45 oranges."
    result = normalize_text(text, "en_US")
    assert "<NUM>" in result
    assert "123" not in result
    assert "45" not in result
    assert "apples" in result
    assert "oranges" in result


def test_normalize_text_with_custom_numeric_token():
    """カスタム数字トークンが使用できることをテスト"""
    text = "Price is 999 dollars."
    result = normalize_text(text, "en_US", normalize_numeric="<NUMBER>")
    assert "<NUMBER>" in result
    assert "999" not in result


def test_normalize_text_without_numeric_replacement():
    """数字を置き換えない場合のテスト"""
    text = "I have 123 apples."
    result = normalize_text(text, "en_US", normalize_numeric=None)
    assert "123" in result
    assert "<NUM>" not in result

    # False でも同様に動作することを確認
    result_false = normalize_text(text, "en_US", normalize_numeric=False)
    assert "123" in result_false
    assert "<NUM>" not in result_false


def test_normalize_text_japanese():
    """日本語テキストの正規化テスト"""
    text = "価格は1234円です。"
    result = normalize_text(text, "ja_JP")
    assert "<NUM>" in result
    assert "1234" not in result
    # 日本語は小文字化されない
    assert "価格" in result


def test_normalize_text_multiple_numbers():
    """複数の数字がすべて置き換えられることをテスト"""
    text = "Year 2024 has 365 days and 12 months."
    result = normalize_text(text, "en_US")
    # すべての数字が <NUM> に置き換えられる
    assert result.count("<NUM>") == 3
    assert "2024" not in result
    assert "365" not in result
    assert "12" not in result


def test_normalize_text_no_numbers():
    """数字がないテキストのテスト"""
    text = "Hello world!"
    result = normalize_text(text, "en_US")
    assert "<NUM>" not in result
    assert "hello" in result  # 小文字化される


def test_normalize_text_whitespace_normalization():
    """空白の正規化が機能することをテスト"""
    text = "Multiple    spaces    here"
    result = normalize_text(text, "en_US")
    # 複数の空白が1つに正規化される
    assert "  " not in result
    assert "multiple spaces here" in result


def test_normalize_text_with_empty_string_numeric():
    """normalize_numeric='' で数値正規化が無効になることをテスト"""
    text = "I have 123 apples and 45 oranges."

    # 空文字列を渡すと数値正規化が無効になる
    result = normalize_text(text, "en_US", normalize_numeric='')
    assert "123" in result
    assert "45" in result
    assert "<NUM>" not in result

    # None の場合と同様の動作
    result_none = normalize_text(text, "en_US", normalize_numeric=None)
    assert result == result_none


def test_normalize_numeric_parameter_variations():
    """normalize_numeric パラメータの様々な値の動作をテスト"""
    text = "Price is 999 dollars."

    # デフォルト値（'<NUM>'）で数値が置き換えられる
    result_default = normalize_text(text, "en_US")
    assert "<NUM>" in result_default
    assert "999" not in result_default

    # カスタムトークンで数値が置き換えられる
    result_custom = normalize_text(text, "en_US", normalize_numeric="<NUMBER>")
    assert "<NUMBER>" in result_custom
    assert "999" not in result_custom

    # None で数値が置き換えられない
    result_none = normalize_text(text, "en_US", normalize_numeric=None)
    assert "999" in result_none
    assert "<NUM>" not in result_none

    # False で数値が置き換えられない
    result_false = normalize_text(text, "en_US", normalize_numeric=False)
    assert "999" in result_false
    assert "<NUM>" not in result_false

    # 空文字列で数値が置き換えられない
    result_empty = normalize_text(text, "en_US", normalize_numeric='')
    assert "999" in result_empty
    assert "<NUM>" not in result_empty
    # None と空文字列は同じ動作
    assert result_none == result_empty
