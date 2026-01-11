#!/usr/bin/env python
"""
対話型ヘルプスクリプト
このスクリプトでは、プロジェクトの各スクリプトのコマンドラインオプションについて詳細な説明を提供します。
"""

import os
import argparse
import time
from typing import Callable
from wcwidth import wcswidth

def print_header(title: str, width: int | None = None) -> None:
    """見出しを表示

    Args:
        title: 表示する見出しテキスト
        width: 表示幅（指定しない場合はtitleの幅に基づいて自動計算）
    """
    # 表示幅を計算（マルチバイト文字に対応）
    if width is None:
        display_width = wcswidth(title)
        if display_width < 0:
            # wcswidthが負の値を返した場合はlen()にフォールバック
            display_width = len(title)
    else:
        display_width = width
    line = "=" * display_width
    print(f"\n{line}")
    print(title)
    print(f"{line}\n")

def print_command(command, description):
    """コマンドとその説明を表示"""
    print(f"  {command}")
    print(f"    {description}")
    print()

def display_tokenizer_help():
    """text_tokenizer.pyのヘルプ情報を表示"""
    print_header("text_tokenizer.py - データの前処理とトークン化")

    print("説明:")
    print("  テキストデータのトークン化と前処理を行います。オプションでデータ拡張も適用できます。")
    print("  データセットをダウンロードし、SentencePieceモデルをトレーニングして、トークン化したデータを保存します。\n")

    print("オプション:")
    print_command("--sample-size SIZE", "使用するサンプル数を指定します。デフォルトは1000、0で全データを使用。")
    print_command("--augment", "データ拡張を有効にします。")
    print_command("--augment-factor FACTOR", "データ拡張の割合を指定します。デフォルトは0.3（元データの30%）。")

    print("使用例:")
    print("  python text_tokenizer.py")
    print("  python text_tokenizer.py --sample-size 5000")
    print("  python text_tokenizer.py --augment --augment-factor 0.5")
    print("  python text_tokenizer.py --sample-size 0  # 全データを使用\n")

def display_train_help():
    """train.pyのヘルプ情報を表示"""
    print_header("train.py - モデルのトレーニング")

    print("説明:")
    print("  Transformerモデルのトレーニングを行います。チェックポイント機能やデータ拡張などの機能があります。\n")

    print("オプション:")
    print_command("--resume", "最新のチェックポイントからトレーニングを再開します。")
    print_command("--checkpoint PATH", "指定したチェックポイントファイルからトレーニングを再開します。")
    print_command("--augment", "トレーニング時にデータ拡張を適用します。")
    print_command("--augment-factor FACTOR", "データ拡張の割合を指定します。デフォルトは0.3。")
    print_command("--epochs N", "トレーニングのエポック数を指定します。")
    print_command("--batch-size SIZE", "バッチサイズを指定します。")
    print_command("--no-wandb", "Weights & Biasesのログ記録を無効にします。")

    print("使用例:")
    print("  python train.py")
    print("  python train.py --augment")
    print("  python train.py --resume")
    print("  python train.py --checkpoint models/checkpoints/checkpoint_epoch_5_20230401.pth")
    print("  python train.py --epochs 30 --batch-size 64")
    print("  python train.py --no-wandb  # WandBログを無効化\n")

def display_predict_help():
    """predict.pyのヘルプ情報を表示"""
    print_header("predict.py - モデルによる翻訳")

    print("説明:")
    print("  トレーニング済みモデルを使用して翻訳を行います。\n")

    print("使用方法:")
    print("  翻訳したいテキストを標準入力から入力します。")
    print("  'exit'と入力すると終了します。\n")

    print("使用例:")
    print("  python predict.py")
    print("  入力: これは翻訳のテストです。")
    print("  出力: This is a translation test.")
    print("  入力: exit  # 終了\n")

def display_augment_help():
    """data_augmentation.pyのヘルプ情報を表示"""
    print_header("data_augmentation.py - データ拡張機能")

    print("説明:")
    print("  テキストデータの拡張機能を提供します。このモジュールは直接実行するよりも、")
    print("  text_tokenizer.pyやtrain.pyから--augmentオプションを使って実行することをお勧めします。\n")

    print("提供する拡張技術:")
    print_command("トークンマスキング", "ランダムに選択したトークンを<unk>トークンに置き換えます。")
    print_command("トークン削除", "ランダムにトークンを削除します。")
    print_command("トークン置換", "ランダムにトークンを別のトークンに置き換えます。")
    print_command("トークン順序入れ替え", "局所的な窓内でトークンの順序をランダムに入れ替えます。")
    print_command("逆翻訳", "テキストを一度別の言語に翻訳した後、元の言語に戻します（トレーニング済みモデルが必要）。")

    print("使用例:")
    print("  python text_tokenizer.py --augment  # 推奨される使用方法")
    print("  python train.py --augment  # 推奨される使用方法\n")

# トピック名から表示関数へのマッピング
TOPIC_FUNCTIONS: dict[str, Callable[[], None]] = {
    'tokenizer': display_tokenizer_help,
    'train': display_train_help,
    'predict': display_predict_help,
    'augment': display_augment_help,
}

# 数字キーからトピック名へのマッピング（対話型メニュー用）
NUMBER_TO_TOPIC = {
    '1': 'tokenizer',
    '2': 'train',
    '3': 'predict',
    '4': 'augment',
    '5': 'all',
}

# メインヘルプメニューを更新
MAIN_MENU = """
利用可能なコマンド:

1. tokenizer  - text_tokenizer.py のヘルプと使用方法
2. train      - train.py のヘルプと使用方法
3. predict    - predict.py のヘルプと使用方法
4. augment    - データ拡張機能のヘルプ
5. all        - すべてのヘルプを表示

選択してください (1-5, または 'q' で終了): """

def display_topic(topic: str | None) -> None:
    """トピックのヘルプ情報を表示するヘルパー関数

    Args:
        topic: 表示するトピック名（'all'、'tokenizer'、'train'、'predict'、'augment'のいずれか、またはNone）
    """
    if topic is None:
        return

    if topic == 'all':
        # すべてのヘルプを表示
        for func in TOPIC_FUNCTIONS.values():
            func()
    elif topic in TOPIC_FUNCTIONS:
        # マッピングされた関数を呼び出し
        TOPIC_FUNCTIONS[topic]()
    else:
        print("無効な選択です。もう一度試してください。")

# メインの処理関数
def main():
    while True:
        os.system('cls' if os.name == 'nt' else 'clear')
        print_header("Transformer翻訳モデル - 対話型ヘルプ")

        print(MAIN_MENU)
        choice = input().strip().lower()

        if choice == 'q':
            break

        os.system('cls' if os.name == 'nt' else 'clear')

        # 数字キーをトピック名に変換
        topic = NUMBER_TO_TOPIC.get(choice, choice)

        display_topic(topic)
        if topic == 'all' or topic in TOPIC_FUNCTIONS:
            input("続行するにはEnterキーを押してください...")
        else:
            time.sleep(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="対話型ヘルプシステム")
    parser.add_argument('topic', nargs='?', choices=['tokenizer', 'train', 'predict', 'augment', 'all'],
                       help="特定のトピックのヘルプを直接表示します")

    args = parser.parse_args()

    if args.topic:
        display_topic(args.topic)
    else:
        main()
