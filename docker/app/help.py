#!/usr/bin/env python
"""
対話型ヘルプスクリプト
このスクリプトでは、プロジェクトの各スクリプトのコマンドラインオプションについて詳細な説明を提供します。
"""

import os
import argparse
import time

def print_header(title):
    """見出しを表示"""
    line = "=" * len(title)
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
    print_command("--enhanced", "相対位置エンコーディングとGLUを使用した強化版モデルを使用します。")
    print_command("--rel-pos-max-dist DIST", "相対位置エンコーディングの最大距離を指定します。デフォルトは64。")

    print("使用例:")
    print("  python train.py")
    print("  python train.py --augment")
    print("  python train.py --enhanced  # 強化版モデル使用")
    print("  python train.py --enhanced --rel-pos-max-dist 128  # 相対位置の最大距離を指定")
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

def enhanced_model_help():
    """enhanced_model.pyのヘルプ情報を表示"""
    print_header("強化版モデル - 相対位置エンコーディングとGLU")

    print("説明:")
    print("  最新の研究に基づいた改良版Transformerモデルを提供します。")
    print("  標準モデルに比べて、特に長いシーケンスや複雑な言語構造に対して優れた性能を発揮します。\n")

    print("主な改良点:")
    print_command("相対位置エンコーディング (RPE)",
                "絶対位置ではなく、トークン間の相対的な位置関係をモデル化します。")
    print_command("", "長いシーケンスや未知の長さのシーケンスにより効果的です。")
    print_command("", "Shaw et al. (2018)とRaffel et al. (2019)の手法に基づいています。")
    print()
    print_command("Gated Linear Units (GLU)",
                "通常のフィードフォワードネットワークの代わりにゲート機構を導入します。")
    print_command("", "情報の流れをより効果的に制御し、重要な特徴を強調します。")
    print_command("", "Dauphin et al. (2017)の論文に基づいています。")

    print("\n使用方法:")
    print("  train.pyの--enhancedオプションを使用して強化版モデルを有効にします。")
    print("  --rel-pos-max-distオプションで相対位置エンコーディングの最大距離を指定できます。\n")

    print("使用例:")
    print("  python train.py --enhanced")
    print("  python train.py --enhanced --rel-pos-max-dist 128  # デフォルトは64\n")

    print("参考文献:")
    print("  - Shaw et al. (2018). \"Self-Attention with Relative Position Representations\"")
    print("  - Raffel et al. (2019). \"Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer\"")
    print("  - Dauphin et al. (2017). \"Language Modeling with Gated Convolutional Networks\"\n")

# メインヘルプメニューを更新
MAIN_MENU = """
利用可能なコマンド:

1. tokenizer  - text_tokenizer.py のヘルプと使用方法
2. train      - train.py のヘルプと使用方法
3. predict    - predict.py のヘルプと使用方法
4. augment    - データ拡張機能のヘルプ
5. all        - すべてのヘルプを表示

選択してください (1-5, または 'q' で終了): """

# METEORヘルプを削除

# display_meteor_help関数を削除

# メインの処理関数からMETEOR関連の処理を削除
def main():
    while True:
        os.system('cls' if os.name == 'nt' else 'clear')
        print("=" * 80)
        print("Transformer翻訳モデル - 対話型ヘルプ")
        print("=" * 80)

        print(MAIN_MENU)
        choice = input().strip().lower()

        if choice == 'q':
            break

        os.system('cls' if os.name == 'nt' else 'clear')

        if choice == '1' or choice == 'tokenizer':
            display_tokenizer_help()
        elif choice == '2' or choice == 'train':
            display_train_help()
        elif choice == '3' or choice == 'predict':
            display_predict_help()
        elif choice == '4' or choice == 'augment':
            display_augment_help()
        elif choice == '5' or choice == 'all':
            display_tokenizer_help()
            display_train_help()
            display_predict_help()
            display_augment_help()
        else:
            print("無効な選択です。もう一度試してください。")
            time.sleep(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="対話型ヘルプシステム")
    parser.add_argument('topic', nargs='?', choices=['tokenizer', 'train', 'predict', 'augment', 'enhanced', 'all'],
                       help="特定のトピックのヘルプを直接表示します")

    args = parser.parse_args()

    if args.topic:
        if args.topic == 'tokenizer':
            display_tokenizer_help()
        elif args.topic == 'train':
            display_train_help()
        elif args.topic == 'predict':
            display_predict_help()
        elif args.topic == 'augment':
            display_augment_help()
        elif args.topic == 'enhanced':
            enhanced_model_help()
        elif args.topic == 'all':
            display_tokenizer_help()
            display_train_help()
            display_predict_help()
            display_augment_help()
            enhanced_model_help()
    else:
        main()
