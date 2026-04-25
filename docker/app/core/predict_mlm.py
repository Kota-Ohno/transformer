#!/usr/bin/env python3
"""
MLM（Masked Language Modeling）推論スクリプト

学習済みモデルを使用してマスクされたトークンを予測します。
"""
import sys
import os
import argparse
import logging
from typing import List, Tuple, Optional

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

# Pythonパスを設定
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models.mlm_model import MLMModel
from utils.masking import create_mlm_mask

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_model(model_path: str, device: torch.device) -> MLMModel:
    """学習済みモデルを読み込み"""
    logger.info(f"モデルを読み込み中: {model_path}")
    model = MLMModel.from_pretrained(model_path)
    model = model.to(device)
    model.eval()
    logger.info("モデル読み込み完了")
    return model


def predict_masked_tokens(
    model: MLMModel,
    tokenizer,
    text: str,
    device: torch.device,
    top_k: int = 5
) -> List[Tuple[str, str, List[Tuple[str, float]]]]:
    """
    マスクされたトークンを予測
    
    Args:
        model: MLMモデル
        tokenizer: トークナイザー
        text: 入力テキスト（[MASK]を含む）
        device: デバイス
        top_k: 上位何個の予測を返すか
        
    Returns:
        [(元のトークン, 予測トークン, [(候補, 確率), ...]), ...]
    """
    # トークナイズ
    encoding = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=512
    )
    
    input_ids = encoding["input_ids"].to(device)
    attention_mask = encoding["attention_mask"].to(device)
    
    # マスク位置を特定
    mask_positions = (input_ids == tokenizer.mask_token_id).nonzero(as_tuple=True)
    
    if len(mask_positions[0]) == 0:
        logger.warning("テキストに[MASK]が含まれていません")
        return []
    
    # 予測
    with torch.no_grad():
        logits = model(input_ids, attention_mask)
    
    results = []
    
    for batch_idx, pos_idx in zip(mask_positions[0].tolist(), mask_positions[1].tolist()):
        # 該当位置のロジットを取得
        token_logits = logits[batch_idx, pos_idx, :]
        
        # Softmaxで確率に変換
        probs = F.softmax(token_logits, dim=-1)
        
        # 上位k個の予測を取得
        top_probs, top_indices = torch.topk(probs, top_k)
        
        # デコード
        candidates = []
        for prob, idx in zip(top_probs.tolist(), top_indices.tolist()):
            token_str = tokenizer.decode([idx], skip_special_tokens=True)
            candidates.append((token_str, prob))
        
        results.append(("[MASK]", candidates[0][0], candidates))
    
    return results


def interactive_mode(model: MLMModel, tokenizer, device: torch.device):
    """対話モード"""
    print("\n=== MLM 対話モード ===")
    print("テキストを入力してください（[MASK]を含む）")
    print("終了するには 'quit' または 'exit' と入力\n")
    
    while True:
        try:
            text = input("入力: ").strip()
            
            if text.lower() in ["quit", "exit", "q"]:
                print("終了します")
                break
            
            if not text:
                continue
            
            if "[MASK]" not in text:
                print("警告: テキストに [MASK] が含まれていません")
                continue
            
            results = predict_masked_tokens(model, tokenizer, text, device)
            
            if results:
                print("\n予測結果:")
                for i, (original, predicted, candidates) in enumerate(results, 1):
                    print(f"  {i}. [MASK] → '{predicted}'")
                    print(f"     候補: {', '.join([f'{t}({p:.2%})' for t, p in candidates])}")
                print()
            else:
                print("予測結果がありません\n")
                
        except KeyboardInterrupt:
            print("\n終了します")
            break
        except Exception as e:
            logger.error(f"エラー: {e}")


def batch_predict(
    model: MLMModel,
    tokenizer,
    texts: List[str],
    device: torch.device,
    output_file: Optional[str] = None
):
    """バッチ予測"""
    results = []
    
    for text in texts:
        predictions = predict_masked_tokens(model, tokenizer, text, device)
        results.append({
            "text": text,
            "predictions": predictions
        })
    
    # 結果を表示
    for result in results:
        print(f"\n入力: {result['text']}")
        if result['predictions']:
            for i, (original, predicted, candidates) in enumerate(result['predictions'], 1):
                print(f"  {i}. [MASK] → '{predicted}'")
        else:
            print("  予測なし")
    
    # ファイルに保存
    if output_file:
        import json
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        logger.info(f"結果を保存しました: {output_file}")


def main():
    parser = argparse.ArgumentParser(description="MLM推論スクリプト")
    
    parser.add_argument("--model-path", type=str, default="models/mlm_model",
                        help="学習済みモデルのパス")
    parser.add_argument("--tokenizer", type=str, default="bert-base-cased",
                        help="使用するトークナイザー")
    parser.add_argument("--text", type=str, default=None,
                        help="予測するテキスト（[MASK]を含む）")
    parser.add_argument("--file", type=str, default=None,
                        help="予測するテキストが記載されたファイル（1行ごと）")
    parser.add_argument("--output", type=str, default=None,
                        help="出力ファイル（JSON形式）")
    parser.add_argument("--top-k", type=int, default=5,
                        help="上位何個の予測を表示するか")
    parser.add_argument("--interactive", action="store_true",
                        help="対話モードで実行")
    parser.add_argument("--device", type=str, default="cpu",
                        help="使用するデバイス（cpu/cuda）")
    
    args = parser.parse_args()
    
    # デバイスを設定
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    logger.info(f"使用デバイス: {device}")
    
    # トークナイザーを読み込み
    logger.info(f"トークナイザーを読み込み中: {args.tokenizer}")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    
    # モデルを読み込み
    model = load_model(args.model_path, device)
    
    # 実行モード
    if args.interactive:
        interactive_mode(model, tokenizer, device)
    elif args.text:
        # 単一テキスト
        results = predict_masked_tokens(model, tokenizer, args.text, device, args.top_k)
        print(f"\n入力: {args.text}")
        if results:
            for i, (original, predicted, candidates) in enumerate(results, 1):
                print(f"\n予測 {i}:")
                print(f"  最確: '{predicted}'")
                print(f"  候補: {', '.join([f'{t}({p:.2%})' for t, p in candidates])}")
        else:
            print("予測結果がありません")
    elif args.file:
        # ファイルから読み込み
        with open(args.file, "r", encoding="utf-8") as f:
            texts = [line.strip() for line in f if line.strip()]
        batch_predict(model, tokenizer, texts, device, args.output)
    else:
        # デフォルト例
        example_texts = [
            "The cat sat on the [MASK] and looked at the birds.",
            "I love to eat [MASK] for breakfast.",
            "The [MASK] is shining brightly today."
        ]
        print("デフォルト例を実行します:\n")
        batch_predict(model, tokenizer, example_texts, device)


if __name__ == "__main__":
    main()
