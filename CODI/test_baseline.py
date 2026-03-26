#!/usr/bin/env python3
"""
基础模型测试脚本 - 测试原始 LLM 的 zero-shot 能力（无 LoRA、无 latent）

用法:
    python test_baseline.py \
        --model_path "/data/yhao/sim-con/modelscope/LLM-Research/Llama-3.2-1B-Instruct" \
        --datasets "gsm8k svamp gsm-hard multi-arith commonsense" \
        --batch_size 32 \
        --output_dir "./results/baseline"
"""

import os
import re
import json
import math
import time
import argparse
from datetime import datetime

import torch
import transformers
from datasets import load_dataset, concatenate_datasets
from src.tokenizer_utils import load_tokenizer_with_fallback
from tqdm import tqdm


# ============================================================
# 数据集配置
# ============================================================
DATASET_CONFIGS = {
    "gsm8k": {
        "hf_id": "zen-E/GSM8k-Aug",
        "split": "test",
        "question_field": "question",
        "answer_field": "answer",
        "answer_type": "number",
    },
    "gsm-hard": {
        "hf_id": "juyoung-trl/gsm-hard",
        "split": "train",
        "question_field": "instruction",
        "answer_field": "response",
        "answer_type": "number",
    },
    "multi-arith": {
        "hf_id": "ChilleD/MultiArith",
        "split": "test",
        "question_field": "question",
        "answer_field": "final_ans",
        "answer_type": "number",
    },
    "svamp": {
        "hf_id": "ChilleD/SVAMP",
        "split": "all",
        "question_field": "question_concat",
        "answer_field": "Answer",
        "answer_type": "number",
    },
    "commonsense": {
        "hf_id": "zen-E/CommonsenseQA-GPT4omini",
        "split": "validation",
        "question_field": "question",
        "answer_field": "answer",
        "answer_type": "choice",
    },
    "asdiv": {
        "hf_id": "EleutherAI/asdiv",
        "split": "validation",
        "question_field": "question",
        "answer_field": "answer",
        "answer_type": "number",
        "has_body": True,  # 需要拼接 body 到问题中
    },
}


def load_dataset_by_name(data_name: str):
    """加载数据集，优先使用本地缓存"""
    config = DATASET_CONFIGS[data_name]
    
    # 尝试离线加载
    os.environ["HF_DATASETS_OFFLINE"] = "1"
    try:
        print(f"[Data] 加载 {data_name} (离线模式)...")
        dataset = load_dataset(config["hf_id"], trust_remote_code=True)
        print(f"[Data] ✓ 从本地缓存加载成功")
    except Exception:
        print(f"[Data] 本地缓存不存在，尝试在线加载...")
        os.environ.pop("HF_DATASETS_OFFLINE", None)
        dataset = load_dataset(config["hf_id"], trust_remote_code=True)
    finally:
        os.environ.pop("HF_DATASETS_OFFLINE", None)
    
    if config["split"] == "all":
        test_set = concatenate_datasets([dataset["train"], dataset["test"]])
    else:
        test_set = dataset[config["split"]]
    
    return test_set, config


def prepare_data(test_set, config):
    """准备问题和答案"""
    questions = []
    answers = []
    has_body = config.get("has_body", False)
    
    for ex in test_set:
        q = ex[config["question_field"]].strip().replace('  ', ' ')
        
        # ASDiv: 拼接 body 到问题中
        if has_body and "body" in ex:
            body = ex["body"].strip()
            q = f"{body} {q}"
        
        questions.append(q)
        
        ans = ex[config["answer_field"]]
        
        # 处理不同类型的答案
        if isinstance(ans, bool):
            answers.append(ans)
        elif ans in ["True", "False"]:
            answers.append(ans == "True")
        elif config["answer_type"] == "choice" and str(ans).strip() in "ABCDE":
            answers.append(str(ans).strip())
        else:
            if "####" in str(ans):
                ans = str(ans).split('####')[-1]
            ans = str(ans).replace(',', '')
            
            # ASDiv 格式: "9 (apples)" -> 提取第一个数字
            if has_body:
                match = re.match(r'^(-?\d+\.?\d*)', str(ans))
                if match:
                    ans = match.group(1)
            
            try:
                answers.append(float(ans))
            except ValueError:
                answers.append(float("inf"))
    
    return questions, answers


def extract_answer(text: str, answer_type: str):
    """从模型输出提取答案"""
    text = text.replace(',', '')
    
    if answer_type == "choice":
        # 选择题
        pred = text.split("The answer is:")[-1].strip()
        if pred and pred[0] in "ABCDE":
            return pred[0]
        match = re.search(r'\b([A-E])\b', text)
        if match:
            return match.group(1)
        return float('inf')
    
    elif answer_type == "boolean":
        text_lower = text.lower()
        if 'yes' in text_lower:
            return True
        elif 'no' in text_lower:
            return False
        return float('inf')
    
    else:
        # 数字答案
        pred = re.findall(r'-?\d+\.?\d*', text)
        if not pred:
            return float('inf')
        return float(pred[-1])


def main():
    parser = argparse.ArgumentParser(description="测试基础 LLM 模型")
    parser.add_argument("--model_path", type=str, required=True, help="模型路径")
    parser.add_argument("--datasets", type=str, default="gsm8k", help="数据集列表，空格分隔")
    parser.add_argument("--batch_size", type=int, default=16, help="批次大小")
    parser.add_argument("--max_new_tokens", type=int, default=256, help="最大生成 token 数")
    parser.add_argument("--output_dir", type=str, default="./results/baseline", help="结果保存目录")
    parser.add_argument("--greedy", action="store_true", default=True, help="使用贪婪解码")
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # ============================================================
    # 加载模型和 tokenizer
    # ============================================================
    print(f"\n{'='*60}")
    print(f"加载模型: {args.model_path}")
    print(f"{'='*60}")
    
    tokenizer = load_tokenizer_with_fallback(
        args.model_path,
        padding_side="left",
        use_fast=False,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    model = transformers.AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model.eval()
    print("✓ 模型加载完成")
    
    # ============================================================
    # 测试每个数据集
    # ============================================================
    datasets_list = args.datasets.split()
    all_results = []
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    for data_name in datasets_list:
        if data_name not in DATASET_CONFIGS:
            print(f"⚠ 跳过未知数据集: {data_name}")
            continue
        
        print(f"\n{'─'*60}")
        print(f"测试数据集: {data_name}")
        print(f"{'─'*60}")
        
        start_time = time.time()
        
        # 加载数据
        test_set, config = load_dataset_by_name(data_name)
        questions, answers = prepare_data(test_set, config)
        print(f"样本数: {len(questions)}")
        
        # 批量推理
        predictions = []
        num_batches = math.ceil(len(questions) / args.batch_size)
        
        for i in tqdm(range(num_batches), desc=f"Testing {data_name}"):
            start_idx = i * args.batch_size
            end_idx = min((i + 1) * args.batch_size, len(questions))
            batch_questions = questions[start_idx:end_idx]
            
            # Tokenize
            inputs = tokenizer(
                batch_questions,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512,
            ).to(device)
            
            # Generate
            with torch.no_grad():
                if args.greedy:
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=args.max_new_tokens,
                        do_sample=False,
                        pad_token_id=tokenizer.pad_token_id,
                    )
                else:
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=args.max_new_tokens,
                        do_sample=True,
                        temperature=0.1,
                        top_k=40,
                        top_p=0.95,
                        pad_token_id=tokenizer.pad_token_id,
                    )
            
            # Decode and extract answers
            for j, output in enumerate(outputs):
                # 只取生成的部分
                generated = output[inputs["input_ids"].shape[1]:]
                text = tokenizer.decode(generated, skip_special_tokens=True)
                pred = extract_answer(text, config["answer_type"])
                predictions.append(pred)
        
        # 计算准确率
        correct = sum(1 for p, g in zip(predictions, answers) if p == g)
        accuracy = correct / len(answers)
        
        elapsed_time = time.time() - start_time
        avg_time_per_sample = elapsed_time / len(answers) if len(answers) > 0 else 0
        
        print(f"✓ {data_name} 准确率: {100*accuracy:.2f}% ({correct}/{len(answers)}) | 耗时: {elapsed_time:.1f}s ({avg_time_per_sample*1000:.1f}ms/样本)")
        
        # 保存结果
        result = {
            "dataset": data_name,
            "accuracy": accuracy,
            "correct": correct,
            "total": len(answers),
            "elapsed_time_sec": round(elapsed_time, 1),
            "ms_per_sample": round(avg_time_per_sample * 1000, 1),
            "model": args.model_path,
            "timestamp": datetime.now().isoformat(),
        }
        all_results.append(result)
        
        # 保存每个数据集的详细结果
        detail_path = os.path.join(args.output_dir, f"{data_name}_results.json")
        with open(detail_path, 'w') as f:
            json.dump({
                "config": result,
                "predictions": [str(p) for p in predictions],
                "answers": [str(a) for a in answers],
            }, f, indent=2)
    
    # ============================================================
    # 汇总结果
    # ============================================================
    print(f"\n{'='*60}")
    print("测试汇总")
    print(f"{'='*60}")
    
    total_time = sum(r['elapsed_time_sec'] for r in all_results)
    total_samples = sum(r['total'] for r in all_results)
    
    for r in all_results:
        print(f"  {r['dataset']:15s}: {100*r['accuracy']:.2f}%  | {r['total']:5d}条 | {r['elapsed_time_sec']:.1f}s ({r['ms_per_sample']:.1f}ms/条)")
    
    if all_results:
        avg_acc = sum(r['accuracy'] for r in all_results) / len(all_results)
        avg_ms_per_sample = (total_time * 1000 / total_samples) if total_samples > 0 else 0
        print(f"  {'─'*60}")
        print(f"  {'平均准确率':15s}: {100*avg_acc:.2f}%")
        print(f"  {'总样本数':15s}: {total_samples}条")
        print(f"  {'总耗时':15s}: {total_time:.1f}s ({total_time/60:.1f}min)")
        print(f"  {'平均耗时':15s}: {avg_ms_per_sample:.1f}ms/条")
    
    # 保存汇总
    summary_path = os.path.join(args.output_dir, "summary.json")
    with open(summary_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\n结果保存至: {args.output_dir}")


if __name__ == "__main__":
    main()
