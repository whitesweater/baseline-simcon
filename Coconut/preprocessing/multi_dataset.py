#!/usr/bin/env python3
"""
多数据集预处理脚本 - 支持 GSM8K, GSM-Hard, MultiArith, SVAMP

功能：
1. 从 Hugging Face 加载数据集
2. 转换为统一的 JSON 格式
3. 生成 iCoT 格式的训练数据

用法：
    python preprocessing/multi_dataset.py --dataset gsm8k --split train
    python preprocessing/multi_dataset.py --dataset gsm-hard --split train
    python preprocessing/multi_dataset.py --dataset multiarith --split test
    python preprocessing/multi_dataset.py --dataset svamp --split all
"""

import json
import argparse
import os
from datasets import load_dataset, concatenate_datasets


# ============================================================
# 数据集配置
# ============================================================
DATASET_CONFIGS = {
    "gsm8k": {
        "hf_id": "gsm8k",
        "hf_name": "main",
        "splits": ["train", "test"],
        "question_field": "question",
        "answer_field": "answer",
        "cot_separator": "####",
    },
    "gsm8k-aug": {
        "hf_id": "zen-E/GSM8k-Aug",
        "hf_name": None,
        "splits": ["train", "test"],
        "question_field": "question",
        "answer_field": "answer",
        "cot_separator": "####",
    },
    "gsm-hard": {
        "hf_id": "juyoung-trl/gsm-hard",
        "hf_name": None,
        "splits": ["train"],
        "question_field": "instruction",
        "answer_field": "response",
        "cot_separator": "####",
    },
    "multiarith": {
        "hf_id": "ChilleD/MultiArith",
        "hf_name": None,
        "splits": ["train", "test"],
        "question_field": "question",
        "answer_field": "final_ans",
        "cot_separator": None,  # 无 CoT
    },
    "svamp": {
        "hf_id": "ChilleD/SVAMP",
        "hf_name": None,
        "splits": ["train", "test"],
        "question_field": "Body",  # 需要拼接
        "answer_field": "Answer",
        "cot_separator": None,  # 无 CoT
    },
}


# ============================================================
# 数据处理函数
# ============================================================
def process_gsm8k(example, config):
    """
    处理 GSM8K 格式数据
    格式: Question ... #### Answer
    """
    question = example[config["question_field"]].strip()
    answer_text = example[config["answer_field"]]
    
    if config["cot_separator"] in answer_text:
        # 有推理步骤
        parts = answer_text.split(config["cot_separator"])
        reasoning = parts[0].strip()
        answer = parts[1].strip().replace(',', '')
        
        # 将推理步骤按句子分割
        steps = [s.strip() for s in reasoning.split('.') if s.strip()]
    else:
        # 无推理步骤
        steps = []
        answer = answer_text.strip().replace(',', '')
    
    return {
        "question": question,
        "steps": steps,
        "answer": answer,
    }


def process_gsm_hard(example, config):
    """
    处理 GSM-Hard 格式数据
    格式与 GSM8K 相同
    """
    return process_gsm8k(example, config)


def process_multiarith(example, config):
    """
    处理 MultiArith 格式数据
    无 CoT，只有问题和答案
    """
    question = example[config["question_field"]].strip()
    answer = str(example[config["answer_field"]]).strip().replace(',', '')
    
    return {
        "question": question,
        "steps": [],  # 无步骤
        "answer": answer,
    }


def process_svamp(example, config):
    """
    处理 SVAMP 格式数据
    需要拼接 Body + Question
    """
    body = example.get("Body", "").strip()
    question_text = example.get("Question", "").strip()
    
    # 拼接完整问题
    if body and question_text:
        question = f"{body} {question_text}"
    else:
        question = body or question_text
    
    answer = str(example[config["answer_field"]]).strip().replace(',', '')
    
    return {
        "question": question,
        "steps": [],  # 无步骤
        "answer": answer,
    }


# ============================================================
# 主处理函数
# ============================================================
PROCESSORS = {
    "gsm8k": process_gsm8k,
    "gsm8k-aug": process_gsm8k,
    "gsm-hard": process_gsm_hard,
    "multiarith": process_multiarith,
    "svamp": process_svamp,
}


def load_and_process_dataset(dataset_name, split):
    """
    加载并处理数据集
    
    Args:
        dataset_name: 数据集名称
        split: 数据集分割（train/test/all）
    
    Returns:
        processed_data: 处理后的数据列表
    """
    if dataset_name not in DATASET_CONFIGS:
        raise ValueError(f"未知数据集: {dataset_name}。支持: {list(DATASET_CONFIGS.keys())}")
    
    config = DATASET_CONFIGS[dataset_name]
    processor = PROCESSORS[dataset_name]
    
    print(f"[Load] 数据集: {dataset_name}")
    print(f"[Load] Hugging Face: {config['hf_id']}")
    print(f"[Load] 分割: {split}")
    
    # 加载数据集
    if config["hf_name"]:
        dataset = load_dataset(config["hf_id"], config["hf_name"])
    else:
        dataset = load_dataset(config["hf_id"])
    
    # 处理分割
    if split == "all":
        # 合并所有分割
        test_set = concatenate_datasets([dataset[s] for s in config["splits"]])
    elif split in dataset:
        test_set = dataset[split]
    else:
        raise ValueError(f"数据集 {dataset_name} 不存在分割: {split}")
    
    print(f"[Load] 样本数: {len(test_set)}")
    
    # 处理数据
    processed_data = []
    for example in test_set:
        try:
            processed = processor(example, config)
            processed_data.append(processed)
        except Exception as e:
            print(f"[Warning] 处理样本失败: {e}")
            continue
    
    print(f"[Process] 成功处理: {len(processed_data)} 样本")
    
    return processed_data


def save_json(data, output_path):
    """保存为 JSON 格式"""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"[Save] JSON: {output_path}")


def save_icot_format(data, output_path):
    """
    保存为 iCoT 文本格式
    格式: question || step1 step2 ... ## answer
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        for sample in data:
            question = sample["question"]
            steps = " ".join(sample["steps"]) if sample["steps"] else ""
            answer = sample["answer"]
            
            # 格式: question || steps ## answer
            line = f"{question} || {steps} ## {answer}\n"
            f.write(line)
    print(f"[Save] iCoT: {output_path}")


def generate_statistics(data, dataset_name):
    """生成数据集统计信息"""
    total = len(data)
    with_steps = sum(1 for d in data if d["steps"])
    avg_steps = sum(len(d["steps"]) for d in data) / total if total > 0 else 0
    
    stats = {
        "dataset": dataset_name,
        "total_samples": total,
        "samples_with_steps": with_steps,
        "samples_without_steps": total - with_steps,
        "avg_steps_per_sample": round(avg_steps, 2),
    }
    
    print("\n" + "="*60)
    print(f"数据集统计: {dataset_name}")
    print("="*60)
    for key, value in stats.items():
        print(f"{key:25s}: {value}")
    print("="*60 + "\n")
    
    return stats


# ============================================================
# 命令行接口
# ============================================================
def main():
    parser = argparse.ArgumentParser(
        description="多数据集预处理脚本 - 支持 GSM8K, GSM-Hard, MultiArith, SVAMP"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=list(DATASET_CONFIGS.keys()),
        help="数据集名称",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        help="数据集分割 (train/test/all)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data",
        help="输出目录",
    )
    parser.add_argument(
        "--format",
        type=str,
        default="both",
        choices=["json", "icot", "both"],
        help="输出格式 (json/icot/both)",
    )
    
    args = parser.parse_args()
    
    # 加载并处理数据
    processed_data = load_and_process_dataset(args.dataset, args.split)
    
    # 生成统计信息
    stats = generate_statistics(processed_data, args.dataset)
    
    # 保存数据
    base_name = f"{args.dataset}_{args.split}"
    
    if args.format in ["json", "both"]:
        json_path = os.path.join(args.output_dir, f"{base_name}.json")
        save_json(processed_data, json_path)
    
    if args.format in ["icot", "both"]:
        icot_path = os.path.join(args.output_dir, f"{base_name}.txt")
        save_icot_format(processed_data, icot_path)
    
    # 保存统计信息
    stats_path = os.path.join(args.output_dir, f"{base_name}_stats.json")
    save_json(stats, stats_path)
    
    print(f"\n处理完成！")
    print(f"数据集: {args.dataset}")
    print(f"分割: {args.split}")
    print(f"样本数: {len(processed_data)}")


if __name__ == "__main__":
    main()
