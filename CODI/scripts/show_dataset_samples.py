#!/usr/bin/env python3
"""
数据集样例展示脚本
为每个指定的数据集输出一个样例（问题、思维链、答案）

优先从本地加载数据，参考 train.py 和 test.py 的数据加载逻辑

用法：
    python scripts/show_dataset_samples.py
    python scripts/show_dataset_samples.py --datasets "gsm8k svamp"
    python scripts/show_dataset_samples.py --output_file outputs/dataset_samples.txt
"""

import os
import json
import argparse

# ============================================================
# 数据集配置（优先使用本地路径）
# ============================================================
# 获取脚本所在目录的父目录作为项目根目录
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

DATASET_CONFIGS = {
    "gsm8k": {
        "hf_id": "zen-E/GSM8k-Aug",
        "split": "test",
        "question_field": "query",
        "cot_field": "reasoning",
        "answer_field": "answer",
        "answer_type": "number",
    },
    "svamp": {
        "hf_id": "ChilleD/SVAMP",
        "split": "all",
        "question_field": "query",
        "cot_field": "reasoning",
        "answer_field": "answer",
        "answer_type": "number",
    },
    "gsm-hard": {
        "hf_id": "juyoung-trl/gsm-hard",
        "split": "train",
        "question_field": "instruction",
        "cot_field": "response",  # gsm-hard 的 response 包含思维链和答案
        "answer_field": "response",
        "answer_type": "number",
    },
    "commonsense": {
        "hf_id": "zen-E/CommonsenseQA-GPT4omini",
        "split": "validation",
        "question_field": "query",
        "cot_field": "reasoning",
        "answer_field": "answer",
        "answer_type": "choice",
    },
    "asdiv": {
        "hf_id": "EleutherAI/asdiv",
        "split": "validation",
        "question_field": "question",
        "cot_field": None,  # asdiv 没有思维链
        "answer_field": "answer",
        "answer_type": "number",
        "has_body": True,
    },
    "multi-arith": {
        "hf_id": "ChilleD/MultiArith",
        "split": "test",
        "question_field": "query",
        "cot_field": "reasoning",
        "answer_field": "answer",
        "answer_type": "number",
    },
    "coin_flip": {
        "question_field": "query",
        "cot_field": "reasoning",
        "answer_field": "answer",
        "answer_type": "boolean",
    },
}


def load_dataset_by_name(data_name: str):
    """加载指定数据集，优先从本地加载"""
    if data_name not in DATASET_CONFIGS:
        raise ValueError(f"未知数据集: {data_name}。支持: {list(DATASET_CONFIGS.keys())}")
    
    config = DATASET_CONFIGS[data_name]
    
    # 优先从本地 JSON 文件加载
    if "local_path" in config:
        local_path = config["local_path"]
        if os.path.exists(local_path):
            print(f"[Data] ✓ 从本地加载: {data_name} ({local_path})")
            with open(local_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            return data, config
        else:
            print(f"[Data] 本地文件不存在: {local_path}，尝试从 HuggingFace 加载")
    
    # 回退到 HuggingFace 加载
    if "hf_id" not in config:
        raise FileNotFoundError(f"本地数据集文件不存在且无 HF 备选: {data_name}")
    
    print(f"[Data] 从 HuggingFace 加载: {data_name} ({config['hf_id']})")
    from datasets import load_dataset, concatenate_datasets
    
    hf_config = config.get("hf_config", None)
    if hf_config:
        dataset = load_dataset(config["hf_id"], hf_config, trust_remote_code=True)
    else:
        dataset = load_dataset(config["hf_id"], trust_remote_code=True)
    
    if config["split"] == "all":
        test_set = concatenate_datasets([dataset["train"], dataset["test"]])
    else:
        test_set = dataset[config["split"]]
    
    # 转换为 list 格式
    data = [dict(item) for item in test_set]
    return data, config


def extract_cot_and_answer(sample: dict, config: dict, data_name: str):
    """从样例中提取思维链和答案"""
    question_field = config["question_field"]
    cot_field = config.get("cot_field")
    answer_field = config["answer_field"]
    has_body = config.get("has_body", False)
    
    # 构建问题
    question = sample.get(question_field, "N/A")
    if has_body and "body" in sample:
        question = f"{sample['body']} {question}"
    
    # 获取思维链
    cot = ""
    if cot_field and cot_field in sample:
        cot = sample[cot_field]
    elif "full_answer" in sample:
        # 某些数据集的完整答案包含思维链
        cot = sample["full_answer"]
    elif "reasoning" in sample:
        cot = sample["reasoning"]
    
    # 获取答案
    answer = sample.get(answer_field, "N/A")
    
    # 特殊处理 gsm-hard: response 包含 "#### answer" 格式
    if data_name == "gsm-hard" and "####" in str(answer):
        parts = str(answer).split("####")
        cot = parts[0].strip()
        answer = parts[1].strip() if len(parts) > 1 else answer
    
    # 特殊处理 asdiv: answer 格式如 "9 (apples)"
    if data_name == "asdiv":
        import re
        match = re.match(r'([\d.]+)', str(answer))
        if match:
            answer = match.group(1)
    
    return question, cot, answer


def format_sample(data_name: str, sample: dict, config: dict, idx: int) -> str:
    """格式化输出样例"""
    question, cot, answer = extract_cot_and_answer(sample, config, data_name)
    
    lines = []
    lines.append(f"\n{'='*70}")
    lines.append(f"📊 数据集: {data_name.upper()} | 样例 #{idx+1}")
    lines.append(f"{'='*70}")
    
    lines.append(f"\n📝 【问题】")
    lines.append(f"{question}")
    
    if cot:
        lines.append(f"\n💭 【思维链/推理过程】")
        lines.append(f"{cot}")
    else:
        lines.append(f"\n💭 【思维链/推理过程】")
        lines.append(f"(该数据集无思维链)")
    
    lines.append(f"\n✅ 【答案】")
    lines.append(f"{answer}")
    
    lines.append(f"\n{'='*70}\n")
    
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="展示数据集样例（问题、思维链、答案）")
    parser.add_argument(
        "--datasets",
        type=str,
        default="gsm8k svamp gsm-hard commonsense asdiv",
        help="数据集列表，空格分隔"
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=1,
        help="每个数据集展示的样例数量"
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default=os.path.join(PROJECT_ROOT, "outputs/dataset_samples.txt"),
        help="输出文件路径"
    )
    
    args = parser.parse_args()
    datasets = args.datasets.split()
    
    print(f"\n🎯 准备展示以下数据集的样例: {datasets}")
    print(f"   每个数据集展示 {args.num_samples} 个样例")
    print(f"   输出文件: {args.output_file}\n")
    
    all_output = []
    all_output.append("=" * 70)
    all_output.append("数据集样例展示")
    all_output.append(f"数据集: {', '.join(datasets)}")
    all_output.append(f"每个数据集样例数: {args.num_samples}")
    all_output.append("=" * 70)
    
    for data_name in datasets:
        try:
            data, config = load_dataset_by_name(data_name)
            print(f"✓ {data_name}: 共 {len(data)} 条数据")
            
            # 展示样例
            for i in range(min(args.num_samples, len(data))):
                sample = data[i]
                output = format_sample(data_name, sample, config, i)
                print(output)
                all_output.append(output)
                
        except Exception as e:
            error_msg = f"\n❌ 加载 {data_name} 失败: {e}\n"
            print(error_msg)
            all_output.append(error_msg)
            import traceback
            traceback.print_exc()
    
    # 保存到文件
    os.makedirs(os.path.dirname(args.output_file) or ".", exist_ok=True)
    with open(args.output_file, 'w', encoding='utf-8') as f:
        f.write("\n".join(all_output))
    print(f"\n📁 结果已保存到: {args.output_file}")


if __name__ == "__main__":
    main()
