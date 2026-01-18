#!/usr/bin/env python3
"""
多数据集测试脚本 - 一次加载模型，测试多个数据集

核心优化：
1. 模型只加载一次，然后在多个数据集上测试
2. 清晰的结果保存结构
3. 自动生成汇总报告

用法：
    python test_multi_dataset.py \
        --model_name_or_path /path/to/base/model \
        --ckpt_dir /path/to/checkpoint \
        --datasets "gsm8k svamp gsm-hard" \
        --num_runs 1
"""

import logging
import math
import re
import os
import json
from dataclasses import dataclass, field
from typing import Dict, Optional, List
from datetime import datetime

import torch
import transformers
from torch.nn import functional as F
from huggingface_hub import hf_hub_download
from peft import LoraConfig, TaskType
from datasets import load_dataset, concatenate_datasets
from safetensors.torch import load_file

from src.model import (
    CODI,
    ModelArguments,
    DataArguments,
    TrainingArguments,
)
from src.trajectory_consistency import TrajectoryConsistencyLoss
from src.trajectory_acceleration import TrajectoryAccelerationLoss
from src.trajectory_action import TrajectoryActionLoss
from src.trajectory_geodesic import TrajectoryGeodesicDeviationLoss


# ============================================================
# 环境配置
# ============================================================
CODI_SAVE_DIR = os.environ.get("CODI_SAVE_DIR", "./outputs")
CODI_RESULT_DIR = os.environ.get("CODI_RESULT_DIR", "./results")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


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
        "split": "all",  # 特殊处理：合并 train 和 test
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
}


# ============================================================
# 工具函数
# ============================================================
def sanitize_for_json(obj):
    """递归处理数据，将 inf/nan 替换为字符串以便 JSON 序列化"""
    import math
    if isinstance(obj, float):
        if math.isinf(obj):
            return "inf" if obj > 0 else "-inf"
        if math.isnan(obj):
            return "nan"
        return obj
    elif isinstance(obj, dict):
        return {k: sanitize_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [sanitize_for_json(item) for item in obj]
    return obj


def save_json(data, filepath):
    """保存 JSON 文件，自动处理 inf/nan"""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(sanitize_for_json(data), f, ensure_ascii=False, indent=2)


def save_jsonl_line(filepath, data):
    """追加一行到 JSONL 文件，自动处理 inf/nan"""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "a", encoding="utf-8") as f:
        f.write(json.dumps(sanitize_for_json(data), ensure_ascii=False) + "\n")


def load_state_dict_from_ckpt(ckpt_dir, token=None):
    """加载 checkpoint"""
    if not ckpt_dir:
        return None
    
    safetensor_local = os.path.join(ckpt_dir, "model.safetensors")
    bin_local = os.path.join(ckpt_dir, "pytorch_model.bin")
    
    if os.path.exists(safetensor_local):
        return load_file(safetensor_local)
    if os.path.exists(bin_local):
        return torch.load(bin_local, map_location="cpu")
    
    # 尝试从 HF Hub 下载
    try:
        safetensor_remote = hf_hub_download(repo_id=ckpt_dir, filename="model.safetensors", token=token)
        return load_file(safetensor_remote)
    except Exception:
        pass
    
    try:
        bin_remote = hf_hub_download(repo_id=ckpt_dir, filename="pytorch_model.bin", token=token)
        return torch.load(bin_remote, map_location="cpu")
    except Exception as e:
        print(f"[Eval] 无法加载 checkpoint: {e}")
        return None


def get_model_name_from_ckpt(ckpt_dir):
    """从 checkpoint 路径提取模型名称"""
    if not ckpt_dir:
        return "base"
    # 取路径的最后几个部分作为名称
    parts = ckpt_dir.rstrip('/').split('/')
    # 尝试找到有意义的名称
    for i, part in enumerate(parts):
        if part in ['trained', 'outputs', 'checkpoints']:
            return '_'.join(parts[i+1:]) if i+1 < len(parts) else parts[-1]
    return parts[-1] if parts else "unknown"


# ============================================================
# 数据集加载
# ============================================================
def load_dataset_by_name(data_name: str):
    """加载指定数据集"""
    if data_name not in DATASET_CONFIGS:
        raise ValueError(f"未知数据集: {data_name}。支持: {list(DATASET_CONFIGS.keys())}")
    
    config = DATASET_CONFIGS[data_name]
    print(f"[Data] 加载数据集: {data_name} ({config['hf_id']})")
    
    dataset = load_dataset(config["hf_id"])
    
    if config["split"] == "all":
        # 特殊处理：合并所有 split
        test_set = concatenate_datasets([dataset["train"], dataset["test"]])
    else:
        test_set = dataset[config["split"]]
    
    return test_set, config


def prepare_questions_and_answers(test_set, config):
    """准备问题和答案"""
    question_field = config["question_field"]
    answer_field = config["answer_field"]
    answer_type = config["answer_type"]
    
    questions = [ex[question_field].strip().replace('  ', ' ') for ex in test_set]
    answers = []
    
    for ex in test_set:
        ans = ex[answer_field]
        
        # 处理布尔值
        if isinstance(ans, bool):
            answers.append(ans)
            continue
        if ans in ["True", "False"]:
            answers.append(ans == "True")
            continue
        
        # 处理选择题
        if answer_type == "choice" and ans in "ABCDE":
            answers.append(ans)
            continue
        
        # 处理数字答案
        if "####" in str(ans):
            ans = str(ans).split('####')[-1]
        ans = str(ans).replace(',', '')
        
        try:
            answers.append(float(ans))
        except ValueError:
            answers.append(float("inf"))
    
    return questions, answers


# ============================================================
# 答案提取
# ============================================================
def extract_answer(sentence: str, answer_type: str):
    """从生成文本中提取答案"""
    sentence = sentence.replace(',', '')
    
    if answer_type == "choice":
        # 选择题：提取 A-E
        pred = sentence.split("The answer is:")[-1].strip()
        if pred and pred[0] in "ABCDE":
            return pred[0]
        for char in "ABCDE":
            if char in sentence:
                return char
        return float('inf')
    
    elif answer_type == "boolean":
        if "True" in sentence:
            return True
        elif "False" in sentence:
            return False
        return float('inf')
    
    else:
        # 数字答案
        pred = re.findall(r'-?\d+\.?\d*', sentence)
        if not pred:
            return float('inf')
        return float(pred[-1])


def compute_accuracy(gold: list, pred: list):
    """计算准确率"""
    acc = sum(1 for p, g in zip(pred, gold) if p == g)
    return acc / len(gold) if gold else 0.0


# ============================================================
# 结果管理
# ============================================================
class ResultsManager:
    """
    结果保存管理器
    
    目录结构：
    results/
    ├── models/
    │   └── {model_name}/
    │       ├── {dataset}/
    │       │   ├── run_{i}/
    │       │   │   ├── predictions.json
    │       │   │   ├── metrics.json
    │       │   │   └── trajectory_stats.jsonl
    │       │   └── summary.json
    │       └── model_summary.json
    ├── datasets/
    │   └── {dataset}/
    │       └── all_models.csv
    └── summary/
        ├── all_results.csv
        └── comparison_matrix.csv
    """
    
    def __init__(self, base_dir: str = None):
        self.base_dir = base_dir or CODI_RESULT_DIR
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 创建目录结构
        self.models_dir = os.path.join(self.base_dir, "models")
        self.datasets_dir = os.path.join(self.base_dir, "datasets")
        self.summary_dir = os.path.join(self.base_dir, "summary")
        
        os.makedirs(self.models_dir, exist_ok=True)
        os.makedirs(self.datasets_dir, exist_ok=True)
        os.makedirs(self.summary_dir, exist_ok=True)
        
        # 存储当前运行的结果
        self.all_results = []
        
        # 全局结果文件路径（追加模式）
        self.global_results_path = os.path.join(self.summary_dir, "all_results.csv")
    
    def get_run_dir(self, model_name: str, dataset: str, run_id: int) -> str:
        """获取运行结果目录"""
        run_dir = os.path.join(self.models_dir, model_name, dataset, f"run_{run_id}")
        os.makedirs(run_dir, exist_ok=True)
        return run_dir
    
    def save_run_results(self, model_name: str, dataset: str, run_id: int,
                         predictions: list, accuracy: float, 
                         questions: list, answers: list, 
                         trajectory_stats: dict = None):
        """保存单次运行结果"""
        run_dir = self.get_run_dir(model_name, dataset, run_id)
        
        # 保存预测结果
        save_json({
            "predictions": predictions,
            "ground_truth": [str(a) for a in answers],
        }, os.path.join(run_dir, "predictions.json"))
        
        # 保存指标
        metrics = {
            "model": model_name,
            "dataset": dataset,
            "run_id": run_id,
            "accuracy": accuracy,
            "total_samples": len(predictions),
            "correct": int(accuracy * len(predictions)),
            "timestamp": self.timestamp,
        }
        save_json(metrics, os.path.join(run_dir, "metrics.json"))
        
        # 保存轨迹统计
        if trajectory_stats:
            save_json(trajectory_stats, os.path.join(run_dir, "trajectory_stats.json"))
        
        # 添加到总结果
        self.all_results.append(metrics)
        
        print(f"[Results] 保存结果: {run_dir}")
        return metrics
    
    def finalize(self):
        """生成最终汇总报告"""
        if not self.all_results:
            print("[Results] 没有结果需要汇总")
            return
        
        # 1. 追加当前结果到全局 CSV（而不是覆盖）
        import csv
        file_exists = os.path.exists(self.global_results_path)
        with open(self.global_results_path, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=self.all_results[0].keys())
            if not file_exists:
                writer.writeheader()
            writer.writerows(self.all_results)
        print(f"[Results] 追加到汇总表: {self.global_results_path}")
        
        # 2. 重新读取所有结果，生成完整的对比矩阵
        self._regenerate_all_summaries()
    
    def _regenerate_all_summaries(self):
        """从全局 CSV 重新生成所有汇总报告"""
        import pandas as pd
        
        if not os.path.exists(self.global_results_path):
            print("[Results] 全局结果文件不存在，跳过汇总生成")
            return
        
        try:
            df = pd.read_csv(self.global_results_path)
            if df.empty:
                print("[Results] 全局结果为空")
                return
        except Exception as e:
            print(f"[Results] 读取全局结果失败: {e}")
            return
        
        print(f"[Results] 从 {len(df)} 条记录生成汇总...")
        
        # 生成对比矩阵
        self._generate_comparison_matrix_from_df(df)
        
        # 生成每个数据集的模型对比
        self._generate_per_dataset_summary_from_df(df)
        
        # 生成每个模型的汇总
        self._generate_per_model_summary_from_df(df)
    
    def _generate_comparison_matrix_from_df(self, df):
        """从 DataFrame 生成对比矩阵"""
        import pandas as pd
        
        try:
            # 计算每个 (模型, 数据集) 的平均准确率
            pivot = df.pivot_table(
                values='accuracy',
                index='model',
                columns='dataset',
                aggfunc=['mean', 'std', 'count']
            )
            
            # 扁平化列名
            pivot.columns = [f"{col[1]}_{col[0]}" for col in pivot.columns]
            pivot = pivot.reset_index()
            
            # 添加平均值列
            acc_cols = [c for c in pivot.columns if c.endswith('_mean')]
            if acc_cols:
                pivot['avg_accuracy'] = pivot[acc_cols].mean(axis=1)
            
            # 按平均准确率排序
            if 'avg_accuracy' in pivot.columns:
                pivot = pivot.sort_values('avg_accuracy', ascending=False)
            
            matrix_path = os.path.join(self.summary_dir, "comparison_matrix.csv")
            pivot.to_csv(matrix_path, index=False, float_format='%.4f')
            print(f"[Results] 对比矩阵: {matrix_path}")
            
            # 打印矩阵
            print("\n" + "="*80)
            print("模型 × 数据集 准确率矩阵")
            print("="*80)
            print(pivot.to_string(index=False))
            print("="*80 + "\n")
        except Exception as e:
            print(f"[Results] 生成对比矩阵失败: {e}")
    
    def _generate_per_dataset_summary_from_df(self, df):
        """从 DataFrame 为每个数据集生成模型对比"""
        import pandas as pd
        
        for dataset in df['dataset'].unique():
            try:
                dataset_df = df[df['dataset'] == dataset]
                
                summary = dataset_df.groupby('model').agg({
                    'accuracy': ['mean', 'std', 'count'],
                    'correct': 'sum',
                    'total_samples': 'first',
                }).round(4)
                
                summary.columns = ['acc_mean', 'acc_std', 'num_runs', 'total_correct', 'samples_per_run']
                summary = summary.sort_values('acc_mean', ascending=False)
                
                dataset_dir = os.path.join(self.datasets_dir, dataset)
                os.makedirs(dataset_dir, exist_ok=True)
                summary_path = os.path.join(dataset_dir, "all_models.csv")
                summary.to_csv(summary_path, float_format='%.4f')
                print(f"[Results] 数据集汇总: {summary_path}")
            except Exception as e:
                print(f"[Results] 生成 {dataset} 汇总失败: {e}")
    
    def _generate_per_model_summary_from_df(self, df):
        """从 DataFrame 为每个模型生成汇总"""
        import pandas as pd
        
        for model in df['model'].unique():
            try:
                model_df = df[df['model'] == model]
                
                summary = model_df.groupby('dataset').agg({
                    'accuracy': ['mean', 'std', 'count'],
                    'correct': 'sum',
                }).round(4)
                
                summary.columns = ['acc_mean', 'acc_std', 'num_runs', 'total_correct']
                
                model_dir = os.path.join(self.models_dir, model)
                os.makedirs(model_dir, exist_ok=True)
                summary_path = os.path.join(model_dir, "model_summary.csv")
                summary.to_csv(summary_path, float_format='%.4f')
                print(f"[Results] 模型汇总: {summary_path}")
            except Exception as e:
                print(f"[Results] 生成 {model} 汇总失败: {e}")

    def _generate_comparison_matrix(self):
        """生成 模型×数据集 对比矩阵"""
        import pandas as pd
        
        df = pd.DataFrame(self.all_results)
        
        # 计算每个 (模型, 数据集) 的平均准确率
        pivot = df.pivot_table(
            values='accuracy',
            index='model',
            columns='dataset',
            aggfunc=['mean', 'std', 'count']
        )
        
        # 扁平化列名
        pivot.columns = [f"{col[1]}_{col[0]}" for col in pivot.columns]
        pivot = pivot.reset_index()
        
        # 添加平均值列
        acc_cols = [c for c in pivot.columns if c.endswith('_mean')]
        if acc_cols:
            pivot['avg_accuracy'] = pivot[acc_cols].mean(axis=1)
        
        matrix_path = os.path.join(self.summary_dir, "comparison_matrix.csv")
        pivot.to_csv(matrix_path, index=False, float_format='%.4f')
        print(f"[Results] 对比矩阵: {matrix_path}")
        
        # 打印矩阵
        print("\n" + "="*80)
        print("模型 × 数据集 准确率矩阵")
        print("="*80)
        print(pivot.to_string(index=False))
        print("="*80 + "\n")
    
    def _generate_per_dataset_summary(self):
        """为每个数据集生成模型对比"""
        import pandas as pd
        
        df = pd.DataFrame(self.all_results)
        
        for dataset in df['dataset'].unique():
            dataset_df = df[df['dataset'] == dataset]
            
            summary = dataset_df.groupby('model').agg({
                'accuracy': ['mean', 'std', 'count'],
                'correct': 'sum',
                'total_samples': 'first',
            }).round(4)
            
            summary.columns = ['acc_mean', 'acc_std', 'num_runs', 'total_correct', 'samples_per_run']
            summary = summary.sort_values('acc_mean', ascending=False)
            
            dataset_dir = os.path.join(self.datasets_dir, dataset)
            os.makedirs(dataset_dir, exist_ok=True)
            summary_path = os.path.join(dataset_dir, "all_models.csv")
            summary.to_csv(summary_path, float_format='%.4f')
            print(f"[Results] 数据集汇总: {summary_path}")
    
    def _generate_per_model_summary(self):
        """为每个模型生成汇总"""
        import pandas as pd
        
        df = pd.DataFrame(self.all_results)
        
        for model in df['model'].unique():
            model_df = df[df['model'] == model]
            
            summary = model_df.groupby('dataset').agg({
                'accuracy': ['mean', 'std', 'count'],
                'correct': 'sum',
            }).round(4)
            
            summary.columns = ['acc_mean', 'acc_std', 'num_runs', 'total_correct']
            
            model_dir = os.path.join(self.models_dir, model)
            os.makedirs(model_dir, exist_ok=True)
            summary_path = os.path.join(model_dir, "model_summary.csv")
            summary.to_csv(summary_path, float_format='%.4f')
            print(f"[Results] 模型汇总: {summary_path}")


# ============================================================
# 主评估类
# ============================================================
class MultiDatasetEvaluator:
    """多数据集评估器：一次加载模型，测试多个数据集"""
    
    def __init__(self, model_args, training_args):
        self.model_args = model_args
        self.training_args = training_args
        self.model = None
        self.tokenizer = None
        self.model_name = get_model_name_from_ckpt(model_args.ckpt_dir)
    
    def load_model(self):
        """加载模型（只调用一次）"""
        print("\n" + "="*80)
        print(f"[Model] 加载模型: {self.model_name}")
        print(f"[Model] Base: {self.model_args.model_name_or_path}")
        print(f"[Model] Checkpoint: {self.model_args.ckpt_dir}")
        print("="*80 + "\n")
        
        # 配置 LoRA
        if self.model_args.lora_init:
            if any(name in self.model_args.model_name_or_path.lower() for name in ["llama", "mistral", "falcon", "qwen"]):
                target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj", "gate_proj"]
            elif any(name in self.model_args.model_name_or_path.lower() for name in ["phi"]):
                target_modules = ["q_proj", "k_proj", "v_proj", "dense", "fc1", "fc2"]
            elif any(name in self.model_args.model_name_or_path.lower() for name in ["gpt2"]):
                target_modules = ["c_attn", "c_proj", 'c_fc']
            else:
                raise ValueError(f"Unsupported model: {self.model_args.model_name_or_path}")
            
            lora_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                inference_mode=False,
                r=self.model_args.lora_r,
                lora_alpha=self.model_args.lora_alpha,
                lora_dropout=0.1,
                target_modules=target_modules,
                init_lora_weights=True,
            )
        else:
            raise NotImplementedError("必须使用 LoRA")
        
        # 创建模型
        self.model = CODI(self.model_args, self.training_args, lora_config)
        
        # 加载 checkpoint
        state_dict = load_state_dict_from_ckpt(self.model_args.ckpt_dir, self.model_args.token)
        if state_dict is not None:
            self.model.load_state_dict(state_dict, strict=False)
            print(f"[Model] 已加载 checkpoint")
        else:
            print(f"[Model] 使用基础模型权重")
        
        self.model.codi.tie_weights()
        
        # 加载 tokenizer
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(
            self.model_args.model_name_or_path,
            token=self.model_args.token,
            model_max_length=self.training_args.model_max_length,
            padding_side="left",
            use_fast=False,
        )
        
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
            self.tokenizer.pad_token_id = self.model.pad_token_id or self.tokenizer.convert_tokens_to_ids('[PAD]')
        
        # 移动到 GPU
        self.model = self.model.to('cuda')
        self.model.to(torch.bfloat16)
        self.model.eval()
        
        print(f"[Model] 模型加载完成，已移至 CUDA (bfloat16)")
    
    def evaluate_dataset(self, data_name: str, batch_size: int = 128) -> dict:
        """在单个数据集上评估"""
        print(f"\n{'─'*80}")
        print(f"[Eval] 数据集: {data_name}")
        print(f"{'─'*80}")
        
        # 加载数据集
        test_set, config = load_dataset_by_name(data_name)
        questions, answers = prepare_questions_and_answers(test_set, config)
        
        print(f"[Eval] 样本数: {len(questions)}")
        
        # 准备 batch
        question_data = self._prepare_batches(questions, batch_size)
        
        # 推理
        predictions = self._run_inference(question_data, config["answer_type"])
        
        # 计算准确率
        accuracy = compute_accuracy(answers, predictions)
        
        print(f"[Eval] 准确率: {100*accuracy:.2f}%")
        
        return {
            "predictions": predictions,
            "answers": answers,
            "questions": questions,
            "accuracy": accuracy,
            "config": config,
        }
    
    def _prepare_batches(self, questions: list, batch_size: int) -> list:
        """准备 batch 数据"""
        num_batches = math.ceil(len(questions) / batch_size)
        batches = []
        
        for i in range(num_batches):
            start = i * batch_size
            end = min((i + 1) * batch_size, len(questions))
            
            batch = self.tokenizer(
                questions[start:end],
                return_tensors="pt",
                padding="longest",
            )
            
            # 添加 BOT token
            if self.training_args.remove_eos:
                bot_tensor = torch.tensor([self.model.bot_id], dtype=torch.long).expand(batch["input_ids"].size(0), 1)
            else:
                bot_tensor = torch.tensor([self.tokenizer.eos_token_id, self.model.bot_id], dtype=torch.long).expand(batch["input_ids"].size(0), 2)
            
            batch["input_ids"] = torch.cat((batch["input_ids"], bot_tensor), dim=1)
            batch["attention_mask"] = torch.cat((batch["attention_mask"], torch.ones_like(bot_tensor)), dim=1)
            
            batches.append(batch.to('cuda'))
        
        return batches
    
    def _run_inference(self, question_data: list, answer_type: str) -> list:
        """运行推理"""
        gen_kwargs = {
            "max_new_tokens": 256,
            "temperature": 0.1,
            "top_k": 40,
            "top_p": 0.95,
            "do_sample": True,
        }
        
        all_predictions = []
        
        for step, batch in enumerate(question_data):
            if step % 10 == 0:
                print(f"[Eval] Batch {step+1}/{len(question_data)}")
            
            batch_size = batch["input_ids"].size(0)
            
            with torch.no_grad():
                # 编码问题
                outputs = self.model.codi(
                    input_ids=batch["input_ids"],
                    use_cache=True,
                    output_hidden_states=True,
                    attention_mask=batch["attention_mask"]
                )
                past_key_values = outputs.past_key_values
                latent_embd = outputs.hidden_states[-1][:, -1, :].unsqueeze(1)
                
                if self.training_args.use_prj:
                    latent_embd = self.model.prj(latent_embd)
                
                # Latent iterations
                for _ in range(self.training_args.inf_latent_iterations):
                    outputs = self.model.codi(
                        inputs_embeds=latent_embd,
                        use_cache=True,
                        output_hidden_states=True,
                        past_key_values=past_key_values
                    )
                    past_key_values = outputs.past_key_values
                    latent_embd = outputs.hidden_states[-1][:, -1, :].unsqueeze(1)
                    
                    if self.training_args.use_prj:
                        latent_embd = self.model.prj(latent_embd)
                
                # 添加 EOT token
                if self.training_args.remove_eos:
                    eot_emb = self.model.get_embd(self.model.codi, self.model.model_name)(
                        torch.tensor([self.model.eot_id], dtype=torch.long, device='cuda')
                    ).unsqueeze(0).expand(batch_size, -1, -1)
                else:
                    eot_emb = self.model.get_embd(self.model.codi, self.model.model_name)(
                        torch.tensor([self.model.eot_id, self.tokenizer.eos_token_id], dtype=torch.long, device='cuda')
                    ).unsqueeze(0).expand(batch_size, -1, -1)
                
                # 生成
                output = eot_emb
                finished = torch.zeros(batch_size, dtype=torch.bool, device="cuda")
                pred_tokens = [[] for _ in range(batch_size)]
                
                for _ in range(gen_kwargs["max_new_tokens"]):
                    out = self.model.codi(
                        inputs_embeds=output,
                        use_cache=True,
                        past_key_values=past_key_values
                    )
                    past_key_values = out.past_key_values
                    logits = out.logits[:, -1, :self.model.codi.config.vocab_size-1]
                    
                    if self.training_args.greedy:
                        next_token_ids = torch.argmax(logits, dim=-1)
                    else:
                        logits /= gen_kwargs["temperature"]
                        probs = F.softmax(logits, dim=-1)
                        next_token_ids = torch.multinomial(probs, num_samples=1).squeeze(-1)
                    
                    for b in range(batch_size):
                        if not finished[b]:
                            pred_tokens[b].append(next_token_ids[b].item())
                            if next_token_ids[b] == self.tokenizer.eos_token_id:
                                finished[b] = True
                    
                    if finished.all():
                        break
                    
                    output = self.model.get_embd(self.model.codi, self.model.model_name)(next_token_ids).unsqueeze(1)
                
                # 解码并提取答案
                for tokens in pred_tokens:
                    decoded = self.tokenizer.decode(tokens, skip_special_tokens=True)
                    pred = extract_answer(decoded, answer_type)
                    all_predictions.append(pred)
        
        return all_predictions


# ============================================================
# 扩展的参数类
# ============================================================
@dataclass
class MultiDatasetArgs:
    """多数据集测试参数"""
    datasets: str = field(
        default="gsm8k",
        metadata={"help": "要测试的数据集，空格分隔。如: 'gsm8k svamp gsm-hard'"}
    )
    num_runs: int = field(
        default=1,
        metadata={"help": "每个数据集运行的次数"}
    )
    result_dir: str = field(
        default=None,
        metadata={"help": "结果保存目录"}
    )


# ============================================================
# 主函数
# ============================================================
def main():
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments, MultiDatasetArgs))
    model_args, data_args, training_args, multi_args = parser.parse_args_into_dataclasses()
    
    # 解析数据集列表
    datasets = multi_args.datasets.split()
    print(f"\n{'='*80}")
    print(f"多数据集测试")
    print(f"{'='*80}")
    print(f"数据集: {datasets}")
    print(f"运行次数: {multi_args.num_runs}")
    print(f"{'='*80}\n")
    
    # 初始化结果管理器
    result_dir = multi_args.result_dir or CODI_RESULT_DIR
    results_manager = ResultsManager(result_dir)
    
    # 初始化评估器并加载模型（只加载一次！）
    evaluator = MultiDatasetEvaluator(model_args, training_args)
    evaluator.load_model()

    def _get_start_run_id(model_name: str, dataset: str) -> int:
        """扫描已有 run_* 目录，返回起始 run_id（最大值 + 1）。"""
        dataset_dir = os.path.join(results_manager.models_dir, model_name, dataset)
        if not os.path.isdir(dataset_dir):
            return 0
        pattern = re.compile(r"^run_(\d+)$")
        max_id = -1
        for name in os.listdir(dataset_dir):
            match = pattern.match(name)
            if not match:
                continue
            run_path = os.path.join(dataset_dir, name)
            if os.path.isdir(run_path):
                max_id = max(max_id, int(match.group(1)))
        return max_id + 1
    
    # 在所有数据集上测试
    for dataset in datasets:
        start_run_id = _get_start_run_id(evaluator.model_name, dataset)
        for i in range(multi_args.num_runs):
            run_id = start_run_id + i
            print(f"\n{'='*80}")
            print(f"测试: {dataset} (Run {i + 1}/{multi_args.num_runs}, ID {run_id})")
            print(f"{'='*80}")
            
            try:
                result = evaluator.evaluate_dataset(dataset, data_args.batch_size)
                
                results_manager.save_run_results(
                    model_name=evaluator.model_name,
                    dataset=dataset,
                    run_id=run_id,
                    predictions=result["predictions"],
                    accuracy=result["accuracy"],
                    questions=result["questions"],
                    answers=result["answers"],
                )
            except Exception as e:
                print(f"[Error] 测试 {dataset} 失败: {e}")
                import traceback
                traceback.print_exc()
    
    # 生成汇总报告
    results_manager.finalize()
    
    print(f"\n{'='*80}")
    print("测试完成！")
    print(f"结果目录: {result_dir}")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
