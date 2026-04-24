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
from decimal import Decimal, InvalidOperation
from dataclasses import dataclass, field
from typing import Dict, Optional, List
from datetime import datetime
import time

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
from src.tokenizer_utils import load_tokenizer_with_fallback


# ============================================================
# 环境配置
# ============================================================
CODI_SAVE_DIR = os.environ.get("CODI_SAVE_DIR", "./outputs")
CODI_RESULT_DIR = os.environ.get("CODI_RESULT_DIR", "./results")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

DEFAULT_GSM8K_AUG_HF_ID = "zen-E/GSM8k-Aug"
DEFAULT_GSM8K_AUG_CACHE_DIR = "/data/yhao/hf_datasets_cache"


def get_dataset_load_kwargs(hf_id: str) -> dict:
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    if hf_id != os.environ.get("CODI_GSM8K_AUG_HF_ID", DEFAULT_GSM8K_AUG_HF_ID):
        return {}

    cache_dir = os.environ.get("CODI_GSM8K_AUG_CACHE_DIR", DEFAULT_GSM8K_AUG_CACHE_DIR)
    os.makedirs(cache_dir, exist_ok=True)
    return {"cache_dir": cache_dir}


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
        "local_path": "./local_datasets/multiarith/eval_42.json",  # 优先使用本地数据
        "hf_id": "ChilleD/MultiArith",  # 备用 HF 数据
        "split": "test",
        "local_question_field": "query",
        "local_answer_field": "answer",
        "hf_question_field": "question",
        "hf_answer_field": "final_ans",
        "answer_type": "number",
    },
    "svamp": {
        "local_path": "./local_datasets/svamp/eval_42.json",  # 优先使用本地数据
        "hf_id": "ChilleD/SVAMP",  # 备用 HF 数据
        "split": "test",
        "local_question_field": "query",
        "local_answer_field": "answer",
        "hf_question_field": "question_concat",
        "hf_answer_field": "Answer",
        "expected_local_size": 200,
        "answer_type": "number",
    },
    "commonsense": {
        "hf_id": "zen-E/CommonsenseQA-GPT4omini",
        "split": "validation",
        "question_field": "question",
        "answer_field": "answer",
        "answer_type": "choice",
    },
    # ========== 新增数据集 ==========
    "strategyqa": {
        "hf_id": "ChilleD/StrategyQA",
        "split": "test",
        "question_field": "question",
        "answer_field": "answer",
        "answer_type": "boolean",  # True/False
    },
    "aqua": {
        "hf_id": "deepmind/aqua_rat",
        "split": "test",
        "question_field": "question",  # 需要特殊处理，拼接 options
        "answer_field": "correct",     # A/B/C/D/E
        "answer_type": "choice",
        "has_options": True,           # 标记需要拼接选项
    },
    "asdiv": {
        "hf_id": "EleutherAI/asdiv",
        "split": "validation",
        "question_field": "question",  # 需要特殊处理，拼接 body
        "answer_field": "answer",      # 格式如 "9 (apples)"，需要提取数字
        "answer_type": "number",
        "has_body": True,              # 标记需要拼接 body
    },
    "du": {
        "hf_id": "lukaemon/bbh",
        "hf_config": "date_understanding",  # 子集名称
        "split": "test",
        "question_field": "input",     # 问题已包含选项
        "answer_field": "target",      # 格式如 "(B)"
        "answer_type": "choice",
        "extract_choice_from_paren": True,  # 从 "(X)" 提取 X
    },
    "coin_flip": {
        "local_path": "./local_datasets/coin_flip/eval_42.json",  # 本地 JSON 文件
        "question_field": "query",
        "answer_field": "answer",
        "answer_type": "boolean",  # yes/no
    },
    "math500": {
        "hf_id": "HuggingFaceH4/MATH-500",
        "split": "test",
        "question_field": "problem",
        "answer_field": "answer",
        "answer_type": "math",
    },
    "aime": {
        "hf_id": "HuggingFaceH4/aime_2024",
        "split": "train",
        "question_field": "problem",
        "answer_field": "answer",
        "answer_type": "math",
    },
}

DATASET_ALIASES = {
    "math-500": "math500",
    "math_500": "math500",
    "math500": "math500",
    "aime": "aime",
    "aime24": "aime",
    "aime_2024": "aime",
    "aime-2024": "aime",
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


def canonicalize_dataset_name(data_name: str) -> str:
    """统一用户输入的数据集别名。"""
    key = data_name.strip().lower()
    return DATASET_ALIASES.get(key, key)


def _extract_last_boxed_content(text: str) -> Optional[str]:
    """提取最后一个 \\boxed{...} / \\fbox{...} 的内容，支持简单嵌套。"""
    last_start = -1
    last_prefix_len = 0
    for prefix in ("\\boxed{", "\\fbox{", "\\framebox{"):
        idx = text.rfind(prefix)
        if idx > last_start:
            last_start = idx
            last_prefix_len = len(prefix)
    if last_start == -1:
        return None

    cursor = last_start + last_prefix_len
    depth = 1
    chars = []
    while cursor < len(text):
        ch = text[cursor]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return "".join(chars)
        chars.append(ch)
        cursor += 1
    return None


def _unwrap_simple_latex_wrappers(text: str) -> str:
    """去掉常见的最外层 LaTeX 样式包装。"""
    patterns = [
        r"^\\text\{(.+)\}$",
        r"^\\textbf\{(.+)\}$",
        r"^\\mathbf\{(.+)\}$",
        r"^\\mathrm\{(.+)\}$",
        r"^\\operatorname\{(.+)\}$",
    ]
    changed = True
    while changed:
        changed = False
        for pattern in patterns:
            match = re.fullmatch(pattern, text.strip(), flags=re.DOTALL)
            if match:
                text = match.group(1).strip()
                changed = True
    return text


def _decimal_to_canonical_string(value: Decimal) -> str:
    """稳定地序列化 Decimal，避免把整数末尾的 0 错误裁掉。"""
    text = format(value.normalize(), "f")
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    if text in {"", "-0"}:
        return "0"
    return text


def _canonicalize_numeric_string(text: str) -> Optional[str]:
    """将简单数字/分数字符串规整成统一格式。"""
    candidate = text.strip()
    if not candidate:
        return None

    latex_frac = re.fullmatch(r"\\frac\{(-?\d+)\}\{(-?\d+)\}", candidate)
    slash_frac = re.fullmatch(r"(-?\d+)\s*/\s*(-?\d+)", candidate)
    simple_number = re.fullmatch(r"-?\d+(?:\.\d+)?", candidate)

    try:
        if latex_frac:
            numerator = Decimal(latex_frac.group(1))
            denominator = Decimal(latex_frac.group(2))
            if denominator == 0:
                return None
            value = numerator / denominator
            return _decimal_to_canonical_string(value)
        if slash_frac:
            numerator = Decimal(slash_frac.group(1))
            denominator = Decimal(slash_frac.group(2))
            if denominator == 0:
                return None
            value = numerator / denominator
            return _decimal_to_canonical_string(value)
        if simple_number:
            value = Decimal(candidate)
            return _decimal_to_canonical_string(value)
    except (InvalidOperation, ZeroDivisionError):
        return None
    return None


def normalize_math_answer(text: str) -> str:
    """将数学数据集答案规整成便于比较的形式。"""
    if text is None:
        return ""

    candidate = str(text).strip()
    boxed = _extract_last_boxed_content(candidate)
    if boxed:
        candidate = boxed.strip()

    candidate = candidate.replace("\\left", "").replace("\\right", "")
    candidate = candidate.replace("\\!", "")
    candidate = candidate.replace("\\,", "")
    candidate = candidate.replace("\\displaystyle", "")
    candidate = candidate.replace("$", "")
    candidate = candidate.replace("\\(", "(").replace("\\)", ")")
    candidate = candidate.replace("\\[", "[").replace("\\]", "]")
    candidate = candidate.replace("^\\circ", "")
    candidate = candidate.replace("^{\\circ}", "")
    candidate = _unwrap_simple_latex_wrappers(candidate)
    candidate = candidate.strip().rstrip(".")

    if candidate.startswith("(") and candidate.endswith(")"):
        inner = candidate[1:-1].strip()
        if "," not in inner:
            candidate = inner

    candidate = re.sub(r"\s+", "", candidate).lower()

    numeric = _canonicalize_numeric_string(candidate)
    if numeric is not None:
        return numeric

    return candidate


def answers_match(pred, gold) -> bool:
    """统一的答案比较逻辑，兼容数字/布尔/选择题/数学表达式。"""
    if isinstance(pred, float) and math.isinf(pred):
        return False
    if isinstance(gold, float) and math.isinf(gold):
        return False
    if isinstance(pred, bool) or isinstance(gold, bool):
        return pred == gold
    if isinstance(pred, (int, float)) and isinstance(gold, (int, float)):
        return pred == gold

    pred_norm = normalize_math_answer(str(pred))
    gold_norm = normalize_math_answer(str(gold))
    if pred_norm == gold_norm:
        return True

    pred_num = _canonicalize_numeric_string(pred_norm)
    gold_num = _canonicalize_numeric_string(gold_norm)
    return pred_num is not None and pred_num == gold_num


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
    data_name = canonicalize_dataset_name(data_name)
    if data_name not in DATASET_CONFIGS:
        raise ValueError(f"未知数据集: {data_name}。支持: {list(DATASET_CONFIGS.keys())}")
    
    config = DATASET_CONFIGS[data_name]
    load_kwargs = get_dataset_load_kwargs(config.get("hf_id", ""))

    def resolve_config_for_source(source: str) -> dict:
        resolved = dict(config)
        question_field_key = f"{source}_question_field"
        answer_field_key = f"{source}_answer_field"
        if question_field_key in resolved:
            resolved["question_field"] = resolved[question_field_key]
        if answer_field_key in resolved:
            resolved["answer_field"] = resolved[answer_field_key]
        return resolved

    def validate_local_json(data: list, resolved_config: dict):
        expected_size = resolved_config.get("expected_local_size")
        if expected_size is not None and len(data) != expected_size:
            print(
                f"[Data] 警告: {data_name} 本地评测集预期 {expected_size} 条，"
                f"实际读取到 {len(data)} 条"
            )

        required_fields = [
            resolved_config["question_field"],
            resolved_config["answer_field"],
        ]
        for idx, ex in enumerate(data):
            missing_fields = [field for field in required_fields if field not in ex]
            if missing_fields:
                raise KeyError(
                    f"{data_name} 本地样本 #{idx} 缺少字段 {missing_fields}，"
                    f"可用字段: {sorted(ex.keys())}"
                )
    
    # 优先支持本地 JSON 文件
    if "local_path" in config:
        local_path = config["local_path"]
        if os.path.exists(local_path):
            resolved_config = resolve_config_for_source("local")
            print(f"[Data] 加载本地数据集: {data_name} ({local_path})")
            with open(local_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            validate_local_json(data, resolved_config)
            # 将 list 转换为 Dataset 格式
            from datasets import Dataset
            test_set = Dataset.from_list(data)
            return test_set, resolved_config
        else:
            # 本地文件不存在，回退到 HuggingFace
            print(f"[Data] 本地文件不存在: {local_path}，尝试从 HuggingFace 加载")
            if "hf_id" not in config:
                raise FileNotFoundError(f"本地数据集文件不存在: {local_path}")
    
    hf_config = config.get("hf_config", None)  # 子集名称（如 BBH 的 date_understanding）
    
    # 优先尝试离线模式加载本地缓存，避免网络超时
    def try_load_dataset(hf_id, hf_config=None):
        """先尝试离线加载缓存，失败后再尝试在线加载"""
        try:
            # 首先尝试离线模式（直接使用本地缓存）
            if hf_config:
                return load_dataset(hf_id, hf_config, **load_kwargs)
            else:
                return load_dataset(hf_id, **load_kwargs)
        except Exception as e:
            print(f"[Data] 在线加载失败: {e}")
            print(f"[Data] 尝试强制使用本地缓存...")
            # 设置离线模式环境变量
            old_offline = os.environ.get("HF_DATASETS_OFFLINE", None)
            os.environ["HF_DATASETS_OFFLINE"] = "1"
            try:
                if hf_config:
                    return load_dataset(hf_id, hf_config, **load_kwargs)
                else:
                    return load_dataset(hf_id, **load_kwargs)
            finally:
                # 恢复环境变量
                if old_offline is None:
                    os.environ.pop("HF_DATASETS_OFFLINE", None)
                else:
                    os.environ["HF_DATASETS_OFFLINE"] = old_offline
    
    # 先尝试离线模式加载（跳过网络检查）
    os.environ["HF_DATASETS_OFFLINE"] = "1"
    try:
        if hf_config:
            print(f"[Data] 加载数据集 (离线优先): {data_name} ({config['hf_id']}/{hf_config})")
            dataset = load_dataset(config["hf_id"], hf_config, **load_kwargs)
        else:
            print(f"[Data] 加载数据集 (离线优先): {data_name} ({config['hf_id']})")
            dataset = load_dataset(config["hf_id"], **load_kwargs)
        print(f"[Data] ✓ 从本地缓存加载成功")
    except Exception as e:
        print(f"[Data] 本地缓存不存在，尝试在线加载: {e}")
        os.environ.pop("HF_DATASETS_OFFLINE", None)
        if hf_config:
            dataset = load_dataset(config["hf_id"], hf_config, **load_kwargs)
        else:
            dataset = load_dataset(config["hf_id"], **load_kwargs)
    finally:
        os.environ.pop("HF_DATASETS_OFFLINE", None)

    resolved_config = resolve_config_for_source("hf")

    if resolved_config["split"] == "all":
        # 特殊处理：合并所有 split
        test_set = concatenate_datasets([dataset["train"], dataset["test"]])
    elif resolved_config["split"] in dataset:
        test_set = dataset[resolved_config["split"]]
    else:
        available_splits = list(dataset.keys())
        if len(available_splits) == 1:
            fallback_split = available_splits[0]
            print(
                f"[Data] 配置 split={resolved_config['split']} 不存在，"
                f"回退到唯一 split: {fallback_split}"
            )
            test_set = dataset[fallback_split]
        else:
            raise KeyError(
                f"数据集 {data_name} 不存在 split={resolved_config['split']}，"
                f"可用: {available_splits}"
            )
    
    return test_set, resolved_config


def prepare_questions_and_answers(test_set, config):
    """准备问题和答案"""
    question_field = config["question_field"]
    answer_field = config["answer_field"]
    answer_type = config["answer_type"]
    has_options = config.get("has_options", False)  # AQuA 需要拼接选项
    has_body = config.get("has_body", False)        # ASDiv 需要拼接 body
    extract_choice_from_paren = config.get("extract_choice_from_paren", False)  # DU: "(B)" -> "B"
    
    questions = []
    answers = []
    
    for ex in test_set:
        # 构建问题
        q = ex[question_field].strip().replace('  ', ' ')
        
        # AQuA: 拼接选项到问题中
        if has_options and "options" in ex:
            options_str = "\n".join(ex["options"])
            q = f"{q}\n\nOptions:\n{options_str}"
        
        # ASDiv: 拼接 body 到问题中
        if has_body and "body" in ex:
            body = ex["body"].strip()
            q = f"{body} {q}"
        
        questions.append(q)
        
        # 处理答案
        ans = ex[answer_field]
        
        # DU: 从 "(B)" 提取 "B"
        if extract_choice_from_paren:
            match = re.match(r'\(([A-F])\)', str(ans).strip())
            if match:
                answers.append(match.group(1))
                continue

        if answer_type == "math":
            answers.append(normalize_math_answer(str(ans)))
            continue
        
        # 处理布尔值 (StrategyQA, coin_flip)
        if isinstance(ans, bool):
            answers.append(ans)
            continue
        if ans in ["True", "False"]:
            answers.append(ans == "True")
            continue
        # 处理 yes/no 字符串 (coin_flip)
        if answer_type == "boolean" or str(ans).lower() in ["yes", "no"]:
            answers.append(str(ans).lower() == "yes")
            continue
        
        # 处理选择题 (AQuA, CommonsenseQA)
        if answer_type == "choice" and str(ans).strip() in "ABCDEF":
            answers.append(str(ans).strip())
            continue
        
        # 处理 ASDiv 格式: "9 (apples)" -> 9.0
        if has_body:
            # 提取第一个数字
            match = re.match(r'^(-?\d+\.?\d*)', str(ans).replace(',', ''))
            if match:
                answers.append(float(match.group(1)))
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
def extract_answer(sentence: str, answer_type: str, question: str = None):
    """从生成文本中提取答案
    
    Args:
        sentence: 模型生成的文本
        answer_type: 答案类型 ("number", "choice", "boolean")
        question: 原始问题（用于选择题时从具体答案匹配回选项字母）
    """
    sentence_clean = sentence.replace(',', '')
    
    if answer_type == "choice":
        # 选择题：只提取 A-F 选项字母，如果结果不是选项字母则返回 inf
        pred = sentence_clean.split("The answer is:")[-1].strip()
        if pred and pred[0] in "ABCDEF":
            return pred[0]
        # 尝试从 $\boxed{X}$ 格式提取
        boxed_match = re.search(r'\\boxed\{([A-F])\}', sentence)
        if boxed_match:
            return boxed_match.group(1)
        # 尝试找最后出现的独立选项字母（只匹配大写 A-F）
        choice_match = re.search(r'\b([A-F])\b', sentence_clean)
        if choice_match:
            return choice_match.group(1)
        
        # 未找到有效选项字母，返回 inf
        return float('inf')
    
    elif answer_type == "boolean":
        # StrategyQA: 答案是 Yes/No 或 True/False
        sentence_lower = sentence.lower()
        # 查找最后出现的 yes/no
        last_yes = sentence_lower.rfind('yes')
        last_no = sentence_lower.rfind('no')
        
        if last_yes == -1 and last_no == -1:
            # 回退到 True/False
            if "True" in sentence:
                return True
            elif "False" in sentence:
                return False
            return float('inf')
        
        # 返回最后出现的那个
        if last_yes > last_no:
            return True
        else:
            return False
    elif answer_type == "math":
        candidates = []
        boxed = _extract_last_boxed_content(sentence)
        if boxed:
            candidates.append(boxed)
        for marker in ["The answer is:", "Final answer:", "Answer:"]:
            idx = sentence.lower().rfind(marker.lower())
            if idx != -1:
                candidates.append(sentence[idx + len(marker):])
        lines = [line.strip() for line in sentence.splitlines() if line.strip()]
        if lines:
            candidates.append(lines[-1])
        candidates.append(sentence)

        for candidate in candidates:
            normalized = normalize_math_answer(candidate)
            if normalized:
                return normalized
        return float('inf')
    
    else:
        # 数字答案
        pred = re.findall(r'-?\d+\.?\d*', sentence_clean)
        if not pred:
            return float('inf')
        return float(pred[-1])


def compute_accuracy(gold: list, pred: list):
    """计算准确率"""
    acc = sum(1 for p, g in zip(pred, gold) if answers_match(p, g))
    return acc / len(gold) if gold else 0.0


def compute_batch_radius_stats(latents_TBD: torch.Tensor, training_args) -> dict:
    """计算单个 batch 的 radius 统计信息
    
    Args:
        latents_TBD: [T, B, D] 形状的张量，T=迭代次数, B=batch_size, D=embedding_dim
        training_args: 训练参数
    
    Returns:
        radius 统计字典
    """
    if latents_TBD is None or latents_TBD.numel() == 0:
        return None
    
    try:
        tc = TrajectoryConsistencyLoss(
            space_type=getattr(training_args, 'trajectory_space_type', 'euclidean'),
            radius_threshold=getattr(training_args, 'trajectory_radius_threshold', 2.0),
            curvature=getattr(training_args, 'trajectory_curvature', 1.0),
        )
        stats = tc.compute_stats(latents_TBD)
        return {
            "radius_max": float(stats['radius_max'].item()),
            "radius_mean": float(stats['radius_mean'].item()),
            "violation_rate": float(stats['violation_rate'].item()),
            "threshold": float(stats['radius_threshold'].item()),
            "batch_size": int(latents_TBD.size(1)),
        }
    except Exception as e:
        print(f"[Radius] 统计失败: {e}")
        return None



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
                         trajectory_stats: dict = None,
                         latents: list = None,
                         elapsed_time: float = None,
                         token_stats: dict = None,
                         raw_outputs: list = None):  # DEBUG: 添加原始输出参数
        """保存单次运行结果"""
        run_dir = self.get_run_dir(model_name, dataset, run_id)
        
        # 保存预测结果
        save_json({
            "predictions": predictions,
            "ground_truth": [str(a) for a in answers],
        }, os.path.join(run_dir, "predictions.json"))
        
        # DEBUG: 保存模型原始输出（用于调试 extract_answer）
        if raw_outputs:
            save_json({
                "raw_outputs": raw_outputs,
                "predictions": predictions,
                "ground_truth": [str(a) for a in answers],
            }, os.path.join(run_dir, "raw_outputs.json"))
            print(f"[DEBUG] 保存原始输出: {os.path.join(run_dir, 'raw_outputs.json')}")
        
        # 保存指标
        ms_per_sample = (elapsed_time * 1000 / len(predictions)) if len(predictions) > 0 else 0
        metrics = {
            "model": model_name,
            "dataset": dataset,
            "run_id": run_id,
            "accuracy": accuracy,
            "total_samples": len(predictions),
            "correct": int(accuracy * len(predictions)),
            "elapsed_time_sec": elapsed_time if elapsed_time else 0,
            "ms_per_sample": round(ms_per_sample, 1),
            "avg_output_tokens": token_stats.get("avg_output_tokens", 0) if token_stats else 0,
            "timestamp": self.timestamp,
        }
        save_json(metrics, os.path.join(run_dir, "metrics.json"))
        
        # 保存轨迹统计
        if trajectory_stats:
            save_json(trajectory_stats, os.path.join(run_dir, "trajectory_stats.json"))
        
        # 保存 latent embeddings（每道题的中间 latent）
        if latents:
            latents_path = os.path.join(run_dir, "latents.json")
            latents_data = {
                "description": "Latent embeddings for each question across iterations",
                "num_samples": len(latents),
                "num_iterations": len(latents[0]) if latents and latents[0] else 0,
                "embedding_dim": len(latents[0][0]) if latents and latents[0] and latents[0][0] else 0,
                "latents": latents,  # [num_samples, num_iterations, embedding_dim]
            }
            save_json(latents_data, latents_path)
            print(f"[Results] 保存 latents: {latents_path} ({len(latents)} 样本, 每样本 {len(latents[0]) if latents else 0} 步)")
        
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
            print("="*80)
            
            # 汇总统计
            total_samples = int(df['total_samples'].sum())
            total_time = df['elapsed_time_sec'].sum()
            avg_ms = (total_time * 1000 / total_samples) if total_samples > 0 else 0
            print(f"\n总样本数: {total_samples}条 | 总耗时: {total_time:.1f}s ({total_time/60:.1f}min) | 平均: {avg_ms:.1f}ms/条")
            print("="*80 + "\n")
            
            # 生成 Token 和耗时汇总
            self._generate_token_time_summary_from_df(df)
            
        except Exception as e:
            print(f"[Results] 生成对比矩阵失败: {e}")
    
    def _generate_token_time_summary_from_df(self, df):
        """生成每个模型在每个数据集上的平均回答 Token 和耗时"""
        import pandas as pd
        
        try:
            if 'avg_output_tokens' not in df.columns:
                print("[Results] 没有 avg_output_tokens 数据可汇总")
                return
            
            # 生成 模型×数据集 的平均回答 Token 矩阵
            pivot_tokens = df.pivot_table(
                values='avg_output_tokens',
                index='model',
                columns='dataset',
                aggfunc='mean'
            ).round(1)
            
            pivot_tokens = pivot_tokens.reset_index()
            
            # 添加总平均列
            token_cols = [c for c in pivot_tokens.columns if c != 'model']
            if token_cols:
                pivot_tokens['avg_all'] = pivot_tokens[token_cols].mean(axis=1).round(1)
            
            # 保存到文件
            token_summary_path = os.path.join(self.summary_dir, "avg_output_tokens_matrix.csv")
            pivot_tokens.to_csv(token_summary_path, index=False, float_format='%.1f')
            print(f"[Results] 平均回答Token矩阵: {token_summary_path}")
            
            # 打印矩阵
            print("\n" + "="*80)
            print("模型 × 数据集 平均回答Token数")
            print("="*80)
            print(pivot_tokens.to_string(index=False))
            print("="*80 + "\n")
            
            # 同样生成耗时矩阵
            if 'elapsed_time_sec' in df.columns:
                pivot_time = df.pivot_table(
                    values='elapsed_time_sec',
                    index='model',
                    columns='dataset',
                    aggfunc='mean'
                ).round(1)
                pivot_time = pivot_time.reset_index()
                time_cols = [c for c in pivot_time.columns if c != 'model']
                if time_cols:
                    pivot_time['total_time'] = pivot_time[time_cols].sum(axis=1).round(1)
                
                time_summary_path = os.path.join(self.summary_dir, "elapsed_time_matrix.csv")
                pivot_time.to_csv(time_summary_path, index=False, float_format='%.1f')
                print(f"[Results] 耗时矩阵: {time_summary_path}")
                
                print("\n" + "="*80)
                print("模型 × 数据集 耗时(秒)")
                print("="*80)
                print(pivot_time.to_string(index=False))
                print("="*80 + "\n")
            
        except Exception as e:
            print(f"[Results] 生成 Token/耗时汇总失败: {e}")
    
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

    def attach_external_model(self, model, tokenizer, model_name: str = None):
        """复用一个已经加载好的模型/tokenizer（用于训练中评测，不再二次加载）。

        要求：
        - `model` 是 CODI 实例（DDP 下请先 `.module` 解包），已经 `.to('cuda')`。
        - `tokenizer` 必须是 padding_side='left' 的，否则推理时 BOT token 会被插到 pad 后面。
        """
        self.model = model
        self.tokenizer = tokenizer
        if getattr(tokenizer, "padding_side", None) != "left":
            print(f"[Eval][WARN] tokenizer.padding_side={tokenizer.padding_side!r}, "
                  f"推理通常要求 'left'，否则 BOT token 位置会错。")
        if model_name:
            self.model_name = model_name
        # 切到 eval；不强行改 dtype/device，因为外部模型已经设置好。
        try:
            self.model.eval()
        except Exception:
            pass
        print(f"[Model] 已挂接外部模型: {self.model_name}")
    
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
        self.tokenizer = load_tokenizer_with_fallback(
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
    
    def evaluate_dataset(self, data_name: str, batch_size: int = 128, save_latents: bool = True) -> dict:
        """在单个数据集上评估"""
        print(f"\n{'─'*80}")
        print(f"[Eval] 数据集: {data_name}")
        print(f"{'─'*80}")
        
        start_time = time.time()
        
        # 加载数据集
        test_set, config = load_dataset_by_name(data_name)
        questions, answers = prepare_questions_and_answers(test_set, config)
        
        print(f"[Eval] 样本数: {len(questions)}")
        
        # 准备 batch
        question_data = self._prepare_batches(questions, batch_size)
        
        # 推理（同时收集 latent embeddings 和 token 统计）
        predictions, all_latents, token_stats, raw_outputs, trajectory_stats = self._run_inference(
            question_data, config["answer_type"], 
            save_latents=save_latents, 
            questions=questions,  # 传递原始问题用于选择题答案匹配
            batch_size=batch_size
        )
        
        # 计算准确率
        accuracy = compute_accuracy(answers, predictions)
        
        elapsed_time = time.time() - start_time
        ms_per_sample = (elapsed_time * 1000 / len(answers)) if len(answers) > 0 else 0
        
        print(f"[Eval] 准确率: {100*accuracy:.2f}%")
        print(f"[Eval] 耗时: {elapsed_time:.2f}s ({ms_per_sample:.1f}ms/样本)")
        print(f"[Eval] 平均回答 Token: {token_stats['avg_output_tokens']:.1f}")
        
        return {
            "predictions": predictions,
            "answers": answers,
            "questions": questions,
            "accuracy": accuracy,
            "config": config,
            "latents": all_latents,  # 每道题的 latent embeddings
            "elapsed_time": elapsed_time,
            "token_stats": token_stats,
            "raw_outputs": raw_outputs,  # DEBUG: 模型原始输出
            "trajectory_stats": trajectory_stats,  # radius/轨迹统计信息
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
    
    def _run_inference(self, question_data: list, answer_type: str, save_latents: bool = True, 
                       questions: list = None, batch_size: int = 1) -> tuple:
        """运行推理，同时收集每道题的 latent embeddings 和 token 统计
        
        Args:
            question_data: tokenized batch 数据
            answer_type: 答案类型
            save_latents: 是否保存完整的 latent embeddings
            questions: 原始问题列表（用于选择题答案匹配）
            batch_size: batch 大小
        
        Note: trajectory_stats 在第一个 batch 推理过程中直接计算（利用 GPU 张量）
        """
        gen_kwargs = {
            "max_new_tokens": 256,
            "temperature": 0.1,
            "top_k": 40,
            "top_p": 0.95,
            "do_sample": True,
        }
        
        all_predictions = []
        all_latents = [] if save_latents else None  # 仅在需要时收集
        all_output_tokens = []
        all_raw_outputs = []
        sample_idx = 0
        
        # 累积所有 batch 的 radius 统计
        all_radius_stats = []
        
        for step, batch in enumerate(question_data):
            if step % 10 == 0:
                print(f"[Eval] Batch {step+1}/{len(question_data)}")
            
            cur_batch_size = batch["input_ids"].size(0)
            
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
                
                # 收集 latent embeddings 用于轨迹统计（仅在第一个 batch 收集，直接使用 GPU 张量）
                latent_embeddings_for_stats = [latent_embd.squeeze(1)]  # [B, D]
                
                # 如需保存完整 latents
                if save_latents:
                    batch_latents = [[latent_embd[b, 0, :].cpu().tolist()] for b in range(cur_batch_size)]
                
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
                    
                    # 收集 latent 用于统计
                    latent_embeddings_for_stats.append(latent_embd.squeeze(1))  # [B, D]
                    
                    if save_latents:
                        for b in range(cur_batch_size):
                            batch_latents[b].append(latent_embd[b, 0, :].cpu().tolist())
                
                # 每个 batch 都计算 radius 统计
                if latent_embeddings_for_stats:
                    latents_TBD = torch.stack(latent_embeddings_for_stats, dim=0)  # [T, B, D]
                    batch_radius = compute_batch_radius_stats(latents_TBD, self.training_args)
                    if batch_radius:
                        all_radius_stats.append(batch_radius)
                        print(f"[Radius] batch={step} max={batch_radius['radius_max']:.4f} mean={batch_radius['radius_mean']:.4f} viol={batch_radius['violation_rate']:.4f}")
                
                # 添加 EOT token
                if self.training_args.remove_eos:
                    eot_emb = self.model.get_embd(self.model.codi, self.model.model_name)(
                        torch.tensor([self.model.eot_id], dtype=torch.long, device='cuda')
                    ).unsqueeze(0).expand(cur_batch_size, -1, -1)
                else:
                    eot_emb = self.model.get_embd(self.model.codi, self.model.model_name)(
                        torch.tensor([self.model.eot_id, self.tokenizer.eos_token_id], dtype=torch.long, device='cuda')
                    ).unsqueeze(0).expand(cur_batch_size, -1, -1)
                
                # 生成
                output = eot_emb
                finished = torch.zeros(cur_batch_size, dtype=torch.bool, device="cuda")
                pred_tokens = [[] for _ in range(cur_batch_size)]
                
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
                    
                    for b in range(cur_batch_size):
                        if not finished[b]:
                            pred_tokens[b].append(next_token_ids[b].item())
                            if next_token_ids[b] == self.tokenizer.eos_token_id:
                                finished[b] = True
                    
                    if finished.all():
                        break
                    
                    output = self.model.get_embd(self.model.codi, self.model.model_name)(next_token_ids).unsqueeze(1)
                
                # 解码并提取答案
                for b, tokens in enumerate(pred_tokens):
                    decoded = self.tokenizer.decode(tokens, skip_special_tokens=True)
                    question_text = questions[sample_idx] if questions else None
                    pred = extract_answer(decoded, answer_type, question=question_text)
                    all_predictions.append(pred)
                    if save_latents:
                        all_latents.append(batch_latents[b])
                    all_output_tokens.append(len(tokens))
                    all_raw_outputs.append(decoded)
                    sample_idx += 1
        
        # Token 统计汇总
        token_stats = {
            "output_tokens_per_sample": all_output_tokens,
            "avg_output_tokens": sum(all_output_tokens) / len(all_output_tokens) if all_output_tokens else 0,
        }
        
        # 汇总所有 batch 的 radius 统计
        trajectory_stats = {}
        if all_radius_stats:
            import numpy as np
            all_max = [s["radius_max"] for s in all_radius_stats]
            all_mean = [s["radius_mean"] for s in all_radius_stats]
            all_viol = [s["violation_rate"] for s in all_radius_stats]
            total_samples = sum(s["batch_size"] for s in all_radius_stats)
            
            trajectory_stats = {
                "num_batches": len(all_radius_stats),
                "total_samples": total_samples,
                "threshold": all_radius_stats[0]["threshold"],
                "radius_max": float(np.max(all_max)),
                "radius_mean": float(np.mean(all_mean)),
                "radius_mean_std": float(np.std(all_mean)),
                "violation_rate_mean": float(np.mean(all_viol)),
                "violation_rate_max": float(np.max(all_viol)),
                "per_batch": all_radius_stats,  # 保留每个 batch 的详细数据
            }
            print(f"\n[Radius 汇总] {len(all_radius_stats)} batches, {total_samples} samples")
            print(f"[Radius 汇总] max={trajectory_stats['radius_max']:.4f}, mean={trajectory_stats['radius_mean']:.4f}, viol_rate={trajectory_stats['violation_rate_mean']:.4f}")
        
        return all_predictions, all_latents, token_stats, all_raw_outputs, trajectory_stats


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
    save_latents: bool = field(
        default=True,
        metadata={"help": "是否保存每道题的 latent embeddings（文件可能很大）"}
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
                # 只在 gsm8k 数据集上保存 latents
                should_save_latents = multi_args.save_latents and dataset == "gsm8k"
                # should_save_latents = False # 目前全部不保存 latent，避免占用过多空间
                result = evaluator.evaluate_dataset(dataset, data_args.batch_size, save_latents=should_save_latents)
                
                results_manager.save_run_results(
                    model_name=evaluator.model_name,
                    dataset=dataset,
                    run_id=run_id,
                    predictions=result["predictions"],
                    accuracy=result["accuracy"],
                    questions=result["questions"],
                    answers=result["answers"],
                    trajectory_stats=result.get("trajectory_stats"),  # radius/轨迹统计
                    latents=result.get("latents") if should_save_latents else None,
                    elapsed_time=result.get("elapsed_time"),
                    token_stats=result.get("token_stats"),
                    raw_outputs=result.get("raw_outputs"),  # DEBUG: 传入原始输出
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
