#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据集下载脚本 - 支持镜像站和代理
"""
from datasets import load_dataset
import os
import sys

# ============================================================
# 镜像和代理配置
# ============================================================
def setup_environment():
    """配置 HuggingFace 镜像和代理"""
    # HuggingFace 镜像（国内）
    os.environ["HF_ENDPOINT"] = os.environ.get("HF_ENDPOINT", "https://hf-mirror.com")
    
    # 超时设置
    os.environ["HF_HUB_HTTP_TIMEOUT"] = "180"
    
    # 如需代理，取消下面两行注释
    # os.environ["HTTP_PROXY"] = "http://127.0.0.1:3128"
    # os.environ["HTTPS_PROXY"] = "http://127.0.0.1:3128"
    
    print(f"[Config] HF_ENDPOINT: {os.environ.get('HF_ENDPOINT')}")
    print(f"[Config] HTTP_PROXY: {os.environ.get('HTTP_PROXY', 'None')}")

# ============================================================
# 数据集配置
# ============================================================
DATASETS = {
    "gsm8k": {
        "hf_id": "zen-E/GSM8k-Aug",
        "description": "GSM8K 数学问题数据集（增强版）"
    },
    "gsm-hard": {
        "hf_id": "juyoung-trl/gsm-hard",
        "description": "GSM8K 困难版本"
    },
    "multi-arith": {
        "hf_id": "ChilleD/MultiArith",
        "description": "MultiArith 多步算术推理数据集"
    },
    "svamp": {
        "hf_id": "ChilleD/SVAMP",
        "description": "SVAMP 数学推理数据集"
    },
    "commonsense": {
        "hf_id": "zen-E/CommonsenseQA-GPT4omini",
        "description": "CommonsenseQA 常识推理数据集"
    }
}

# ============================================================
# 下载函数
# ============================================================
def download_dataset(dataset_name, cache_dir=None):
    """
    下载指定数据集
    
    Args:
        dataset_name: 数据集名称
        cache_dir: 缓存目录
    """
    if dataset_name not in DATASETS:
        print(f"❌ 未知数据集: {dataset_name}")
        print(f"可用数据集: {', '.join(DATASETS.keys())}")
        return False
    
    config = DATASETS[dataset_name]
    print(f"\n{'='*70}")
    print(f"📦 下载数据集: {dataset_name}")
    print(f"   描述: {config['description']}")
    print(f"   HF ID: {config['hf_id']}")
    print(f"{'='*70}\n")
    
    try:
        dataset = load_dataset(config["hf_id"], cache_dir=cache_dir)
        print(f"\n✅ 下载成功: {dataset_name}")
        print(f"   Splits: {list(dataset.keys())}")
        for split_name, split_data in dataset.items():
            print(f"   - {split_name}: {len(split_data)} 样本")
        return True
    except Exception as e:
        print(f"\n❌ 下载失败: {dataset_name}")
        print(f"   错误: {e}")
        return False

def download_all(cache_dir=None):
    """下载所有数据集"""
    print(f"\n{'='*70}")
    print("📦 批量下载所有数据集")
    print(f"{'='*70}\n")
    
    results = {}
    for dataset_name in DATASETS.keys():
        success = download_dataset(dataset_name, cache_dir)
        results[dataset_name] = success
    
    # 汇总
    print(f"\n{'='*70}")
    print("📊 下载汇总")
    print(f"{'='*70}")
    success_count = sum(results.values())
    total_count = len(results)
    for name, success in results.items():
        status = "✅" if success else "❌"
        print(f"{status} {name}")
    print(f"\n成功: {success_count}/{total_count}")
    print(f"{'='*70}\n")

# ============================================================
# 主程序
# ============================================================
if __name__ == "__main__":
    setup_environment()
    
    # 从环境变量获取缓存目录
    cache_dir = os.environ.get("CODI_CACHE_DIR", None)
    if cache_dir:
        print(f"[Config] Cache dir: {cache_dir}")
        os.makedirs(cache_dir, exist_ok=True)
    
    if len(sys.argv) > 1:
        # 下载指定数据集
        for dataset_name in sys.argv[1:]:
            download_dataset(dataset_name, cache_dir)
    else:
        # 下载所有数据集
        download_all(cache_dir)
