#!/usr/bin/env python3
"""
测试多数据集加载功能

用法: python preprocessing/test_dataset_loading.py
"""

import sys
sys.path.append('.')

from transformers import AutoTokenizer
from dataset import get_dataset_by_name, DATASET_PATHS


def test_dataset(dataset_name, split, tokenizer):
    """测试单个数据集加载"""
    print(f"\n{'='*60}")
    print(f"测试: {dataset_name}/{split}")
    print(f"{'='*60}")
    
    try:
        dataset = get_dataset_by_name(dataset_name, split, tokenizer, max_size=10)
        
        print(f"✅ 加载成功")
        print(f"样本数: {len(dataset)}")
        
        if len(dataset) > 0:
            print(f"\n示例样本:")
            sample = dataset[0]
            print(f"  question_tokenized 长度: {len(sample['question_tokenized'])}")
            print(f"  steps_tokenized 数量: {len(sample['steps_tokenized'])}")
            print(f"  answer_tokenized 长度: {len(sample['answer_tokenized'])}")
        
        return True
    except Exception as e:
        print(f"❌ 失败: {e}")
        return False


def main():
    print("\n" + "="*60)
    print("多数据集加载功能测试")
    print("="*60)
    
    # 加载 tokenizer
    print("\n加载 tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    print(f"✅ tokenizer 加载成功")
    
    # 测试所有数据集
    results = {}
    for dataset_name, splits in DATASET_PATHS.items():
        results[dataset_name] = {}
        for split in splits.keys():
            success = test_dataset(dataset_name, split, tokenizer)
            results[dataset_name][split] = success
    
    # 汇总结果
    print("\n" + "="*60)
    print("测试结果汇总")
    print("="*60)
    
    total_tests = 0
    passed_tests = 0
    
    for dataset_name, splits in results.items():
        print(f"\n{dataset_name}:")
        for split, success in splits.items():
            status = "✅" if success else "❌"
            print(f"  {split:10s}: {status}")
            total_tests += 1
            if success:
                passed_tests += 1
    
    print(f"\n{'='*60}")
    print(f"总计: {passed_tests}/{total_tests} 通过")
    print(f"{'='*60}\n")
    
    if passed_tests == total_tests:
        print("🎉 所有测试通过！")
        return 0
    else:
        print("⚠️  部分测试失败，请检查数据文件是否存在")
        print("提示: 运行 bash preprocessing/process_all_datasets.sh")
        return 1


if __name__ == "__main__":
    exit(main())
