#!/usr/bin/env python3
"""
从已保存的 metrics.json 文件重建 all_results.csv 和汇总报告
"""
import os
import json
import glob
import pandas as pd
from datetime import datetime

def rebuild_summary(results_base="results"):
    """从 models/ 目录下的 metrics.json 文件重建汇总"""
    
    models_dir = os.path.join(results_base, "models")
    summary_dir = os.path.join(results_base, "summary")
    datasets_dir = os.path.join(results_base, "datasets")
    
    os.makedirs(summary_dir, exist_ok=True)
    os.makedirs(datasets_dir, exist_ok=True)
    
    # 查找所有 metrics.json 文件
    pattern = os.path.join(models_dir, "*", "*", "run_*", "metrics.json")
    metrics_files = glob.glob(pattern)
    
    print(f"找到 {len(metrics_files)} 个 metrics.json 文件")
    
    if not metrics_files:
        print("没有找到任何结果文件")
        return
    
    # 收集所有结果
    all_results = []
    for filepath in metrics_files:
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
            
            # 从路径解析信息
            # results/models/{model}/{dataset}/run_{i}/metrics.json
            parts = filepath.split(os.sep)
            run_idx = parts.index("run_0") if "run_0" in parts else -2
            
            # 向上解析
            for i, part in enumerate(parts):
                if part == "models" and i + 3 < len(parts):
                    model = parts[i + 1]
                    dataset = parts[i + 2]
                    run_part = parts[i + 3]
                    run_id = int(run_part.replace("run_", ""))
                    break
            else:
                # 备用解析
                model = data.get("model", "unknown")
                dataset = data.get("dataset", "unknown")
                run_id = data.get("run_id", 0)
            
            all_results.append({
                "model": model,
                "dataset": dataset,
                "run_id": run_id,
                "accuracy": data.get("accuracy", 0),
                "correct": data.get("correct", 0),
                "total_samples": data.get("total", data.get("total_samples", 0)),
                "timestamp": data.get("timestamp", ""),
            })
            
        except Exception as e:
            print(f"读取 {filepath} 失败: {e}")
    
    if not all_results:
        print("没有有效的结果数据")
        return
    
    # 创建 DataFrame
    df = pd.DataFrame(all_results)
    
    # 保存 all_results.csv
    all_results_path = os.path.join(summary_dir, "all_results.csv")
    df.to_csv(all_results_path, index=False, float_format='%.4f')
    print(f"\n保存全部结果到: {all_results_path}")
    print(f"共 {len(df)} 条记录")
    
    # 生成对比矩阵
    print("\n" + "="*80)
    print("模型 × 数据集 准确率矩阵")
    print("="*80)
    
    pivot = df.pivot_table(
        values='accuracy',
        index='model',
        columns='dataset',
        aggfunc=['mean', 'std']
    )
    
    # 扁平化列名
    pivot.columns = [f"{col[1]}_{col[0]}" for col in pivot.columns]
    pivot = pivot.reset_index()
    
    # 计算平均准确率
    mean_cols = [c for c in pivot.columns if c.endswith('_mean')]
    if mean_cols:
        pivot['avg_accuracy'] = pivot[mean_cols].mean(axis=1)
        pivot = pivot.sort_values('avg_accuracy', ascending=False)
    
    matrix_path = os.path.join(summary_dir, "comparison_matrix.csv")
    pivot.to_csv(matrix_path, index=False, float_format='%.4f')
    print(f"保存对比矩阵到: {matrix_path}\n")
    
    # 打印矩阵
    print(pivot.to_string(index=False))
    print("="*80)
    
    # 为每个数据集生成汇总
    print("\n各数据集模型排名:")
    for dataset in df['dataset'].unique():
        dataset_df = df[df['dataset'] == dataset]
        summary = dataset_df.groupby('model').agg({
            'accuracy': ['mean', 'std', 'count'],
            'correct': 'sum',
        }).round(4)
        summary.columns = ['acc_mean', 'acc_std', 'num_runs', 'total_correct']
        summary = summary.sort_values('acc_mean', ascending=False)
        
        # 保存
        dataset_out_dir = os.path.join(datasets_dir, dataset)
        os.makedirs(dataset_out_dir, exist_ok=True)
        summary.to_csv(os.path.join(dataset_out_dir, "all_models.csv"), float_format='%.4f')
        
        print(f"\n[{dataset}]")
        print(summary.to_string())
    
    # 为每个模型生成汇总
    print("\n\n各模型在不同数据集上的表现:")
    for model in df['model'].unique():
        model_df = df[df['model'] == model]
        summary = model_df.groupby('dataset').agg({
            'accuracy': ['mean', 'std'],
            'correct': 'sum',
        }).round(4)
        summary.columns = ['acc_mean', 'acc_std', 'total_correct']
        
        # 保存
        model_out_dir = os.path.join(models_dir, model)
        os.makedirs(model_out_dir, exist_ok=True)
        summary.to_csv(os.path.join(model_out_dir, "model_summary.csv"), float_format='%.4f')
        
        print(f"\n[{model}]")
        print(summary.to_string())
    
    print("\n\n汇总重建完成!")
    print(f"  - 全部结果: {all_results_path}")
    print(f"  - 对比矩阵: {matrix_path}")
    print(f"  - 数据集汇总: {datasets_dir}/<dataset>/all_models.csv")
    print(f"  - 模型汇总: {models_dir}/<model>/model_summary.csv")


if __name__ == "__main__":
    import sys
    results_base = sys.argv[1] if len(sys.argv) > 1 else "results"
    rebuild_summary(results_base)
