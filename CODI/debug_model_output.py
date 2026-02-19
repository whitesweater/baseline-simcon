#!/usr/bin/env python3
"""
调试脚本：保存模型在不同数据集上的原始输出文本
用于分析为什么 extract_answer() 无法正确提取答案

运行示例:
python debug_model_output.py \
    --model_name_or_path /data/yhao/sim-con/modelscope/LLM-Research/Llama-3.2-1B-Instruct \
    --ckpt_dir /data/yhao/baseline/CODI/final_use_model_codi_sim_sircl/codi \
    --datasets du commonsense strategyqa aqua \
    --num_samples 10 \
    --output_file debug_outputs.json
"""

import argparse
import json
import os
import sys
import torch
from datasets import load_dataset
from transformers import AutoTokenizer
from peft import PeftModel

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.model import CODI

# 数据集配置（与 test_multi_dataset.py 保持一致）
DATASET_CONFIGS = {
    "gsm8k": {
        "hf_name": "openai/gsm8k",
        "split": "test",
        "question_key": "question",
        "answer_key": "answer",
        "answer_type": "number",
    },
    "multiarith": {
        "hf_name": "ChilleD/MultiArith",
        "split": "test",
        "question_key": "question",
        "answer_key": "final_ans",
        "answer_type": "number",
    },
    "svamp": {
        "hf_name": "ChilleD/SVAMP",
        "split": "test",
        "question_key": "Body",
        "answer_key": "Answer",
        "answer_type": "number",
        "has_question_field": True,
    },
    "asdiv": {
        "hf_name": "EleutherAI/asdiv",
        "split": "test",
        "question_key": "body",
        "answer_key": "answer",
        "answer_type": "number",
        "has_body": True,
    },
    "commonsense": {
        "hf_name": "zen-E/CommonsenseQA-GPT4omini",
        "split": "train",
        "question_key": "question",
        "answer_key": "answerKey",
        "answer_type": "choice",
    },
    "strategyqa": {
        "hf_name": "ChilleD/StrategyQA",
        "split": "test",
        "question_key": "question",
        "answer_key": "answer",
        "answer_type": "boolean",
    },
    "aqua": {
        "hf_name": "deepmind/aqua_rat",
        "split": "test",
        "question_key": "question",
        "answer_key": "correct",
        "answer_type": "choice",
        "has_options": True,
    },
    "du": {
        "hf_name": "lukaemon/bbh",
        "hf_config": "date_understanding",
        "split": "test",
        "question_key": "input",
        "answer_key": "target",
        "answer_type": "choice",
        "extract_choice_from_paren": True,
    },
}


def load_model(model_name_or_path, ckpt_dir, num_latent=8, inf_latent_iterations=8):
    """加载CODI模型"""
    print(f"Loading model from {model_name_or_path}")
    print(f"Loading checkpoint from {ckpt_dir}")
    
    # 加载tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # 加载模型
    model = CODI(
        model_name_or_path=model_name_or_path,
        num_latent=num_latent,
        lora_init=False,
        use_decoder=False,
        use_prj=False,
    )
    
    # 加载LoRA权重
    if os.path.exists(ckpt_dir):
        print(f"Loading LoRA weights from {ckpt_dir}")
        model.model = PeftModel.from_pretrained(model.model, ckpt_dir)
        model.model = model.model.merge_and_unload()
    
    model.eval()
    model.cuda()
    
    return model, tokenizer


def load_dataset_samples(dataset_name, num_samples=10):
    """加载数据集样本"""
    config = DATASET_CONFIGS[dataset_name]
    
    print(f"Loading dataset: {dataset_name} ({config['hf_name']})")
    
    # 加载数据集
    if "hf_config" in config:
        dataset = load_dataset(config["hf_name"], config["hf_config"], split=config["split"])
    else:
        dataset = load_dataset(config["hf_name"], split=config["split"])
    
    # 限制样本数量
    if len(dataset) > num_samples:
        dataset = dataset.select(range(num_samples))
    
    samples = []
    for item in dataset:
        question = item[config["question_key"]]
        answer = item[config["answer_key"]]
        
        # 处理特殊字段
        if config.get("has_question_field"):
            question = f"{item.get('Body', '')} {item.get('Question', '')}"
        if config.get("has_body"):
            question = f"{item.get('body', '')} {item.get('question', '')}"
        if config.get("has_options"):
            options = item.get("options", [])
            if options:
                options_str = " ".join([f"({chr(65+i)}) {opt}" for i, opt in enumerate(options)])
                question = f"{question}\nOptions: {options_str}"
        if config.get("extract_choice_from_paren"):
            # 从 "(B)" 格式提取 "B"
            import re
            match = re.search(r'\(([A-F])\)', str(answer))
            if match:
                answer = match.group(1)
        
        # 处理布尔答案
        if config["answer_type"] == "boolean":
            if isinstance(answer, bool):
                answer = "Yes" if answer else "No"
            elif str(answer).lower() in ["true", "1"]:
                answer = "Yes"
            elif str(answer).lower() in ["false", "0"]:
                answer = "No"
        
        samples.append({
            "question": question.strip(),
            "answer": str(answer).strip(),
            "answer_type": config["answer_type"],
        })
    
    return samples


def generate_response(model, tokenizer, question, inf_latent_iterations=8, max_new_tokens=256):
    """生成模型响应"""
    # 构建输入
    bot_token = "<bot>"
    prompt = f"{question} {bot_token}"
    
    # Tokenize
    inputs = tokenizer(prompt, return_tensors="pt", padding=True)
    input_ids = inputs["input_ids"].cuda()
    attention_mask = inputs["attention_mask"].cuda()
    
    # 获取特殊token ID
    bot_id = tokenizer.convert_tokens_to_ids(bot_token)
    eot_id = tokenizer.convert_tokens_to_ids("<eot>")
    if eot_id is None or eot_id == tokenizer.unk_token_id:
        eot_id = tokenizer.eos_token_id
    
    # 找到 bot token 位置
    bot_positions = (input_ids == bot_id).nonzero(as_tuple=True)
    if len(bot_positions[1]) > 0:
        latent_start_pos = bot_positions[1][0].item()
    else:
        latent_start_pos = input_ids.shape[1] - 1
    
    # 生成隐藏状态（latent iterations）
    with torch.no_grad():
        hidden_states = model.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True
        ).hidden_states[-1]
        
        # 获取 latent position 的 hidden state
        latent_hidden = hidden_states[:, latent_start_pos:latent_start_pos+1, :]
        
        # 进行多次 latent iteration
        for _ in range(inf_latent_iterations):
            outputs = model.model(
                inputs_embeds=latent_hidden,
                output_hidden_states=True
            )
            latent_hidden = outputs.hidden_states[-1]
        
        # 准备生成
        # 将 latent hidden state 拼接回原始输入
        input_embeds = model.model.get_input_embeddings()(input_ids)
        input_embeds[:, latent_start_pos:latent_start_pos+1, :] = latent_hidden
        
        # 使用 generate 生成答案
        generated_ids = model.model.generate(
            inputs_embeds=input_embeds,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    
    # 解码输出
    full_output = tokenizer.decode(generated_ids[0], skip_special_tokens=False)
    new_output = tokenizer.decode(generated_ids[0][input_ids.shape[1]:], skip_special_tokens=True)
    
    return {
        "full_output": full_output,
        "new_tokens_only": new_output.strip(),
        "input_prompt": prompt,
    }


def simple_generate(model, tokenizer, question, max_new_tokens=256):
    """简化的生成方法（不使用latent iteration，用于对比）"""
    prompt = f"Question: {question}\nAnswer:"
    
    inputs = tokenizer(prompt, return_tensors="pt", padding=True)
    input_ids = inputs["input_ids"].cuda()
    attention_mask = inputs["attention_mask"].cuda()
    
    with torch.no_grad():
        generated_ids = model.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    
    full_output = tokenizer.decode(generated_ids[0], skip_special_tokens=False)
    new_output = tokenizer.decode(generated_ids[0][input_ids.shape[1]:], skip_special_tokens=True)
    
    return {
        "full_output": full_output,
        "new_tokens_only": new_output.strip(),
        "input_prompt": prompt,
    }


def main():
    parser = argparse.ArgumentParser(description="Debug model outputs")
    parser.add_argument("--model_name_or_path", type=str, required=True,
                        help="Base model path")
    parser.add_argument("--ckpt_dir", type=str, required=True,
                        help="Checkpoint directory with LoRA weights")
    parser.add_argument("--datasets", type=str, nargs="+", 
                        default=["du", "commonsense", "strategyqa", "aqua"],
                        help="Datasets to test")
    parser.add_argument("--num_samples", type=int, default=10,
                        help="Number of samples per dataset")
    parser.add_argument("--num_latent", type=int, default=8,
                        help="Number of latent tokens")
    parser.add_argument("--inf_latent_iterations", type=int, default=8,
                        help="Number of latent iterations during inference")
    parser.add_argument("--output_file", type=str, default="debug_outputs.json",
                        help="Output file path")
    parser.add_argument("--use_simple_generate", action="store_true",
                        help="Use simple generation without latent iterations")
    
    args = parser.parse_args()
    
    # 加载模型
    model, tokenizer = load_model(
        args.model_name_or_path,
        args.ckpt_dir,
        args.num_latent,
        args.inf_latent_iterations
    )
    
    results = {}
    
    for dataset_name in args.datasets:
        if dataset_name not in DATASET_CONFIGS:
            print(f"Warning: Unknown dataset {dataset_name}, skipping")
            continue
        
        print(f"\n{'='*60}")
        print(f"Testing dataset: {dataset_name}")
        print(f"{'='*60}")
        
        samples = load_dataset_samples(dataset_name, args.num_samples)
        dataset_results = []
        
        for i, sample in enumerate(samples):
            print(f"\n--- Sample {i+1}/{len(samples)} ---")
            print(f"Question: {sample['question'][:200]}...")
            print(f"Expected Answer: {sample['answer']} (type: {sample['answer_type']})")
            
            try:
                if args.use_simple_generate:
                    output = simple_generate(model, tokenizer, sample["question"])
                else:
                    output = generate_response(
                        model, tokenizer, sample["question"],
                        args.inf_latent_iterations
                    )
                
                print(f"Model Output: {output['new_tokens_only'][:500]}")
                
                dataset_results.append({
                    "question": sample["question"],
                    "expected_answer": sample["answer"],
                    "answer_type": sample["answer_type"],
                    "model_output": output["new_tokens_only"],
                    "full_output": output["full_output"],
                    "input_prompt": output["input_prompt"],
                })
            except Exception as e:
                print(f"Error: {e}")
                dataset_results.append({
                    "question": sample["question"],
                    "expected_answer": sample["answer"],
                    "answer_type": sample["answer_type"],
                    "error": str(e),
                })
        
        results[dataset_name] = dataset_results
    
    # 保存结果
    output_path = args.output_file
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"\n{'='*60}")
    print(f"Results saved to: {output_path}")
    print(f"{'='*60}")
    
    # 打印摘要
    print("\n=== Summary ===")
    for dataset_name, dataset_results in results.items():
        print(f"\n{dataset_name}:")
        for i, r in enumerate(dataset_results[:3]):  # 只打印前3个
            if "error" in r:
                print(f"  [{i+1}] ERROR: {r['error']}")
            else:
                print(f"  [{i+1}] Expected: {r['expected_answer']}")
                print(f"       Output: {r['model_output'][:100]}...")


if __name__ == "__main__":
    main()
