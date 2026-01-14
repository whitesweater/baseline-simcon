#    Copyright 2023 Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

import logging
import math
import re
import os
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence

import torch
import transformers
from torch.nn import functional as F
import json
import numpy as np

from peft import PeftModel, LoraConfig, TaskType, get_peft_model
from datasets import load_dataset, concatenate_datasets
from accelerate.utils import set_seed
from safetensors.torch import load_file

from src.model import (
    CODI,
    ModelArguments,
    DataArguments,
    TrainingArguments,
)
from src.trajectory_consistency import TrajectoryConsistencyLoss

# ============================================================
# 环境配置：优先从环境变量读取路径
# ============================================================
CODI_SAVE_DIR = os.environ.get("CODI_SAVE_DIR", "/hpc2hdd/home/yhao481/jhupload/baseline/CODI/outputs")
CODI_RESULT_DIR = os.environ.get("CODI_RESULT_DIR", os.path.join(CODI_SAVE_DIR, "../result"))

do_print = True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

def save_jsonl_line(filepath, data):
    """
    将一条字典数据追加写入到 JSONL 文件中。

    参数:
        filepath (str): 目标 JSONL 文件路径。
        data (dict): 要写入的数据，必须是可序列化为 JSON 的字典。
    """
    if not isinstance(data, dict):
        raise ValueError("data 必须是一个字典")

    with open(filepath, "a", encoding="utf-8") as f:
        json_line = json.dumps(data, ensure_ascii=False)
        f.write(json_line + "\n")

def read_json(file_path):
    """
    从指定路径读取JSON文件并返回对应的Python对象。
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
            return data
    except Exception as e:
        print(f"读取JSON文件时出错: {e}")
        return None

def write_json(data, file_path):
    """
    将Python对象写入指定路径的JSON文件中。
    """
    try:
        with open(file_path, 'w', encoding='utf-8') as file:
            json.dump(data, file, ensure_ascii=False, indent=4)
    except Exception as e:
        print(f"写入JSON文件时出错: {e}")

def _load_eval_set(data_args):
    question_name = "question"
    answer_name = "answer"
    if "gsm-hard" == data_args.data_name:
        test_set = read_json('/mnt/shared-storage-user/weixilin/MLLM/coconut/data/gsm8k_hard_format.json')
    elif "multi-arith" == data_args.data_name:
        test_set = read_json('/mnt/shared-storage-user/weixilin/MLLM/coconut/data/multiarith_format.json')
    elif "svamp" == data_args.data_name:
        test_set = read_json('/mnt/shared-storage-user/weixilin/MLLM/coconut/data/svamp_format.json')
    elif "commonsense" == data_args.data_name:
        dataset = load_dataset("zen-E/CommonsenseQA-GPT4omini")
        test_set = dataset['validation']
    elif "gsm8k" == data_args.data_name:
        test_set = load_dataset("zen-E/GSM8k-Aug")["test"]
    else:
        raise NotImplementedError
    return test_set, question_name, answer_name


def _build_answer_list(test_set, answer_name):
    answer = []
    for example in test_set:
        example = example[answer_name]
        if isinstance(example, bool):
            answer.append(example)
            continue
        if example in ["True", "False"]:
            answer.append(example == "True")
            continue
        if example in "ABCDE":
            answer.append(example)
            continue
        if "####" in example:
            ans = example.split('####')[-1]
        else:
            ans = example
        ans = ans.replace(',', '')
        try:
            ans = float(ans)
        except ValueError:
            ans = float("inf")
        answer.append(ans)
    return answer


def _eval_core(model, tokenizer, model_args, data_args, training_args, do_print_flag=False):
    global do_print
    do_print = do_print_flag

    device = next(model.parameters()).device
    bs = data_args.eval_batch_size or data_args.batch_size

    test_set, question_name, answer_name = _load_eval_set(data_args)
    logging.warning("Formatting inputs...")
    question = [f"{example[question_name].strip().replace('  ', ' ')}" for example in test_set]
    answer = _build_answer_list(test_set, answer_name)

    logging.warning("Tokenizing inputs...")
    eval_step = math.ceil(len(question)/bs)
    logging.warning(f"Total example: {len(question)} | eval batch size: {bs}eval steps: {eval_step}")

    question_data = []
    for i in range(eval_step):
        if i < eval_step - 1:
            batch = tokenizer(
                question[i*bs: (i+1)*bs],
                return_tensors="pt",
                padding="longest",
            )
        else:
            batch = tokenizer(
                question[i*bs:],
                return_tensors="pt",
                padding="longest",
            )
        if training_args.remove_eos:
            bot_tensor = torch.tensor([model.bot_id], dtype=torch.long).expand(batch["input_ids"].size(0), 1)
        else:
            bot_tensor = torch.tensor([tokenizer.eos_token_id, model.bot_id], dtype=torch.long).expand(batch["input_ids"].size(0), 2)
        batch["input_ids"] = torch.cat((batch["input_ids"], bot_tensor), dim=1)
        batch["attention_mask"] = torch.cat((batch["attention_mask"], torch.ones_like(bot_tensor)), dim=1)
        batch['input_len'] = len(batch['input_ids'][0])
        question_data.append(batch.to(device))

    model.eval()
    gen_kwargs = {
        "max_new_tokens": 256,
        "temperature":0.1,
        "top_k": 40,
        "top_p": 0.95,
        "do_sample": True,
    }

    ans_pred_list = []
    len_cot = []

    for step, batch in enumerate(question_data):
        batch_size = batch["input_ids"].size(0)
        with torch.no_grad():
            past_key_values = None
            outputs = model.codi(input_ids=batch["input_ids"], use_cache=True, output_hidden_states=True, past_key_values=past_key_values, attention_mask=batch["attention_mask"])
            past_key_values = outputs.past_key_values
            latent_embd = outputs.hidden_states[-1][:, -1, :].unsqueeze(1)

            if training_args.use_prj:
                latent_embd = model.prj(latent_embd)

            if training_args.remove_eos:
                eot_emb = model.get_embd(model.codi, model.model_name)(torch.tensor([model.eot_id], dtype=torch.long, device=device)).unsqueeze(0).to(device)
            else:
                eot_emb = model.get_embd(model.codi, model.model_name)(torch.tensor([model.eot_id, tokenizer.eos_token_id], dtype=torch.long, device=device)).unsqueeze(0).to(device)
            eot_emb = eot_emb.expand(batch["input_ids"].size(0), -1, -1)

            output = eot_emb
            finished = torch.zeros(batch_size, dtype=torch.bool, device=device)
            pred_tokens = [[] for _ in range(batch_size)]

            for _ in range(gen_kwargs["max_new_tokens"]):
                out = model.codi(
                        inputs_embeds=output,
                        output_hidden_states=False,
                        attention_mask=None,
                        use_cache=True,
                        output_attentions=False,
                        past_key_values=past_key_values
                    )
                past_key_values = out.past_key_values
                logits = out.logits[:, -1, :model.codi.config.vocab_size-1]

                if training_args.greedy:
                    next_token_ids = torch.argmax(logits, dim=-1).squeeze(-1)
                else:
                    logits /= gen_kwargs["temperature"]
                    if gen_kwargs["top_k"] > 1:
                        top_k_values, _ = torch.topk(logits, gen_kwargs["top_k"], dim=-1)
                        min_top_k_value = top_k_values[:, -1].unsqueeze(-1)
                        logits[logits < min_top_k_value] = -float("inf")
                    if gen_kwargs["top_p"] < 1.0:
                        sorted_logit, sorted_indices = torch.sort(logits, descending=True, dim=-1)
                        cumulative_probs = torch.cumsum(torch.nn.functional.softmax(sorted_logit, dim=-1), dim=-1)
                        sorted_indices_to_remove = cumulative_probs > gen_kwargs["top_p"]
                        if sorted_indices_to_remove.any():
                            sorted_indices_to_remove = sorted_indices_to_remove.roll(1, dims=-1)
                            sorted_indices_to_remove[:, 0] = False
                        for b in range(logits.size(0)):
                            logits[b, sorted_indices[b, sorted_indices_to_remove[b]]] = -float("inf")
                    probs = torch.nn.functional.softmax(logits, dim=-1)
                    next_token_ids = torch.multinomial(probs, num_samples=1).squeeze(-1)

                for b in range(batch_size):
                    if not finished[b]:
                        pred_tokens[b].append(next_token_ids[b].item())
                        if next_token_ids[b] == tokenizer.eos_token_id:
                            finished[b] = True
                if finished.all():
                    break

                output = model.get_embd(model.codi, model.model_name)(next_token_ids).unsqueeze(1).to(device)

            for mini_step, pred_token in enumerate(pred_tokens):
                len_cot.append(len(pred_token))
                decoded_pred = tokenizer.decode(pred_token, skip_special_tokens=True)
                if do_print:
                    print(f"Question {step*bs+mini_step} Starts...")
                    print(f"Q: {question[step*bs+mini_step]}")
                    print(decoded_pred)
                    print(f"Question {step*bs+mini_step} Ends")
                    print(f"Prediction={extract_answer_number(decoded_pred, data_args.data_name)}; Groundtruth={answer[step*bs+mini_step]}")
                    print("")
                ans_pred_list.append(extract_answer_number(decoded_pred, data_args.data_name))

    os.makedirs(CODI_RESULT_DIR, exist_ok=True)
    result_json_path = os.path.join(CODI_RESULT_DIR, f"{data_args.data_name}.json")
    write_json({"ans": ans_pred_list}, result_json_path)

    accuracy = compute_accuracy(answer, ans_pred_list)
    print(f"[Eval] {data_args.data_name} accuracy: {100*accuracy:.2f}% | avg COT len: {sum(len_cot)/len(len_cot)}")
    print(f"Results saved to: {result_json_path}")
    return 100*accuracy


def evaluation(model_args, data_args, training_args):
    if model_args.lora_init:
        task_type = TaskType.CAUSAL_LM
        if any(name in model_args.model_name_or_path.lower() for name in ["llama", "mistral", "falcon", "qwen"]):
            target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj", "gate_proj"]
        elif any(name in model_args.model_name_or_path.lower() for name in ["phi"]):
            target_modules = ["q_proj", "k_proj", "v_proj", "dense", "fc1", "fc2"]
        elif any(name in model_args.model_name_or_path.lower() for name in ["gpt2"]):
            target_modules = ["c_attn", "c_proj", 'c_fc']
        else:
            raise ValueError(f"Only support LLAMA, Mistral, Falcon, Phi-2, but got {model_args.model_name_or_path}.")
        lora_config = LoraConfig(
            task_type=task_type,
            inference_mode=False,
            r=model_args.lora_r,
            lora_alpha=model_args.lora_alpha,
            lora_dropout=0.1,
            target_modules=target_modules,
            init_lora_weights=True,
        )
    else:
        raise NotImplementedError

    model = CODI(model_args, training_args, lora_config)
    try:
        state_dict = load_file(os.path.join(model_args.ckpt_dir, "model.safetensors"))
    except Exception:
        state_dict = torch.load(os.path.join(model_args.ckpt_dir, "pytorch_model.bin"))
    model.load_state_dict(state_dict, strict=False)
    model.codi.tie_weights()

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        token=model_args.token,
        model_max_length=training_args.model_max_length,
        padding_side="left",
        use_fast=False,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        tokenizer.pad_token_id = model.pad_token_id
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids('[PAD]')

    model = model.to('cuda').to(torch.bfloat16)
    return _eval_core(model, tokenizer, model_args, data_args, training_args, do_print_flag=do_print)


def evaluation_with_model(model, tokenizer, model_args, data_args, training_args, do_print_flag=False):
    return _eval_core(model, tokenizer, model_args, data_args, training_args, do_print_flag)


def extract_answer_number(sentence: str, data_name: str = "gsm8k") -> float:
    """
    从模型生成的句子中提取答案。
    
    Args:
        sentence: 模型生成的文本
        data_name: 数据集名称，用于确定答案提取策略
    
    Returns:
        提取的答案（数字、字母或布尔值）
    """
    sentence = sentence.replace(',', '')
    pred = [s for s in re.findall(r'-?\d+\.?\d*', sentence)]
    if not pred:
        #   # CommonsenseQA: 答案是 A-E 选项
        # if "commonsense" in data_name:
        #     pred_str = sentence.split("The answer is:")[-1].strip()
        #     if pred_str and pred_str[0] in "ABCDE":
        #         return pred_str[0]
        #     return float('inf')
        # # ProntoQA / Strategy: 答案是布尔值
        # elif "strategy" in data_name or "prontoqa" in data_name.lower():
        
        
        if "commonsense" in data_args.data_name:
            pred = sentence.split("The answer is:")[-1].strip()
            if pred[0] not in "ABCDE":
                raise ValueError
            return pred[0]
        elif "strategy" in data_args.data_name or "prontoqa" in data_args.data_name.lower():
            if "True" in sentence:
                return True
            elif "False" in sentence:
                return False
            else:
                raise ValueError
        return float('inf')

    # use the last number as the answer
    pred_answer = float(pred[-1])

    return pred_answer


def compute_accuracy(gold: list, pred: list):
    acc = 0.0
    for p, g in zip(pred, gold):
        if isinstance(p, list):
            if g in p:
                acc += 1
        else:
            if p == g:
                acc += 1

    return acc / len(gold)


if __name__ == "__main__":
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    accu_list = []
    for i in range(training_args.inf_num_iterations):
        accu = evaluation(model_args, data_args, training_args)
        accu_list.append(accu)
    print(f"Average accuracy over {training_args.inf_num_iterations} sampling: {sum(accu_list)/len(accu_list)}")
