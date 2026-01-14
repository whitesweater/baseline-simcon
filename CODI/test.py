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
from src.trajectory_acceleration import TrajectoryAccelerationLoss
from src.trajectory_action import TrajectoryActionLoss
from src.trajectory_geodesic import TrajectoryGeodesicDeviationLoss

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


def _stack_latents(latent_embeddings_for_consistency):
    if not latent_embeddings_for_consistency:
        return None
    return torch.stack(latent_embeddings_for_consistency, dim=0)  # [T,B,D]


def log_radius_stats(latents_TBD, training_args, step, batch_size, do_print, radius_log_path):
    try:
        tc = TrajectoryConsistencyLoss(
            space_type=training_args.trajectory_space_type,
            radius_threshold=training_args.trajectory_radius_threshold,
            curvature=training_args.trajectory_curvature,
        )
        stats = tc.compute_stats(latents_TBD)
        if do_print:
            print(
                f"[Radius] batch={step} max={stats['radius_max'].item():.4f} "
                f"mean={stats['radius_mean'].item():.4f} "
                f"viol_rate={stats['violation_rate'].item():.4f} "
                f"thr={stats['radius_threshold'].item()}"
            )
        if radius_log_path:
            os.makedirs(os.path.dirname(radius_log_path), exist_ok=True)
            save_jsonl_line(radius_log_path, {
                "batch": int(step),
                "radius_max": float(stats['radius_max'].item()),
                "radius_mean": float(stats['radius_mean'].item()),
                "violation_rate": float(stats['violation_rate'].item()),
                "radius_threshold": float(stats['radius_threshold'].item()),
                "num_latent_steps": int(latents_TBD.size(0)),
                "batch_size": int(batch_size),
            })
    except Exception as e:
        print(f"[Radius] 统计失败: {e}")


def log_accel_stats(latents_TBD, training_args, step, batch_size, do_print, accel_log_path):
    try:
        ta = TrajectoryAccelerationLoss(
            max_acceleration=training_args.trajectory_max_acceleration
        )
        acc_stats = ta.compute_stats(latents_TBD)
        if do_print:
            print(
                f"[Accel] batch={step} max={acc_stats['accel_max'].item():.4f} "
                f"mean={acc_stats['accel_mean'].item():.4f} "
                f"viol_rate={acc_stats['violation_rate'].item():.4f} "
                f"thr={acc_stats['max_acceleration'].item()}"
            )
        if accel_log_path:
            os.makedirs(os.path.dirname(accel_log_path), exist_ok=True)
            save_jsonl_line(accel_log_path, {
                "batch": int(step),
                "accel_max": float(acc_stats['accel_max'].item()),
                "accel_mean": float(acc_stats['accel_mean'].item()),
                "accel_std": float(acc_stats['accel_std'].item()),
                "violation_rate": float(acc_stats['violation_rate'].item()),
                "max_acceleration": float(acc_stats['max_acceleration'].item()),
                "num_latent_steps": int(latents_TBD.size(0)),
                "batch_size": int(batch_size),
            })
    except Exception as e:
        print(f"[Accel] 统计失败: {e}")


def log_action_stats(latents_TBD, training_args, step, batch_size, do_print, action_log_path):
    try:
        ta = TrajectoryActionLoss(
            lambda_energy=training_args.trajectory_action_lambda_energy,
            lambda_length=training_args.trajectory_action_lambda_length,
        )
        action_stats = ta.compute_stats(latents_TBD)
        if do_print:
            print(
                f"[Action] batch={step} total_max={action_stats['total_max'].item():.4f} "
                f"total_mean={action_stats['total_mean'].item():.4f} "
                f"kin_mean={action_stats['kinetic_mean'].item():.4f} "
                f"pot_mean={action_stats['potential_mean'].item():.4f} "
                f"lambda_e={action_stats['lambda_energy'].item()} "
                f"lambda_l={action_stats['lambda_length'].item()}"
            )
        if action_log_path:
            os.makedirs(os.path.dirname(action_log_path), exist_ok=True)
            save_jsonl_line(action_log_path, {
                "batch": int(step),
                "total_max": float(action_stats['total_max'].item()),
                "total_mean": float(action_stats['total_mean'].item()),
                "kinetic_mean": float(action_stats['kinetic_mean'].item()),
                "potential_mean": float(action_stats['potential_mean'].item()),
                "lambda_energy": float(action_stats['lambda_energy'].item()),
                "lambda_length": float(action_stats['lambda_length'].item()),
                "num_latent_steps": int(latents_TBD.size(0)),
                "batch_size": int(batch_size),
            })
    except Exception as e:
        print(f"[Action] 统计失败: {e}")


def log_geodesic_stats(latents_TBD, training_args, step, batch_size, do_print, geodesic_log_path):
    try:
        tg = TrajectoryGeodesicDeviationLoss(
            curvature=training_args.trajectory_curvature
        )
        geo_stats = tg.compute_stats(latents_TBD)
        if do_print:
            print(
                f"[Geodesic] batch={step} dev_max={geo_stats['deviation_max'].item():.4f} "
                f"dev_mean={geo_stats['deviation_mean'].item():.4f}"
            )
        if geodesic_log_path:
            os.makedirs(os.path.dirname(geodesic_log_path), exist_ok=True)
            save_jsonl_line(geodesic_log_path, {
                "batch": int(step),
                "deviation_max": float(geo_stats['deviation_max'].item()),
                "deviation_mean": float(geo_stats['deviation_mean'].item()),
                "num_latent_steps": int(latents_TBD.size(0)),
                "batch_size": int(batch_size),
            })
    except Exception as e:
        print(f"[Geodesic] 统计失败: {e}")

def evaluation(model_args, data_args, training_args):
    print("[Eval] Init model config...")
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
    # import pdb; pdb.set_trace()
    model = CODI(model_args, training_args, lora_config)
    #if "llama" in model_args.model_name_or_path:
    #    model.codi.resize_token_embeddings(128261)
    # import pdb; pdb.set_trace()
    print("[Eval] Loading checkpoint...")
    try:
        state_dict = load_file(os.path.join(model_args.ckpt_dir, "model.safetensors"))
    except Exception:
        state_dict = torch.load(os.path.join(model_args.ckpt_dir, "pytorch_model.bin"))
    
    # new_state_dict = { k.replace("coconut", "codi"): v for k, v in state_dict.items() }
    # torch.save(new_state_dict, "/scratch/prj/inf_multimodal_qa/scratch_tmp/transfer/pytorch_model.bin")
    model.load_state_dict(state_dict, strict=False)
    model.codi.tie_weights()
    
    tokenizer_path = model_args.model_name_or_path 
    # tokenizer_path = '/mnt/shared-storage-user/mllm/shared/weixilin/gpt2'
    print("[Eval] Loading tokenizer...")
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        tokenizer_path,
        token=model_args.token,
        model_max_length=training_args.model_max_length,
        padding_side="left",
        use_fast=False,
    )

    if tokenizer.pad_token_id is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        tokenizer.pad_token_id = model.pad_token_id
        if tokenizer.pad_token_id is None: # error handling
            tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids('[PAD]')

    device = "cuda"
    model = model.to('cuda')
    model.to(torch.bfloat16)
    print("[Eval] Model moved to CUDA and set to bfloat16.")
    
    # 从环境变量获取结果输出路径
    radius_log_path = os.path.join(CODI_RESULT_DIR, f"radius_{data_args.data_name}.jsonl")
    accel_log_path = os.path.join(CODI_RESULT_DIR, f"accel_{data_args.data_name}.jsonl")
    action_log_path = os.path.join(CODI_RESULT_DIR, f"action_{data_args.data_name}.jsonl")
    geodesic_log_path = os.path.join(CODI_RESULT_DIR, f"geodesic_{data_args.data_name}.jsonl")
    ######################
    #      dataset       #
    ######################
    logging.warning("[Eval] Downloading/Loading Data...")
    question_name = "question"
    answer_name = "answer"
    if "gsm-hard" == data_args.data_name:
        # dataset = load_dataset("juyoung-trl/gsm-hard")
        # test_set = dataset['train']
        # question_name = "instruction"
        # answer_name = "response"
        test_set = read_json('/mnt/shared-storage-user/weixilin/MLLM/coconut/data/gsm8k_hard_format.json')
    elif "multi-arith" == data_args.data_name:
        # dataset = load_dataset("ChilleD/MultiArith")
        # test_set = dataset['test']
        # answer_name = "final_ans"
        test_set = read_json('/mnt/shared-storage-user/weixilin/MLLM/coconut/data/multiarith_format.json')
    elif "svamp" == data_args.data_name:
        # dataset = load_dataset("ChilleD/SVAMP")
        # test_set = concatenate_datasets([dataset["train"], dataset["test"]])
        # question_name = "question_concat"
        # answer_name = "Answer"
        test_set = read_json('/mnt/shared-storage-user/weixilin/MLLM/coconut/data/svamp_format.json')
    elif "commonsense" == data_args.data_name:
        dataset = load_dataset("zen-E/CommonsenseQA-GPT4omini")
        test_set = dataset['validation']
    elif "gsm8k" == data_args.data_name:
        #  hf_id = "zen-E/GSM8k-Aug-NL" if "full" in name else "/data/yhao/sim-con/datasets--zen-E--GSM8k-Aug"
        # test_set = load_dataset("/data/yhao/sim-con/datasets--zen-E--GSM8k-Aug")['test']
        # hf_id = "zen-E/GSM8k-Aug-NL" if "full" in name else "zen-E/GSM8k-Aug"
        test_set = load_dataset("zen-E/GSM8k-Aug")["test"]
        # test_set = read_json('/mnt/shared-storage-user/weixilin/MLLM/coconut/data/gsm_test_clean.json')
        # import pdb; pdb.set_trace()
        # print()
    else:
        raise NotImplementedError

    logging.warning("[Eval] Formatting inputs...")
    question = [f"{example[question_name].strip().replace('  ', ' ')}" for example in test_set]
    answer = []

    # get numerical answer
    for example in test_set:
        example = example[answer_name]
        if isinstance(example, bool):
            answer.append(example)
            continue
        if example in ["True", "False"]:
            if example == "True":
                ans = True
            else:
                ans = False
            answer.append(ans)
            continue
        if example in "ABCDE":
            answer.append(example)
            continue
        if "####" in example:
            ans = example.split('####')[-1]
        else:
            ans = example
        ans = ans.replace(',', '')  # handle numbers like 2,000
        try:
            ans = float(ans)
        except ValueError:
            ans = float("inf")
        answer.append(ans)

    logging.warning("[Eval] Tokenizing inputs...")
    eval_step = math.ceil(len(question)/data_args.batch_size)
    logging.warning(f"Total example: {len(question)} | eval batch size: {data_args.batch_size}"
                    f"eval steps: {eval_step}")
    
    question_data = []
    # import pdb; pdb.set_trace()
    print("[Eval] Building batches...")
    for i in range(eval_step):
        if i < eval_step - 1:
            batch = tokenizer(
                question[i*data_args.batch_size: (i+1)*data_args.batch_size],
                return_tensors="pt",
                padding="longest",
            )
        else:
            batch = tokenizer(
                question[i*data_args.batch_size:],
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
    print("[Eval] Starting inference loop...")
    gen_kwargs = {
        "max_new_tokens": 256,
        "temperature":0.1,
        "top_k": 40,
        "top_p": 0.95,
        "do_sample": True,
    }

    ans_pred_list = []
    ans_pred_list_accu_at_n_passes = []
    attention_map_weights = []
    attention_to_latents_against_len_sum = []
    attention_to_latents_against_len_count = []
    #set_seed(42)
    gating_probs_sums = None
    len_cot = []
    model.eval()
    attn_to_latent_list = []
    if model_args.soft_weight:
        embedding_matrix = model.codi.get_base_model().model.embed_tokens.weight.data.to(model.codi.device)
        vocab_center = embedding_matrix.mean(dim=0)
    
    for step, batch in enumerate(question_data):
        if step % 10 == 0:
            print(f"[Eval] Processing batch {step+1}/{len(question_data)}")
        batch_size = batch["input_ids"].size(0)
        with torch.no_grad():
            # encode the question
            past_key_values = None
            outputs = model.codi(input_ids=batch["input_ids"], use_cache=True, output_hidden_states=True, past_key_values=past_key_values, attention_mask=batch["attention_mask"])
            past_key_values = outputs.past_key_values
            latent_embd = outputs.hidden_states[-1][:, -1, :].unsqueeze(1)
            # Collect latent embeddings across iterations for radius stats
            latent_embeddings_for_consistency = []

            if training_args.use_prj:
                latent_embd = model.prj(latent_embd)

            if model_args.soft_weight:
                soft_probs = F.softmax(model.codi.lm_head(latent_embd).to(model.codi.device), dim=-1)
                soft_embeds = (soft_probs.squeeze(dim=1).unsqueeze(dim=-1) * embedding_matrix.unsqueeze(dim=0)).sum(dim=0)

                soft_probs_expanded = soft_probs.squeeze(dim=1)
                soft_embeds = (soft_probs_expanded.unsqueeze(dim=2) * embedding_matrix).sum(dim=1)  # Shape: [128, 2048]
                if training_args.use_prj:
                    soft_embeds = model.prj(soft_embeds)
                latent_embd = latent_embd + model_args.soft_weight * soft_embeds
            # Append first latent after projection/soft-weight
            latent_embeddings_for_consistency.append(latent_embd.squeeze(1))  # [B,D]
            
            inf_latent_iterations = training_args.inf_latent_iterations
            for i in range(inf_latent_iterations):
                # decode the latent embeddings
                outputs = model.codi(inputs_embeds=latent_embd, use_cache=True, output_hidden_states=True, past_key_values=past_key_values)
                past_key_values = outputs.past_key_values
                latent_embd = outputs.hidden_states[-1][:, -1, :].unsqueeze(1)
                
                if training_args.use_prj:
                    latent_embd = model.prj(latent_embd)
                
                if model_args.soft_weight:
                    soft_probs = F.softmax(model.codi.lm_head(latent_embd).to(model.codi.device), dim=-1)
                    soft_embeds = (soft_probs.squeeze(dim=1).unsqueeze(dim=-1) * embedding_matrix.unsqueeze(dim=0)).sum(dim=0)
                    
                    soft_probs_expanded = soft_probs.squeeze(dim=1)
                    soft_embeds = (soft_probs_expanded.unsqueeze(dim=2) * embedding_matrix).sum(dim=1)  # Shape: [128, 2048]
                    # import pdb; pdb.set_trace()
                    if training_args.use_prj:
                        soft_embeds = model.prj(soft_embeds)

                    latent_embd = latent_embd + model_args.soft_weight * soft_embeds
                # Append subsequent latents
                latent_embeddings_for_consistency.append(latent_embd.squeeze(1))

            latents_TBD = _stack_latents(latent_embeddings_for_consistency)
            if latents_TBD is not None:
                log_radius_stats(latents_TBD, training_args, step, batch_size, do_print, radius_log_path)
                log_accel_stats(latents_TBD, training_args, step, batch_size, do_print, accel_log_path)
                log_action_stats(latents_TBD, training_args, step, batch_size, do_print, action_log_path)
                log_geodesic_stats(latents_TBD, training_args, step, batch_size, do_print, geodesic_log_path)

            if training_args.remove_eos:
                eot_emb = model.get_embd(model.codi, model.model_name)(torch.tensor([model.eot_id], dtype=torch.long, device='cuda')).unsqueeze(0).to(device)
            else:
                eot_emb = model.get_embd(model.codi, model.model_name)(torch.tensor([model.eot_id, tokenizer.eos_token_id], dtype=torch.long, device='cuda')).unsqueeze(0).to(device)
            
            eot_emb = eot_emb.expand(batch["input_ids"].size(0), -1, -1)

            output = eot_emb
            
            seq_len = 0
            finished = torch.zeros(batch_size, dtype=torch.bool, device="cuda")  # Track EOS for each sequence
            pred_tokens = [[] for _ in range(batch_size)]
            for i in range(gen_kwargs["max_new_tokens"]):
                seq_len += 1
                # import pdb; pdb.set_trace()
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

                # implement the sampling process
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
                        cumulative_probs = torch.cumsum(F.softmax(sorted_logit, dim=-1), dim=-1)

                        sorted_indices_to_remove = cumulative_probs > gen_kwargs["top_p"]
                        if sorted_indices_to_remove.any():
                            sorted_indices_to_remove = sorted_indices_to_remove.roll(1, dims=-1)
                            sorted_indices_to_remove[:, 0] = False

                        for b in range(logits.size(0)):
                            logits[b, sorted_indices[b, sorted_indices_to_remove[b]]] = -float("inf")
                    
                    probs = F.softmax(logits, dim=-1)
                    next_token_ids = torch.multinomial(probs, num_samples=1).squeeze(-1)

                # Handle EOS for each sequence
                for b in range(batch_size):
                    if not finished[b]:
                        pred_tokens[b].append(next_token_ids[b].item())
                        if next_token_ids[b] == tokenizer.eos_token_id:
                            finished[b] = True

                # Break if all sequences have finished
                if finished.all():
                    break

                #output = model.codi.get_base_model().transformer.wte(next_token_ids).unsqueeze(1).to(device)
                output = model.get_embd(model.codi, model.model_name)(next_token_ids).unsqueeze(1).to(device)

            for mini_step, pred_token in enumerate(pred_tokens):
                # import pdb; pdb.set_trace()
                len_cot.append(len(pred_token))
                decoded_pred = tokenizer.decode(pred_token, skip_special_tokens=True)
                # output_dir=f'/mnt/shared-storage-user/weixilin/MLLM/coconut/codi/inference_tokens_list/{'-'.join(model_args.ckpt_dir.split('/')[5:])}/{data_args.data_name}.jsonl'
                # os.makedirs(os.path.dirname(output_dir), exist_ok=True)
                # save_jsonl_line(output_dir, {'list': tokenizer.encode(cot_output+answer_output, add_special_tokens=False), 'length': len(tokenizer.encode(cot_output+answer_output, add_special_tokens=False))})
                # import pdb; pdb.set_trace()
                # Extract the numbers in sentences 
                if do_print:
                    print(f"Question {step*data_args.batch_size+mini_step} Starts...")
                    print(f"Q: {question[step*data_args.batch_size+mini_step]}")
                    print(decoded_pred)
                    print(f"Question {step*data_args.batch_size+mini_step} Ends")
                    print(f"Prediction={extract_answer_number(decoded_pred, data_args.data_name)}; Groundtruth={answer[step*data_args.batch_size+mini_step]}")
                    print("")
                ans_pred_list.append(extract_answer_number(decoded_pred, data_args.data_name))
    # 保存结果到环境变量指定的目录
    os.makedirs(CODI_RESULT_DIR, exist_ok=True)
    result_json_path = os.path.join(CODI_RESULT_DIR, f"{data_args.data_name}.json")
    write_json({"ans": ans_pred_list}, result_json_path)
    
    accuracy = compute_accuracy(answer, ans_pred_list)

    print(f"adapter: {model_args.adapter_name_or_path} | GSM8K test accuracy: {100*accuracy:.2f}% | ")
    print(f"average length of COT: {sum(len_cot)/len(len_cot)}")
    print(f"Results saved to: {result_json_path}")
    
    if model_args.save_ablation:
        ablation_path = os.path.join(CODI_RESULT_DIR, f"{data_args.data_name}_ablation.jsonl")
        save_jsonl_line(ablation_path, {
            'model_name': '-'.join(model_args.ckpt_dir.split('/')[5:]),
            'data_name': data_args.data_name,
            'soft_weight': model_args.soft_weight,
            'acc.': accuracy
        })
    return 100*accuracy


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
