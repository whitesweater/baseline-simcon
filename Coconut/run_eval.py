# Multi-dataset evaluation script for Coconut/COT-SFT models
# Adapted from /data/yhao/Bs/coconut/run_eval.py

import torch
import torch.distributed
import torch.optim as optim
from transformers import AutoModelForCausalLM, AutoTokenizer

from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
import torch.distributed as dist
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from transformers.models.llama.modeling_llama import LlamaDecoderLayer
from transformers.models.gpt2.modeling_gpt2 import GPT2Block

try:
    from transformers.models.qwen2.modeling_qwen2 import Qwen2DecoderLayer
except ImportError:
    Qwen2DecoderLayer = None

try:
    from transformers.models.qwen3.modeling_qwen3 import Qwen3DecoderLayer
except ImportError:
    Qwen3DecoderLayer = None

from coconut import Coconut, CoconutGPT_Same_Word_Embedding
from dataset import (
    get_dataset,
    get_question_latent_dataset,
    get_cot_latent_dataset,
    MyCollator,
)

from tqdm import tqdm
import os, sys
import yaml
import json
import argparse
import functools
import re
from datetime import datetime
from datasets import load_dataset, concatenate_datasets
from utils import Config, set_seed


def dist_ready():
    return dist.is_available() and dist.is_initialized()


def unwrap_model(model):
    return model.module if hasattr(model, "module") else model


DATASET_CONFIGS = {
    'gsm8k':       'data/gsm_test.json',
    'gsm_hard':    'data/gsm-hard_train.json',
    'multi_arith': 'data/multi-arith_test.json',
    'svamp':       'data/svamp_all.json',
    'asdiv':       'data/asdiv_test.json',
    'math500':     'data/math500_test.json',
    'aime':        'data/aime_train.json',
}

DATASET_PRESETS = {
    'basic':  ['gsm8k', 'svamp', 'gsm_hard'],
    'math':   ['gsm8k', 'svamp', 'gsm_hard', 'asdiv'],
    'hard':   ['math500', 'aime'],
    'all':    list(DATASET_CONFIGS.keys()),
}

HF_DATASET_CONFIGS = {
    'gsm_hard': {
        'hf_id': 'juyoung-trl/gsm-hard',
        'split': 'train',
        'question_field': 'instruction',
        'answer_field': 'response',
        'answer_type': 'number',
    },
    'svamp': {
        'hf_id': 'ChilleD/SVAMP',
        'split': 'all',
        'question_field': 'question_concat',
        'answer_field': 'Answer',
        'answer_type': 'number',
    },
    'asdiv': {
        'hf_id': 'EleutherAI/asdiv',
        'split': 'validation',
        'question_field': 'body',
        'question_field_2': 'question',
        'answer_field': 'answer',
        'answer_type': 'number',
    },
    'math500': {
        'hf_id': 'HuggingFaceH4/MATH-500',
        'split': 'test',
        'question_field': 'problem',
        'answer_field': 'answer',
        'answer_type': 'string',
    },
    'aime': {
        'hf_id': 'HuggingFaceH4/aime_2024',
        'split': 'train',
        'question_field': 'problem',
        'answer_field': 'answer',
        'answer_type': 'number',
    },
}


def _normalize_answer_text(ans, answer_type):
    if answer_type == 'string':
        return str(ans).strip()

    ans = str(ans).replace(",", "").strip()
    if "####" in ans:
        ans = ans.split("####")[-1].strip()
    try:
        num = float(ans)
        return str(int(num)) if num == int(num) else str(num)
    except ValueError:
        return ans


def materialize_hf_eval_dataset(dataset_name, output_root):
    if dataset_name not in HF_DATASET_CONFIGS:
        raise ValueError(f"No HF config found for dataset: {dataset_name}")

    config = HF_DATASET_CONFIGS[dataset_name]
    dataset = load_dataset(config['hf_id'])
    if config['split'] == 'all':
        split_dataset = concatenate_datasets([dataset['train'], dataset['test']])
    else:
        split_dataset = dataset[config['split']]

    os.makedirs(output_root, exist_ok=True)
    out_path = os.path.join(output_root, f"{dataset_name}_{config['split']}.json")

    rows = []
    for ex in split_dataset:
        question = str(ex[config['question_field']]).strip()
        if config.get('question_field_2'):
            question = question + " " + str(ex[config['question_field_2']]).strip()
        answer = _normalize_answer_text(ex[config['answer_field']], config['answer_type'])
        rows.append({
            'question': question,
            'steps': [],
            'answer': answer,
        })

    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)
    return out_path


def expand_datasets(dataset_args):
    if dataset_args is None:
        return None
    datasets = []
    for d in dataset_args:
        if d in DATASET_PRESETS:
            for name in DATASET_PRESETS[d]:
                if name in DATASET_CONFIGS:
                    datasets.append((name, DATASET_CONFIGS[name]))
        elif d in DATASET_CONFIGS:
            datasets.append((d, DATASET_CONFIGS[d]))
        elif os.path.exists(d):
            name = os.path.splitext(os.path.basename(d))[0]
            datasets.append((name, d))
        else:
            print(f"Warning: Unknown dataset '{d}', skipping")
    seen = set()
    unique = []
    for item in datasets:
        if item[1] not in seen:
            seen.add(item[1])
            unique.append(item)
    return unique


def normalize_eval_answer(answer):
    text = str(answer).replace(",", "").strip()
    if "####" in text:
        text = text.split("####")[-1].strip()
    return text


def extract_last_number(text):
    matches = re.findall(r"[-+]?(?:\d+\.\d+|\d+|\.\d+)", text.replace(",", ""))
    if not matches:
        return text.replace(",", "").strip()
    value = matches[-1]
    sign = ""
    if value.startswith(("+", "-")):
        sign = "-" if value[0] == "-" else ""
        value = value[1:]
    if "." in value:
        left, right = value.split(".", 1)
        left = left.lstrip("0") or "0"
        right = right.rstrip("0")
        normalized = f"{left}.{right}" if right else left
    else:
        normalized = value.lstrip("0") or "0"
    if normalized == "0":
        return "0"
    return sign + normalized


def extract_answer_output(text, extraction):
    text = text.strip()
    if extraction == "numeric_last":
        for marker in ("####", "###"):
            if marker in text:
                text = text.split(marker)[-1].strip()
                break
        boxed = re.findall(r"\\boxed\{([^{}]+)\}", text)
        if boxed:
            text = boxed[-1]
        return extract_last_number(text)
    return text.split("#")[-1].replace(",", "").strip()


def format_eval_question(question, prompt_template):
    question = str(question).strip()
    if prompt_template in (None, "", "plain", "coconut"):
        return question
    if prompt_template == "answer":
        return f"Question: {question}\nAnswer:"
    if prompt_template == "step_by_step":
        return f"Question: {question}\nLet's think step by step."
    if prompt_template == "hash_final":
        return (
            f"Question: {question}\n"
            "Let's think step by step. At the end, write the final answer after ####."
        )
    raise ValueError(f"Unknown eval_prompt_template: {prompt_template}")


def evaluate_single_dataset(
    parallel_model, tokenizer, configs, dataset_path, dataset_name,
    latent_id, start_id, end_id, collator, rank, world_size,
    scheduled_stage=0
):
    with open(dataset_path) as f:
        data = json.load(f)

    eval_max_size = getattr(configs, 'eval_max_size', None)
    if eval_max_size:
        data = data[: int(eval_max_size)]

    prompt_template = getattr(configs, 'eval_prompt_template', 'plain')
    tokenization_path = dataset_path
    if prompt_template not in (None, "", "plain", "coconut") or eval_max_size:
        cache_root = os.path.join(configs.save_path, configs.name, "_prompt_cache")
        os.makedirs(cache_root, exist_ok=True)
        safe_dataset = re.sub(r"[^A-Za-z0-9_.-]+", "_", dataset_name)
        safe_prompt = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(prompt_template or "plain"))
        tokenization_path = os.path.join(
            cache_root,
            f"{safe_dataset}_{safe_prompt}_{len(data)}.json",
        )
        if rank == 0:
            formatted_data = []
            for row in data:
                formatted = dict(row)
                formatted["question"] = format_eval_question(
                    row["question"],
                    prompt_template,
                )
                formatted_data.append(formatted)
            with open(tokenization_path, "w", encoding="utf-8") as out_f:
                json.dump(formatted_data, out_f, ensure_ascii=False, indent=2)
        if dist_ready():
            dist.barrier()

    question_val = [d["question"] for d in data]
    answers_val = [normalize_eval_answer(d["answer"]) for d in data]
    cot_val = ["\n".join(d.get("steps", [])) for d in data]

    base_dataset_valid = get_dataset(
        tokenization_path, tokenizer, max_size=32 if configs.debug else 100000000
    )

    if "math500" in dataset_path.lower() or "aime" in dataset_path.lower():
        max_new_tokens = 256
    elif "gsm" in dataset_path.lower() or "arith" in dataset_path.lower() or "svamp" in dataset_path.lower():
        max_new_tokens = 128
    else:
        max_new_tokens = 128

    dataset_gen_val = get_question_latent_dataset(
        scheduled_stage,
        base_dataset_valid,
        configs,
        start_id,
        latent_id,
        end_id,
        no_special_marker=configs.cot or configs.no_cot or configs.no_thoughts,
    )

    if world_size > 1:
        rank_indices = list(range(rank, len(dataset_gen_val), world_size))
        eval_dataset = torch.utils.data.Subset(dataset_gen_val, rank_indices)
    else:
        eval_dataset = dataset_gen_val

    def left_pad_generation_collator(features):
        max_length = max(len(feature["input_ids"]) for feature in features)
        input_ids = []
        attention_mask = []
        for feature in features:
            pad_len = max_length - len(feature["input_ids"])
            input_ids.append([tokenizer.pad_token_id] * pad_len + feature["input_ids"])
            attention_mask.append([0] * pad_len + feature["attention_mask"])
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "idx": torch.tensor([feature["idx"] for feature in features], dtype=torch.long),
        }

    eval_batch_size = getattr(configs, 'eval_batch_size', 1)
    eval_collator = left_pad_generation_collator if eval_batch_size > 1 else collator
    valid_gen_dataloader = torch.utils.data.DataLoader(
        eval_dataset,
        num_workers=1,
        pin_memory=True,
        batch_size=eval_batch_size,
        collate_fn=eval_collator,
    )

    total_length = len(eval_dataset)
    pbar = tqdm(
        colour="blue", desc=f"{dataset_name}", total=total_length,
        dynamic_ncols=True, disable=(rank != 0)
    )

    cor = torch.tensor(0, device=rank)
    cor_cot = torch.tensor(0, device=rank)
    total = torch.tensor(0, device=rank)

    with torch.no_grad():
        generation_model = unwrap_model(parallel_model)
        generation_model.eval()
        model_config = getattr(generation_model, "config", None)
        model_max_length = None
        for attr in ("max_position_embeddings", "n_positions"):
            value = getattr(model_config, attr, None)
            if isinstance(value, int) and value > 0:
                model_max_length = value
                break
        for idx, batch in enumerate(valid_gen_dataloader):
            test_indices = batch["idx"].detach().cpu().tolist()

            batch = {
                k: v.to(rank)
                for k, v in batch.items()
                if v is not None and k not in ["idx", "position_ids"]
            }

            batch_count = len(test_indices)
            total += batch_count

            effective_max_new_tokens = max_new_tokens
            if model_max_length is not None:
                input_len = int(batch["input_ids"].shape[-1])
                if input_len >= model_max_length:
                    keep_len = max(1, model_max_length - 1)
                    for key in ("input_ids", "attention_mask"):
                        if key in batch and batch[key].dim() == 2:
                            batch[key] = batch[key][:, -keep_len:]
                    input_len = int(batch["input_ids"].shape[-1])
                effective_max_new_tokens = max(
                    1, min(max_new_tokens, model_max_length - input_len)
                )

            outputs = generation_model.generate(
                **batch,
                max_new_tokens=effective_max_new_tokens,
                synced_gpus=not configs.only_eval,
            )

            if getattr(configs, 'decode_generated_only', False):
                prompt_width = int(batch["input_ids"].shape[-1])
                decode_tokens = outputs[:, prompt_width:]
            else:
                decode_tokens = outputs
            text_outputs = tokenizer.batch_decode(decode_tokens, skip_special_tokens=True)
            for offset, (test_idx, text_output) in enumerate(zip(test_indices, text_outputs)):
                answer = answers_val[test_idx]
                answer_cot = cot_val[test_idx]
                answer_output = extract_answer_output(
                    text_output,
                    getattr(configs, 'answer_extraction', 'hash'),
                )
                cot_output = ("\n".join(text_output.split("\n")[1:])).split("#")[0].strip()

                if idx == 0 and offset < 3 and rank == 0:
                    print(f"\n[{dataset_name}] Example {offset+1}:")
                    print(f"  Question: {question_val[test_idx][:100]}...")
                    print(f"  Expected: '{answer}'")
                    print(f"  Got:      '{answer_output}'")

                cor += (answer_output == answer)
                cor_cot += (cot_output == answer_cot)

            pbar.update(batch_count)
            acc = float(cor.detach().float() / total.detach().float())
            pbar.set_description(f"{dataset_name} Acc: {acc:.4f}")

    pbar.close()

    if dist_ready():
        dist.all_reduce(cor, op=dist.ReduceOp.SUM)
        dist.all_reduce(cor_cot, op=dist.ReduceOp.SUM)
        dist.all_reduce(total, op=dist.ReduceOp.SUM)

    return {
        'dataset': dataset_name,
        'dataset_path': dataset_path,
        'eval_prompt_template': prompt_template,
        'eval_max_size': eval_max_size,
        'accuracy': cor.item() / total.item() if total.item() > 0 else 0,
        'correct': int(cor.item()),
        'total': int(total.item()),
        'cot_exact_match': cor_cot.item() / total.item() if total.item() > 0 else 0,
    }


def main():
    parser = argparse.ArgumentParser(description="Multi-dataset evaluation")
    parser.add_argument("config_file")
    parser.add_argument("--datasets", "-d", nargs="*", default=None,
                        help="Dataset names or presets: basic, math, hard, all. "
                             "Default: uses val_path from config.")
    args = parser.parse_args()

    if {"LOCAL_RANK", "RANK", "WORLD_SIZE"} <= set(os.environ):
        dist.init_process_group("nccl")
        local_rank = int(os.environ["LOCAL_RANK"])
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
    else:
        local_rank = 0
        rank = 0
        world_size = 1
    torch.cuda.set_device(local_rank)

    with open(args.config_file) as f:
        config_dict = yaml.safe_load(f)

    if rank == 0:
        print("Config:", config_dict)

    configs = Config(config_dict)
    set_seed(configs.seed)
    save_dir = os.path.join(configs.save_path, configs.name)
    os.makedirs(save_dir, exist_ok=True)

    model = AutoModelForCausalLM.from_pretrained(configs.model_id)
    tokenizer = AutoTokenizer.from_pretrained(configs.model_id)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.add_tokens("<|start-latent|>")
    tokenizer.add_tokens("<|end-latent|>")
    tokenizer.add_tokens("<|latent|>")
    latent_id = tokenizer.convert_tokens_to_ids("<|latent|>")
    start_id = tokenizer.convert_tokens_to_ids("<|start-latent|>")
    end_id = tokenizer.convert_tokens_to_ids("<|end-latent|>")

    if configs.load_model_path not in (None, "None", "null"):
        saved_weights = torch.load(
            configs.load_model_path, map_location=torch.device(rank)
        )
        if configs.coconut and not any(
            k.startswith("base_causallm") for k in saved_weights.keys()
        ):
            print(model.load_state_dict(saved_weights, strict=False))
        elif not configs.coconut:
            print(model.load_state_dict(saved_weights, strict=False))
        # coconut loading from coconut checkpoint handled below

    if not (configs.cot or getattr(configs, 'no_thoughts', False) or getattr(configs, 'no_cot', False)):
        model.resize_token_embeddings(len(tokenizer))
        embeddings = model.get_input_embeddings()
        target_id = tokenizer.convert_tokens_to_ids("<<")
        for token_id in [latent_id, start_id, end_id]:
            target_embedding = embeddings.weight.data[target_id]
            embeddings.weight.data[token_id] = target_embedding
            lm_head = model.lm_head
            lm_head.weight.data[token_id] = lm_head.weight.data[target_id]

    if getattr(configs, 'no_thoughts', False):
        configs.c_thought = 0
        configs.coconut = False

    if configs.coconut:
        if getattr(configs, 'mode', '') == 'coconutgpt_same_word_embedding':
            model = CoconutGPT_Same_Word_Embedding(model, latent_id, start_id, end_id, tokenizer.eos_token_id)
        else:
            model = Coconut(model, latent_id, start_id, end_id, tokenizer.eos_token_id)

    if configs.load_model_path not in (None, "None", "null"):
        if configs.coconut and any(
            k.startswith("base_causallm") for k in saved_weights.keys()
        ):
            print(model.load_state_dict(saved_weights, strict=False))

    model = model.to(rank)

    # Build auto-wrap policy for FSDP
    wrap_cls = {LlamaDecoderLayer}
    if Qwen2DecoderLayer is not None:
        wrap_cls.add(Qwen2DecoderLayer)
    if Qwen3DecoderLayer is not None:
        wrap_cls.add(Qwen3DecoderLayer)

    auto_wrap_policy = functools.partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls=wrap_cls,
    )

    if getattr(configs, 'bf16', False):
        model.to(torch.bfloat16)

    # Eval uses plain single-process mode when launched without torchrun.
    parallel_model = DDP(model, device_ids=[rank]) if world_size > 1 else model
    del model

    if rank == 0:
        print(parallel_model)

    # Determine datasets
    eval_datasets = expand_datasets(args.datasets)
    if eval_datasets is None:
        eval_datasets = [("gsm8k", configs.val_path)]

    if rank == 0:
        print(f"\nDatasets to evaluate: {[d[0] for d in eval_datasets]}")

    collator = MyCollator(tokenizer, latent_id=latent_id, label_pad_token_id=-100)
    scheduled_stage = 0 if (configs.cot or getattr(configs, 'no_cot', False)) \
        else getattr(configs, 'resume', 0) // getattr(configs, 'epochs_per_stage', 1)

    all_results = []

    for dataset_name, dataset_path in eval_datasets:
        force_hf = os.environ.get("COCONUT_FORCE_HF", "").lower() in {"1", "true", "yes"}
        if force_hf and dataset_name in HF_DATASET_CONFIGS:
            hf_cache_root = os.path.join("data", "hf_eval_cache")
            dataset_path = materialize_hf_eval_dataset(dataset_name, hf_cache_root)
            if rank == 0:
                print(f"[HF] Materialized {dataset_name} to {dataset_path}")

        if not os.path.exists(dataset_path):
            if rank == 0:
                if dataset_name in HF_DATASET_CONFIGS:
                    hf_cache_root = os.path.join("data", "hf_eval_cache")
                    dataset_path = materialize_hf_eval_dataset(dataset_name, hf_cache_root)
                    print(f"[HF] Local file missing; materialized {dataset_name} to {dataset_path}")
                else:
                    print(f"Warning: {dataset_path} not found, skipping {dataset_name}")
                    continue
            else:
                if dataset_name not in HF_DATASET_CONFIGS:
                    continue
                hf_cache_root = os.path.join("data", "hf_eval_cache")
                dataset_path = materialize_hf_eval_dataset(dataset_name, hf_cache_root)

        if rank == 0:
            print(f"\n{'#'*60}")
            print(f"# Evaluating: {dataset_name}")
            print(f"# Path: {dataset_path}")
            print(f"{'#'*60}")

        result = evaluate_single_dataset(
            parallel_model, tokenizer, configs, dataset_path, dataset_name,
            latent_id, start_id, end_id, collator, rank, world_size,
            scheduled_stage=scheduled_stage
        )
        all_results.append(result)

        if rank == 0:
            print(f"\n>>> {dataset_name}: {result['accuracy']:.4f} ({result['correct']}/{result['total']})")

    # Summary
    if rank == 0 and all_results:
        print(f"\n{'='*70}")
        print("MULTI-DATASET EVALUATION SUMMARY")
        print(f"{'='*70}")
        print(f"Model: {configs.load_model_path}")
        print(f"{'='*70}")
        print(f"{'Dataset':<20} {'Accuracy':>10} {'Correct':>10} {'Total':>10}")
        print(f"{'-'*70}")

        for r in all_results:
            print(f"{r['dataset']:<20} {r['accuracy']:>10.4f} {r['correct']:>10} {r['total']:>10}")

        print(f"{'='*70}")
        avg_acc = sum(r['accuracy'] for r in all_results) / len(all_results)
        print(f"{'Average':<20} {avg_acc:>10.4f}")

        results_file = os.path.join(
            save_dir,
            f"multi_eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )
        with open(results_file, 'w') as f:
            json.dump({
                'model': configs.load_model_path,
                'model_id': configs.model_id,
                'mode': 'coconut' if configs.coconut else 'cot',
                'answer_extraction': getattr(configs, 'answer_extraction', 'hash'),
                'decode_generated_only': getattr(configs, 'decode_generated_only', False),
                'eval_batch_size': getattr(configs, 'eval_batch_size', 1),
                'eval_prompt_template': getattr(configs, 'eval_prompt_template', 'plain'),
                'eval_max_size': getattr(configs, 'eval_max_size', None),
                'timestamp': datetime.now().isoformat(),
                'results': all_results,
                'average_accuracy': avg_acc,
            }, f, indent=2)
        print(f"\nResults saved to: {results_file}")

    if dist_ready():
        dist.barrier()


if __name__ == "__main__":
    main()
