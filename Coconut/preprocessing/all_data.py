# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

import json
import argparse

from datasets import load_dataset, concatenate_datasets


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
        "split": "all",
        "question_field": "question_concat",
        "answer_field": "Answer",
        "answer_type": "number",
    },
}


def load_dataset_by_name(data_name: str):
    if data_name not in DATASET_CONFIGS:
        raise ValueError(
            f"未知数据集: {data_name}。支持: {list(DATASET_CONFIGS.keys())}"
        )

    config = DATASET_CONFIGS[data_name]
    dataset = load_dataset(config["hf_id"])

    if config["split"] == "all":
        test_set = concatenate_datasets([dataset["train"], dataset["test"]])
    else:
        test_set = dataset[config["split"]]

    return test_set, config


def prepare_questions_and_answers(test_set, config):
    question_field = config["question_field"]
    answer_field = config["answer_field"]
    answer_type = config["answer_type"]

    questions = [ex[question_field].strip().replace("  ", " ") for ex in test_set]
    answers = []

    for ex in test_set:
        ans = ex[answer_field]

        if isinstance(ans, bool):
            answers.append(ans)
            continue
        if ans in ["True", "False"]:
            answers.append(ans == "True")
            continue

        if answer_type == "choice" and ans in "ABCDE":
            answers.append(ans)
            continue

        if "####" in str(ans):
            ans = str(ans).split("####")[-1]
        ans = str(ans).replace(",", "")

        try:
            answers.append(float(ans))
        except ValueError:
            answers.append(float("inf"))

    return questions, answers


def main(dataset, split=None, output=None):
    """
    Convert dataset to JSON format with CODI-aligned processing.
    Args:
        dataset (str): The dataset name.
        split (str): The dataset split (must match CODI setting).
        output (str): Output json path.
    """
    test_set, config = load_dataset_by_name(dataset)

    if split is not None and split != config["split"]:
        raise ValueError(
            f"数据集 {dataset} 只支持 split='{config['split']}'，"
            f"当前传入: '{split}'"
        )

    questions, answers = prepare_questions_and_answers(test_set, config)
    data = [
        {
            "question": question,
            "steps": [],
            "answer": answer,
        }
        for question, answer in zip(questions, answers)
    ]

    output_path = output or f"data/{dataset}_{config['split']}.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert dataset to JSON format (CODI-aligned)."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=list(DATASET_CONFIGS.keys()),
        help="数据集名称",
    )
    parser.add_argument(
        "--split",
        type=str,
        default=None,
        help="数据集划分（必须与 CODI 一致）",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="输出 JSON 路径，默认 data/{dataset}_{split}.json",
    )
    args = parser.parse_args()
    main(args.dataset, args.split, args.output)
