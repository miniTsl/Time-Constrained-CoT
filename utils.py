import os
import json
import random
import json
import os
import numpy as np
from pathlib import Path
from typing import Iterable, Union, Any
from transformers import AutoTokenizer
from examples import get_examples
import torch

def gen_budget_list(budget, data_name):
    if budget <0:
        return [-1]
    else: 
        if data_name == "gsm8k":
            budget_list = []
            for i in range(25, 500, 25):
                budget_list.append(i)
            for i in range(500, 1001, 50):
                budget_list.append(i)
        elif data_name == "math":
            budget_list = []
            for i in range(25, 1000, 25):
                budget_list.append(i)
            for i in range(1000, 1501, 50):
                budget_list.append(i)
        return budget_list


def load_data_with_cropped_cot(full_cot_path, args):
    TERMINATOR="\n\n**Final Answer**\n"
    samples = list(load_jsonl(full_cot_path))
    full_cots = [sample["code"][0] for sample in samples]
    
    # use tokenizer to batch crop full_cots
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, padding_side="right")
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    full_cots_tokens = tokenizer(full_cots, return_tensors="pt", padding=True).input_ids
    cot_lengths = (full_cots_tokens != tokenizer.pad_token_id).sum(dim=1)
    truncate_lengths = torch.minimum(cot_lengths, torch.tensor(args.budget))
    mask = torch.arange(full_cots_tokens.shape[1])[None, :] <= truncate_lengths[:, None]
    # Apply mask to get truncated sequences
    part_cots_tokens = full_cots_tokens.masked_fill(~mask, tokenizer.pad_token_id)
    part_cots = tokenizer.batch_decode(part_cots_tokens, skip_special_tokens=True)
    # if full_cot is less than or equal to budget, consider it as processed
    processed_samples = [sample for sample in samples if cot_lengths[sample["idx"]] <= args.budget]
    # Create index mapping for samples that need processing
    process_indices = [i for i, sample in enumerate(samples) if cot_lengths[sample["idx"]] > args.budget]
    # Filter samples and part_cots using the same indices
    samples = [samples[i] for i in process_indices]
    part_cots = [part_cots[i] for i in process_indices]
    
    # Modify samples in-place
    for sample, part_cot in zip(samples, part_cots):
        prompt = sample["prompt"]
        
        if args.prompt_type == "coarse-to-fine-qwen" or args.prompt_type == "qwen25-math-cot":
            sample["prompt"] = prompt.replace("<|im_start|>assistant\n",
                                            "<|im_start|>assistant\n" + part_cot)
            sample["prompt"] += TERMINATOR
        elif args.prompt_type == "mathstral-step-by-step" or args.prompt_type == "mathstral-coarse-to-fine":
            sample["prompt"] = prompt.replace("[/INST]",
                                            "[/INST] " + part_cot)
            sample["prompt"] += TERMINATOR
        else:
            pass
        sample.pop("code")
        sample.pop("pred")
        sample.pop("report")
        sample.pop("score")
        
    return samples, processed_samples

def set_output_path(args, data_name):
    # args.output_dir defines experiment path,such as outputs/12_25
    output_dir = os.path.join(args.output_dir, args.model_name_or_path, args.prompt_type)
    out_file_prefix = f"{args.split}_{args.prompt_type}_{args.num_test_sample}_seed{args.seed}_t{args.temperature}"
    if args.budget > 0 :
        out_file = f"{output_dir}/{data_name}/{out_file_prefix}_s{args.start}_e{args.end}_b{int(args.budget)}.jsonl"
    else:
        out_file = f"{output_dir}/{data_name}/{out_file_prefix}_s{args.start}_e{args.end}.jsonl"
    os.makedirs(f"{output_dir}/{data_name}", exist_ok=True)
    return out_file_prefix, output_dir, out_file

def set_seed(seed: int = 42) -> None:
    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"Random seed set as {seed}")


def load_jsonl(file: Union[str, Path]) -> Iterable[Any]:
    with open(file, "r", encoding="utf-8") as f:
        for line in f:
            try:
                yield json.loads(line)
            except:
                print("Error in loading:", line)
                exit()


def save_jsonl(samples, save_path):
    # ensure path
    folder = os.path.dirname(save_path)
    os.makedirs(folder, exist_ok=True)

    with open(save_path, "w", encoding="utf-8") as f:
        for sample in samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")
    print("Saved to", save_path)


def lower_keys(example):
    new_example = {}
    for key, value in example.items():
        if key != key.lower():
            new_key = key.lower()
            new_example[new_key] = value
        else:
            new_example[key] = value
    return new_example


EXAMPLES = get_examples()


def load_prompt(data_name, prompt_type, num_shots):
    if not num_shots:
        return []

    if data_name in ["gsm_hard", "svamp", "tabmwp", "asdiv", "mawps"]:
        data_name = "gsm8k"
    if data_name in ["math_oai", "hungarian_exam", "math-oai", "aime24", "amc23"]:
        data_name = "math"
    if data_name in ["sat_math"]:
        data_name = "mmlu_stem"
    if data_name in [
        "gaokao2024_I",
        "gaokao2024_II",
        "gaokao_math_qa",
        "gaokao2024_mix",
        "cn_middle_school",
    ]:
        data_name = "gaokao"

    if prompt_type in ["tool-integrated"]:
        prompt_type = "tora"

    return EXAMPLES[data_name][:num_shots]


PROMPT_TEMPLATES = {
    "direct": ("Question: {input}\nAnswer: ", "{output}", "\n\n"),
    "cot": ("Question: {input}\nAnswer: ", "{output}", "\n\n\n"),
    "pal": ("Question: {input}\n\n", "{output}", "\n---\n"),
    "tool-integrated": ("Question: {input}\n\nSolution:\n", "{output}", "\n---\n"),
    "self-instruct": ("<|user|>\n{input}\n<|assistant|>\n", "{output}", "\n"),
    "tora": ("<|user|>\n{input}\n<|assistant|>\n", "{output}", "\n"),
    "wizard_zs": (
        "### Instruction:\n{input}\n\n### Response: Let's think step by step.",
        "{output}",
        "\n\n\n",
    ),
    "platypus_fs": (
        "### Instruction:\n{input}\n\n### Response:\n",
        "{output}",
        "\n\n\n",
    ),
    "deepseek-math": (
        "User: {input}\nPlease reason step by step, "
        "and put your final answer within \\boxed{{}}.\n\nAssistant:",
        "{output}",
        "\n\n\n",
    ),
    "kpmath": (
        "User: Please reason step by step and put your final answer at the end "
        'with "The answer is: ".\n\n{input}\n\nAssistant:',
        "{output}",
    ),
    "jiuzhang": (
        "## Question\n{input}\n\n## Solution\n",
        "{output}",
        "\n\n\n",
    ),
    "jiuzhang_tora": (
        "## Question\n{input}\n\n## Code Solution\n",
        "{output}",
        "\n\n\n",
    ),
    "jiuzhang_nl": (
        "## Question\n{input}\n\n## Natural Language Solution\n",
        "{output}",
        "\n\n\n",
    ),
    "mmiqc": (
        'Please solve the following problem and put your answer at the end with "The answer is: ".\n\n{input}\n\n',
        "{output}",
        "\n\n\n",
    ),
    "abel": (
        "Question:\n{input}\nAnswer:\nLet's think step by step.\n",
        "{output}",
        "\n\n",
    ),
    "shepherd": ("{input}\n", "{output}", "\n\n\n"),
    "qwen-boxed": (
        "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
        "<|im_start|>user\n{input}\nPlease reason step by step, and put your final answer within \\boxed{{}}.<|im_end|>\n"
        "<|im_start|>assistant\n",
        "{output}",
        "\n\n",
    ),
    "qwen25-math-cot": (
        "<|im_start|>system\nPlease reason step by step, and put your final answer within \\boxed{{}} when done reasoning or early-stop keyword **Final Answer** appears.<|im_end|>\n"
        "<|im_start|>user\n{input}<|im_end|>\n"
        "<|im_start|>assistant\n",
        "{output}",
        "\n\n",
    ),
    "coarse-to-fine-qwen": (
        "<|im_start|>system\nSolve the task by following format:\n**Coarse Reasoning**\nShort analysis and an answer. Focus on efficiency and simplicity.\n\n**Fine Reasoning**\nDetailed analysis step by step and a refined answer. Focus on accuracy and correctness.\n\n**Final Answer** \nYour final answer within \\boxed{{}} when done reasoning or early-stop keyword **Final Answer** appears.<|im_end|>\n"
        "<|im_start|>user\n{input}<|im_end|>\n"
        "<|im_start|>assistant\n",
        "{output}",
        "\n\n", 
    ),
    "qwen25-step-by-step-hard": (
        "<|im_start|>system\nPlease reason step by step, and put your final answer within \\boxed{{}}.\n<|im_end|>\n"
        "<|im_start|>user\n{input}<|im_end|>\n"
        "<|im_start|>assistant\n",
        "{output}",
        "\n\n",
    ),
    "mathstral-step-by-step-hard": (
        "<s>[INST] Please reason step by step, and put your final answer within \\boxed{{}}.\n\n{input}[/INST]",
        "{output}",
        "\n\n",
    ),
    "mathstral-step-by-step": (
        "<s>[INST] Please reason step by step, and put your final answer within \\boxed{{}} when done reasoning or early-stop keyword **Final Answer** appears.\n\n{input}[/INST]",
        "{output}",
        "\n\n",
    ),
    "mathstral-coarse-to-fine": (
        "<s>[INST] Solve the task by following format:\n**Coarse Reasoning**\nShort analysis and an answer. Focus on efficiency and simplicity.\n\n**Fine Reasoning**\nDetailed analysis step by step and a refined answer. Focus on accuracy and correctness.\n\n**Final Answer** \nYour final answer within \\boxed{{}} when done reasoning or early-stop keyword **Final Answer** appears.\n\n{input}[/INST]",
        "{output}",
        "\n\n", 
    ),
    "internlm-math-fs": ("Question:{input}\nAnswer:", "{output}", "\n"),
    "internlm-math-chat": (
        "<|im_start|>user\n{input}<|im_end|>\n" "<|im_start|>assistant\n",
        "{output}",
        "\n\n",
    ),
    "mistral": (
        "[INST] {input}[/INST]",
        "{output}",
        "\n\n",
    ),
    "numina": ("### Problem: {input}\n### Solution:", " {output}", "\n\n"),
}


def construct_prompt(example, data_name, args):
    if args.adapt_few_shot and data_name in [
        "gaokao2024_I",
        "gaokao2024_II",
        "gaokao_math_qa",
        "gaokao2024_mix",
        "cn_middle_school",
    ]:
        demos = load_prompt(data_name, args.prompt_type, 5)
    else:
        demos = load_prompt(data_name, args.prompt_type, args.num_shots)
    prompt_type = args.prompt_type
    if prompt_type == "platypus_fs":
        prompt_type = "cot"
    if prompt_type == "tool-integrated":
        prompt_type = "tora"

    prompt_temp = PROMPT_TEMPLATES[args.prompt_type]

    input_template, output_template, splitter = (
        prompt_temp[0],
        prompt_temp[1],
        prompt_temp[2],
    )
    if args.prompt_type == "qwen25-math-cot":
        # Hotfix to support putting all demos into a single turn
        demo_prompt = splitter.join([q + "\n" + a for q, a in demos])
    else:
        demo_prompt = splitter.join(
            [
                input_template.format(input=q) + output_template.format(output=a)
                for q, a in demos
            ]
        )
    context = input_template.format(input=example["question"])
    if len(demo_prompt) == 0 or (
        args.adapt_few_shot and example["gt_ans"] not in ["A", "B", "C", "D", "E"]
    ):
        full_prompt = context
    else:
        if args.prompt_type == "qwen25-math-cot":
            # Hotfix to supportting put all demos into a single turn
            full_prompt = demo_prompt + splitter + example["question"]
            full_prompt = input_template.format(input=full_prompt)
        else:
            full_prompt = demo_prompt + splitter + context

    if args.prompt_type == "platypus_fs":
        full_prompt_temp = (
            "Below is an instruction that describes a task. "
            "Write a response that appropriately completes the request.\n\n"
            "### Instruction:\n{instruction}\n\n### Response:\n"
        )
        full_prompt = full_prompt_temp.format(instruction=full_prompt)

    if prompt_type == "tora":
        full_prompt = (
            """Integrate step-by-step reasoning and Python code to solve math problems using the following guidelines:

- Analyze the question and write functions to solve the problem; the function should not take any arguments.
- Present the final result in LaTeX using a `\boxed{}` without any units.
- Utilize the `pi` symbol and `Rational`` from Sympy for $\pi$ and fractions, and simplify all fractions and square roots without converting them to decimal values.

Here are some examples you may refer to:

---

"""
            + full_prompt
        )

    return full_prompt.strip(" ")  # important!


key_map = {
    "gt": "Ground Truth",
    "pred": "Prediction",
    "gt_cot": "Reference CoT",
    "score": "Score",
}


def show_sample(sample, print_all_preds=False):
    print("==" * 20)
    for key in ["idx", "type", "level", "dataset"]:
        if key in sample:
            # capitalize
            print("{}: {}".format(key[0].upper() + key[1:], sample[key]))
    print("Question:", repr(sample["question"]))
    if "code" in sample:
        if print_all_preds:
            for code in sample["code"]:
                print("-" * 20)
                print("code:", code)
            print("Execution:", sample["report"])
        else:
            print("Solution:\n", sample["code"][0])
            print("Execution:", sample["report"][0])
    if "pred" in sample:
        print("Prediction:", repr(sample["pred"][0]))
    for key in ["gt", "score", "unit", "gt_cot"]:
        if key in sample:
            _key = key_map.get(key, key)
            print("{}: {}".format(_key, repr(sample[key])))
    print()

if __name__ == "__main__":
    full_cot_path = "outputs/12_26/Qwen/Qwen2.5-7B-Instruct/coarse-to-fine-qwen/gsm8k/test_coarse-to-fine-qwen_-1_seed0_t0.0_s0_e-1.jsonl"
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str, default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--prompt_type", type=str, default="coarse-to-fine-qwen")
    args = parser.parse_args()
    budget = 500
    samples = ""
    # print(get_appended_prompt(full_cot_path, samples, args, budget)[0])
    # # print element by element
    # for i, element in enumerate(get_appended_prompt(full_cot_path, samples, args, budget)):
    #     print(f"Element {i}: {element}")
