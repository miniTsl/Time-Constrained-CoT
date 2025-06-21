import random
import os
import argparse
import time
from vllm import LLM, SamplingParams
from datetime import datetime
from tqdm import tqdm
from transformers import AutoTokenizer

from evaluate import evaluate
from utils.utils import set_seed, load_jsonl, save_jsonl, construct_prompt, set_output_path, gen_budget_list, load_data_with_cropped_cot
# from parser import *
from original_parser import *
from trajectory import *
from data_loader import load_data
from python_executor import PythonExecutor
from model_utils import load_hf_lm_and_tokenizer, generate_completions


def is_multi_choice(answer):
    for c in answer:
        if c not in ["A", "B", "C", "D", "E"]:
            return False
    return True


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--budget", type=float, default=-1, help="flag to indicate whether to use budget")
    parser.add_argument("--output_budget", type=float, default=-1, help="budget of tokens to use for generation")
    parser.add_argument("--data_names", default="gsm8k", type=str)
    parser.add_argument("--data_dir", default="./data", type=str)
    parser.add_argument("--model_name_or_path", default="Qwen/QwQ-32B-Preview", type=str)
    parser.add_argument("--output_dir", default="Qwen/QwQ-32B-Preview/math_eval", type=str)
    parser.add_argument("--prompt_type", default="qwen25-math-cot", type=str)
    parser.add_argument("--split", default="test", type=str)
    parser.add_argument("--num_test_sample", default=-1, type=int)  # -1 for full data
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--start", default=0, type=int)
    parser.add_argument("--end", default=-1, type=int)
    parser.add_argument("--temperature", default=0, type=float)
    parser.add_argument("--n_sampling", default=1, type=int)
    parser.add_argument("--top_p", default=1, type=float)
    parser.add_argument("--max_tokens_per_call", default=4096, type=int)
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--use_vllm", action="store_true")
    parser.add_argument("--save_outputs", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--use_safetensors", action="store_true")
    parser.add_argument("--num_shots", type=int, default=0)
    parser.add_argument("--apply_chat_template", action="store_true", help="Apply chat template to prompt.",)
    parser.add_argument("--pipeline_parallel_size", type=int, default=1)
    parser.add_argument("--adapt_few_shot", action="store_true", help="Few shot for multiple-choice questions, zero shot for others.",)
    args = parser.parse_args()
    args.top_p = (1 if args.temperature == 0 else args.top_p)  # top_p must be 1 when using greedy sampling (vllm)
    return args


def setup(args):
    # load model
    available_gpus = os.environ["CUDA_VISIBLE_DEVICES"].split(",")
    if args.use_vllm:
        if args.model_name_or_path == "microsoft/Phi-3-small-128k-instruct":
            # Phi-3-small doesn't support prefix caching or chunked prefill because it uses BlockSparse Attention, which is relatively new.
            llm = LLM(
                model=args.model_name_or_path,
                tensor_parallel_size=len(available_gpus) // args.pipeline_parallel_size,
                pipeline_parallel_size=args.pipeline_parallel_size,
                trust_remote_code=True,
                gpu_memory_utilization=0.9,
                enable_chunked_prefill=False,
                max_model_len = 10240 if "gemma" not in args.model_name_or_path and "Qwen2.5-Math" not in args.model_name_or_path else None,
                max_num_seqs = 512
            )
        else:
            llm = LLM(
                model=args.model_name_or_path,
                tensor_parallel_size=len(available_gpus) // args.pipeline_parallel_size,
                pipeline_parallel_size=args.pipeline_parallel_size,
                trust_remote_code=True,
                gpu_memory_utilization=0.9,
                max_model_len = 10240 if "gemma" not in args.model_name_or_path and "Qwen2.5-Math" not in args.model_name_or_path else None,
                max_num_seqs = 512
            )
        tokenizer = None
        if args.apply_chat_template:
            tokenizer = AutoTokenizer.from_pretrained(
                args.model_name_or_path, trust_remote_code=True
            )
    else:
        llm, tokenizer = load_hf_lm_and_tokenizer(
            model_name_or_path=args.model_name_or_path,
            use_fast_tokenizer=True,
            use_safetensors=args.use_safetensors,
        )

    # infer & eval
    data_list = args.data_names.split(",")
    for data_name in data_list:
        budget_list = gen_budget_list(args.budget, data_name, args.model_name_or_path, args.prompt_type)
        for budget in budget_list:
            print("\n" + "-" * 50)
            print("Budget list:", budget_list, " Budget length:", len(budget_list))
            print("Current budget:", budget)
            args.output_budget = budget
            if budget > 0 and "hard" in args.prompt_type:
                args.max_tokens_per_call = budget   # hard crop
            elif budget > 0 and "hard" not in args.prompt_type:
                args.max_tokens_per_call = 25   # models should summarize the answer with 25 new tokens
            result = main(llm, tokenizer, data_name, args)
            print("-" * 50)
            print(f"Data: {data_name}")
            print(f"Budget: {budget}")
            print(f"Accuracy: {result['acc']:.1f}\n")


def prepare_data(data_name, args):
    # get out_file_prefix, output_dir and out_file
    out_file_prefix, output_dir, out_file = set_output_path(args, data_name)
    
    # if outfile exists, return
    if os.path.exists(out_file):
        return out_file_prefix, output_dir, out_file
    
    # if original run of soft crop without budget or using hard crop with budget, load samples from original dataset
    if args.budget < 0 or "hard" in args.prompt_type:
        examples = load_data(data_name, args.split, args.data_dir)
        # sample `num_test_sample` from dataset， -1 for full data
        if args.num_test_sample > 0:
            examples = examples[: args.num_test_sample]
        # shuffle
        if args.shuffle:
            random.seed(datetime.now().timestamp())
            random.shuffle(examples)
        # select start and end
        examples = examples[args.start : len(examples) if args.end == -1 else args.end]
        # load all processed samples
        processed_samples = []
        # if not args.overwrite:
        #     processed_files = [
        #         f
        #         for f in os.listdir(f"{output_dir}/{data_name}/")
        #         if f.endswith(".jsonl") and f.startswith(out_file_prefix)
        #     ]
        #     for f in processed_files:
        #         processed_samples.extend(
        #             list(load_jsonl(f"{output_dir}/{data_name}/{f}"))
        #         )
        # dedepulicate
        processed_samples = {sample["idx"]: sample for sample in processed_samples}
        processed_idxs = list(processed_samples.keys())
        processed_samples = list(processed_samples.values())
        examples = [example for example in examples if example["idx"] not in processed_idxs]
    else:
        # append cropped CoT to samples
        full_cot_path = out_file.replace("_b" + str(args.output_budget), "")
        examples, processed_samples = load_data_with_cropped_cot(full_cot_path, args)
        
    return examples, processed_samples, out_file


def main(llm, tokenizer, data_name, args):
    examples, processed_samples, out_file = prepare_data(data_name, args)
    # if outfile exists, return
    if os.path.exists(out_file):
        print(f"File {out_file} exists, skipping inference.")
        with open(out_file.replace(".jsonl", "_metrics.json"), "r") as f:
            result_json = json.load(f)
        return result_json
    
    print("-" * 50)
    print("data:", data_name, ", samples to infer:", len(examples))
    if len(examples) > 0:
        print("smaple No.0:", examples[0])

    # init python executor
    if "pal" in args.prompt_type:
        executor = PythonExecutor(get_answer_expr="solution()")
    else:
        executor = PythonExecutor(get_answer_from_stdout=True)
    
    # if original run of soft crop without budget or using hard crop with budget, still needs processing
    if args.budget < 0 or "hard" in args.prompt_type:
        samples = []
        print("\nProcessing", len(examples), "examples", "=" * 50)
        # process each example
        for example in tqdm(examples, total=len(examples)):
            idx = example["idx"]
            # parse question and answer
            example["question"] = parse_question(example, data_name)
            if example["question"] == "":
                continue
            gt_cot, gt_ans = parse_ground_truth(example, data_name)
            example["gt_ans"] = gt_ans
            full_prompt = construct_prompt(example, data_name, args)

            sample = {
                "idx": idx,
                "question": example["question"],
                "gt_cot": gt_cot,
                "gt": gt_ans,
                "prompt": full_prompt,
            }

            # add remain fields
            for key in [
                "level",
                "subject",
                "type",
                "unit",
                "solution_type",
                "choices",
                "solution",
                "ques_type",
                "ans_type",
                "answer_type",
                "dataset",
                "subfield",
                "filed",
                "theorem",
                "answer",
            ]:
                if key in example:
                    sample[key] = example[key]
            samples.append(sample)
    else:
        samples = examples
    
    if len(samples) > 0:
        print("\nPrompt of No.0 sample:")
        print(samples[0]["prompt"])
    
    # repeat n times
    input_prompts = [sample["prompt"] for sample in samples for _ in range(args.n_sampling)]
    
    if args.apply_chat_template:
        input_prompts = [
            tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt.strip()}],
                tokenize=False,
                add_generation_prompt=True,
            )
            for prompt in input_prompts
        ]
    
    remain_prompts = input_prompts
    remain_prompts = [(i, prompt) for i, prompt in enumerate(remain_prompts)]
    end_prompts = []

    max_func_call = 1 if args.prompt_type in ["cot", "pal"] else 1

    stop_words = ["</s>", "<|im_end|>", "<|endoftext|>"]

    if args.prompt_type in ["cot"]:
        stop_words.append("\n\nQuestion:")
    if args.prompt_type in ["pal", "tool-integrated", "jiuzhang_tora"]:
        stop_words.extend(["\n\n---", "```output"])
    elif args.prompt_type in ["wizard_zs", "platypus_fs"]:
        stop_words.extend(["Instruction", "Response"])
    elif "jiuzhang" in args.prompt_type:
        stop_words.append("\n\n## Question")
    elif "numina" in args.prompt_type:
        stop_words.append("\n### Problem")
    elif "pure" in args.prompt_type:
        stop_words.append("\n\n\n")

    if "Phi-3" in args.model_name_or_path and "mini" in args.model_name_or_path:
        stop_words.append("Question")
    
    # start inference
    # measure time use
    start_time = time.time()
    for epoch in range(max_func_call):
        print("-" * 20, "Epoch", epoch)
        current_prompts = remain_prompts
        if len(current_prompts) == 0:
            break

        prompts = [item[1] for item in current_prompts]
        # spilt prompts into chunks to avoid OOM
        num_prompts = len(prompts)
        chunk_size = min(num_prompts, 1000)
        outputs = []

        for i in range(0, num_prompts, chunk_size):
            chunk = prompts[i:i + chunk_size]  # 获取当前的 chunk
            if args.use_vllm:
                print("Using max_tokens_per_call:", args.max_tokens_per_call)
                chunk_outputs = llm.generate(
                    chunk,
                    SamplingParams(
                        temperature=args.temperature,
                        top_p=args.top_p,
                        max_tokens=args.max_tokens_per_call,
                        n=1,
                        stop=stop_words,
                        stop_token_ids=(
                            [151645, 151643]
                            if "qwen2" in args.model_name_or_path.lower()
                            else None
                        ),
                    ),
                )

                chunk_outputs = sorted(
                    chunk_outputs, key=lambda x: int(x.request_id)
                )  # sort outputs by request_id
                outputs.extend([output.outputs[0].text for output in chunk_outputs])
            else:
                chunk_outputs = generate_completions(
                    model=llm,
                    tokenizer=tokenizer,
                    prompts=chunk,
                    max_new_tokens=args.max_tokens_per_call,
                    batch_size=256,
                    stop_id_sequences=stop_words,
                )
                outputs.extend(chunk_outputs)

        assert len(outputs) == len(current_prompts)

        # process all outputs
        remain_prompts = []
        remain_codes = []
        # for (i, query), output in zip(current_prompts, outputs):
        #     output = output.rstrip()
        #     query += output # append output to query prompt
        #     if args.prompt_type == "pal":
        #         remain_prompts.append((i, query))
        #         if "```python" in output:
        #             output = extract_program(query)
        #         remain_codes.append(output)
        #     elif args.prompt_type == "cot":
        #         end_prompts.append((i, query))
        #     elif "boxed" not in output and output.endswith("```"):
        #         program = extract_program(query)
        #         remain_prompts.append((i, query))
        #         remain_codes.append(program)
        #     else:
        #         end_prompts.append((i, query))

        # # execute the remain prompts
        # remain_results = executor.batch_apply(remain_codes)
        # for k in range(len(remain_prompts)):
        #     i, query = remain_prompts[k]
        #     res, report = remain_results[k]
        #     exec_result = res if res else report
        #     if "pal" in args.prompt_type:
        #         exec_result = "\\boxed{" + exec_result + "}"
        #     exec_result = f"\n```output\n{exec_result}\n```\n"
        #     query += exec_result
        #     # not end
        #     if epoch == max_func_call - 1:
        #         query += "\nReach max function call limit."
        #     remain_prompts[k] = (i, query)

        # now, we consider pure cot without any tool call
        for (i, query), output in zip(current_prompts, outputs):
            output = output.rstrip()
            query += output # append output to query prompt
            end_prompts.append((i, query))

    # unsolved samples
    print("Unsolved samples:", len(remain_prompts))
    end_prompts.extend(remain_prompts)
    # sort by idx
    end_prompts = sorted(end_prompts, key=lambda x: x[0])

    # remove input_prompt from end_prompt
    codes = []
    assert len(input_prompts) == len(end_prompts)
    for i in range(len(input_prompts)):
        _, end_prompt = end_prompts[i]
        code = end_prompt.split(input_prompts[i])[-1].strip()
        for stop_word in stop_words:
            if stop_word in code:
                code = code.split(stop_word)[0].strip()
        codes.append(code)

    # extract preds
    results = [
        run_execute(executor, code, args.prompt_type, data_name) for code in codes
    ]
    time_use = time.time() - start_time

    # put results back to examples
    all_samples = []
    for i, sample in enumerate(samples):
        code = codes[i * args.n_sampling : (i + 1) * args.n_sampling]
        result = results[i * args.n_sampling : (i + 1) * args.n_sampling]
        preds = [item[0] for item in result]
        reports = [item[1] for item in result]
        for j in range(len(preds)):
            # if preds[j] is None, pass
            if preds[j] is None:
                continue
            if sample["gt"] in ["A", "B", "C", "D", "E"] and preds[j] not in [
                "A",
                "B",
                "C",
                "D",
                "E",
            ]:
                preds[j] = choice_answer_clean(code[j])
            elif is_multi_choice(sample["gt"]) and not is_multi_choice(preds[j]):
                # remove any non-choice char
                preds[j] = "".join(
                    [c for c in preds[j] if c in ["A", "B", "C", "D", "E"]]
                )

        # sample.pop("prompt")  # save the prompt for debug
        sample.update({"code": code, "pred": preds, "report": reports})
        all_samples.append(sample)
    
    # add processed samples
    all_samples.extend(processed_samples)
    all_samples, result_json = evaluate(
        samples=all_samples,
        data_name=data_name,
        prompt_type=args.prompt_type,
        execute=True,
    )

    # save outputs
    if len(processed_samples) <= len(all_samples) and args.save_outputs:
        save_jsonl(all_samples, out_file)

    result_json["time_use_in_second"] = time_use
    result_json["time_use_in_minite"] = (
        f"{int(time_use // 60)}:{int(time_use % 60):02d}"
    )
    result_json["budget"] = args.output_budget
    inferenced_sample_num = len(all_samples) - len(processed_samples)
    result_json["inferenced_sample_num"] = inferenced_sample_num
    
    with open(
        out_file.replace(".jsonl", "_metrics.json"), "w"
    ) as f:
        json.dump(result_json, f, indent=4)
    return result_json


if __name__ == "__main__":
    args = parse_args()
    set_seed(args.seed)
    setup(args)