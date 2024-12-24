import random
import os
import argparse
import time
from vllm import LLM, SamplingParams
from datetime import datetime
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

from evaluate import evaluate
from utils import set_seed, load_jsonl, save_jsonl, construct_prompt, load_json, dump_json
from parser import *
from trajectory import *
from data_loader import load_data
from python_executor import PythonExecutor
from model_utils import load_hf_lm_and_tokenizer, generate_completions
import json
# from incontext.utils import load_jsonl, save_jsonl, load_json, dump_json


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ratio", type=float, default=-1, help="ratio of cot to use for generation")
    parser.add_argument("--data_names", default="math", type=str)
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
    parser.add_argument("--max_tokens_per_call", default=8192, type=int)
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--use_vllm", action="store_true")
    parser.add_argument("--save_outputs", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--use_safetensors", action="store_true")
    parser.add_argument("--num_shots", type=int, default=0)
    parser.add_argument("--apply_chat_template", action="store_true", help="Apply chat template to prompt.",)
    parser.add_argument("--pipeline_parallel_size", type=int, default=1)
    parser.add_argument("--adapt_few_shot", action="store_true", help="Few shot for multiple-choice questions, zero shot for others.",)

    parser.add_argument("--cot_nums", type=int, default=10)
    parser.add_argument("--merge_cots", action="store_true", help="whether it is the stage to merge 10 cot paths into one prompt, query LLM about the final result")

    args = parser.parse_args()
    args.top_p = (1 if args.temperature == 0 else args.top_p)  # top_p must be 1 when using greedy sampling (vllm)
    if args.ratio > 0:
        args.max_tokens_per_call = 50
    return args

def get_merged_cots(cot_answers, ratio):
    merged_thoughts = ''
    for i in range(len(cot_answers)):
        merged_thoughts += f'Reasoning path {i}: {cot_answers[i]}\n'

def crop_cot(cot, ratio):
    cut_cot = cot[:int(len(cot)*ratio)]
    # 将prompt中的<|im_start|>assistant\n换成新内容
    full_prompt = full_prompt.replace("<|im_start|>assistant\n", "<|im_start|>assistant\n" + cut_cot + "\n\nFinal answer within \\boxed{{}}:\n")

def prepare_data(data_name, args, max_samples=None, cot_id=None):
    examples = load_data(data_name, args.split, args.data_dir)

    if os.path.exists(f'test_{max_samples}_{data_name}.json'):
        sample_ids = load_json(f'test_{max_samples}_{data_name}.json')
    else:
        sample_ids = random.sample(range(len(examples)), max_samples)
        dump_json(f'test_{max_samples}_{data_name}.json', sample_ids)
    examples = [examples[i] for i in sample_ids]

    # # sample `num_test_sample` from dataset， -1 for full data
    # if args.num_test_sample > 0:
    #     # examples = random.sample(examples, min(args.num_test_sample, len(examples)))
    #     examples = examples[: args.num_test_sample]

    # shuffle
    if args.shuffle:
        random.seed(datetime.now().timestamp())
        random.shuffle(examples)

    # select start and end
    # examples = examples[args.start : len(examples) if args.end == -1 else args.end]

    # get out_file name
    dt_string = datetime.now().strftime("%m-%d_%H-%M")
    model_name = "/".join(args.model_name_or_path.split("/")[-2:])
    out_file_prefix = f"{args.split}_{args.prompt_type}_{args.num_test_sample}_seed{args.seed}_t{args.temperature}"
    output_dir = args.output_dir
    if not os.path.exists(output_dir):
        output_dir = f"outputs/12_20/{output_dir}"
    if not cot_id:
        if args.ratio > 0 :
            out_file = f"{output_dir}/{data_name}/{out_file_prefix}_s{args.start}_e{args.end}_r{args.ratio}.jsonl"
        else:
            out_file = f"{output_dir}/{data_name}/{out_file_prefix}_s{args.start}_e{args.end}.jsonl"
    else:
        if args.ratio > 0 :
            out_file = f"{output_dir}/{data_name}/{out_file_prefix}_s{args.start}_e{args.end}_r{args.ratio}_cot_{cot_id}.jsonl"
        else:
            out_file = f"{output_dir}/{data_name}/{out_file_prefix}_s{args.start}_e{args.end}_cot_{cot_id}.jsonl"
    os.makedirs(f"{output_dir}/{data_name}", exist_ok=True)

    # load all processed samples
    processed_samples = []
    if not args.overwrite:
        processed_files = [
            f
            for f in os.listdir(f"{output_dir}/{data_name}/")
            if f.endswith(".jsonl") and f.startswith(out_file_prefix)
        ]
        for f in processed_files:
            processed_samples.extend(
                list(load_jsonl(f"{output_dir}/{data_name}/{f}"))
            )

    # dedepulicate
    processed_samples = {sample["idx"]: sample for sample in processed_samples}
    processed_idxs = list(processed_samples.keys())
    processed_samples = list(processed_samples.values())
    examples = [example for example in examples if example["idx"] not in processed_idxs]
    return examples, processed_samples, out_file


def setup(args):
    # load model
    available_gpus = os.environ["CUDA_VISIBLE_DEVICES"].split(",")
    if args.use_vllm:
        llm = LLM(
            model=args.model_name_or_path,
            tensor_parallel_size=len(available_gpus) // args.pipeline_parallel_size,
            pipeline_parallel_size=args.pipeline_parallel_size,
            trust_remote_code=True,
            gpu_memory_utilization=0.8
        )
        tokenizer = None
        if args.apply_chat_template:
            tokenizer = AutoTokenizer.from_pretrained(
                args.model_name_or_path, trust_remote_code=True
            )
    else:
        llm, tokenizer = load_hf_lm_and_tokenizer(
            model_name_or_path=args.model_name_or_path,
            load_in_half=True,
            use_fast_tokenizer=True,
            use_safetensors=args.use_safetensors,
        )
    
    for cot_id in range(0, args.cot_nums):
        print('*'*30, 'cot_id:', cot_id, '*'*30)
        # infer & eval
        data_list = args.data_names.split(",")
        results = []
        for data_name in data_list:
            results.append(main(llm, tokenizer, data_name, args, cot_id))

        # add "avg" result to data_list and results
        data_list.append("avg")
        results.append(
            {
                "acc": sum([result["acc"] for result in results]) / len(results),
            }
        )

        # print all results
        pad = max([len(data_name) for data_name in data_list])
        print("\t".join(data_name.ljust(pad, " ") for data_name in data_list))
        print("\t".join([f"{result['acc']:.1f}".ljust(pad, " ") for result in results]))


def is_multi_choice(answer):
    for c in answer:
        if c not in ["A", "B", "C", "D", "E"]:
            return False
    return True

def parse_similar_examples_in_train(problem, train_set, all_similar_examples_for_test_set, coarse_to_fine_cots, topk=10):
    if problem in all_similar_examples_for_test_set.keys():
        similar_ids = all_similar_examples_for_test_set[problem]
        try:
            similar_cots = [coarse_to_fine_cots[int(i)] for i in similar_ids]
        except:
            import pdb;pdb.set_trace()
        similar_questions = [train_set[i]["problem"] for i in similar_ids]
        if topk > len(similar_ids):
            return similar_questions, similar_cots
        else:
            return similar_questions[:topk], similar_cots[:topk]
    else:
        return None, None


def main(llm, tokenizer, data_name, args, cot_id):
    examples, processed_samples, out_file = prepare_data(data_name, args, max_samples=128, cot_id=cot_id)

    math_test_neighbors = load_json('data/math-processedv2/math_test_neighbours.json')
    # for k, v in math_test_neighbors.items():
    #     math_test_neighbors[k] = [int(i) for i in v]
    tmp_math_cots_coarse2fine = list(load_jsonl('data/math-processedv2/math_train_coarse2fine.jsonl'))
    math_cots_coarse2fine = {}
    for cot_dict in tmp_math_cots_coarse2fine:
        math_cots_coarse2fine[int(list(cot_dict.keys())[0])] = list(cot_dict.values())[0]
    math_train_data = list(load_jsonl('data/math/train.jsonl'))

    print("\n" + "-" * 50)
    print("data:", data_name, ", remain samples:", len(examples))
    if len(examples) > 0:
        print(examples[0])

    # init python executor
    if "pal" in args.prompt_type:
        executor = PythonExecutor(get_answer_expr="solution()")
    else:
        executor = PythonExecutor(get_answer_from_stdout=True)

    if args.ratio > 0 :
        done_samples_path = f"outputs/12_20/" + args.output_dir + "/" + data_name + "/" + args.split + "_" + args.prompt_type + "_" + str(args.num_test_sample) + "_seed" + str(args.seed) + "_t" + str(args.temperature) + "_s" + str(args.start) + "_e" + str(args.end)  + ".jsonl"
        done_samples = list(load_jsonl(done_samples_path))
    else:
        done_samples = []
    done_samples = {sample["idx"]: sample for sample in done_samples}
    
    samples = []
    print("\nProcessing", len(examples), "examples", "=" * 50)
    for example in tqdm(examples, total=len(examples)):
        idx = example["idx"]

        # parse question and answer
        example["question"] = parse_question(example, data_name)
        if example["question"] == "":
            continue
        gt_cot, gt_ans = parse_ground_truth(example, data_name)
        example["gt_ans"] = gt_ans

        similar_questions, similar_cots = parse_similar_examples_in_train(example["problem"], math_train_data, math_test_neighbors, math_cots_coarse2fine)
        demos = [(similar_questions[cot_id], similar_cots[cot_id])]
        full_prompt = construct_prompt(example, data_name, args, demos)
        # # add ratio part of complete cot
        if args.ratio > 0 :
            done_cot = done_samples[idx]["code"][0]
            cut_cot = done_cot[:int(len(done_cot)*args.ratio)]
            # 将prompt中的<|im_start|>assistant\n换成新内容
            full_prompt = full_prompt.replace("<|im_start|>assistant\n", "<|im_start|>assistant\n" + cut_cot + "\n\nFinal answer within \\boxed{{}}:\n")

        if idx == args.start:
            print(full_prompt)

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

    max_func_call = 1 if args.prompt_type in ["cot", "pal"] else 4

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

    # start inference
    # measure time use
    start_time = time.time()
    for epoch in range(max_func_call):
        print("-" * 20, "Epoch", epoch)
        current_prompts = remain_prompts
        if len(current_prompts) == 0:
            break

        # get all outputs
        prompts = [item[1] for item in current_prompts]
        # 为了防止内存爆炸，将prompts分成4份，每份调用一次vllm
        num_prompts = len(prompts)
        if num_prompts < 300:
            chunk_size = num_prompts
        else:
            chunk_size = (num_prompts + 9) // 10  # 计算每一份的大小，确保包含所有的 prompts
        outputs = []
        
        for i in range(0, num_prompts, chunk_size):
            chunk = prompts[i:i + chunk_size]  # 获取当前的 chunk
            if args.use_vllm:
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
                    batch_size=16,
                    stop_id_sequences=stop_words,
                )
                outputs.extend(chunk_outputs)
        assert len(outputs) == len(current_prompts)

        # process all outputs
        remain_prompts = []
        remain_codes = []
        for (i, query), output in zip(current_prompts, outputs):
            output = output.rstrip()
            query += output
            if args.prompt_type == "pal":
                remain_prompts.append((i, query))
                if "```python" in output:
                    output = extract_program(query)
                remain_codes.append(output)
            elif args.prompt_type == "cot":
                end_prompts.append((i, query))
            elif "boxed" not in output and output.endswith("```"):
                program = extract_program(query)
                remain_prompts.append((i, query))
                remain_codes.append(program)
            else:
                end_prompts.append((i, query))

        # execute the remain prompts
        remain_results = executor.batch_apply(remain_codes)
        for k in range(len(remain_prompts)):
            i, query = remain_prompts[k]
            res, report = remain_results[k]
            exec_result = res if res else report
            if "pal" in args.prompt_type:
                exec_result = "\\boxed{" + exec_result + "}"
            exec_result = f"\n```output\n{exec_result}\n```\n"
            query += exec_result
            # not end
            if epoch == max_func_call - 1:
                query += "\nReach max function call limit."
            remain_prompts[k] = (i, query)

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
    if len(processed_samples) < len(all_samples) and args.save_outputs:
        save_jsonl(all_samples, out_file)

    result_json["time_use_in_second"] = time_use
    result_json["time_use_in_minite"] = (
        f"{int(time_use // 60)}:{int(time_use % 60):02d}"
    )

    with open(
        out_file.replace(".jsonl", f"_{args.prompt_type}_metrics.json"), "w"
    ) as f:
        json.dump(result_json, f, indent=4)
    return result_json


if __name__ == "__main__":
    args = parse_args()
    set_seed(args.seed)
    setup(args)
