from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import time
import pandas as pd
import numpy as np
import argparse
import matplotlib.pyplot as plt
from matplotlib import rcParams
from scipy import interpolate
# from utils.model_list import __MODEL_LIST__

"""model_list = [
    "Qwen/QwQ-32B-Preview",
    # "Skywork/Skywork-o1-Open-Llama-3.1-8B", 
    # "PowerInfer/SmallThinker-3B-Preview",
    "NovaSky-AI/Sky-T1-32B-Preview",
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
    "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
    "Qwen/Qwen2.5-Math-1.5B-Instruct",
    "Qwen/Qwen2.5-Math-7B-Instruct",
    "mistralai/Mathstral-7B-v0.1",
    "Qwen/Qwen2.5-32B-Instruct",
    "Qwen/Qwen2.5-14B-Instruct",
    "Qwen/Qwen2.5-7B-Instruct",
    "Qwen/Qwen2.5-3B-Instruct",
    "Qwen/Qwen2.5-1.5B-Instruct",
    "mistralai/Mistral-Small-Instruct-2409",
    "mistralai/Mistral-Nemo-Instruct-2407",
    "mistralai/Ministral-8B-Instruct-2410",
    "google/gemma-2-27b-it",
    "google/gemma-2-9b-it",
    "google/gemma-2-2b-it",
    "microsoft/Phi-3-medium-128k-instruct",
    "microsoft/Phi-3-small-128k-instruct",
    "microsoft/Phi-3.5-mini-instruct",
    "microsoft/Phi-3-mini-128k-instruct",
    "microsoft/phi-4",
    "meta-llama/Llama-3.1-8B-Instruct",
    "meta-llama/Llama-3.2-3B-Instruct",
    "meta-llama/Llama-3.2-1B-Instruct",
]"""

model_list = [
    "Qwen/QwQ-32B-Preview",
    "NovaSky-AI/Sky-T1-32B-Preview",
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
    "google/gemma-2-27b-it",
    "google/gemma-2-9b-it",
]

def measure_inference_latency(model, tokenizer, tokenized_ids, target_output_tokens, num_runs=3):
    # Prepare input
    input_length = tokenized_ids.shape[1]
    
    # Measure latency
    latencies = []
    for _ in range(num_runs):
        while True:  # Keep generating until we get exact output length
            torch.cuda.synchronize()
            start_time = time.perf_counter_ns()  # Use nanoseconds for timing
            
            output_ids = model.generate(
                tokenized_ids,
                max_new_tokens=target_output_tokens,
                do_sample=True,  # Enable sampling for varied outputs
                temperature=0.9
            )
            
            torch.cuda.synchronize()
            end_time = time.perf_counter_ns()  # Use nanoseconds for timing
            
            output_ids = output_ids[0][input_length:]
            # Check if output length matches target
            actual_output_length = len(output_ids)
            print("actual_output_length: ", actual_output_length)
            # decode and print the output
            output_text = tokenizer.decode(output_ids, skip_special_tokens=True)
            print("output_text: ", output_text)
            if actual_output_length == target_output_tokens:
                latencies.append((end_time - start_time) / 1_000_000_000)  # Convert to seconds
                break
            
    # Return average latency in seconds
    return np.mean(latencies)

def construct_input_of_length(tokenizer, base_text, prompt_v1_length, target_length, begin_token, device):
    current_length = prompt_v1_length
    
    # Calculate how many more tokens we need
    tokens_needed = target_length - current_length
    print("tokens_needed: ", tokens_needed)
    if tokens_needed <= 0:
        prompt = [{"role": "user", "content": base_text}]
        return tokenizer.apply_chat_template(
            prompt,
            add_generation_prompt=True,
            return_tensors="pt"
        ).to(device)
    
    # Add required number of "begin" words
    num_begin_needed = tokens_needed // begin_token
    print("num_begin_needed: ", num_begin_needed)
    extended_text = base_text + " " + " ".join(["begin"] * num_begin_needed)
    
    # Fine-tune length if needed
    while True:
        prompt = [{"role": "user", "content": extended_text}]
        tokenized_ids = tokenizer.apply_chat_template(
            prompt,
            add_generation_prompt=True,
            return_tensors="pt"
        ).to(device)
        current_length = tokenized_ids.shape[1]
        
        if current_length == target_length:
            return tokenized_ids
        elif current_length < target_length:
            extended_text += " begin"
        else:
            # Remove last begin if we went over
            extended_text = " ".join(extended_text.split()[:-1])

def main(checkpoint, device):
    # Model initialization
    model_name = checkpoint
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        torch_dtype="auto",
        trust_remote_code=True
    )

    # Define input token lengths and output token lengths to test
    input_tokens = [128]
    output_tokens = [64,128,256,512,1024]

    base_text = "Write a long story about a cat for 5k words."
    
    # Calculate initial prompt length
    base_prompt = [{"role": "user", "content": base_text}]
    prompt_v1_tokenized_ids = tokenizer.apply_chat_template(
        base_prompt,
        add_generation_prompt=True,
        return_tensors="pt"
    ).to(model.device)
    prompt_v1_length = prompt_v1_tokenized_ids.shape[1]
    print("prompt_v1_length: ", prompt_v1_length)
    
    # Calculate token length of " begin" once
    begin_prompt = [{"role": "user", "content": base_text + " begin"}]
    begin_tokenized_ids = tokenizer.apply_chat_template(
        begin_prompt,
        add_generation_prompt=True,
        return_tensors="pt"
    ).to(model.device)
    begin_prompt_length = begin_tokenized_ids.shape[1]
    print("begin_prompt_length: ", begin_prompt_length)

    begin_token = begin_prompt_length - prompt_v1_length
    print("begin_token: ", begin_token)
    
    # warmup
    for i in range(2):
        print("warmup run: ", i)
        _ = model.generate(
            prompt_v1_tokenized_ids,
            max_new_tokens=50,
            do_sample=True,
            temperature=0.9
        )
    
    # Create results matrix
    results = []
    
    # Measure latency for each combination
    for input_len in input_tokens:
        row = []
        # Create input text of exactly desired length
        tokenized_ids = construct_input_of_length(tokenizer, base_text, prompt_v1_length, input_len, begin_token, model.device)

        print(f"Testing input length: {input_len} tokens")
        
        for output_len in output_tokens:
            print(f"Output length: {output_len} tokens")
            latency = measure_inference_latency(
                model,
                tokenizer,
                tokenized_ids,
                output_len
            )
            row.append(latency)
        results.append(row)
    
    # Create DataFrame and save to CSV
    df = pd.DataFrame(
        results,
        index=[f"{tokens} tokens" for tokens in input_tokens],
        columns=[f"{tokens} tokens" for tokens in output_tokens]
    )
    
    # Save results
    df.to_csv("./0209/" + checkpoint.split("/")[-1] + "_" + device + ".csv")
    print("\nResults saved to ./0209/" + checkpoint.split("/")[-1] + "_" + device + ".csv")
    print(df)
    
    plt.figure(figsize=(8, 6))

    for row in df.index:
        x = np.array([int(t.split()[0]) for t in df.columns])
        y = df.loc[row].values
        f = interpolate.interp1d(x, y, kind='cubic')
        xnew = np.linspace(x.min(), x.max(), 100)
        ynew = f(xnew)
        plt.plot(xnew, ynew, label=row)

    plt.xlabel('output_tokens')
    plt.ylabel('time')
    plt.title('latency_mapping')
    plt.legend(title="input_tokens")
    plt.grid(True)

    model_name_split = model_name.split("/", 1)[1]  # 分割字符串，去掉前半部分
    plt.savefig("./0209/" + model_name_split + "_output_plot_special.png")
    
    del model, tokenizer
    torch.cuda.empty_cache()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="qwen2.5-32b", help="checkpoint to use")
    parser.add_argument("--device", type=str, default="a800", help="device to run on", required=False)
    
    args = parser.parse_args()
    for model in model_list:
        print("Running model: ", model)
        main(model, args.device)