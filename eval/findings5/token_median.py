import os
import json
import random
import json
import os
import numpy as np
from pathlib import Path
from typing import Iterable, Union, Any
from transformers import AutoTokenizer
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import sys
import pandas as pd
import statistics
sys.path.append("..") 
from prompts import PROMPT_TEMPLATES,CHAT_TEMPLATE_FORMATS

file_dir = "/data03/sunyi/time_constrained_cot/outputs/2_6"

# model_list = [
#     "NovaSky-AI/Sky-T1-32B-Preview",
#     "Qwen/QwQ-32B-Preview",
#     "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
#     "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
#     "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
#     "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
#     "Qwen/Qwen2.5-32B-Instruct",
#     "Qwen/Qwen2.5-14B-Instruct", 
#     "Qwen/Qwen2.5-7B-Instruct", 
#     "Qwen/Qwen2.5-3B-Instruct", 
#     "Qwen/Qwen2.5-1.5B-Instruct",
#     "Qwen/Qwen2.5-Math-1.5B-Instruct",
#     "Qwen/Qwen2.5-Math-7B-Instruct",
#     "mistralai/Mistral-Small-Instruct-2409",
#     "mistralai/Mistral-Nemo-Instruct-2407",
#     "mistralai/Ministral-8B-Instruct-2410",
#     "mistralai/Mathstral-7B-v0.1",
#     "google/gemma-2-27b-it",
#     "google/gemma-2-9b-it",
#     "google/gemma-2-2b-it",
#     "microsoft/phi-4",
#     "microsoft/Phi-3-medium-128k-instruct",
#     "microsoft/Phi-3-small-128k-instruct",
#     "microsoft/Phi-3-mini-128k-instruct",
#     "microsoft/Phi-3.5-mini-instruct",
# ]

model_list = [
    "microsoft/Phi-3-small-128k-instruct",
    #"Qwen/QwQ-32B-Preview": "qwen",
    #"Qwen/Qwen2.5-32B-Instruct": "qwen",
    #"Qwen/Qwen2.5-14B-Instruct": "qwen",
    #"Qwen/Qwen2.5-7B-Instruct": "qwen",
    #"Qwen/Qwen2.5-3B-Instruct": "qwen",
    "Qwen/Qwen2.5-1.5B-Instruct",
    #"Qwen/Qwen2.5-Math-1.5B-Instruct": "qwen-math",
    #"Qwen/Qwen2.5-Math-7B-Instruct": "qwen-math",
    #"internlm/internlm2_5-1_8b-chat": "internlm",
    #"internlm/internlm2_5-7b-chat": "internlm",
    #"internlm/internlm2_5-20b-chat": "internlm",
    "google/gemma-2-2b-it",
    #"google/gemma-2-9b-it": "gemma",
    #"google/gemma-2-27b-it": "gemma",
    "mistralai/Mathstral-7B-v0.1",
    #"mistralai/Ministral-8B-Instruct-2410": "mistral",
    #"mistralai/Mistral-Nemo-Instruct-2407": "mistral",
    #"mistralai/Mistral-Small-Instruct-2409": "mistral",
    "microsoft/phi-4",
    "microsoft/Phi-3-medium-128k-instruct",
    #"microsoft/Phi-3.5-mini-instruct": "phi3mini",
    "microsoft/Phi-3-mini-128k-instruct",
    #"NovaSky-AI/Sky-T1-32B-Preview": "qwen",
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
    # "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B": "deepseek-r1-distill",
    # "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B": "deepseek-r1-distill",
    # "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B": "deepseek-r1-distill",
    # "deepseek-ai/DeepSeek-R1-Distill-Llama-8B": "deepseek-r1-distill",
    #"meta-llama/Llama-3.2-3B-Instruct": "llama",
    "meta-llama/Llama-3.2-1B-Instruct",
    #"meta-llama/Llama-3.1-8B-Instruct": "llama"
]

model_list_o1 = [
    "NovaSky-AI/Sky-T1-32B-Preview",
    "Qwen/QwQ-32B-Preview",
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
]

model_list_not_o1=[
    "Qwen/Qwen2.5-32B-Instruct",
    "Qwen/Qwen2.5-14B-Instruct", 
    "Qwen/Qwen2.5-7B-Instruct", 
    "Qwen/Qwen2.5-3B-Instruct", 
    "Qwen/Qwen2.5-1.5B-Instruct",
    "Qwen/Qwen2.5-Math-1.5B-Instruct",
    "Qwen/Qwen2.5-Math-7B-Instruct",
    "mistralai/Mistral-Small-Instruct-2409",
    "mistralai/Mistral-Nemo-Instruct-2407",
    "mistralai/Ministral-8B-Instruct-2410",
    "mistralai/Mathstral-7B-v0.1",
    "google/gemma-2-27b-it",
    "google/gemma-2-9b-it",
    "google/gemma-2-2b-it",
    "microsoft/phi-4",
    "microsoft/Phi-3-medium-128k-instruct",
    "microsoft/Phi-3-small-128k-instruct",
    "microsoft/Phi-3-mini-128k-instruct",
    "microsoft/Phi-3.5-mini-instruct",
]

PROMP_LIST = ["-c2f","-sbs","-aav"]

MODEL_SERIES_MAP = {
    "Qwen/QwQ-32B-Preview": "qwen",
    "Qwen/Qwen2.5-32B-Instruct": "qwen",
    "Qwen/Qwen2.5-14B-Instruct": "qwen",
    "Qwen/Qwen2.5-7B-Instruct": "qwen",
    "Qwen/Qwen2.5-3B-Instruct": "qwen",
    "Qwen/Qwen2.5-1.5B-Instruct": "qwen",
    "Qwen/Qwen2.5-Math-1.5B-Instruct": "qwen-math",
    "Qwen/Qwen2.5-Math-7B-Instruct": "qwen-math",
    "internlm/internlm2_5-1_8b-chat": "internlm",
    "internlm/internlm2_5-7b-chat": "internlm",
    "internlm/internlm2_5-20b-chat": "internlm",
    "google/gemma-2-2b-it": "gemma",
    "google/gemma-2-9b-it": "gemma",
    "google/gemma-2-27b-it": "gemma",
    "mistralai/Mathstral-7B-v0.1": "mistral",
    "mistralai/Ministral-8B-Instruct-2410": "mistral",
    "mistralai/Mistral-Nemo-Instruct-2407": "mistral",
    "mistralai/Mistral-Small-Instruct-2409": "mistral",
    "microsoft/phi-4": "phi4",
    "microsoft/Phi-3-medium-128k-instruct": "phi3medium",
    "microsoft/Phi-3-small-128k-instruct": "phi3small",
    "microsoft/Phi-3.5-mini-instruct": "phi3mini",
    "microsoft/Phi-3-mini-128k-instruct": "phi3mini",
    "NovaSky-AI/Sky-T1-32B-Preview": "qwen",
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B": "deepseek-r1-distill",
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B": "deepseek-r1-distill",
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B": "deepseek-r1-distill",
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B": "deepseek-r1-distill",
    "deepseek-ai/DeepSeek-R1-Distill-Llama-8B": "deepseek-r1-distill",
    "meta-llama/Llama-3.2-3B-Instruct": "llama",
    "meta-llama/Llama-3.2-1B-Instruct": "llama",
    "meta-llama/Llama-3.1-8B-Instruct": "llama"
}

o1_like_models = [
            "Qwen/QwQ-32B-Preview", 
            "Skywork/Skywork-o1-Open-Llama-3.1-8B", 
            "PowerInfer/SmallThinker-3B-Preview",
            "NovaSky-AI/Sky-T1-32B-Preview", 
            "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
            "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
            "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
            "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
            "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
]

Model_Name_Map = {
    "Qwen/QwQ-32B-Preview": "QwQ",
    "NovaSky-AI/Sky-T1-32B-Preview": "Sky-T1",
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B": "DRD-Qwen-1.5B",
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B": "DRD-Qwen-7B",
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B": "DRD-Qwen-14B",
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B": "DRD-Qwen-32B",
    "deepseek-ai/DeepSeek-R1-Distill-Llama-8B": "DRD-Llama-8B",
    "Qwen/Qwen2.5-Math-1.5B-Instruct": "Qwen2.5-Math-1.5B",
    "Qwen/Qwen2.5-Math-7B-Instruct": "Qwen2.5-Math-7B",
    "mistralai/Mathstral-7B-v0.1": "Mathstral-7B",
    "Qwen/Qwen2.5-32B-Instruct": "Qwen2.5-32B",
    "Qwen/Qwen2.5-14B-Instruct": "Qwen2.5-14B",
    "Qwen/Qwen2.5-7B-Instruct": "Qwen2.5-7B",
    "Qwen/Qwen2.5-3B-Instruct": "Qwen2.5-3B",
    "Qwen/Qwen2.5-1.5B-Instruct": "Qwen2.5-1.5B",
    "mistralai/Mistral-Small-Instruct-2409": "Mistral-Small",
    "mistralai/Mistral-Nemo-Instruct-2407": "Mistral-Nemo",
    "mistralai/Ministral-8B-Instruct-2410": "Ministral-8B",
    "google/gemma-2-27b-it": "Gemma-27B",
    "google/gemma-2-9b-it": "Gemma-9B",
    "google/gemma-2-2b-it": "Gemma-2B",
    "microsoft/Phi-3-medium-128k-instruct": "Phi-3-Medium",
    "microsoft/Phi-3-small-128k-instruct": "Phi-3-Small",
    "microsoft/Phi-3.5-mini-instruct": "Phi-3.5-Mini",
    "microsoft/Phi-3-mini-128k-instruct": "Phi-3-Mini",
    "microsoft/phi-4": "Phi-4",
    "meta-llama/Llama-3.1-8B-Instruct": "Llama-3.1-8B",
    "meta-llama/Llama-3.2-3B-Instruct": "Llama-3.2-3B",
    "meta-llama/Llama-3.2-1B-Instruct": "Llama-3.2-1B",
}

MODEL_SERIES_PROMPT_TYPE_MAP = {
    "qwen": ["qwen" + prompt for prompt in PROMP_LIST],
    "qwen-math": ["qwen-math" + prompt for prompt in PROMP_LIST],
    "internlm": ["internlm"+prompt for prompt in PROMP_LIST],
    "mistral": ["mistral"+prompt for prompt in PROMP_LIST],
    "gemma": ["gemma"+prompt for prompt in PROMP_LIST],
    "phi3mini": ["phi3mini"+prompt for prompt in PROMP_LIST],
    "phi3small": ["phi3small"+prompt for prompt in PROMP_LIST],
    "phi3medium": ["phi3medium"+prompt for prompt in PROMP_LIST],
    "phi4": ["phi4"+prompt for prompt in PROMP_LIST],
    "deepseek-r1-distill": ["deepseek-r1-distill"+prompt for prompt in PROMP_LIST],
    "llama": ["llama"+prompt for prompt in PROMP_LIST]
}

datasets =["gsm8k","math500"]

num_samples = {
    "gsm8k":1319,
    "math500":500
}

def load_data_with_cropped_cot(model,prompt,dataset):
    print(model,prompt,dataset)
    full_cot_dir = f"{file_dir}/{model}/{prompt}/{dataset}"
    if not os.path.exists(full_cot_dir):
        return "文件夹不存在"
    all_files = os.listdir(full_cot_dir)
    jsonl_files = [f for f in all_files if f.endswith('.jsonl')]
    full_cot_path = os.path.join(full_cot_dir, jsonl_files[0])
    samples = list(load_jsonl(full_cot_path))
    questions = [sample["question"] for sample in samples]
    print(PROMPT_TEMPLATES[prompt][0])
    prompts = [
        PROMPT_TEMPLATES[prompt][0].format(input=question,token_budget = 1000)
        for question in questions
    ]
    #print(full_cots)
    # use tokenizer to batch crop full_cots
    tokenizer = AutoTokenizer.from_pretrained(model, padding_side="right", trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    
    full_cots_tokens = tokenizer(prompts, return_tensors="pt", padding=True).input_ids
    row_lengths = [torch.sum(row != tokenizer.pad_token_id).item() for row in full_cots_tokens]
    #print(row_lengths)
    #print(row_lengths)
    median = statistics.median(row_lengths)
    df.loc[MODEL_SERIES_MAP[model],f"{dataset}-{prompt.split('-')[-1]}"] = median
    print(df)

def load_jsonl(file: Union[str, Path]) -> Iterable[Any]:
    with open(file, "r", encoding="utf-8") as f:
        for line in f:
            try:
                yield json.loads(line)
            except:
                print("Error in loading:", line)
                exit()
                
if __name__ == "__main__":
    data = {
        "gsm8k-sbs": [None, None, None, None, None, None, None, None, None],  # 初始化为空
        "gsm8k-c2f": [None, None, None, None, None, None, None, None, None],
        "gsm8k-aav": [None, None, None, None, None, None, None, None, None],
        "math500-sbs": [None, None, None, None, None, None, None, None, None],
        "math500-c2f": [None, None, None, None, None, None, None, None, None],
        "math500-aav": [None, None, None, None, None, None, None, None, None]
    }
    df = pd.DataFrame(data,index=["mistral","qwen","phi3mini","phi3small","phi3medium","phi4","llama","gemma","deepseek-r1-distill"])
    for model in model_list:
        #prompt类型
        for prompt in MODEL_SERIES_PROMPT_TYPE_MAP[MODEL_SERIES_MAP[model]]:
            #数据类型
            for dataset in datasets:
                load_data_with_cropped_cot(model,prompt,dataset)
                
    df.to_csv('toeken_median.csv', index=False)