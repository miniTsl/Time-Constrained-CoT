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

file_dir = "/data03/sunyi/time_constrained_cot/outputs/2_6"

model_list = [
    #"NovaSky-AI/Sky-T1-32B-Preview",
    "meta-llama/Llama-3.1-8B-Instruct",
    "Qwen/Qwen2.5-7B-Instruct", 
    "Qwen/QwQ-32B-Preview",
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

PROMP_LIST = ["-sbs-hard"]


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

def load_data_with_cropped_cot(full_cot_dir_path,model):
    
    files = os.listdir(full_cot_dir_path)
    file_list = [f for f in files if os.path.isfile(os.path.join(full_cot_dir_path, f))]

    #o1为8192
    if model in o1_like_models:
        budgets = 8192
    else:
        budgets = 4096
        
    #找到目标文件
    filtered_files = [f for f in file_list if str(budgets) in f and f.endswith(".jsonl")]
    if len(filtered_files) != 1 :
        raise ValueError(f"Error: Multiple matching files found: {filtered_files}")
    else:
        selected_file = filtered_files[0]
    #print(selected_file)
    #提取code
    samples = list(load_jsonl(full_cot_dir_path+"/"+selected_file))
    full_cots = [sample["code"] for sample in samples]
    full_cots = [item[0] for item in full_cots]
    
    # use tokenizer to batch crop full_cots
    tokenizer = AutoTokenizer.from_pretrained(model,trust_remote_code=True, padding_side="right")
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    full_cots_tokens = tokenizer(full_cots, return_tensors="pt", padding=True).input_ids
    #统计
    token_lengths = [torch.sum(row != tokenizer.eos_token_id).item() for row in full_cots_tokens]
    
    plt.boxplot(token_lengths,showfliers=False,positions=[cnt])
    
    
def load_jsonl(file: Union[str, Path]) -> Iterable[Any]:
    print(file)
    with open(file, "r", encoding="utf-8") as f:
        for line in f:
            try:
                yield json.loads(line)
            except:
                print("Error in loading:", line)
                exit()
                
if __name__ == "__main__":
    for dataset in datasets:
        plt.figure(figsize=(7, 6))
        x_list = []
        cnt=1
        for model in model_list:
            for prompt in MODEL_SERIES_PROMPT_TYPE_MAP[MODEL_SERIES_MAP[model]]:
                load_data_with_cropped_cot("/data03/sunyi/time_constrained_cot/outputs/2_6"+"/"+model+"/"+prompt+"/"+dataset,model)
                cnt=cnt+1
                #print(model+"_"+prompt+"_"+dataset)
                x_list.append(Model_Name_Map[model])
                
        plt.title('Boxplot of Token Lengths')
        plt.ylabel('Token Length',fontsize=15)
        plt.xticks(range(1, len(x_list) + 1), x_list, fontsize=15)
        plt.yticks(fontsize=15)
        plt.savefig(dataset+'_boxplot.png', dpi=300)
        plt.show()
        plt.close()