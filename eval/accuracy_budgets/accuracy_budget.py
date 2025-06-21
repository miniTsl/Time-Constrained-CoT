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
import random
import matplotlib
import re

file_dir = "/data03/sunyi/time_constrained_cot/outputs/2_6"

model_list = [
    "NovaSky-AI/Sky-T1-32B-Preview",
    "Qwen/QwQ-32B-Preview",
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
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

PROMP_LIST = ["-c2f"]

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

def equally_spaced_select(counts, labels, num=6):
    # 检查列表长度是否足够
    if len(counts) < num or len(labels) < num:
        print("列表长度不足，无法取出指定数量的元素。")
        return [], []
    # 计算间距
    step = (len(counts) - 1) // (num - 1)
    # 等间距选取元素
    selected_counts = [counts[i * step] for i in range(num)]
    selected_labels = [labels[i * step] for i in range(num)]
    return selected_counts, selected_labels

if __name__ == "__main__":
    #模型名
    #gsm8k_random_idx = [random.randint(0, num_samples["gsm8k"]-1) for _ in range(5)]
    gsm8k_random_idx = [51]
    math500_random_idx = [random.randint(0, num_samples["math500"]-1) for _ in range(5)]
    print(gsm8k_random_idx,math500_random_idx)
    for model in model_list:
        #prompt类型
        for prompt in MODEL_SERIES_PROMPT_TYPE_MAP[MODEL_SERIES_MAP[model]]:
            #数据类型
            for dataset in datasets:
                plt.figure(figsize=(15, 6)) 
                if dataset == "gsm8k":
                    random_idx = gsm8k_random_idx
                else:
                    random_idx = math500_random_idx
                folder_path = f"/data03/sunyi/time_constrained_cot/outputs/2_6/{model}/{prompt}/{dataset}"
                if not os.path.exists(folder_path):
                    print(f"文件夹不存在: {folder_path}")
                    continue
                #遍历不同budget
                labels=[]
                counts=[]
                for file_name in os.listdir(folder_path):
                    if file_name.endswith('.jsonl'):
                        file_path = os.path.join(folder_path, file_name)
                        count = []
                        with open(file_path, 'r') as f:
                            for line in f:
                                data = json.loads(line)
                                #print(data['score'],data['idx'])
                                if data['idx'] in random_idx :
                                    if data['score'] == [True]:
                                        count.append(1)
                                    else:
                                        count.append(-1)
                        #print(file_name)
                        match = re.search(r'b(\d+)', file_name)
                        if match == None:
                            continue
                        counts.append(count)
                        labels.append(int(match.group(1)))
                        combined = list(zip(labels, counts))

                        sorted_combined = sorted(combined, key=lambda x: x[0])

                        sorted_labels, sorted_counts = zip(*sorted_combined)

                        labels = list(sorted_labels)
                        counts = list(sorted_counts)

                #print(labels)
                #print(counts)
                #print(labels)
                
                counts, labels = equally_spaced_select(counts, labels)
                
                group_gap = 1.0
                bar_gap = 0.1
                
                n = len(counts)
                m = len(counts[0])
                
                bar_width = (1 - (m - 1) * bar_gap) / m
                # 起始位置
                group_starts = np.arange(n) * (m * bar_width + (m - 1) * bar_gap + group_gap)
                # 颜色
                colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
                
                for i in range(n):
                    group_data = counts[i]
                    group_x = group_starts[i] + np.arange(m) * (bar_width + bar_gap)
                    for j in range(m):
                        plt.bar(group_x[j], group_data[j], width=bar_width, color=colors[j])
                    
                plt.xticks(group_starts + (m * bar_width + (m - 1) * bar_gap) / 2, labels,fontsize=10)
                plt.yticks([-1, 0, 1])
                plt.title(Model_Name_Map[model]+prompt+dataset)
                plt.xlabel('token budgets')
                plt.ylabel('Scores')
                
                save_folder_path = "accuracy_budgets/"+model+"/"+prompt
                if not os.path.exists(save_folder_path):
                    os.makedirs(save_folder_path)
                plt.savefig(save_folder_path+"/"+dataset, dpi=300, bbox_inches='tight')
                matplotlib.pyplot.close()