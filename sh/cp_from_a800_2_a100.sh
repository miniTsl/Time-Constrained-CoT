model_list=(
    "mistralai/Mistral-7B-Instruct-v0.3"
    "mistralai/Ministral-8B-Instruct-2410"
    "mistralai/Mistral-Nemo-Instruct-2407"
    "mistralai/Mistral-Small-Instruct-2409"
    "mistralai/Mathstral-7B-v0.1"
    
    "Qwen/Qwen2.5-32B-Instruct"
    "Qwen/Qwen2.5-14B-Instruct"
    "Qwen/Qwen2.5-7B-Instruct"
    "Qwen/Qwen2.5-3B-Instruct"
    "Qwen/Qwen2.5-1.5B-Instruct"
    "Qwen/Qwen2.5-Math-1.5B-Instruct"
    "Qwen/Qwen2.5-Math-7B-Instruct"
    "Qwen/QwQ-32B-Preview"
    
    "microsoft/Phi-3-mini-128k-instruct"
    "microsoft/Phi-3-small-128k-instruct"
    "microsoft/Phi-3-medium-128k-instruct"
    "microsoft/Phi-3.5-mini-instruct"
    "microsoft/phi-4"
    
    "meta-llama/Llama-3.2-3B-Instruct"
    "meta-llama/Llama-3.2-1B-Instruct"
    "meta-llama/Llama-3.1-8B-Instruct"
    
    "google/gemma-2-9b-it"
    "google/gemma-2-2b-it"

    "AI-MO/NuminaMath-7B-CoT"

    "internlm/internlm2_5-1_8b-chat"
    "internlm/internlm2_5-7b-chat"
    "internlm/internlm2_5-20b-chat"
    "internlm/internlm2-math-plus-1_8b"
    "internlm/internlm2-math-plus-20b"
    "internlm/internlm2-math-plus-7b"
    
    "deepseek-ai/deepseek-math-7b-instruct"
    "deepseek-ai/deepseek-math-7b-rl"
    
    "PowerInfer/SmallThinker-3B-Preview"

    "Skywork/Skywork-o1-Open-Llama-3.1-8B"
)

for model in "${model_list[@]}"; do
    rsync -avz --info=progress2  -e  "ssh -p 2233" /data03/sunyi/hf_cache/hub/models--microsoft--Phi-3.5-mini-instruct sunyi@10.0.0.10:/data/sunyi/hf_cache/hub/models--microsoft--Phi-3.5-mini-instruct
done