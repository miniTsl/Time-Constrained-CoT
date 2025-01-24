#!/bin/bash

# 模型列表
models=(
# Qwen/Qwen2.5-14B-Instruct
# Qwen/Qwen2.5-32B-Instruct
# internlm/internlm2-math-plus-7b
# internlm/internlm2-math-plus-20b
# AI-MO/NuminaMath-7B-CoT
# Skywork/Skywork-o1-Open-Llama-3.1-8B
# deepseek-ai/deepseek-math-7b-instruct
# deepseek-ai/deepseek-math-7b-rl
# Skywork/Skywork-o1-Open-PRM-Qwen-2.5-1.5B
# Skywork/Skywork-o1-Open-PRM-Qwen-2.5-7B
# PowerInfer/SmallThinker-3B-Preview
# internlm/internlm2-math-plus-1_8b
# deepseek-ai/DeepSeek-V2-Lite-Chat
# internlm/internlm2_5-20b-chat
# internlm/internlm2_5-7b-chat
# internlm/internlm2_5-1_8b-chat
# internlm/internlm2-chat-7b
# internlm/internlm2-chat-20b
# internlm/internlm2-chat-1_8b
# mistralai/Mistral-7B-Instruct-v0.3
# mistralai/Mistral-Nemo-Instruct-2407
# mistralai/Mistral-Small-Instruct-2409
# mistralai/Ministral-8B-Instruct-2410
# microsoft/phi-4
# microsoft/Phi-3-mini-128k-instruct
# microsoft/Phi-3-small-128k-instruct
# microsoft/Phi-3-medium-128k-instruct
# microsoft/Phi-3.5-mini-instruct
# NovaSky-AI/Sky-T1-32B-Preview
# google/gemma-2-27b-it
# THUDM/glm-4-9b-chat
# THUDM/glm-edge-1.5b-chat
# THUDM/glm-edge-4b-chat
# 01-ai/Yi-1.5-6B-Chat
# 01-ai/Yi-1.5-9B-Chat-16K
# 01-ai/Yi-1.5-34B-Chat-16K
# meta-llama/Llama-3.3-70B-Instruct
deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B
deepseek-ai/DeepSeek-R1-Distill-Qwen-7B
deepseek-ai/DeepSeek-R1-Distill-Qwen-14B
deepseek-ai/DeepSeek-R1-Distill-Qwen-32B
deepseek-ai/DeepSeek-R1-Distill-Llama-8B
)

# 遍历每个模型并下载
for model in "${models[@]}"
do
   ./hf_guohong.sh $model  --exclude  "*.ckpt" "*.pth" #--token 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 #"consolidated.safetensors" # --token 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 
done
