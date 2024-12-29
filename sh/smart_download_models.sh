#!/bin/bash

# 模型列表
models=(
Qwen/Qwen2.5-14B-Instruct
Qwen/Qwen2.5-32B-Instruct
internlm/internlm2-math-plus-7b
internlm/internlm2-math-plus-20b
AI-MO/NuminaMath-7B-CoT
)

# 遍历每个模型并下载
for model in "${models[@]}"
do
   ./hf_guohong.sh $model  --exclude "*.bin" "*.ckpt" "*.pth" # --token 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 
done
