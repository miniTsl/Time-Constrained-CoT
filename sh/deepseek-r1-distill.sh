# 定义模型列表为数组
MODEL_LIST=(
    deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B
    deepseek-ai/DeepSeek-R1-Distill-Qwen-7B
    deepseek-ai/DeepSeek-R1-Distill-Qwen-14B
    deepseek-ai/DeepSeek-R1-Distill-Qwen-32B
    deepseek-ai/DeepSeek-R1-Distill-Llama-8B

)

PROMPT_TYPE_LIST=(
    # "quick"
    # "direct"
    "sbs"
    # "c2f"
    # "aav"
    # "kf"
    # "o1-mimic-hard-user"
    "sbs-hard"
    # "direct-hard"
    # "quick-hard"
    # "c2f-hard"
    # "aav-hard"
)

PROMPT_PREFIX="deepseek-r1-distill"
OUTPUT_DIR=/data03/sunyi/time_constrained_cot/outputs/1_10


# 遍历模型列表
for MODEL_NAME_OR_PATH in "${MODEL_LIST[@]}"; do
    export CUDA_VISIBLE_DEVICES="1,2"
    echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES}"
    echo "Processing model: ${MODEL_NAME_OR_PATH}"
    
    for PROMPT_TYPE in "${PROMPT_TYPE_LIST[@]}"; do
        # hard
        if [[ ${PROMPT_TYPE} == *"hard"* ]]; then
            BUDGET=1
            bash sh/eval.sh ${PROMPT_PREFIX}-${PROMPT_TYPE} ${MODEL_NAME_OR_PATH} ${BUDGET} ${OUTPUT_DIR}
        else
            BUDGET=-1
            bash sh/eval.sh ${PROMPT_PREFIX}-${PROMPT_TYPE} ${MODEL_NAME_OR_PATH} ${BUDGET} ${OUTPUT_DIR}
            BUDGET=1
            bash sh/eval.sh ${PROMPT_PREFIX}-${PROMPT_TYPE} ${MODEL_NAME_OR_PATH} ${BUDGET} ${OUTPUT_DIR}
        fi
    done
done