# 定义模型列表为数组
MODEL_LIST=(
    # Qwen/QwQ-32B-Preview 
    # Qwen/Qwen2.5-32B-Instruct 
    # Qwen/Qwen2.5-14B-Instruct
    # Qwen/Qwen2.5-7B-Instruct 
    # Qwen/Qwen2.5-3B-Instruct 
    # Qwen/Qwen2.5-1.5B-Instruct
    NovaSky-AI/Sky-T1-32B-Preview

)

PROMPT_TYPE_LIST=(
    "quick"
    "direct"
    "sbs"
    "c2f"
    "aav"
    # "kf"
    # "o1-mimic-hard-user"
    "sbs-hard"
    "direct-hard"
    "quick-hard"
    "c2f-hard"
    "aav-hard"
)

PROMPT_PREFIX="qwen"
OUTPUT_DIR=/data03/sunyi/time_constrained_cot/outputs/1_10


# 遍历模型列表
for MODEL_NAME_OR_PATH in "${MODEL_LIST[@]}"; do
    export CUDA_VISIBLE_DEVICES="0,4,5,6"
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