# 定义模型列表为数组
MODEL_LIST=(
    # Qwen/QwQ-32B-Preview 
    # Qwen/Qwen2.5-32B-Instruct 
    Qwen/Qwen2.5-14B-Instruct
)
# Qwen/Qwen2.5-7B-Instruct Qwen/Qwen2.5-3B-Instruct Qwen/Qwen2.5-1.5B-Instruct)

# 遍历模型列表
for MODEL_NAME_OR_PATH in "${MODEL_LIST[@]}"; do
    export CUDA_VISIBLE_DEVICES="4,5,6,7"
    echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES}"
    echo "Processing model: ${MODEL_NAME_OR_PATH}"
    
    # # step-by-step hard
    # PROMPT_TYPE="qwen-sbs-hard"
    # OUTPUT_DIR=outputs/1_10
    # BUDGET=1
    # bash sh/eval_hard.sh ${PROMPT_TYPE} ${MODEL_NAME_OR_PATH} ${BUDGET} ${OUTPUT_DIR}

    # step-by-step
    PROMPT_TYPE="qwen-sbs"
    BUDGET=-1
    OUTPUT_DIR=outputs/1_10
    bash sh/eval.sh ${PROMPT_TYPE} ${MODEL_NAME_OR_PATH} ${BUDGET} ${OUTPUT_DIR}

    BUDGET=1
    bash sh/eval.sh ${PROMPT_TYPE} ${MODEL_NAME_OR_PATH} ${BUDGET} ${OUTPUT_DIR}


    # coarse-to-fine
    PROMPT_TYPE="qwen-c2f"
    BUDGET=-1
    OUTPUT_DIR=outputs/1_10
    bash sh/eval.sh ${PROMPT_TYPE} ${MODEL_NAME_OR_PATH} ${BUDGET} ${OUTPUT_DIR}

    BUDGET=1
    bash sh/eval.sh ${PROMPT_TYPE} ${MODEL_NAME_OR_PATH} ${BUDGET} ${OUTPUT_DIR}


    # # knowledge-first
    # PROMPT_TYPE="qwen-kf"
    # BUDGET=-1
    # OUTPUT_DIR=outputs/1_10
    # bash sh/eval.sh ${PROMPT_TYPE} ${MODEL_NAME_OR_PATH} ${BUDGET} ${OUTPUT_DIR}

    # BUDGET=1
    # bash sh/eval.sh ${PROMPT_TYPE} ${MODEL_NAME_OR_PATH} ${BUDGET} ${OUTPUT_DIR}


    # # answer-and-verify
    # PROMPT_TYPE="qwen-aav"
    # BUDGET=-1
    # OUTPUT_DIR=outputs/1_10
    # bash sh/eval.sh ${PROMPT_TYPE} ${MODEL_NAME_OR_PATH} ${BUDGET} ${OUTPUT_DIR}

    # BUDGET=1
    # bash sh/eval.sh ${PROMPT_TYPE} ${MODEL_NAME_OR_PATH} ${BUDGET} ${OUTPUT_DIR}

done