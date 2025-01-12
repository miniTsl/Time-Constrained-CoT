# 定义模型列表为数组
MODEL_LIST=(
    "mistralai/Mistral-7B-Instruct-v0.3"
    "mistralai/Ministral-8B-Instruct-2410"
    # "mistralai/Mistral-Nemo-Instruct-2407"
    # "mistralai/Mistral-Small-Instruct-2409"
    "mistralai/Mathstral-7B-v0.1"
)

# 遍历模型列表
for MODEL_NAME_OR_PATH in "${MODEL_LIST[@]}"; do
    export CUDA_VISIBLE_DEVICES="2,3"
    echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES}"
    echo "Processing model: ${MODEL_NAME_OR_PATH}"
    
    # step-by-step hard
    PROMPT_TYPE="mistral-sbs-hard"
    BUDGET=1
    OUTPUT_DIR=outputs/1_10

    bash sh/eval_hard.sh ${PROMPT_TYPE} ${MODEL_NAME_OR_PATH} ${BUDGET} ${OUTPUT_DIR}


    # step-by-step
    PROMPT_TYPE="mistral-sbs"
    BUDGET=-1
    OUTPUT_DIR=outputs/1_10
    bash sh/eval.sh ${PROMPT_TYPE} ${MODEL_NAME_OR_PATH} ${BUDGET} ${OUTPUT_DIR}

    BUDGET=1
    bash sh/eval.sh ${PROMPT_TYPE} ${MODEL_NAME_OR_PATH} ${BUDGET} ${OUTPUT_DIR}


    # coarse-to-fine
    PROMPT_TYPE="mistral-c2f"
    OUTPUT_DIR=outputs/1_10
    BUDGET=-1
    bash sh/eval.sh ${PROMPT_TYPE} ${MODEL_NAME_OR_PATH} ${BUDGET} ${OUTPUT_DIR}

    BUDGET=1
    bash sh/eval.sh ${PROMPT_TYPE} ${MODEL_NAME_OR_PATH} ${BUDGET} ${OUTPUT_DIR}


    # knowledge-first
    PROMPT_TYPE="mistral-kf"
    OUTPUT_DIR=outputs/1_10
    BUDGET=-1
    bash sh/eval.sh ${PROMPT_TYPE} ${MODEL_NAME_OR_PATH} ${BUDGET} ${OUTPUT_DIR}

    BUDGET=1
    bash sh/eval.sh ${PROMPT_TYPE} ${MODEL_NAME_OR_PATH} ${BUDGET} ${OUTPUT_DIR}

    # aav
    PROMPT_TYPE="mistral-aav"
    OUTPUT_DIR=outputs/1_10
    BUDGET=-1
    bash sh/eval.sh ${PROMPT_TYPE} ${MODEL_NAME_OR_PATH} ${BUDGET} ${OUTPUT_DIR}

    BUDGET=1
    bash sh/eval.sh ${PROMPT_TYPE} ${MODEL_NAME_OR_PATH} ${BUDGET} ${OUTPUT_DIR}
done
