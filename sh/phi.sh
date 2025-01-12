# 定义模型列表为数组
MODEL_LIST=(
    "microsoft/Phi-3-mini-128k-instruct"
    "microsoft/Phi-3-small-128k-instruct"
    "microsoft/Phi-3-medium-128k-instruct"
    "microsoft/Phi-3.5-mini-instruct"
    "microsoft/phi-4"
)

# 遍历模型列表
for MODEL_NAME_OR_PATH in "${MODEL_LIST[@]}"; do
    export CUDA_VISIBLE_DEVICES="0,1,2,3"
    echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES}"
    echo "Processing model: ${MODEL_NAME_OR_PATH}"
    
    # set prompt_prefix, phi3mini or phi3small or phi3medium or phi4
    if [[ ${MODEL_NAME_OR_PATH} == *"mini"* ]]; then
        PROMPT_PREFIX="phi3mini"
    elif [[ ${MODEL_NAME_OR_PATH} == *"small"* ]]; then
        PROMPT_PREFIX="phi3small"
    elif [[ ${MODEL_NAME_OR_PATH} == *"medium"* ]]; then
        PROMPT_PREFIX="phi3medium"
    elif [[ ${MODEL_NAME_OR_PATH} == *"phi-4"* ]]; then
        PROMPT_PREFIX="phi4"
    fi

    # # step-by-step hard
    # PROMPT_TYPE="${PROMPT_PREFIX}-sbs-hard"
    # BUDGET=1
    # OUTPUT_DIR=outputs/1_10

    # bash sh/eval_hard.sh ${PROMPT_TYPE} ${MODEL_NAME_OR_PATH} ${BUDGET} ${OUTPUT_DIR}


    # step-by-step
    PROMPT_TYPE="${PROMPT_PREFIX}-sbs"
    BUDGET=-1
    OUTPUT_DIR=outputs/1_10
    bash sh/eval.sh ${PROMPT_TYPE} ${MODEL_NAME_OR_PATH} ${BUDGET} ${OUTPUT_DIR}

    # BUDGET=1
    # bash sh/eval.sh ${PROMPT_TYPE} ${MODEL_NAME_OR_PATH} ${BUDGET} ${OUTPUT_DIR}


    # # coarse-to-fine
    # PROMPT_TYPE="${PROMPT_PREFIX}-c2f"
    # OUTPUT_DIR=outputs/1_10
    # BUDGET=-1
    # bash sh/eval.sh ${PROMPT_TYPE} ${MODEL_NAME_OR_PATH} ${BUDGET} ${OUTPUT_DIR}

    # # BUDGET=1
    # # bash sh/eval.sh ${PROMPT_TYPE} ${MODEL_NAME_OR_PATH} ${BUDGET} ${OUTPUT_DIR}


    # # knowledge-first
    # PROMPT_TYPE="${PROMPT_PREFIX}-kf"
    # OUTPUT_DIR=outputs/1_10
    # BUDGET=-1
    # bash sh/eval.sh ${PROMPT_TYPE} ${MODEL_NAME_OR_PATH} ${BUDGET} ${OUTPUT_DIR}

    # # BUDGET=1
    # # bash sh/eval.sh ${PROMPT_TYPE} ${MODEL_NAME_OR_PATH} ${BUDGET} ${OUTPUT_DIR}

    # # aav
    # PROMPT_TYPE="${PROMPT_PREFIX}-aav"
    # OUTPUT_DIR=outputs/1_10
    # BUDGET=-1
    # bash sh/eval.sh ${PROMPT_TYPE} ${MODEL_NAME_OR_PATH} ${BUDGET} ${OUTPUT_DIR}

    # # BUDGET=1
    # # bash sh/eval.sh ${PROMPT_TYPE} ${MODEL_NAME_OR_PATH} ${BUDGET} ${OUTPUT_DIR}
done
