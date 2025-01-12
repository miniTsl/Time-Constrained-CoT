# 定义模型列表为数组
MODEL_LIST=(
    "google/gemma-2-9b-it"
    # "google/gemma-2-2b-it"
)

# 遍历模型列表
for MODEL_NAME_OR_PATH in "${MODEL_LIST[@]}"; do
    export CUDA_VISIBLE_DEVICES="0,1"
    echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES}"
    echo "Processing model: ${MODEL_NAME_OR_PATH}"
    
    # step-by-step hard
    PROMPT_TYPE="gemma-sbs-hard"
    BUDGET=1
    OUTPUT_DIR=outputs/1_10

    bash sh/eval_hard.sh ${PROMPT_TYPE} ${MODEL_NAME_OR_PATH} ${BUDGET} ${OUTPUT_DIR}


    # step-by-step
    PROMPT_TYPE="gemma-sbs"
    BUDGET=-1
    OUTPUT_DIR=outputs/1_10
    bash sh/eval.sh ${PROMPT_TYPE} ${MODEL_NAME_OR_PATH} ${BUDGET} ${OUTPUT_DIR}

    BUDGET=1
    bash sh/eval.sh ${PROMPT_TYPE} ${MODEL_NAME_OR_PATH} ${BUDGET} ${OUTPUT_DIR}


    # coarse-to-fine
    PROMPT_TYPE="gemma-c2f"
    OUTPUT_DIR=outputs/1_10
    BUDGET=-1
    bash sh/eval.sh ${PROMPT_TYPE} ${MODEL_NAME_OR_PATH} ${BUDGET} ${OUTPUT_DIR}

    BUDGET=1
    bash sh/eval.sh ${PROMPT_TYPE} ${MODEL_NAME_OR_PATH} ${BUDGET} ${OUTPUT_DIR}


    # knowledge-first
    PROMPT_TYPE="gemma-kf"
    OUTPUT_DIR=outputs/1_10
    BUDGET=-1
    bash sh/eval.sh ${PROMPT_TYPE} ${MODEL_NAME_OR_PATH} ${BUDGET} ${OUTPUT_DIR}

    BUDGET=1
    bash sh/eval.sh ${PROMPT_TYPE} ${MODEL_NAME_OR_PATH} ${BUDGET} ${OUTPUT_DIR}

    # aav
    PROMPT_TYPE="gemma-aav"
    OUTPUT_DIR=outputs/1_10
    BUDGET=-1
    bash sh/eval.sh ${PROMPT_TYPE} ${MODEL_NAME_OR_PATH} ${BUDGET} ${OUTPUT_DIR}

    BUDGET=1
    bash sh/eval.sh ${PROMPT_TYPE} ${MODEL_NAME_OR_PATH} ${BUDGET} ${OUTPUT_DIR}
done
