# 定义模型列表为数组
MODEL_LIST=(PowerInfer/SmallThinker-3B-Preview)

# 遍历模型列表
for MODEL_NAME_OR_PATH in "${MODEL_LIST[@]}"; do
    export CUDA_VISIBLE_DEVICES="1,2,3,4"
    echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES}"
    echo "Processing model: ${MODEL_NAME_OR_PATH}"
    

    # original
    PROMPT_TYPE="smallthinker-step-by-step"
    BUDGET=-1
    OUTPUT_DIR=outputs/12_26

    bash sh/eval.sh ${PROMPT_TYPE} ${MODEL_NAME_OR_PATH} ${BUDGET} ${OUTPUT_DIR}

    BUDGET=1
    bash sh/eval.sh ${PROMPT_TYPE} ${MODEL_NAME_OR_PATH} ${BUDGET} ${OUTPUT_DIR}


    # coarse-to-fine
    PROMPT_TYPE="smallthinker-coarse-to-fine"
    BUDGET=-1
    OUTPUT_DIR=outputs/12_26
    bash sh/eval.sh ${PROMPT_TYPE} ${MODEL_NAME_OR_PATH} ${BUDGET} ${OUTPUT_DIR}

    BUDGET=1
    bash sh/eval.sh ${PROMPT_TYPE} ${MODEL_NAME_OR_PATH} ${BUDGET} ${OUTPUT_DIR}

    # hard
    PROMPT_TYPE="smallthinker-step-by-step-hard"
    OUTPUT_DIR=outputs/12_26
    BUDGET=1
    bash sh/eval_hard.sh ${PROMPT_TYPE} ${MODEL_NAME_OR_PATH} ${BUDGET} ${OUTPUT_DIR}

done