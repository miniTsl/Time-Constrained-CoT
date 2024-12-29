# 定义模型列表为数组
MODEL_LIST=(Qwen/Qwen2.5-32B-Instruct Qwen/Qwen2.5-14B-Instruct Qwen/Qwen2.5-7B-Instruct Qwen/Qwen2.5-3B-Instruct Qwen/Qwen2.5-1.5B-Instruct Qwen/Qwen2.5-0.5B-Instruct)

# 遍历模型列表
for MODEL_NAME_OR_PATH in "${MODEL_LIST[@]}"; do
    # 如果是0.5B的模型，则只使用6和7两张GPU，否则使用4，5，6，7四张GPU
    if [[ ${MODEL_NAME_OR_PATH} == *"0.5B"* ]]; then
        export CUDA_VISIBLE_DEVICES="1,2"
    else
        export CUDA_VISIBLE_DEVICES="1,2,3,4"
    fi
    echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES}"
    echo "Processing model: ${MODEL_NAME_OR_PATH}"
    

    # # original
    # PROMPT_TYPE="qwen25-math-cot"
    # BUDGET=-1
    # OUTPUT_DIR=outputs/12_26

    # bash sh/eval.sh ${PROMPT_TYPE} ${MODEL_NAME_OR_PATH} ${BUDGET} ${OUTPUT_DIR}

    # BUDGET=1
    # bash sh/eval.sh ${PROMPT_TYPE} ${MODEL_NAME_OR_PATH} ${BUDGET} ${OUTPUT_DIR}


    # # coarse-to-fine
    # PROMPT_TYPE="coarse-to-fine-qwen"
    # BUDGET=-1
    # OUTPUT_DIR=outputs/12_26
    # bash sh/eval.sh ${PROMPT_TYPE} ${MODEL_NAME_OR_PATH} ${BUDGET} ${OUTPUT_DIR}

    # BUDGET=1
    # bash sh/eval.sh ${PROMPT_TYPE} ${MODEL_NAME_OR_PATH} ${BUDGET} ${OUTPUT_DIR}

    # hard
    PROMPT_TYPE="qwen25-step-by-step-hard"
    OUTPUT_DIR=outputs/12_26
    BUDGET=1
    bash sh/eval.sh ${PROMPT_TYPE} ${MODEL_NAME_OR_PATH} ${BUDGET} ${OUTPUT_DIR}

done