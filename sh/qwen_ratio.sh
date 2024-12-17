# 定义模型列表为数组
MODEL_LIST=(Qwen/Qwen2.5-7B-Instruct Qwen/Qwen2.5-3B-Instruct Qwen/Qwen2.5-1.5B-Instruct Qwen/Qwen2.5-0.5B-Instruct)

# 遍历模型列表
for MODEL_NAME_OR_PATH in "${MODEL_LIST[@]}"; do
    # 如果是0.5B的模型，则只使用6和7两张GPU，否则使用4，5，6，7四张GPU
    if [[ ${MODEL_NAME_OR_PATH} == *"0.5B"* ]]; then
        export CUDA_VISIBLE_DEVICES="6,7"
    else
        export CUDA_VISIBLE_DEVICES="4,5,6,7"
    fi
    echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES}"
    echo "Processing model: ${MODEL_NAME_OR_PATH}"
    
    # original
    PROMPT_TYPE="qwen25-math-cot"
    RATIO=-1
    OUTPUT_DIR=${MODEL_NAME_OR_PATH}/$PROMPT_TYPE

    # bash sh/eval.sh ${PROMPT_TYPE} ${MODEL_NAME_OR_PATH} ${RATIO} ${OUTPUT_DIR}

    for RATIO in $(seq 0.05 0.05 1.0); do
        echo "Processing ratio: ${RATIO}"
        bash sh/eval.sh ${PROMPT_TYPE} ${MODEL_NAME_OR_PATH} $RATIO ${OUTPUT_DIR}
    done


    # corse-to-fine-structured
    PROMPT_TYPE="corse-to-fine-structured"
    RATIO=-1
    OUTPUT_DIR=${MODEL_NAME_OR_PATH}/$PROMPT_TYPE

    # bash sh/eval.sh ${PROMPT_TYPE} ${MODEL_NAME_OR_PATH} ${RATIO} ${OUTPUT_DIR}

    for RATIO in $(seq 0.05 0.05 1.0); do
        echo "Processing ratio: ${RATIO}"
        bash sh/eval.sh ${PROMPT_TYPE} ${MODEL_NAME_OR_PATH} $RATIO ${OUTPUT_DIR}
    done

done