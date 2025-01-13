# 定义模型列表为数组
MODEL_LIST=(
    "microsoft/phi-4"
    "microsoft/Phi-3-medium-128k-instruct"
    # "microsoft/Phi-3-small-128k-instruct"
    # "microsoft/Phi-3.5-mini-instruct"
    # "microsoft/Phi-3-mini-128k-instruct"
)

PROMPT_TYPE_LIST=(
    "sbs"
    "c2f"
    "aav"
    "kf"
    "sbs-hard"
)


for MODEL_NAME_OR_PATH in "${MODEL_LIST[@]}"; do
    export CUDA_VISIBLE_DEVICES="2,3"
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

    OUTPUT_DIR=/data03/sunyi/time_constrained_cot/outputs/1_10

    for PROMPT_TYPE in "${PROMPT_TYPE_LIST[@]}"; do
        # step-by-step hard
        if [[ ${PROMPT_TYPE} == "sbs-hard" ]]; then
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