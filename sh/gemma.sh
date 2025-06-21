# 定义模型列表为数组
MODEL_LIST=(
    google/gemma-2-27b-it
    google/gemma-2-9b-it
    # google/gemma-2-2b-it
)

PROMPT_TYPE_LIST=(
    # "c2f-budget"
    # "aav-budget"
    # "sbs-budget"
    # "sbs-budget-hard"
    # "quick"
    # "direct"
    "sbs"
    "c2f"
    "aav"
    # "kf"
    # "sbs-hard"
    # "direct-hard"
    # "quick-hard"
    # "c2f-hard"
    # "aav-hard"
)

PROMPT_PREFIX="gemma"
OUTPUT_DIR=/data03/sunyi/time_constrained_cot/outputs/2_6

# 遍历模型列表
for MODEL_NAME_OR_PATH in "${MODEL_LIST[@]}"; do
    export CUDA_VISIBLE_DEVICES="2,3"
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