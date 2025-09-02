# 定义模型列表为数组
MODEL_LIST=(
    meta-llama/Llama-3.2-3B-Instruct
    # meta-llama/Llama-3.2-1B-Instruct
    meta-llama/Llama-3.1-8B-Instruct
    # meta-llama/Llama-3.1-70B-Instruct
)

PROMPT_TYPE_LIST=(
    # "quick"
    # "direct"
    "sbs"
    "c2f"
    "aav"
    # "kf"
    # "o1-mimic-hard-user"
    # "sbs-hard"
    # "direct-hard"
    # "quick-hard"
    # "c2f-hard"
    # "aav-hard"
)

OUTPUT_DIR=/data03/sunyi/time_constrained_cot/outputs/2_6


# 遍历模型列表
for MODEL_NAME_OR_PATH in "${MODEL_LIST[@]}"; do
    export CUDA_VISIBLE_DEVICES="2,3"
    echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES}"
    echo "Processing model: ${MODEL_NAME_OR_PATH}"
    PROMPT_PREFIX="llama"

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