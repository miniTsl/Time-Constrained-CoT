# 定义模型列表为数组
MODEL_LIST=(
    "mistralai/Mistral-7B-Instruct-v0.3"
    "mistralai/Ministral-8B-Instruct-2410"
    "mistralai/Mistral-Nemo-Instruct-2407"
    "mistralai/Mistral-Small-Instruct-2409"
    "mistralai/Mathstral-7B-v0.1"
)


PROMPT_TYPE_LIST=(
    "quick"
    "direct"
    "sbs"
    "c2f"
    "aav"
    "kf"
    "sbs-hard"
    "direct-hard"
    "quick-hard"
)

PROMPT_PREFIX="mistral"
OUTPUT_DIR=/data03/sunyi/time_constrained_cot/outputs/1_10

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
