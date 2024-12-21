# 定义模型列表为数组
# MODEL_LIST=(Qwen/Qwen2.5-7B-Instruct Qwen/Qwen2.5-3B-Instruct Qwen/Qwen2.5-1.5B-Instruct Qwen/Qwen2.5-0.5B-Instruct)
# MODEL_LIST=(/data03/sunyi/hf_cache/hub/models--Qwen--Qwen2.5-0.5B-Instruct/snapshots/7ae557604adf67be50417f59c2c2f167def9a775, /data03/sunyi/hf_cache/hub/models--Qwen--Qwen2.5-7B-Instruct/snapshots/bb46c15ee4bb56c5b63245ef50fd7637234d6f75, /data03/sunyi/hf_cache/hub/models--Qwen--Qwen2.5-3B-Instruct/snapshots/aa8e72537993ba99e69dfaafa59ed015b17504d1)
MODEL_LIST=(/data03/sunyi/hf_cache/hub/models--Qwen--Qwen2.5-3B-Instruct/snapshots/aa8e72537993ba99e69dfaafa59ed015b17504d1)
TEMPERATURE=0.6

# 遍历模型列表
for MODEL_NAME_OR_PATH in "${MODEL_LIST[@]}"; do
    # 如果是0.5B的模型，则只使用6和7两张GPU，否则使用4，5，6，7四张GPU
    if [[ ${MODEL_NAME_OR_PATH} == *"0.5B"* ]]; then
        export CUDA_VISIBLE_DEVICES="6, 7"
    else
        export CUDA_VISIBLE_DEVICES="7"
    fi
    echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES}"
    echo "Processing model: ${MODEL_NAME_OR_PATH}"
    
    # # original
    # PROMPT_TYPE="qwen25-math-cot"
    # RATIO=-1

    MODEL_NAME=$(echo "$MODEL_NAME_OR_PATH" | tr '/' '\n' | grep '2.5' | head -n 1)
    OUTPUT_DIR=${MODEL_NAME}/$PROMPT_TYPE

    # bash sh/eval.sh ${PROMPT_TYPE} ${MODEL_NAME_OR_PATH} ${RATIO} ${OUTPUT_DIR} ${TEMPERATURE}

    # for RATIO in $(seq 0.2 0.2 1.0); do
    #     echo "Processing ratio: ${RATIO}"
    #     bash sh/eval.sh ${PROMPT_TYPE} ${MODEL_NAME_OR_PATH} $RATIO ${OUTPUT_DIR} ${TEMPERATURE}
    # done


    # corse-to-fine-structured
    # PROMPT_TYPE="corse-to-fine-structured"
    PROMPT_TYPE="in-context-corse-to-fine"
    RATIO=-1

    bash sh/eval.sh ${PROMPT_TYPE} ${MODEL_NAME_OR_PATH} ${RATIO} ${OUTPUT_DIR} ${TEMPERATURE}

    # for RATIO in $(seq 0.2 0.2 1.0); do
    #     echo "Processing ratio: ${RATIO}"
    #     bash sh/eval.sh ${PROMPT_TYPE} ${MODEL_NAME_OR_PATH} $RATIO ${OUTPUT_DIR} ${TEMPERATURE}
    # done

done