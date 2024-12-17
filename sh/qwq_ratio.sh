MODEL_NAME_OR_PATH="Qwen/QwQ-32B-Preview"

# # original
# PROMPT_TYPE="qwen25-math-cot"
# RATIO=-1
# OUTPUT_DIR=${MODEL_NAME_OR_PATH}/$PROMPT_TYPE

# # bash sh/eval.sh ${PROMPT_TYPE} ${MODEL_NAME_OR_PATH} ${RATIO} ${OUTPUT_DIR}

# for RATIO in $(seq 0.05 0.05 1.0); do
#     echo "Processing ratio: ${RATIO}"
#     bash sh/eval.sh ${PROMPT_TYPE} ${MODEL_NAME_OR_PATH} $RATIO ${OUTPUT_DIR}
# done


# coarse-to-fine-structured
PROMPT_TYPE="corse-to-fine-structured"
RATIO=-1
OUTPUT_DIR=${MODEL_NAME_OR_PATH}/$PROMPT_TYPE

# bash sh/eval.sh ${PROMPT_TYPE} ${MODEL_NAME_OR_PATH} ${RATIO} ${OUTPUT_DIR}

for RATIO in $(seq 0.5 0.05 1.0); do
    echo "Processing ratio: ${RATIO}"
    bash sh/eval.sh ${PROMPT_TYPE} ${MODEL_NAME_OR_PATH} $RATIO ${OUTPUT_DIR}
done