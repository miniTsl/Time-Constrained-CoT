path="../outputs/12_11/Qwen/QwQ-32B-Preview/coarse-to-fine-structured/gsm8k"
# 将这个path下的文件名中的"corse"都换成"coarse"
for file in "$path"/*; do
    mv "$file" "${file//corse/coarse}"
done

