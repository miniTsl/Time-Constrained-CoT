from transformers import AutoModelForCausalLM, AutoTokenizer
import time

model_name = "Qwen/Qwen2.5-7B-Instruct"

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(model_name)

input_template = "<|im_start|>system\n{system_prompt}<|im_end|>\n"
input_template += "<|im_start|>user\n{user_prompt}<|im_end|>\n"
input_template += "<|im_start|>assistant\n{assistant_prompt}\n"

system_prompt = """Solve the task by following format:
**Coarse Reasoning**
Short analysis and an answer. Focus on efficiency and simplicity.

**Fine Reasoning**
Detailed analysis step by step and a refined answer. Focus on accuracy and correctness.

**Final Answer** 
Your final answer within \\boxed{{}} when done reasoning or early-stop keyword "**Final Answer**" appears.
"""

user_prompt = """Daphne has a rope that is 60 meters long. She wants to use it to mark the boundary of a circle whose radius is an integer. What's the largest possible radius for her circle, in meters?"""

with open("response.txt", "r") as f:
    full_cot = f.read()

ratio = 0.4
assistant_cot = ""
if ratio > 0.0:
    assistant_cot = full_cot[:int(len(full_cot) * ratio)]
    assistant_cot += "\n\n**Final Answer** \n"

# # read assistant_cot from file
# with open("response.txt", "r") as f:
#     assistant_cot = f.read()

prompt = input_template.format(system_prompt=system_prompt, user_prompt=user_prompt, assistant_prompt=assistant_cot)
# assistant_cot = """
# To determine the probability that the sum of two fair 6-sided dice is 9, we need to follow these steps:

# 1. **Identify the total number of possible outcomes when rolling two dice:**
#    Each die has 6 faces, and each face can land on any of the numbers 1 through 6. Therefore, the total number of possible outcomes when rolling two dice is:
#    \[
#    6 \times 6 = 36
#    \]

# 2. **Identify the favorable outcomes where the sum is 9:**
#    We need to find all pairs of numbers \((a, b)\) such that \(a + b = 9\), where \(a\) and \(b\) are the numbers rolled on the first and second die, respectively. Let's list all such pairs:
#    - If \(a = 3\), then \(b = 6\) (pair: \((3, 6)\))
#    - If \(a = 4\), then \(b = 5\) (pair: \((4, 5)\))
#    - If \(a = 5\), then \(b = 4\) (pair: \((5, 4)\))
#    - If \(a = 6\), then \(b = 3\) (pair: \((6, 3)\))

#    There are 4 favorable outcomes.

# 3. **Calculate the probability:**

# ----

# Thus the final answer is:
# """

# messages = [
#     {"role": "system", "content": "Please reason step by step, and put your final answer within \\boxed{}."},
#     {"role": "user", "content": prompt},
# ]

# # apply_chat_template(): Converts a list of dictionaries with "role" and "content" keys to a list of token ids. This method is intended for use with chat models, and will read the tokenizer's chat_template attribute to determine the format and control tokens to use when converting.
# text = tokenizer.apply_chat_template(
#     messages,
#     tokenize=False, # Whether to tokenize the output. If False, the output will be a string.
#     add_generation_prompt=True  # If this is set, a prompt with the token(s) that indicate the start of an assistant message will be appended to the formatted output. 
# )

# print("text after apply_chat_template:")
# print("-" * 100)
# print(text)

# messages_with_cot = [
#     {"role": "system", "content": "Please reason step by step, and put your final answer within \\boxed{}."},
#     {"role": "user", "content": prompt},
#     {"role": "assistant", "content": assistant_cot}
# ]

# # apply_chat_template(): Converts a list of dictionaries with "role" and "content" keys to a list of token ids. This method is intended for use with chat models, and will read the tokenizer's chat_template attribute to determine the format and control tokens to use when converting.
# text_with_cot = tokenizer.apply_chat_template(
#     messages_with_cot,
#     tokenize=False, # Whether to tokenize the output. If False, the output will be a string.
#     add_generation_prompt=False  # If this is set, a prompt with the token(s) that indicate the start of an assistant message will be appended to the formatted output. 
# )

# print("text_with_cot after apply_chat_template:")
# print("-" * 100)
# print(text_with_cot)

print("formatted input prompt:")
print("-" * 100)
print(prompt)

model_inputs = tokenizer([prompt], return_tensors="pt").to("cuda")

start_time = time.perf_counter_ns()
generated_ids = model.generate(
    **model_inputs,
    do_sample=False,
    max_new_tokens=4096
)
end_time = time.perf_counter_ns()

generated_ids = [
    output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
]

response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

print("response:")
print("-" * 100)
print(response)
print("\n" + "-" * 100)
print(f"Time taken: {(end_time - start_time) / 1e9} seconds")
print(f"Generated tokens: {len(generated_ids[0])}")




# import os
# from typing import Iterable, Any, Union
# from pathlib import Path
# import json
# from transformers import AutoModelForCausalLM, AutoTokenizer
# import numpy as np

# model_name = "Qwen/Qwen2.5-7B-Instruct"
# tokenizer = AutoTokenizer.from_pretrained(model_name)

# def load_jsonl(file: Union[str, Path]) -> Iterable[Any]:
#     with open(file, "r", encoding="utf-8") as f:
#         for line in f:
#             try:
#                 yield json.loads(line)
#             except:
#                 print("Error in loading:", line)
#                 exit()

# path0 = "Qwen2.5-Math/evaluation/outputs/Qwen/Qwen2.5-7B-Instruct/qwen25-math-cot/math/test_qwen25-math-cot_-1_seed0_t0.0_s0_e-1.jsonl"
# path1 = "Qwen2.5-Math/evaluation/outputs/Qwen/Qwen2.5-7B-Instruct/corse-to-fine-structured-debug/math/test_corse-to-fine-structured-debug_-1_seed0_t0.0_s0_e-1.jsonl"

# code_list0 = []
# code_list1 = []
# for sample in load_jsonl(path0): # answer number ends with \n####
#     code_list0.append(sample["code"][0])
# for sample in load_jsonl(path1): # answer number ends with \n####
#     code_list1.append(sample["code"][0])

# code_token_count0 = []
# code_token_count1 = []
# for code in code_list0:
#     code_token_count0.append(len(tokenizer.encode(code)))
# for code in code_list1:
#     code_token_count1.append(len(tokenizer.encode(code)))
# # print(code_token_count0)
# print(max(code_token_count0))
# print(min(code_token_count0))
# print(sum(code_token_count0) / len(code_token_count0))
# # 中位数
# print(np.median(code_token_count0))

# # print(code_token_count1)
# print(max(code_token_count1))
# print(min(code_token_count1))
# print(sum(code_token_count1) / len(code_token_count1))
# # 中位数
# print(np.median(code_token_count1))