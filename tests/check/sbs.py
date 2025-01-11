
import os 
os.environ["TOKENIZERS_PARALLELISM"] = "false"
from transformers import AutoTokenizer, AutoModelForCausalLM
model_list = [
    "mistralai/Mistral-7B-Instruct-v0.3",
    "mistralai/Ministral-8B-Instruct-2410",
    "mistralai/Mistral-Nemo-Instruct-2407",
    "mistralai/Mistral-Small-Instruct-2409",
    "mistralai/Mathstral-7B-v0.1",
    
    "Qwen/Qwen2.5-32B-Instruct",
    "Qwen/Qwen2.5-14B-Instruct",
    "Qwen/Qwen2.5-7B-Instruct",
    "Qwen/Qwen2.5-3B-Instruct",
    "Qwen/Qwen2.5-1.5B-Instruct",
    "Qwen/Qwen2.5-Math-1.5B-Instruct",
    "Qwen/Qwen2.5-Math-7B-Instruct",
    "Qwen/QwQ-32B-Preview",
    
    "microsoft/Phi-3-mini-128k-instruct",
    "microsoft/Phi-3-small-128k-instruct",
    "microsoft/Phi-3-medium-128k-instruct",
    "microsoft/Phi-3.5-mini-instruct",
    "microsoft/phi-4",
    
    "meta-llama/Llama-3.2-3B-Instruct",
    "meta-llama/Llama-3.2-1B-Instruct",
    "meta-llama/Llama-3.1-8B-Instruct",
    
    "google/gemma-2-9b-it",
    "google/gemma-2-2b-it",

    "AI-MO/NuminaMath-7B-CoT",

    "internlm/internlm2_5-1_8b-chat",
    "internlm/internlm2_5-7b-chat",
    "internlm/internlm2_5-20b-chat",
    "internlm/internlm2-math-plus-1_8b",
    "internlm/internlm2-math-plus-20b",
    "internlm/internlm2-math-plus-7b",
    
    "deepseek-ai/deepseek-math-7b-instruct",
    "deepseek-ai/deepseek-math-7b-rl",
    
    "PowerInfer/SmallThinker-3B-Preview",

    "Skywork/Skywork-o1-Open-Llama-3.1-8B"
]
print(len(model_list))


# First define all unique chat templates
CHAT_TEMPLATE_FORMATS = {
    "mistral_format": "<s>[INST] {system_message}\n\n{user_message}[/INST]",
    
    "qwen_format": "<|im_start|>system\n{system_message}<|im_end|>\n<|im_start|>user\n{user_message}<|im_end|>\n<|im_start|>assistant\n",
    
    "phi3mini_format": "<|system|>\n{system_message}<|end|>\n<|user|>\n{user_message}<|end|>\n<|assistant|>\n",
    
    "phi3small_format": "<|endoftext|><|system|>\n{system_message}<|end|>\n<|user|>\n{user_message}<|end|>\n<|assistant|>\n",
    
    "phi3medium_format": "<|user|>\n{user_message}<|end|>\n<|assistant|>\n",
    
    "phi4_format": "<|im_start|>system<|im_sep|>{system_message}<|im_end|><|im_start|>user<|im_sep|>{user_message}<|im_end|><|im_start|>assistant<|im_sep|>",
    
    "llama_format": "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{system_message}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{user_message}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n",
    
    "gemma_format": "<bos><start_of_turn>user\n{user_message}<end_of_turn>\n<start_of_turn>model\n",
    
    "numina_format": "### Problem: {user_message}\n### Solution: ",
    
    "internlm_format": "<s><|im_start|>system\n{system_message}<|im_end|>\n<|im_start|>user\n{user_message}<|im_end|>\n<|im_start|>assistant\n",
    
    "deepseek_format": "<｜begin▁of▁sentence｜>{system_message}\n\nUser: {user_message}\n\nAssistant:"
}

# Then map each model to its template format
MODEL_TO_TEMPLATE = {
    # Mistral Family
    "mistralai/Mistral-7B-Instruct-v0.3": CHAT_TEMPLATE_FORMATS["mistral_format"],
    "mistralai/Ministral-8B-Instruct-2410": CHAT_TEMPLATE_FORMATS["mistral_format"],
    "mistralai/Mistral-Nemo-Instruct-2407": CHAT_TEMPLATE_FORMATS["mistral_format"],
    "mistralai/Mistral-Small-Instruct-2409": CHAT_TEMPLATE_FORMATS["mistral_format"],
    "mistralai/Mathstral-7B-v0.1": CHAT_TEMPLATE_FORMATS["mistral_format"],
    
    # Qwen Family
    "Qwen/Qwen2.5-32B-Instruct": CHAT_TEMPLATE_FORMATS["qwen_format"],
    "Qwen/Qwen2.5-14B-Instruct": CHAT_TEMPLATE_FORMATS["qwen_format"],
    "Qwen/Qwen2.5-7B-Instruct": CHAT_TEMPLATE_FORMATS["qwen_format"],
    "Qwen/Qwen2.5-3B-Instruct": CHAT_TEMPLATE_FORMATS["qwen_format"],
    "Qwen/Qwen2.5-1.5B-Instruct": CHAT_TEMPLATE_FORMATS["qwen_format"],
    "Qwen/Qwen2.5-0.5B-Instruct": CHAT_TEMPLATE_FORMATS["qwen_format"],
    "Qwen/Qwen2.5-Math-7B-Instruct": CHAT_TEMPLATE_FORMATS["qwen_format"],
    "Qwen/Qwen2.5-Math-1.5B-Instruct": CHAT_TEMPLATE_FORMATS["qwen_format"],
    "Qwen/QwQ-32B-Preview": CHAT_TEMPLATE_FORMATS["qwen_format"],
    
    # Phi Family
    "microsoft/Phi-3-mini-128k-instruct": CHAT_TEMPLATE_FORMATS["phi3mini_format"],
    "microsoft/Phi-3.5-mini-instruct": CHAT_TEMPLATE_FORMATS["phi3mini_format"],
    "microsoft/Phi-3-small-128k-instruct": CHAT_TEMPLATE_FORMATS["phi3small_format"],
    "microsoft/Phi-3-medium-128k-instruct": CHAT_TEMPLATE_FORMATS["phi3medium_format"],
    "microsoft/phi-4": CHAT_TEMPLATE_FORMATS["phi4_format"],
    
    # Llama Family
    "meta-llama/Llama-3.2-3B-Instruct": CHAT_TEMPLATE_FORMATS["llama_format"],
    "meta-llama/Llama-3.2-1B-Instruct": CHAT_TEMPLATE_FORMATS["llama_format"],
    "meta-llama/Llama-3.1-8B-Instruct": CHAT_TEMPLATE_FORMATS["llama_format"],
    
    # Gemma Family
    "google/gemma-2-9b-it": CHAT_TEMPLATE_FORMATS["gemma_format"],
    "google/gemma-2-2b-it": CHAT_TEMPLATE_FORMATS["gemma_format"],
    
    # NuminaMath
    "AI-MO/NuminaMath-7B-CoT": CHAT_TEMPLATE_FORMATS["numina_format"],
    
    # InternLM Family
    "internlm/internlm2_5-1_8b-chat": CHAT_TEMPLATE_FORMATS["internlm_format"],
    "internlm/internlm2_5-7b-chat": CHAT_TEMPLATE_FORMATS["internlm_format"],
    "internlm/internlm2_5-20b-chat": CHAT_TEMPLATE_FORMATS["internlm_format"],
    "internlm/internlm2-math-plus-1_8b": CHAT_TEMPLATE_FORMATS["internlm_format"],
    "internlm/internlm2-math-plus-7b": CHAT_TEMPLATE_FORMATS["internlm_format"],
    "internlm/internlm2-math-plus-20b": CHAT_TEMPLATE_FORMATS["internlm_format"],
    
    # DeepSeek Math
    "deepseek-ai/deepseek-math-7b-instruct": CHAT_TEMPLATE_FORMATS["deepseek_format"],
    "deepseek-ai/deepseek-math-7b-rl": CHAT_TEMPLATE_FORMATS["deepseek_format"],
    
    # SmallThinker
    "PowerInfer/SmallThinker-3B-Preview": CHAT_TEMPLATE_FORMATS["qwen_format"],  # Uses same format as Qwen
    
    # Skywork
    "Skywork/Skywork-o1-Open-Llama-3.1-8B": CHAT_TEMPLATE_FORMATS["llama_format"]
}


sbs_prompt = """Please reason step by step, and put your final answer within \\boxed{{}} when done reasoning or early-stop keyword **Final Answer** appears."""

# # question 1
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# with open("sbs_q1.txt", "a") as f:
#     for checkpoint in model_list:
#         print("<<< checkpoint: ", checkpoint)
#         f.write("<<< checkpoint: " + checkpoint + "\n")
#         tokenizer = AutoTokenizer.from_pretrained(checkpoint, trust_remote_code=True)
#         model = AutoModelForCausalLM.from_pretrained(
#             checkpoint,
#             trust_remote_code=True,
#             device_map="cuda",
#             torch_dtype="auto"
#         )

#         question = "What is $10.0000198\\cdot 5.9999985401\\cdot 6.9999852$ to the nearest whole number"
#         if "gemma" in checkpoint:
#             prompt = MODEL_TO_TEMPLATE[checkpoint].format(user_message=sbs_prompt + "\n\n" + question)
#         else:
#             prompt = MODEL_TO_TEMPLATE[checkpoint].format(system_message=sbs_prompt, user_message=question)
#         inputs = tokenizer([prompt], return_tensors="pt").to(model.device)
#         out = model.generate(**inputs, max_new_tokens=4096, do_sample=False)
#         generated_ids = out[0][len(inputs.input_ids[0]):]
#         response = tokenizer.decode(generated_ids, skip_special_tokens=True)
#         print("\n<<< response:")
#         print(response)
#         print("-" * 100 + "\n")
#         f.write(response + "\n")
#         f.write("-" * 100 + "\n\n")
#         del tokenizer
#         del model



# # question 2
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
# with open("sbs_q2.txt", "w") as f:
#     for checkpoint in model_list:
#         print("<<< checkpoint: ", checkpoint)
#         f.write("<<< checkpoint: " + checkpoint + "\n")
#         tokenizer = AutoTokenizer.from_pretrained(checkpoint, trust_remote_code=True)
#         model = AutoModelForCausalLM.from_pretrained(
#             checkpoint,
#             trust_remote_code=True,
#             device_map="cuda",
#             torch_dtype="auto"
#         )
#         question = "Find the point of intersection of the line\n\\[\\frac{x - 2}{3} = \\frac{y + 1}{4} = \\frac{z - 2}{12}\\]and $x - y + z = 5.$"
#         if "gemma" in checkpoint:
#             prompt = MODEL_TO_TEMPLATE[checkpoint].format(user_message=sbs_prompt + "\n\n" + question)
#         else:
#             prompt = MODEL_TO_TEMPLATE[checkpoint].format(system_message=sbs_prompt, user_message=question)
#         inputs = tokenizer([prompt], return_tensors="pt").to(model.device)
#         out = model.generate(**inputs, max_new_tokens=4096, do_sample=False)
#         generated_ids = out[0][len(inputs.input_ids[0]):]
#         response = tokenizer.decode(generated_ids, skip_special_tokens=True)
#         print("\n<<< response:")
#         print(response)
#         print("-" * 100 + "\n")
#         f.write(response + "\n")
#         f.write("-" * 100 + "\n\n")
#         del tokenizer
#         del model


# question 3
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
with open("sbs_q3.txt", "w") as f:
    for checkpoint in model_list:
        print("<<< checkpoint: ", checkpoint)
        f.write("<<< checkpoint: " + checkpoint + "\n")
        tokenizer = AutoTokenizer.from_pretrained(checkpoint, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            checkpoint,
            trust_remote_code=True,
            device_map="cuda",
            torch_dtype="auto"
        )
        question = "How many prime numbers less than 100 have a units digit of 3?"
        if "gemma" in checkpoint:
            prompt = MODEL_TO_TEMPLATE[checkpoint].format(user_message=sbs_prompt + "\n\n" + question)
        else:
            prompt = MODEL_TO_TEMPLATE[checkpoint].format(system_message=sbs_prompt, user_message=question)
        inputs = tokenizer([prompt], return_tensors="pt").to(model.device)
        out = model.generate(**inputs, max_new_tokens=4096, do_sample=False)
        generated_ids = out[0][len(inputs.input_ids[0]):]
        response = tokenizer.decode(generated_ids, skip_special_tokens=True)
        print("\n<<< response:")
        print(response)
        print("-" * 100 + "\n")
        f.write(response + "\n")
        f.write("-" * 100 + "\n\n")
        del tokenizer
        del model