# Step-by-step hard
sbs_hard = """Please reason step by step, and put your final answer within \\boxed{{}}."""

# Step-by-step
sbs = """Please reason step by step, and put your final answer within \\boxed{{}} when done reasoning or early-stop keyword **Final Answer** appears."""

# Coarse-to-fine
c2f = """Solve the task by following format:
**Coarse Reasoning**
Short analysis and an answer. Focus on efficiency and simplicity.

**Fine Reasoning**
Detailed analysis step by step and a refined answer. Focus on accuracy and correctness.

**Final Answer** 
Your final answer within \\boxed{{}} when done reasoning or early-stop keyword **Final Answer** appears."""

# Knowledge First
kf = """Solve the task by following format:
**Knowledge to Use**
List theorem and methods that are useful for solving the problem.

**Reasoning**
Step by step analysis using above knowledge.

**Final Answer**
Your final answer within \\boxed{{}} when done reasoning or early-stop keyword **Final Answer** appears."""


# Answer and Verify
aav = """Solve the task by following format:
**Quick Answer**
Give an initial answer based on intuition or quick calculation.

**Verification**
Give a revised answer through reasoning step by step to correct potential mistakes.

**Final Answer**
Your final answer within \\boxed{{}} when done reasoning or early-stop keyword **Final Answer** appears.
"""


# First define all unique chat templates
CHAT_TEMPLATE_FORMATS = {
    "mistral_format": "<s>[INST] {system_message}\n\n{input}[/INST]",
    
    "qwen_format": "<|im_start|>system\n{system_message}<|im_end|>\n<|im_start|>user\n{input}<|im_end|>\n<|im_start|>assistant\n",
    
    "phi3mini_format": "<|system|>\n{system_message}<|end|>\n<|user|>\n{input}<|end|>\n<|assistant|>\n",
    
    "phi3small_format": "<|endoftext|><|system|>\n{system_message}<|end|>\n<|user|>\n{input}<|end|>\n<|assistant|>\n",
    
    "phi3medium_format": "<|user|>\n{input}<|end|>\n<|assistant|>\n",
    
    "phi4_format": "<|im_start|>system<|im_sep|>{system_message}<|im_end|><|im_start|>user<|im_sep|>{input}<|im_end|><|im_start|>assistant<|im_sep|>",
    
    "llama_format": "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{system_message}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{input}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n",
    
    "gemma_format": "<bos><start_of_turn>user\n{input}<end_of_turn>\n<start_of_turn>model\n",
    
    "numina_format": "### Problem: {input}\n### Solution: ",
    
    "internlm_format": "<s><|im_start|>system\n{system_message}<|im_end|>\n<|im_start|>user\n{input}<|im_end|>\n<|im_start|>assistant\n",
    
    "deepseek_format": "<｜begin▁of▁sentence｜>{system_message}\n\nUser: {input}\n\nAssistant:"
}


PROMPT_TEMPLATES = {
    # "direct": ("Question: {input}\nAnswer: ", "{output}", "\n\n"),
    # "cot": ("Question: {input}\nAnswer: ", "{output}", "\n\n\n"),
    # "pal": ("Question: {input}\n\n", "{output}", "\n---\n"),
    # "tool-integrated": ("Question: {input}\n\nSolution:\n", "{output}", "\n---\n"),
    # "self-instruct": ("<|user|>\n{input}\n<|assistant|>\n", "{output}", "\n"),
    # "tora": ("<|user|>\n{input}\n<|assistant|>\n", "{output}", "\n"),
    # "wizard_zs": (
    #     "### Instruction:\n{input}\n\n### Response: Let's think step by step.",
    #     "{output}",
    #     "\n\n\n",
    # ),
    # "platypus_fs": (
    #     "### Instruction:\n{input}\n\n### Response:\n",
    #     "{output}",
    #     "\n\n\n",
    # ),
    # "deepseek-math": (
    #     "User: {input}\nPlease reason step by step, "
    #     "and put your final answer within \\boxed{{}}.\n\nAssistant:",
    #     "{output}",
    #     "\n\n\n",
    # ),
    # "kpmath": (
    #     "User: Please reason step by step and put your final answer at the end "
    #     'with "The answer is: ".\n\n{input}\n\nAssistant:',
    #     "{output}",
    # ),
    # "jiuzhang": (
    #     "## Question\n{input}\n\n## Solution\n",
    #     "{output}",
    #     "\n\n\n",
    # ),
    # "jiuzhang_tora": (
    #     "## Question\n{input}\n\n## Code Solution\n",
    #     "{output}",
    #     "\n\n\n",
    # ),
    # "jiuzhang_nl": (
    #     "## Question\n{input}\n\n## Natural Language Solution\n",
    #     "{output}",
    #     "\n\n\n",
    # ),
    # "mmiqc": (
    #     'Please solve the following problem and put your answer at the end with "The answer is: ".\n\n{input}\n\n',
    #     "{output}",
    #     "\n\n\n",
    # ),
    # "abel": (
    #     "Question:\n{input}\nAnswer:\nLet's think step by step.\n",
    #     "{output}",
    #     "\n\n",
    # ),
    # "shepherd": ("{input}\n", "{output}", "\n\n\n"),
    # "qwen-boxed": (
    #     "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
    #     "<|im_start|>user\n{input}\nPlease reason step by step, and put your final answer within \\boxed{{}}.<|im_end|>\n"
    #     "<|im_start|>assistant\n",
    #     "{output}",
    #     "\n\n",
    # ),
    # ************** Qwen Series **************
    "qwen-sbs-hard": (
        CHAT_TEMPLATE_FORMATS["qwen_format"].replace("{system_message}", sbs_hard),
        "{output}",
        "\n\n",
    ),
    "qwen-sbs": (
        CHAT_TEMPLATE_FORMATS["qwen_format"].replace("{system_message}", sbs),
        "{output}",
        "\n\n",
    ),
    "qwen-c2f": (
        CHAT_TEMPLATE_FORMATS["qwen_format"].replace("{system_message}", c2f),
        "{output}",
        "\n\n", 
    ),
    "qwen-kf": (
        CHAT_TEMPLATE_FORMATS["qwen_format"].replace("{system_message}", kf),
        "{output}",
        "\n\n", 
    ),
    "qwen-aav": (
        CHAT_TEMPLATE_FORMATS["qwen_format"].replace("{system_message}", aav),
        "{output}",
        "\n\n", 
    ),
    # ************** Mistral Series **************
    "mistral-sbs-hard": (
        CHAT_TEMPLATE_FORMATS["mistral_format"].replace("{system_message}", sbs_hard),
        "{output}",
        "\n\n",
    ),
    "mistral-sbs": (
        CHAT_TEMPLATE_FORMATS["mistral_format"].replace("{system_message}", sbs),
        "{output}",
        "\n\n",
    ),
    "mistral-c2f": (
        CHAT_TEMPLATE_FORMATS["mistral_format"].replace("{system_message}", c2f),
        "{output}",
        "\n\n", 
    ),
    "mistral-kf": (
        CHAT_TEMPLATE_FORMATS["mistral_format"].replace("{system_message}", kf),
        "{output}",
        "\n\n", 
    ),
    "mistral-aav": (
        CHAT_TEMPLATE_FORMATS["mistral_format"].replace("{system_message}", aav),
        "{output}",
        "\n\n", 
    ),
    # ************** Phi Series **************
    ## phi3mini
    "phi3mini-sbs-hard": (
        CHAT_TEMPLATE_FORMATS["phi3mini_format"].replace("{system_message}", sbs_hard),
        "{output}",
        "\n\n",
    ),
    "phi3mini-sbs": (
        CHAT_TEMPLATE_FORMATS["phi3mini_format"].replace("{system_message}", sbs),
        "{output}",
        "\n\n",
    ),
    "phi3mini-c2f": (
        CHAT_TEMPLATE_FORMATS["phi3mini_format"].replace("{system_message}", c2f),
        "{output}",
        "\n\n", 
    ),
    "phi3mini-kf": (
        CHAT_TEMPLATE_FORMATS["phi3mini_format"].replace("{system_message}", kf),
        "{output}",
        "\n\n", 
    ),
    "phi3mini-aav": (
        CHAT_TEMPLATE_FORMATS["phi3mini_format"].replace("{system_message}", aav),
        "{output}",
        "\n\n", 
    ),
    
    ## phi3small
    "phi3small-sbs-hard": (
        CHAT_TEMPLATE_FORMATS["phi3small_format"].replace("{system_message}", sbs_hard),
        "{output}",
        "\n\n",
    ),
    "phi3small-sbs": (
        CHAT_TEMPLATE_FORMATS["phi3small_format"].replace("{system_message}", sbs),
        "{output}",
        "\n\n",
    ),
    "phi3small-c2f": (
        CHAT_TEMPLATE_FORMATS["phi3small_format"].replace("{system_message}", c2f),
        "{output}",
        "\n\n", 
    ),  
    "phi3small-kf": (
        CHAT_TEMPLATE_FORMATS["phi3small_format"].replace("{system_message}", kf),
        "{output}",
        "\n\n", 
    ),
    "phi3small-aav": (
        CHAT_TEMPLATE_FORMATS["phi3small_format"].replace("{system_message}", aav),
        "{output}",
        "\n\n", 
    ),
    
    ## phi3medium
    "phi3medium-sbs-hard": (
        CHAT_TEMPLATE_FORMATS["phi3medium_format"].replace("{input}", sbs_hard + "\n\n" + "{input}"),
        "{output}",
        "\n\n",
    ),
    "phi3medium-sbs": (
        CHAT_TEMPLATE_FORMATS["phi3medium_format"].replace("{input}", sbs + "\n\n" + "{input}"),
        "{output}",
        "\n\n",
    ),
    "phi3medium-c2f": (
        CHAT_TEMPLATE_FORMATS["phi3medium_format"].replace("{input}", c2f + "\n\n" + "{input}"),
        "{output}",
        "\n\n", 
    ),
    "phi3medium-kf": (
        CHAT_TEMPLATE_FORMATS["phi3medium_format"].replace("{input}", kf + "\n\n" + "{input}"),
        "{output}",
        "\n\n", 
    ),
    "phi3medium-aav": (
        CHAT_TEMPLATE_FORMATS["phi3medium_format"].replace("{input}", aav + "\n\n" + "{input}"),
        "{output}",
        "\n\n", 
    ),
    
    ## phi4
    "phi4-sbs-hard": (
        CHAT_TEMPLATE_FORMATS["phi4_format"].replace("{system_message}", sbs_hard),
        "{output}",
        "\n\n",
    ),
    "phi4-sbs": (
        CHAT_TEMPLATE_FORMATS["phi4_format"].replace("{system_message}", sbs),
        "{output}",
        "\n\n",
    ),
    "phi4-c2f": (
        CHAT_TEMPLATE_FORMATS["phi4_format"].replace("{system_message}", c2f),
        "{output}",
        "\n\n", 
    ),
    "phi4-kf": (
        CHAT_TEMPLATE_FORMATS["phi4_format"].replace("{system_message}", kf),
        "{output}",
        "\n\n", 
    ),
    "phi4-aav": (
        CHAT_TEMPLATE_FORMATS["phi4_format"].replace("{system_message}", aav),
        "{output}",
        "\n\n", 
    ),
    
    # ************** Gemma Series **************
    "gemma-sbs-hard": (
        CHAT_TEMPLATE_FORMATS["gemma_format"].replace("{input}", sbs_hard + "\n\n" + "{input}"),
        "{output}",
        "\n\n",
    ),
    "gemma-sbs": (
        CHAT_TEMPLATE_FORMATS["gemma_format"].replace("{input}", sbs + "\n\n" + "{input}"),
        "{output}",
        "\n\n",
    ),
    "gemma-c2f": (
        CHAT_TEMPLATE_FORMATS["gemma_format"].replace("{input}", c2f + "\n\n" + "{input}"),
        "{output}",
        "\n\n", 
    ),
    "gemma-kf": (
        CHAT_TEMPLATE_FORMATS["gemma_format"].replace("{input}", kf + "\n\n" + "{input}"),
        "{output}",
        "\n\n", 
    ),
    "gemma-aav": (
        CHAT_TEMPLATE_FORMATS["gemma_format"].replace("{input}", aav + "\n\n" + "{input}"),
        "{output}",
        "\n\n", 
    ),
    
    # # ************** Skywork **************
    # "skywork-step-by-step": (
    #     "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n" + step_by_step + "<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{input}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n",
    #     "{output}",
    #     "\n\n",
    # ),
    # "skywork-coarse-to-fine": (
    #     "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n" + coarse_to_fine + "<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{input}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n",
    #     "{output}",
    #     "\n\n",
    # ),
    # "skywork-step-by-step-hard": (
    #     "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n" + step_by_step_hard + "<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{input}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n",
    #     "{output}",
    #     "\n\n",
    # ),
    # # ************** DeepSeek **************
    # "deepseek-step-by-step": (
    #     "<｜begin▁of▁sentence｜>" + step_by_step + "\n\n"
    #     "User: {input}\n\n"
    #     "Assistant:",
    #     "{output}",
    #     "\n\n",
    # ),
    # "deepseek-coarse-to-fine": (
    #     "<｜begin▁of▁sentence｜>" + coarse_to_fine + "\n\n"
    #     "User: {input}\n\n"
    #     "Assistant:",
    #     "{output}",
    #     "\n\n",
    # ),
    # "deepseek-step-by-step-hard": (
    #     "<｜begin▁of▁sentence｜>" + step_by_step_hard + "\n\n"
    #     "User: {input}\n\n"
    #     "Assistant:",
    #     "{output}",
    #     "\n\n",
    # ),
    # # ************** PowerInfer **************
    # "smallthinker-step-by-step": (
    #     "<|im_start|>system\n" + step_by_step + "<|im_end|>\n"
    #     "<|im_start|>user\n{input}<|im_end|>\n"
    #     "<|im_start|>assistant\n",
    #     "{output}",
    #     "\n\n",
    # ),
    # "smallthinker-coarse-to-fine": (
    #     "<|im_start|>system\n" + coarse_to_fine + "<|im_end|>\n"
    #     "<|im_start|>user\n{input}<|im_end|>\n"
    #     "<|im_start|>assistant\n",
    #     "{output}",
    #     "\n\n",
    # ),
    # "smallthinker-step-by-step-hard": (
    #     "<|im_start|>system\n" + step_by_step_hard + "<|im_end|>\n"
    #     "<|im_start|>user\n{input}<|im_end|>\n"
    #     "<|im_start|>assistant\n",
    #     "{output}",
    #     "\n\n",
    # ),
    # # ************** InternLM **************
    # "internlm-math-fs": ("Question:{input}\nAnswer:", "{output}", "\n"),
    # "internlm-math-chat": (
    #     "<|im_start|>user\n{input}<|im_end|>\n" "<|im_start|>assistant\n",
    #     "{output}",
    #     "\n\n",
    # ),
    # "mistral": (
    #     "[INST] {input}[/INST]",
    #     "{output}",
    #     "\n\n",
    # ),
    # "numina": ("### Problem: {input}\n### Solution:", " {output}", "\n\n"),
}
