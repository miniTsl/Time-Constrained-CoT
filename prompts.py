## directly terminate at token budget
# # Quick hard
# quick_hard = """Give an answer based on intuition or quick calculation. Put your answer within \\boxed{{}}."""




# # Direct hard
# direct_hard = """Solve the problem and put your final answer within \\boxed{{}}."""




# # Step-by-step hard
# sbs_hard = """Please reason step by step. 
# Conclude with: 
# Therefore, the final answer is: \\boxed{{[answer]}}.
# Where [answer] is just the final number or expression that solves the problem."""

# Step-by-step hard for mmlu_stem
sbs_hard = """Please reason step by step. 
Conclude with: 
Therefore, the final choice is: [answer].
Where [answer] is just the choice of the multiple choices that solves the problem."""




# # Coarse-to-fine hard (original)
# c2f_hard = """Use the following pattern to solve the problem:
# **Coarse-Grained Reasoning**
# Provide a brief analysis and initial answer, focusing on efficiency and conciseness.

# **Fine-Grained Reasoning**
# Provide detailed reasoning step by step and a refined answer, focusing on correctness and rigor.

# Conclude with: 
# Therefore, the final answer is: \\boxed{{[answer]}}.
# Where [answer] is just the final number or expression that solves the problem."""

# # Coarse-to-fine hard (modified)
# c2f_hard = """Use the following pattern to solve the problem:
# **Coarse-Grained Reasoning**
# Provide a brief analysis of the problem, focusing on efficiency and conciseness.

# **Fine-Grained Reasoning**
# Provide detailed reasoning step by step of the problem, focusing on correctness and rigor.

# Conclude with: 
# Therefore, the final answer is: \\boxed{{[answer]}}.
# Where [answer] is just the final number or expression that solves the problem."""

# Coarse-to-fine hard for mmlu_stem
c2f_hard = """Use the following pattern to solve the problem:
**Coarse-Grained Reasoning**
Provide a brief analysis of the problem, focusing on efficiency and conciseness.

**Fine-Grained Reasoning**
Provide detailed reasoning step by step of the problem, focusing on correctness and rigor.

Conclude with: 
Therefore, the final choice is: [answer].
Where [answer] is just the choice of the multiple choices that solves the problem."""




# # Answer and Verify hard
# aav_hard = """Use the following pattern to solve the problem:
# **Quick Answer**
# Provide an initial answer based on intuition or quick calculation.

# **Verification**
# Provide a revised answer through reasoning step by step. Correct previous mistakes, if any.

# Conclude with: 
# Therefore, the final answer is: \\boxed{{[answer]}}.
# Where [answer] is just the final number or expression that solves the problem."""

# Answer and Verify hard for mmlu_stem
aav_hard = """Use the following pattern to solve the problem:
**Quick Answer**
Provide an initial answer based on intuition or quick calculation.

**Verification**
Provide a revised answer through reasoning step by step. Correct previous mistakes, if any.

Conclude with: 
Therefore, the final choice is: [answer].
Where [answer] is just the choice of the multiple choices that solves the problem."""




# # Knowledge First hard
# kf_hard = """Solve the task by following format:
# **Knowledge to Use**
# List theorem and methods that are useful for solving the problem.

# **Reasoning**
# Step by step analysis using above knowledge.

# **Final Answer**
# Your final answer within \\boxed{{}}."""




# # early stop and conclude before budget
# early_stop = """\n\nNotice: When you are interrupted by the keyword **Time’s Up!**, stop reasoning immediately.
# Based on your reasoning so far, conclude with: 
# Therefore, the final answer is: \\boxed{{[answer]}}.
# Where [answer] is just the final number or expression that solves the problem."""

# early stop and conclude before budget for mmlu_stem
early_stop = """\n\nNotice: When you are interrupted by the keyword **Time’s Up!**, stop reasoning immediately.
Based on your reasoning so far, conclude with: 
Therefore, the final choice is: [answer].
Where [answer] is just the choice of the multiple choices that solves the problem."""




# # quick
# quick = quick_hard + early_stop

# # direct
# direct = direct_hard + early_stop

# Step-by-step
sbs = sbs_hard + early_stop

# Coarse-to-fine
c2f = c2f_hard + early_stop

# Answer and Verify
aav = aav_hard + early_stop

# # Knowledge First
# kf = kf_hard + early_stop




## append output token limit onto prompts
# budget_limit = "\n\nMake sure your response uses less than {token_budget} tokens."
# # sbs-budget-hard
# sbs_budget_hard = sbs_hard + budget_limit

## combine early-stopping and appending output token limit onto prompts
# # sbs-budget
# sbs_budget = sbs_hard + early_stop + budget_limit

# # c2f-budget
# c2f_budget = c2f_hard + early_stop + budget_limit

# # aav-budget
# aav_budget = aav_hard + early_stop + budget_limit




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
    
    "deepseek-r1-distill_format" : "<｜begin▁of▁sentence｜><｜User｜>{input}<｜Assistant｜>"    # Avoid adding a system prompt; all instructions should be contained within the user prompt.
}


PROMPT_TEMPLATES = {
    # ************** Qwen Series **************
    ## qwen-math
    # "qwen-math-sbs-budget-hard": (
    #     CHAT_TEMPLATE_FORMATS["qwen_format"].replace("{system_message}", sbs_budget_hard),
    #     "{output}",
    #     "\n\n",
    # ),
    # "qwen-math-quick-hard": (
    #     CHAT_TEMPLATE_FORMATS["qwen_format"].replace("{system_message}", quick_hard),
    #     "{output}",
    #     "\n\n",
    # ),
    # "qwen-math-direct-hard": (
    #     CHAT_TEMPLATE_FORMATS["qwen_format"].replace("{system_message}", direct_hard),
    #     "{output}",
    #     "\n\n",
    # ),
    "qwen-math-sbs-hard": (
        CHAT_TEMPLATE_FORMATS["qwen_format"].replace("{system_message}", sbs_hard),
        "{output}",
        "\n\n",
    ),
    "qwen-math-c2f-hard": (
        CHAT_TEMPLATE_FORMATS["qwen_format"].replace("{system_message}", c2f_hard),
        "{output}",
        "\n\n",
    ),
    "qwen-math-aav-hard": (
        CHAT_TEMPLATE_FORMATS["qwen_format"].replace("{system_message}", aav_hard),
        "{output}",
        "\n\n",
    ),
    # "qwen-math-kf-hard": (
    #     CHAT_TEMPLATE_FORMATS["qwen_format"].replace("{system_message}", kf_hard),
    #     "{output}",
    #     "\n\n",
    # ),
    # "qwen-math-quick": (
    #     CHAT_TEMPLATE_FORMATS["qwen_format"].replace("{system_message}", quick),
    #     "{output}",
    #     "\n\n",
    # ),
    # "qwen-math-direct": (
    #     CHAT_TEMPLATE_FORMATS["qwen_format"].replace("{system_message}", direct),
    #     "{output}",
    #     "\n\n",
    # ),
    "qwen-math-sbs": (
        CHAT_TEMPLATE_FORMATS["qwen_format"].replace("{system_message}", sbs),
        "{output}",
        "\n\n",
    ),
    "qwen-math-c2f": (
        CHAT_TEMPLATE_FORMATS["qwen_format"].replace("{system_message}", c2f),
        "{output}",
        "\n\n",
    ),
    "qwen-math-aav": (
        CHAT_TEMPLATE_FORMATS["qwen_format"].replace("{system_message}", aav),
        "{output}",
        "\n\n",
    ),
    # "qwen-math-kf": (
    #     CHAT_TEMPLATE_FORMATS["qwen_format"].replace("{system_message}", kf),
    #     "{output}",
    #     "\n\n",
    # ),
    
    ## qwen2.5
    # "qwen-sbs-budget-hard": (
    #     CHAT_TEMPLATE_FORMATS["qwen_format"].replace("{system_message}", sbs_budget_hard),
    #     "{output}",
    #     "\n\n",
    # ),
    # "qwen-sbs-budget": (
    #     CHAT_TEMPLATE_FORMATS["qwen_format"].replace("{system_message}", sbs_budget),
    #     "{output}",
    #     "\n\n",
    # ),
    # "qwen-c2f-budget": (
    #     CHAT_TEMPLATE_FORMATS["qwen_format"].replace("{system_message}", c2f_budget),
    #     "{output}",
    #     "\n\n",
    # ),
    # "qwen-aav-budget": (
    #     CHAT_TEMPLATE_FORMATS["qwen_format"].replace("{system_message}", aav_budget),
    #     "{output}",
    #     "\n\n",
    # ),
    # "qwen-quick-hard": (
    #     CHAT_TEMPLATE_FORMATS["qwen_format"].replace("{system_message}", quick_hard),
    #     "{output}",
    #     "\n\n",
    # ),
    # "qwen-direct-hard": (
    #     CHAT_TEMPLATE_FORMATS["qwen_format"].replace("{system_message}", direct_hard),
    #     "{output}",
    #     "\n\n",
    # ),
    "qwen-sbs-hard": (
        CHAT_TEMPLATE_FORMATS["qwen_format"].replace("{system_message}", sbs_hard),
        "{output}",
        "\n\n",
    ),
    "qwen-c2f-hard": (
        CHAT_TEMPLATE_FORMATS["qwen_format"].replace("{system_message}", c2f_hard),
        "{output}",
        "\n\n",
    ),
    "qwen-aav-hard": (
        CHAT_TEMPLATE_FORMATS["qwen_format"].replace("{system_message}", aav_hard),
        "{output}",
        "\n\n",
    ),
    # "qwen-kf-hard": (
    #     CHAT_TEMPLATE_FORMATS["qwen_format"].replace("{system_message}", kf_hard),
    #     "{output}",
    #     "\n\n",
    # ),
    # "qwen-quick": (
    #     CHAT_TEMPLATE_FORMATS["qwen_format"].replace("{system_message}", quick),
    #     "{output}",
    #     "\n\n",
    # ),
    # "qwen-direct": (
    #     CHAT_TEMPLATE_FORMATS["qwen_format"].replace("{system_message}", direct),
    #     "{output}",
    #     "\n\n",
    # ),
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
    "qwen-aav": (
        CHAT_TEMPLATE_FORMATS["qwen_format"].replace("{system_message}", aav),
        "{output}",
        "\n\n", 
    ),
    # "qwen-kf": (
    #     CHAT_TEMPLATE_FORMATS["qwen_format"].replace("{system_message}", kf),
    #     "{output}",
    #     "\n\n", 
    # ),
    
    # ************** Mistral Series **************
    # "mistral-sbs-budget-hard": (
    #     CHAT_TEMPLATE_FORMATS["mistral_format"].replace("{system_message}", sbs_budget_hard),
    #     "{output}",
    #     "\n\n",
    # ),
    # "mistral-sbs-budget": (
    #     CHAT_TEMPLATE_FORMATS["mistral_format"].replace("{system_message}", sbs_budget),
    #     "{output}",
    #     "\n\n",
    # ),
    # "mistral-c2f-budget": (
    #     CHAT_TEMPLATE_FORMATS["mistral_format"].replace("{system_message}", c2f_budget),
    #     "{output}",
    #     "\n\n",
    # ),
    # "mistral-aav-budget": (
    #     CHAT_TEMPLATE_FORMATS["mistral_format"].replace("{system_message}", aav_budget),
    #     "{output}",
    #     "\n\n",
    # ),
    # "mistral-quick-hard": (
    #     CHAT_TEMPLATE_FORMATS["mistral_format"].replace("{system_message}", quick_hard),
    #     "{output}",
    #     "\n\n",
    # ),
    # "mistral-direct-hard": (
    #     CHAT_TEMPLATE_FORMATS["mistral_format"].replace("{system_message}", direct_hard),
    #     "{output}",
    #     "\n\n",
    # ),
    "mistral-sbs-hard": (
        CHAT_TEMPLATE_FORMATS["mistral_format"].replace("{system_message}", sbs_hard),
        "{output}",
        "\n\n",
    ),
    "mistral-c2f-hard": (
        CHAT_TEMPLATE_FORMATS["mistral_format"].replace("{system_message}", c2f_hard),
        "{output}",
        "\n\n",
    ),
    "mistral-aav-hard": (
        CHAT_TEMPLATE_FORMATS["mistral_format"].replace("{system_message}", aav_hard),
        "{output}",
        "\n\n",
    ),
    # "mistral-kf-hard": (
    #     CHAT_TEMPLATE_FORMATS["mistral_format"].replace("{system_message}", kf_hard),
    #     "{output}",
    #     "\n\n",
    # ),
    # "mistral-quick": (
    #     CHAT_TEMPLATE_FORMATS["mistral_format"].replace("{system_message}", quick),
    #     "{output}",
    #     "\n\n",
    # ),
    # "mistral-direct": (
    #     CHAT_TEMPLATE_FORMATS["mistral_format"].replace("{system_message}", direct),
    #     "{output}",
    #     "\n\n",
    # ),
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
    "mistral-aav": (
        CHAT_TEMPLATE_FORMATS["mistral_format"].replace("{system_message}", aav),
        "{output}",
        "\n\n", 
    ),
    # "mistral-kf": (
    #     CHAT_TEMPLATE_FORMATS["mistral_format"].replace("{system_message}", kf),
    #     "{output}",
    #     "\n\n", 
    # ),
    
    # ************** Phi Series **************
    # ## phi3
    # # "phi3-sbs-budget-hard": (
    # #     CHAT_TEMPLATE_FORMATS["phi3_format"].replace("{input}", sbs_budget_hard + "\n\n" + "{input}"),
    # #     "{output}",
    # #     "\n\n",
    # # ),
    # # "phi3-quick-hard": (
    # #     CHAT_TEMPLATE_FORMATS["phi3_format"].replace("{input}", quick_hard + "\n\n" + "{input}"),
    # #     "{output}",
    # #     "\n\n",
    # # ),
    # # "phi3-direct-hard": (
    # #     CHAT_TEMPLATE_FORMATS["phi3_format"].replace("{input}", direct_hard + "\n\n" + "{input}"),
    # #     "{output}",
    # #     "\n\n",
    # # ),
    # "phi3-sbs-hard": (
    #     CHAT_TEMPLATE_FORMATS["phi3_format"].replace("{input}", sbs_hard + "\n\n" + "{input}"),
    #     "{output}",
    #     "\n\n",
    # ),
    # "phi3-c2f-hard": (
    #     CHAT_TEMPLATE_FORMATS["phi3_format"].replace("{input}", c2f_hard + "\n\n" + "{input}"),
    #     "{output}",
    #     "\n\n",
    # ),
    # "phi3-aav-hard": (
    #     CHAT_TEMPLATE_FORMATS["phi3_format"].replace("{input}", aav_hard + "\n\n" + "{input}"),
    #     "{output}",
    #     "\n\n",
    # ),
    # # "phi3-kf-hard": (
    # #     CHAT_TEMPLATE_FORMATS["phi3_format"].replace("{input}", kf_hard + "\n\n" + "{input}"),
    # #     "{output}",
    # #     "\n\n",
    # # ),
    # # "phi3-quick": (
    # #     CHAT_TEMPLATE_FORMATS["phi3_format"].replace("{input}", quick + "\n\n" + "{input}"),
    # #     "{output}",
    # #     "\n\n",
    # # ),
    # # "phi3-direct": (
    # #     CHAT_TEMPLATE_FORMATS["phi3_format"].replace("{input}", direct + "\n\n" + "{input}"),
    # #     "{output}",
    # #     "\n\n",
    # # ),
    # "phi3-sbs": (
    #     CHAT_TEMPLATE_FORMATS["phi3_format"].replace("{input}", sbs + "\n\n" + "{input}"),
    #     "{output}",
    #     "\n\n",
    # ),
    # "phi3-c2f": (
    #     CHAT_TEMPLATE_FORMATS["phi3_format"].replace("{input}", c2f + "\n\n" + "{input}"),
    #     "{output}",
    #     "\n\n", 
    # ),
    # "phi3-aav": (
    #     CHAT_TEMPLATE_FORMATS["phi3_format"].replace("{input}", aav + "\n\n" + "{input}"),
    #     "{output}",
    #     "\n\n", 
    # ),
    # # "phi3-kf": (
    # #     CHAT_TEMPLATE_FORMATS["phi3_format"].replace("{input}", kf + "\n\n" + "{input}"),
    # #     "{output}",
    # #     "\n\n", 
    # # ),
    
    ## phi3mini
    # "phi3mini-sbs-budget-hard": (
    #     CHAT_TEMPLATE_FORMATS["phi3mini_format"].replace("{system_message}", sbs_budget_hard),
    #     "{output}",
    #     "\n\n",
    # ),
    # "phi3mini-sbs-budget": (
    #     CHAT_TEMPLATE_FORMATS["phi3mini_format"].replace("{system_message}", sbs_budget),
    #     "{output}",
    #     "\n\n",
    # ),
    # "phi3mini-c2f-budget": (
    #     CHAT_TEMPLATE_FORMATS["phi3mini_format"].replace("{system_message}", c2f_budget),
    #     "{output}",
    #     "\n\n",
    # ),
    # "phi3mini-aav-budget": (
    #     CHAT_TEMPLATE_FORMATS["phi3mini_format"].replace("{system_message}", aav_budget),
    #     "{output}",
    #     "\n\n",
    # ),
    # "phi3mini-quick-hard": (
    #     CHAT_TEMPLATE_FORMATS["phi3mini_format"].replace("{system_message}", quick_hard),
    #     "{output}",
    #     "\n\n",
    # ),
    # "phi3mini-direct-hard": (
    #     CHAT_TEMPLATE_FORMATS["phi3mini_format"].replace("{system_message}", direct_hard),
    #     "{output}",
    #     "\n\n",
    # ),
    "phi3mini-sbs-hard": (
        CHAT_TEMPLATE_FORMATS["phi3mini_format"].replace("{system_message}", sbs_hard),
        "{output}",
        "\n\n",
    ),
    "phi3mini-c2f-hard": (
        CHAT_TEMPLATE_FORMATS["phi3mini_format"].replace("{system_message}", c2f_hard),
        "{output}",
        "\n\n",
    ),
    "phi3mini-aav-hard": (
        CHAT_TEMPLATE_FORMATS["phi3mini_format"].replace("{system_message}", aav_hard),
        "{output}",
        "\n\n",
    ),
    # "phi3mini-kf-hard": (
    #     CHAT_TEMPLATE_FORMATS["phi3mini_format"].replace("{system_message}", kf_hard),
    #     "{output}",
    #     "\n\n",
    # ),
    # "phi3mini-quick": (
    #     CHAT_TEMPLATE_FORMATS["phi3mini_format"].replace("{system_message}", quick),
    #     "{output}",
    #     "\n\n",
    # ),
    # "phi3mini-direct": (
    #     CHAT_TEMPLATE_FORMATS["phi3mini_format"].replace("{system_message}", direct),
    #     "{output}",
    #     "\n\n",
    # ),
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
    "phi3mini-aav": (
        CHAT_TEMPLATE_FORMATS["phi3mini_format"].replace("{system_message}", aav),
        "{output}",
        "\n\n", 
    ),
    # "phi3mini-kf": (
    #     CHAT_TEMPLATE_FORMATS["phi3mini_format"].replace("{system_message}", kf),
    #     "{output}",
    #     "\n\n", 
    # ),
    
    ## phi3small
    # "phi3small-sbs-budget-hard": (
    #     CHAT_TEMPLATE_FORMATS["phi3small_format"].replace("{system_message}", sbs_budget_hard),
    #     "{output}",
    #     "\n\n",
    # ),
    # "phi3small-sbs-budget": (
    #     CHAT_TEMPLATE_FORMATS["phi3small_format"].replace("{system_message}", sbs_budget),
    #     "{output}",
    #     "\n\n",
    # ),
    # "phi3small-c2f-budget": (
    #     CHAT_TEMPLATE_FORMATS["phi3small_format"].replace("{system_message}", c2f_budget),
    #     "{output}",
    #     "\n\n",
    # ),
    # "phi3small-aav-budget": (
    #     CHAT_TEMPLATE_FORMATS["phi3small_format"].replace("{system_message}", aav_budget),
    #     "{output}",
    #     "\n\n",
    # ),
    # "phi3small-quick-hard": (
    #     CHAT_TEMPLATE_FORMATS["phi3small_format"].replace("{system_message}", quick_hard),
    #     "{output}",
    #     "\n\n",
    # ),
    # "phi3small-direct-hard": (
    #     CHAT_TEMPLATE_FORMATS["phi3small_format"].replace("{system_message}", direct_hard),
    #     "{output}",
    #     "\n\n",
    # ),
    "phi3small-sbs-hard": (
        CHAT_TEMPLATE_FORMATS["phi3small_format"].replace("{system_message}", sbs_hard),
        "{output}",
        "\n\n",
    ),
    "phi3small-c2f-hard": (
        CHAT_TEMPLATE_FORMATS["phi3small_format"].replace("{system_message}", c2f_hard),
        "{output}",
        "\n\n",
    ),
    "phi3small-aav-hard": (
        CHAT_TEMPLATE_FORMATS["phi3small_format"].replace("{system_message}", aav_hard),
        "{output}",
        "\n\n",
    ),
    # "phi3small-kf-hard": (
    #     CHAT_TEMPLATE_FORMATS["phi3small_format"].replace("{system_message}", kf_hard),
    #     "{output}",
    #     "\n\n",
    # ),
    # "phi3small-quick": (
    #     CHAT_TEMPLATE_FORMATS["phi3small_format"].replace("{system_message}", quick),
    #     "{output}",
    #     "\n\n",
    # ),
    # "phi3small-direct": (
    #     CHAT_TEMPLATE_FORMATS["phi3small_format"].replace("{system_message}", direct),
    #     "{output}",
    #     "\n\n",
    # ),
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
    "phi3small-aav": (
        CHAT_TEMPLATE_FORMATS["phi3small_format"].replace("{system_message}", aav),
        "{output}",
        "\n\n", 
    ),
    # "phi3small-kf": (
    #     CHAT_TEMPLATE_FORMATS["phi3small_format"].replace("{system_message}", kf),
    #     "{output}",
    #     "\n\n", 
    # ),
    
    ## phi3medium
    # "phi3medium-sbs-budget-hard": (
    #     CHAT_TEMPLATE_FORMATS["phi3medium_format"].replace("{input}", sbs_budget_hard + "\n\n" + "{input}"),
    #     "{output}",
    #     "\n\n",
    # ),
    # "phi3medium-sbs-budget": (
    #     CHAT_TEMPLATE_FORMATS["phi3medium_format"].replace("{input}", sbs_budget + "\n\n" + "{input}"),
    #     "{output}",
    #     "\n\n",
    # ),
    # "phi3medium-c2f-budget": (
    #     CHAT_TEMPLATE_FORMATS["phi3medium_format"].replace("{input}", c2f_budget + "\n\n" + "{input}"),
    #     "{output}",
    #     "\n\n",
    # ),
    # "phi3medium-aav-budget": (
    #     CHAT_TEMPLATE_FORMATS["phi3medium_format"].replace("{input}", aav_budget + "\n\n" + "{input}"),
    #     "{output}",
    #     "\n\n",
    # ),
    # "phi3medium-quick-hard": (
    #     CHAT_TEMPLATE_FORMATS["phi3medium_format"].replace("{input}", quick_hard + "\n\n" + "{input}"),
    #     "{output}",
    #     "\n\n",
    # ),
    # "phi3medium-direct-hard": (
    #     CHAT_TEMPLATE_FORMATS["phi3medium_format"].replace("{input}", direct_hard + "\n\n" + "{input}"),
    #     "{output}",
    #     "\n\n",
    # ),
    "phi3medium-sbs-hard": (
        CHAT_TEMPLATE_FORMATS["phi3medium_format"].replace("{input}", sbs_hard + "\n\n" + "{input}"),
        "{output}",
        "\n\n",
    ),
    "phi3medium-c2f-hard": (
        CHAT_TEMPLATE_FORMATS["phi3medium_format"].replace("{input}", c2f_hard + "\n\n" + "{input}"),
        "{output}",
        "\n\n",
    ),
    "phi3medium-aav-hard": (
        CHAT_TEMPLATE_FORMATS["phi3medium_format"].replace("{input}", aav_hard + "\n\n" + "{input}"),
        "{output}",
        "\n\n",
    ),
    # "phi3medium-kf-hard": (
    #     CHAT_TEMPLATE_FORMATS["phi3medium_format"].replace("{input}", kf_hard + "\n\n" + "{input}"),
    #     "{output}",
    #     "\n\n",
    # ),
    # "phi3medium-quick": (
    #     CHAT_TEMPLATE_FORMATS["phi3medium_format"].replace("{input}", quick + "\n\n" + "{input}"),
    #     "{output}",
    #     "\n\n",
    # ),
    # "phi3medium-direct": (
    #     CHAT_TEMPLATE_FORMATS["phi3medium_format"].replace("{input}", direct + "\n\n" + "{input}"),
    #     "{output}",
    #     "\n\n",
    # ),
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
    "phi3medium-aav": (
        CHAT_TEMPLATE_FORMATS["phi3medium_format"].replace("{input}", aav + "\n\n" + "{input}"),
        "{output}",
        "\n\n", 
    ),
    # "phi3medium-kf": (
    #     CHAT_TEMPLATE_FORMATS["phi3medium_format"].replace("{input}", kf + "\n\n" + "{input}"),
    #     "{output}",
    #     "\n\n", 
    # ),
    
    ## phi4
    # "phi4-sbs-budget-hard": (
    #     CHAT_TEMPLATE_FORMATS["phi4_format"].replace("{system_message}", sbs_budget_hard),
    #     "{output}",
    #     "\n\n",
    # ),
    # "phi4-sbs-budget": (
    #     CHAT_TEMPLATE_FORMATS["phi4_format"].replace("{system_message}", sbs_budget),
    #     "{output}",
    #     "\n\n",
    # ),
    # "phi4-c2f-budget": (
    #     CHAT_TEMPLATE_FORMATS["phi4_format"].replace("{system_message}", c2f_budget),
    #     "{output}",
    #     "\n\n",
    # ),
    # "phi4-aav-budget": (
    #     CHAT_TEMPLATE_FORMATS["phi4_format"].replace("{system_message}", aav_budget),
    #     "{output}",
    #     "\n\n",
    # ),
    # "phi4-quick-hard": (
    #     CHAT_TEMPLATE_FORMATS["phi4_format"].replace("{system_message}", quick_hard),
    #     "{output}",
    #     "\n\n",
    # ),
    # "phi4-direct-hard": (
    #     CHAT_TEMPLATE_FORMATS["phi4_format"].replace("{system_message}", direct_hard),
    #     "{output}",
    #     "\n\n",
    # ),
    "phi4-sbs-hard": (
        CHAT_TEMPLATE_FORMATS["phi4_format"].replace("{system_message}", sbs_hard),
        "{output}",
        "\n\n",
    ),
    "phi4-c2f-hard": (
        CHAT_TEMPLATE_FORMATS["phi4_format"].replace("{system_message}", c2f_hard),
        "{output}",
        "\n\n",
    ),
    "phi4-aav-hard": (
        CHAT_TEMPLATE_FORMATS["phi4_format"].replace("{system_message}", aav_hard),
        "{output}",
        "\n\n",
    ),
    # "phi4-kf-hard": (
    #     CHAT_TEMPLATE_FORMATS["phi4_format"].replace("{system_message}", kf_hard),
    #     "{output}",
    #     "\n\n",
    # ),
    # "phi4-quick": (
    #     CHAT_TEMPLATE_FORMATS["phi4_format"].replace("{system_message}", quick),
    #     "{output}",
    #     "\n\n",
    # ),
    # "phi4-direct": (
    #     CHAT_TEMPLATE_FORMATS["phi4_format"].replace("{system_message}", direct),
    #     "{output}",
    #     "\n\n",
    # ),
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
    "phi4-aav": (
        CHAT_TEMPLATE_FORMATS["phi4_format"].replace("{system_message}", aav),
        "{output}",
        "\n\n", 
    ),
    # "phi4-kf": (
    #     CHAT_TEMPLATE_FORMATS["phi4_format"].replace("{system_message}", kf),
    #     "{output}",
    #     "\n\n", 
    # ),
    
    # ************** Gemma Series **************
    # "gemma-sbs-budget-hard": (
    #     CHAT_TEMPLATE_FORMATS["gemma_format"].replace("{input}", sbs_budget_hard + "\n\n" + "{input}"),
    #     "{output}",
    #     "\n\n",
    # ),
    # "gemma-sbs-budget": (
    #     CHAT_TEMPLATE_FORMATS["gemma_format"].replace("{input}", sbs_budget + "\n\n" + "{input}"),
    #     "{output}",
    #     "\n\n",
    # ),
    # "gemma-c2f-budget": (
    #     CHAT_TEMPLATE_FORMATS["gemma_format"].replace("{input}", c2f_budget + "\n\n" + "{input}"),
    #     "{output}",
    #     "\n\n",
    # ),
    # "gemma-aav-budget": (
    #     CHAT_TEMPLATE_FORMATS["gemma_format"].replace("{input}", aav_budget + "\n\n" + "{input}"),
    #     "{output}",
    #     "\n\n",
    # ),
    # "gemma-quick-hard": (
    #     CHAT_TEMPLATE_FORMATS["gemma_format"].replace("{input}", quick_hard + "\n\n" + "{input}"),
    #     "{output}",
    #     "\n\n",
    # ),
    # "gemma-direct-hard": (
    #     CHAT_TEMPLATE_FORMATS["gemma_format"].replace("{input}", direct_hard + "\n\n" + "{input}"),
    #     "{output}",
    #     "\n\n",
    # ),
    "gemma-sbs-hard": (
        CHAT_TEMPLATE_FORMATS["gemma_format"].replace("{input}", sbs_hard + "\n\n" + "{input}"),
        "{output}",
        "\n\n",
    ),
    "gemma-c2f-hard": (
        CHAT_TEMPLATE_FORMATS["gemma_format"].replace("{input}", c2f_hard + "\n\n" + "{input}"),
        "{output}",
        "\n\n",
    ),
    "gemma-aav-hard": (
        CHAT_TEMPLATE_FORMATS["gemma_format"].replace("{input}", aav_hard + "\n\n" + "{input}"),
        "{output}",
        "\n\n",
    ),
    # "gemma-kf-hard": (
    #     CHAT_TEMPLATE_FORMATS["gemma_format"].replace("{input}", kf_hard + "\n\n" + "{input}"),
    #     "{output}",
    #     "\n\n",
    # ),
    # "gemma-quick": (
    #     CHAT_TEMPLATE_FORMATS["gemma_format"].replace("{input}", quick + "\n\n" + "{input}"),
    #     "{output}",
    #     "\n\n",
    # ),
    # "gemma-direct": (
    #     CHAT_TEMPLATE_FORMATS["gemma_format"].replace("{input}", direct + "\n\n" + "{input}"),
    #     "{output}",
    #     "\n\n",
    # ),
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
    "gemma-aav": (
        CHAT_TEMPLATE_FORMATS["gemma_format"].replace("{input}", aav + "\n\n" + "{input}"),
        "{output}",
        "\n\n", 
    ),
    # "gemma-kf": (
    #     CHAT_TEMPLATE_FORMATS["gemma_format"].replace("{input}", kf + "\n\n" + "{input}"),
    #     "{output}",
    #     "\n\n", 
    # ),
    
    # ************** Llama Series **************
    # "llama-sbs-budget-hard": (
    #     CHAT_TEMPLATE_FORMATS["llama_format"].replace("{system_message}", sbs_budget_hard),
    #     "{output}",
    #     "\n\n",
    # ),
    # "llama-quick-hard": (
    #     CHAT_TEMPLATE_FORMATS["llama_format"].replace("{system_message}", quick_hard),
    #     "{output}",
    #     "\n\n",
    # ),
    # "llama-direct-hard": (
    #     CHAT_TEMPLATE_FORMATS["llama_format"].replace("{system_message}", direct_hard),
    #     "{output}",
    #     "\n\n",
    # ),
    "llama-sbs-hard": (
        CHAT_TEMPLATE_FORMATS["llama_format"].replace("{system_message}", sbs_hard),
        "{output}",
        "\n\n",
    ),
    "llama-c2f-hard": (
        CHAT_TEMPLATE_FORMATS["llama_format"].replace("{system_message}", c2f_hard),
        "{output}",
        "\n\n",
    ),
    "llama-aav-hard": (
        CHAT_TEMPLATE_FORMATS["llama_format"].replace("{system_message}", aav_hard),
        "{output}",
        "\n\n",
    ),
     # "llama-quick": (
    #     CHAT_TEMPLATE_FORMATS["llama_format"].replace("{system_message}", quick),
    #     "{output}",
    #     "\n\n",
    # ),
    # "llama-direct": (
    #     CHAT_TEMPLATE_FORMATS["llama_format"].replace("{system_message}", direct),
    #     "{output}",
    #     "\n\n",
    # ),
    "llama-sbs": (
        CHAT_TEMPLATE_FORMATS["llama_format"].replace("{system_message}", sbs),
        "{output}",
        "\n\n",
    ),
    "llama-c2f": (
        CHAT_TEMPLATE_FORMATS["llama_format"].replace("{system_message}", c2f),
        "{output}",
        "\n\n", 
    ),
    "llama-aav": (
        CHAT_TEMPLATE_FORMATS["llama_format"].replace("{system_message}", aav),
        "{output}",
        "\n\n", 
    ),
    # "llama-kf": (
    #     CHAT_TEMPLATE_FORMATS["llama_format"].replace("{system_message}", kf),
    #     "{output}",
    #     "\n\n", 
    # ),
    
    # ************** DeepSeek-R1-Distill **************
    ## DeepSeek-R1-Distill models output <think> xxx </think> part first
    # "deepseek-r1-distill-sbs-budget-hard": (
    #     CHAT_TEMPLATE_FORMATS["deepseek-r1-distill_format"].replace("{input}", sbs_budget_hard + "\n\n" + "{input}"),
    #     "{output}",
    #     "\n\n",
    # ),
    "deepseek-r1-distill-sbs-hard": (
        CHAT_TEMPLATE_FORMATS["deepseek-r1-distill_format"].replace("{input}", sbs_hard + "\n\n" + "{input}"),
        "{output}",
        "\n\n",
    ),
    "deepseek-r1-distill-c2f-hard": (
        CHAT_TEMPLATE_FORMATS["deepseek-r1-distill_format"].replace("{input}", c2f_hard + "\n\n" + "{input}"),
        "{output}",
        "\n\n",
    ),
    "deepseek-r1-distill-aav-hard": (
        CHAT_TEMPLATE_FORMATS["deepseek-r1-distill_format"].replace("{input}", aav_hard + "\n\n" + "{input}"),
        "{output}",
        "\n\n",
    ),
    "deepseek-r1-distill-sbs": (
        CHAT_TEMPLATE_FORMATS["deepseek-r1-distill_format"].replace("{input}", sbs + "\n\n" + "{input}"),
        "{output}",
        "\n\n",
    ),
    "deepseek-r1-distill-c2f": (
        CHAT_TEMPLATE_FORMATS["deepseek-r1-distill_format"].replace("{input}", c2f + "\n\n" + "{input}"),
        "{output}",
        "\n\n",
    ),
    "deepseek-r1-distill-aav": (
        CHAT_TEMPLATE_FORMATS["deepseek-r1-distill_format"].replace("{input}", aav + "\n\n" + "{input}"),
        "{output}",
        "\n\n",
    )
}
