# o1-mimic doesn't help
# o1-mimic_hard
# o1_mimic_hard = """For the given math problem, you MUST engage in a thorough, logical, and systematic thought process before responding.

# CORE THINKING SEQUENCE

# Initial Engagement

# When you first encounter a math problem, do the following:
# 	1.	Rephrase the problem in your own words to ensure you understand it.
# 	2.	Identify key concepts and mathematical operations involved.
# 	3.	Consider the context and what the problem is asking for.
# 	4.	Clarify any unknowns and what information you have available.
# 	5.	Spot any potential ambiguities or edge cases that may need further clarification.

# Problem Space Exploration

# After understanding the question, explore the problem deeply:
# 	1.	Break down the problem into smaller, manageable parts.
# 	2.	Identify any constraints or special conditions (e.g., assumptions, domain limitations).
# 	3.	Define the goal clearly—what exactly do you need to find or prove?
# 	4.	Map out the required steps to solve the problem, considering what’s needed at each stage.

# Hypothesis Generation

# Before committing to a specific approach:
# 	1.	Generate multiple solution paths or methods.
# 	2.	Evaluate the merits of each approach based on available data, simplicity, and efficiency.
# 	3.	Consider alternative ways to view the problem or potential simplifications.

# Solution Process

# Work through the problem methodically:
# 	1.	Start from basics and move step by step through each part of the solution.
# 	2.	Look for patterns or symmetries that could simplify the work.
# 	3.	Question initial assumptions or steps as you go—math problems can often reveal deeper insights as you probe them.
# 	4.	Make connections between different parts of the problem, refining your understanding as you progress.

# Testing and Verification

# As you solve the problem:
# 	1.	Double-check your assumptions at every stage.
# 	2.	Test intermediate results against known facts or smaller examples.
# 	3.	Verify the consistency of your reasoning and computations.
# 	4.	Consider special cases or edge conditions that might challenge your conclusions.

# Error Recognition and Correction

# If mistakes arise:
# 	1.	Recognize and acknowledge the flaw in reasoning.
# 	2.	Analyze why the error occurred and adjust your process accordingly.
# 	3.	Correct the logic or calculations and integrate new insights into the solution.

# Knowledge Synthesis

# Build a coherent solution:
# 	1.	Connect different mathematical concepts or facts used throughout the process.
# 	2.	Ensure that all relevant aspects of the problem are addressed.
# 	3.	Develop a clear and concise response that ties everything together logically.

# Response Preparation

# Before finalizing the response:
# 	1.	Review all steps taken and ensure every part of the problem is addressed.
# 	2.	Provide a clear, step-by-step solution that matches the complexity of the problem.
# 	3.	If applicable, explain the reasoning behind each step so the process is transparent.
# 	4.	Anticipate possible follow-up questions and address any unclear points preemptively.

# Essential Thinking Characteristics
# 	•	Authenticity: Your reasoning must feel natural, evolving as you progress through the problem.
# 	•	Depth: Your analysis should deepen progressively, starting from basic facts and building toward more complex insights.
# 	•	Balance: Maintain the balance between considering edge cases and keeping the solution streamlined.
# 	•	Clear Communication: The final answer must be structured, logical, and presented with clear reasoning.

# Final Output

# Your solution should be clear, detailed, and logically consistent. It should answer the question fully while leaving no step or idea unexplored. 

# Your final answer should be put within \\boxed{{}}."""
# o1_mimic_hard = """You should use chain of thought and reason step by step to fully understand the problem.
# During your reasoning process:
# You should recognize and correct your potential mistakes;
# You should break down tricky steps into simpler ones;
# You should try a different approach when the current one isn’t working;
# You can reason and solve the problem from different perspectives to make sure you have a comprehensive understanding of the problem.

# Put your final answer within \\boxed{{}}."""


## direct truncation by setting max new tokens
# quick hard
quick_hard = """Give an answer based on intuition or quick calculation. Put your answer within \\boxed{{}}."""

# direct hard
direct_hard = """Solve the problem and put your final answer within \\boxed{{}}."""

# Step-by-step hard
sbs_hard = """Please reason step by step. 
Conclude with: 
Therefore, the final answer is: \\boxed{{[answer]}}.
Where [answer] is just the final number or expression that solves the problem."""

# Coarse-to-fine hard
c2f_hard = """Use the following pattern to solve the problem:
**Coarse-Grained Reasoning**
Provide a brief analysis and initial answer, focusing on efficiency and conciseness.

**Fine-Grained Reasoning**
Provide detailed reasoning step by step and a refined answer, focusing on correctness and rigor.

Conclude with: 
Therefore, the final answer is: \\boxed{{[answer]}}.
Where [answer] is just the final number or expression that solves the problem."""

# Knowledge First hard
kf_hard = """Solve the task by following format:
**Knowledge to Use**
List theorem and methods that are useful for solving the problem.

**Reasoning**
Step by step analysis using above knowledge.

**Final Answer**
Your final answer within \\boxed{{}}."""

# Answer and Verify hard
aav_hard = """Solve the task by following format:
**Quick Answer**
Give an initial answer based on intuition or quick calculation.

**Verification**
Give a revised answer through reasoning step by step to correct potential mistakes.

**Final Answer**
Your final answer within \\boxed{{}}."""


# early truncation and using termination tokens
early_stop = """\n\nNotice: When you are interupted by the keyword **Time’s Up!**, stop reasoning immediately.
Based on your reasoning so far, conclude with: 
Therefore, the final answer is: \\boxed{{[answer]}}.
Where [answer] is just the final number or expression that solves the problem."""

# quick
quick = """Give an answer based on intuition or quick calculation. Put your answer within \\boxed{{}} when done or early-stop keyword **Early Stop** appears."""

# direct
direct = """Solve the problem and put your final answer within \\boxed{{}} when done or early-stop keyword **Final Answer** appears."""

# Step-by-step
sbs = sbs_hard + early_stop

# Coarse-to-fine
c2f = c2f_hard + early_stop

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
    
    "phi3_format": "<|user|>\n{input}<|end|>\n<|assistant|>\n",
    
    "phi3mini_format": "<|system|>\n{system_message}<|end|>\n<|user|>\n{input}<|end|>\n<|assistant|>\n",
    
    "phi3small_format": "<|endoftext|><|system|>\n{system_message}<|end|>\n<|user|>\n{input}<|end|>\n<|assistant|>\n",
    
    "phi3medium_format": "<|user|>\n{input}<|end|>\n<|assistant|>\n",
    
    "phi4_format": "<|im_start|>system<|im_sep|>{system_message}<|im_end|><|im_start|>user<|im_sep|>{input}<|im_end|><|im_start|>assistant<|im_sep|>",
    
    "llama_format": "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{system_message}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{input}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n",
    
    "gemma_format": "<bos><start_of_turn>user\n{input}<end_of_turn>\n<start_of_turn>model\n",
    
    "numina_format": "### Problem: {input}\n### Solution: ",
    
    "internlm_format": "<s><|im_start|>system\n{system_message}<|im_end|>\n<|im_start|>user\n{input}<|im_end|>\n<|im_start|>assistant\n",
    
    "deepseek_format": "<｜begin▁of▁sentence｜>{system_message}\n\nUser: {input}\n\nAssistant:",
    
    "deepseek-r1-distill_format" : "<｜begin▁of▁sentence｜><｜User｜>{input}<｜Assistant｜>"    # Avoid adding a system prompt; all instructions should be contained within the user prompt.
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
    ## qwen-math
    "qwen-math-quick-hard": (
        CHAT_TEMPLATE_FORMATS["qwen_format"].replace("{system_message}", "You are a math expert.").replace("{input}", quick_hard + "\n\n" + "{input}"),
        "{output}",
        "\n\n",
    ),
    "qwen-math-direct-hard": (
        CHAT_TEMPLATE_FORMATS["qwen_format"].replace("{system_message}", "You are a math expert.").replace("{input}", direct_hard + "\n\n" + "{input}"),
        "{output}",
        "\n\n",
    ),
    "qwen-math-sbs-hard": (
        CHAT_TEMPLATE_FORMATS["qwen_format"].replace("{system_message}", "You are a math expert.").replace("{input}", sbs_hard + "\n\n" + "{input}"),
        "{output}",
        "\n\n",
    ),
    "qwen-math-c2f-hard": (
        CHAT_TEMPLATE_FORMATS["qwen_format"].replace("{system_message}", "You are a math expert.").replace("{input}", c2f_hard + "\n\n" + "{input}"),
        "{output}",
        "\n\n",
    ),
    "qwen-math-kf-hard": (
        CHAT_TEMPLATE_FORMATS["qwen_format"].replace("{system_message}", "You are a math expert.").replace("{input}", kf_hard + "\n\n" + "{input}"),
        "{output}",
        "\n\n",
    ),
    "qwen-math-aav-hard": (
        CHAT_TEMPLATE_FORMATS["qwen_format"].replace("{system_message}", "You are a math expert.").replace("{input}", aav_hard + "\n\n" + "{input}"),
        "{output}",
        "\n\n",
    ),
    
    "qwen-math-quick": (
        CHAT_TEMPLATE_FORMATS["qwen_format"].replace("{system_message}", "You are a math expert.").replace("{input}", quick + "\n\n" + "{input}"),
        "{output}",
        "\n\n",
    ),
    "qwen-math-direct": (
        CHAT_TEMPLATE_FORMATS["qwen_format"].replace("{system_message}", "You are a math expert.").replace("{input}", direct + "\n\n" + "{input}"),
        "{output}",
        "\n\n",
    ),
    "qwen-math-sbs": (
        CHAT_TEMPLATE_FORMATS["qwen_format"].replace("{system_message}", "You are a math expert.").replace("{input}", sbs + "\n\n" + "{input}"),
        "{output}",
        "\n\n",
    ),
    "qwen-math-c2f": (
        CHAT_TEMPLATE_FORMATS["qwen_format"].replace("{system_message}", "You are a math expert.").replace("{input}", c2f + "\n\n" + "{input}"),
        "{output}",
        "\n\n",
    ),
    "qwen-math-kf": (
        CHAT_TEMPLATE_FORMATS["qwen_format"].replace("{system_message}", "You are a math expert.").replace("{input}", kf + "\n\n" + "{input}"),
        "{output}",
        "\n\n",
    ),
    "qwen-math-aav": (
        CHAT_TEMPLATE_FORMATS["qwen_format"].replace("{system_message}", "You are a math expert.").replace("{input}", aav + "\n\n" + "{input}"),
        "{output}",
        "\n\n",
    ),
    
    ## qwen2.5
    # "qwen-o1-mimic-hard-user": (
    #     CHAT_TEMPLATE_FORMATS["qwen_format"].replace("{system_message}", "You are a math problem solver.").replace("{input}", o1_mimic_hard + "\n\n" + "{input}"),
    #     "{output}",
    #     "\n\n",
    # ),
    # "qwen-o1-mimic-hard": (
    #     CHAT_TEMPLATE_FORMATS["qwen_format"].replace("{system_message}", o1_mimic_hard),
    #     "{output}",
    #     "\n\n",
    # ),
    "qwen-quick-hard": (
        CHAT_TEMPLATE_FORMATS["qwen_format"].replace("{system_message}", quick_hard),
        "{output}",
        "\n\n",
    ),
    "qwen-direct-hard": (
        CHAT_TEMPLATE_FORMATS["qwen_format"].replace("{system_message}", direct_hard),
        "{output}",
        "\n\n",
    ),
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
    "qwen-kf-hard": (
        CHAT_TEMPLATE_FORMATS["qwen_format"].replace("{system_message}", kf_hard),
        "{output}",
        "\n\n",
    ),
    "qwen-aav-hard": (
        CHAT_TEMPLATE_FORMATS["qwen_format"].replace("{system_message}", aav_hard),
        "{output}",
        "\n\n",
    ),
    "qwen-quick": (
        CHAT_TEMPLATE_FORMATS["qwen_format"].replace("{system_message}", quick),
        "{output}",
        "\n\n",
    ),
    "qwen-direct": (
        CHAT_TEMPLATE_FORMATS["qwen_format"].replace("{system_message}", direct),
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
    "mistral-quick-hard": (
        CHAT_TEMPLATE_FORMATS["mistral_format"].replace("{system_message}", quick_hard),
        "{output}",
        "\n\n",
    ),
    "mistral-direct-hard": (
        CHAT_TEMPLATE_FORMATS["mistral_format"].replace("{system_message}", direct_hard),
        "{output}",
        "\n\n",
    ),
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
    "mistral-kf-hard": (
        CHAT_TEMPLATE_FORMATS["mistral_format"].replace("{system_message}", kf_hard),
        "{output}",
        "\n\n",
    ),
    "mistral-aav-hard": (
        CHAT_TEMPLATE_FORMATS["mistral_format"].replace("{system_message}", aav_hard),
        "{output}",
        "\n\n",
    ),
    "mistral-quick": (
        CHAT_TEMPLATE_FORMATS["mistral_format"].replace("{system_message}", quick),
        "{output}",
        "\n\n",
    ),
    "mistral-direct": (
        CHAT_TEMPLATE_FORMATS["mistral_format"].replace("{system_message}", direct),
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
    ## phi3
    "phi3-quick-hard": (
        CHAT_TEMPLATE_FORMATS["phi3_format"].replace("{input}", quick_hard + "\n\n" + "{input}"),
        "{output}",
        "\n\n",
    ),
    "phi3-direct-hard": (
        CHAT_TEMPLATE_FORMATS["phi3_format"].replace("{input}", direct_hard + "\n\n" + "{input}"),
        "{output}",
        "\n\n",
    ),
    "phi3-sbs-hard": (
        CHAT_TEMPLATE_FORMATS["phi3_format"].replace("{input}", sbs_hard + "\n\n" + "{input}"),
        "{output}",
        "\n\n",
    ),
    "phi3-c2f-hard": (
        CHAT_TEMPLATE_FORMATS["phi3_format"].replace("{input}", c2f_hard + "\n\n" + "{input}"),
        "{output}",
        "\n\n",
    ),
    "phi3-kf-hard": (
        CHAT_TEMPLATE_FORMATS["phi3_format"].replace("{input}", kf_hard + "\n\n" + "{input}"),
        "{output}",
        "\n\n",
    ),
    "phi3-aav-hard": (
        CHAT_TEMPLATE_FORMATS["phi3_format"].replace("{input}", aav_hard + "\n\n" + "{input}"),
        "{output}",
        "\n\n",
    ),
    "phi3-quick": (
        CHAT_TEMPLATE_FORMATS["phi3_format"].replace("{input}", quick + "\n\n" + "{input}"),
        "{output}",
        "\n\n",
    ),
    "phi3-direct": (
        CHAT_TEMPLATE_FORMATS["phi3_format"].replace("{input}", direct + "\n\n" + "{input}"),
        "{output}",
        "\n\n",
    ),
    "phi3-sbs": (
        CHAT_TEMPLATE_FORMATS["phi3_format"].replace("{input}", sbs + "\n\n" + "{input}"),
        "{output}",
        "\n\n",
    ),
    "phi3-c2f": (
        CHAT_TEMPLATE_FORMATS["phi3_format"].replace("{input}", c2f + "\n\n" + "{input}"),
        "{output}",
        "\n\n", 
    ),
    "phi3-kf": (
        CHAT_TEMPLATE_FORMATS["phi3_format"].replace("{input}", kf + "\n\n" + "{input}"),
        "{output}",
        "\n\n", 
    ),
    "phi3-aav": (
        CHAT_TEMPLATE_FORMATS["phi3_format"].replace("{input}", aav + "\n\n" + "{input}"),
        "{output}",
        "\n\n", 
    ),
    
    ## phi3mini
    "phi3mini-quick-hard": (
        CHAT_TEMPLATE_FORMATS["phi3mini_format"].replace("{system_message}", quick_hard),
        "{output}",
        "\n\n",
    ),
    "phi3mini-direct-hard": (
        CHAT_TEMPLATE_FORMATS["phi3mini_format"].replace("{system_message}", direct_hard),
        "{output}",
        "\n\n",
    ),
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
    "phi3mini-kf-hard": (
        CHAT_TEMPLATE_FORMATS["phi3mini_format"].replace("{system_message}", kf_hard),
        "{output}",
        "\n\n",
    ),
    "phi3mini-aav-hard": (
        CHAT_TEMPLATE_FORMATS["phi3mini_format"].replace("{system_message}", aav_hard),
        "{output}",
        "\n\n",
    ),
    "phi3mini-quick": (
        CHAT_TEMPLATE_FORMATS["phi3mini_format"].replace("{system_message}", quick),
        "{output}",
        "\n\n",
    ),
    "phi3mini-direct": (
        CHAT_TEMPLATE_FORMATS["phi3mini_format"].replace("{system_message}", direct),
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
    "phi3small-quick-hard": (
        CHAT_TEMPLATE_FORMATS["phi3small_format"].replace("{system_message}", quick_hard),
        "{output}",
        "\n\n",
    ),
    "phi3small-direct-hard": (
        CHAT_TEMPLATE_FORMATS["phi3small_format"].replace("{system_message}", direct_hard),
        "{output}",
        "\n\n",
    ),
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
    "phi3small-kf-hard": (
        CHAT_TEMPLATE_FORMATS["phi3small_format"].replace("{system_message}", kf_hard),
        "{output}",
        "\n\n",
    ),
    "phi3small-aav-hard": (
        CHAT_TEMPLATE_FORMATS["phi3small_format"].replace("{system_message}", aav_hard),
        "{output}",
        "\n\n",
    ),
    "phi3small-quick": (
        CHAT_TEMPLATE_FORMATS["phi3small_format"].replace("{system_message}", quick),
        "{output}",
        "\n\n",
    ),
    "phi3small-direct": (
        CHAT_TEMPLATE_FORMATS["phi3small_format"].replace("{system_message}", direct),
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
    "phi3medium-quick-hard": (
        CHAT_TEMPLATE_FORMATS["phi3medium_format"].replace("{input}", quick_hard + "\n\n" + "{input}"),
        "{output}",
        "\n\n",
    ),
    "phi3medium-direct-hard": (
        CHAT_TEMPLATE_FORMATS["phi3medium_format"].replace("{input}", direct_hard + "\n\n" + "{input}"),
        "{output}",
        "\n\n",
    ),
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
    "phi3medium-kf-hard": (
        CHAT_TEMPLATE_FORMATS["phi3medium_format"].replace("{input}", kf_hard + "\n\n" + "{input}"),
        "{output}",
        "\n\n",
    ),
    "phi3medium-aav-hard": (
        CHAT_TEMPLATE_FORMATS["phi3medium_format"].replace("{input}", aav_hard + "\n\n" + "{input}"),
        "{output}",
        "\n\n",
    ),
    "phi3medium-quick": (
        CHAT_TEMPLATE_FORMATS["phi3medium_format"].replace("{input}", quick + "\n\n" + "{input}"),
        "{output}",
        "\n\n",
    ),
    "phi3medium-direct": (
        CHAT_TEMPLATE_FORMATS["phi3medium_format"].replace("{input}", direct + "\n\n" + "{input}"),
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
    "phi4-quick-hard": (
        CHAT_TEMPLATE_FORMATS["phi4_format"].replace("{system_message}", quick_hard),
        "{output}",
        "\n\n",
    ),
    "phi4-direct-hard": (
        CHAT_TEMPLATE_FORMATS["phi4_format"].replace("{system_message}", direct_hard),
        "{output}",
        "\n\n",
    ),
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
    "phi4-kf-hard": (
        CHAT_TEMPLATE_FORMATS["phi4_format"].replace("{system_message}", kf_hard),
        "{output}",
        "\n\n",
    ),
    "phi4-aav-hard": (
        CHAT_TEMPLATE_FORMATS["phi4_format"].replace("{system_message}", aav_hard),
        "{output}",
        "\n\n",
    ),
    "phi4-quick": (
        CHAT_TEMPLATE_FORMATS["phi4_format"].replace("{system_message}", quick),
        "{output}",
        "\n\n",
    ),
    "phi4-direct": (
        CHAT_TEMPLATE_FORMATS["phi4_format"].replace("{system_message}", direct),
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
    "gemma-quick-hard": (
        CHAT_TEMPLATE_FORMATS["gemma_format"].replace("{input}", quick_hard + "\n\n" + "{input}"),
        "{output}",
        "\n\n",
    ),
    "gemma-direct-hard": (
        CHAT_TEMPLATE_FORMATS["gemma_format"].replace("{input}", direct_hard + "\n\n" + "{input}"),
        "{output}",
        "\n\n",
    ),
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
    "gemma-kf-hard": (
        CHAT_TEMPLATE_FORMATS["gemma_format"].replace("{input}", kf_hard + "\n\n" + "{input}"),
        "{output}",
        "\n\n",
    ),
    "gemma-aav-hard": (
        CHAT_TEMPLATE_FORMATS["gemma_format"].replace("{input}", aav_hard + "\n\n" + "{input}"),
        "{output}",
        "\n\n",
    ),
    "gemma-quick": (
        CHAT_TEMPLATE_FORMATS["gemma_format"].replace("{input}", quick + "\n\n" + "{input}"),
        "{output}",
        "\n\n",
    ),
    "gemma-direct": (
        CHAT_TEMPLATE_FORMATS["gemma_format"].replace("{input}", direct + "\n\n" + "{input}"),
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
    
    # ************** Llama Series **************
    "llama-quick": (
        CHAT_TEMPLATE_FORMATS["llama_format"].replace("{system_message}", quick),
        "{output}",
        "\n\n",
    ),
    "llama-direct": (
        CHAT_TEMPLATE_FORMATS["llama_format"].replace("{system_message}", direct),
        "{output}",
        "\n\n",
    ),
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
    "llama-kf": (
        CHAT_TEMPLATE_FORMATS["llama_format"].replace("{system_message}", kf),
        "{output}",
        "\n\n", 
    ),
    "llama-sbs-hard": (
        CHAT_TEMPLATE_FORMATS["llama_format"].replace("{system_message}", sbs_hard),
        "{output}",
        "\n\n",
    ),
    "llama-direct-hard": (
        CHAT_TEMPLATE_FORMATS["llama_format"].replace("{system_message}", direct_hard),
        "{output}",
        "\n\n",
    ),
    "llama-quick-hard": (
        CHAT_TEMPLATE_FORMATS["llama_format"].replace("{system_message}", quick_hard),
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
    
    # ************** DeepSeek-R1-Distill **************
    ## DeepSeek-R1-Distill models output <think> xxx </think> part first
    "deepseek-r1-distill-sbs-hard": (
        CHAT_TEMPLATE_FORMATS["deepseek-r1-distill_format"].replace("{input}", sbs_hard + "\n\n" + "{input}"),
        "{output}",
        "\n\n",
    ),
    "deepseek-r1-distill-sbs": (
        CHAT_TEMPLATE_FORMATS["deepseek-r1-distill_format"].replace("{input}", sbs + "\n\n" + "{input}"),
        "{output}",
        "\n\n",
    ),
    
    # ************** InternLM Series**************
    ## internlm2.5
    "internlm-quick": (
        CHAT_TEMPLATE_FORMATS["internlm_format"].replace("{system_message}", quick),
        "{output}",
        "\n\n",
    ),
    "internlm-direct": (
        CHAT_TEMPLATE_FORMATS["internlm_format"].replace("{system_message}", direct),
        "{output}",
        "\n\n",
    ),
    "internlm-sbs": (
        CHAT_TEMPLATE_FORMATS["internlm_format"].replace("{system_message}", sbs),
        "{output}",
        "\n\n",
    ),
    "internlm-c2f": (
        CHAT_TEMPLATE_FORMATS["internlm_format"].replace("{system_message}", c2f),
        "{output}",
        "\n\n",
    ),
    "internlm-aav": (
        CHAT_TEMPLATE_FORMATS["internlm_format"].replace("{system_message}", aav),
        "{output}",
        "\n\n",
    ),
    "internlm-kf": (
        CHAT_TEMPLATE_FORMATS["internlm_format"].replace("{system_message}", kf),
        "{output}",
        "\n\n",
    ),
    "internlm-sbs-hard": (
        CHAT_TEMPLATE_FORMATS["internlm_format"].replace("{system_message}", sbs_hard),
        "{output}",
        "\n\n",
    ),
    "internlm-direct-hard": (
        CHAT_TEMPLATE_FORMATS["internlm_format"].replace("{system_message}", direct_hard),
        "{output}",
        "\n\n",
    ),
    "internlm-quick-hard": (
        CHAT_TEMPLATE_FORMATS["internlm_format"].replace("{system_message}", quick_hard),
        "{output}",
        "\n\n",
    ),
    "internlm-c2f-hard": (
        CHAT_TEMPLATE_FORMATS["internlm_format"].replace("{system_message}", c2f_hard),
        "{output}",
        "\n\n",
    ),
    "internlm-aav-hard": (
        CHAT_TEMPLATE_FORMATS["internlm_format"].replace("{system_message}", aav_hard),
        "{output}",
        "\n\n",
    )
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
    
}