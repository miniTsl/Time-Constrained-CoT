step_by_step_hard = "Please reason step by step, and put your final answer within \\boxed{{}}."

step_by_step = "Please reason step by step, and put your final answer within \\boxed{{}} when done reasoning or early-stop keyword **Final Answer** appears."

coarse_to_fine = "Solve the task by following format:\n**Coarse Reasoning**\nShort analysis and an answer. Focus on efficiency and simplicity.\n\n**Fine Reasoning**\nDetailed analysis step by step and a refined answer. Focus on accuracy and correctness.\n\n**Final Answer** \nYour final answer within \\boxed{{}} when done reasoning or early-stop keyword **Final Answer** appears."



PROMPT_TEMPLATES = {
    "direct": ("Question: {input}\nAnswer: ", "{output}", "\n\n"),
    "cot": ("Question: {input}\nAnswer: ", "{output}", "\n\n\n"),
    "pal": ("Question: {input}\n\n", "{output}", "\n---\n"),
    "tool-integrated": ("Question: {input}\n\nSolution:\n", "{output}", "\n---\n"),
    "self-instruct": ("<|user|>\n{input}\n<|assistant|>\n", "{output}", "\n"),
    "tora": ("<|user|>\n{input}\n<|assistant|>\n", "{output}", "\n"),
    "wizard_zs": (
        "### Instruction:\n{input}\n\n### Response: Let's think step by step.",
        "{output}",
        "\n\n\n",
    ),
    "platypus_fs": (
        "### Instruction:\n{input}\n\n### Response:\n",
        "{output}",
        "\n\n\n",
    ),
    "deepseek-math": (
        "User: {input}\nPlease reason step by step, "
        "and put your final answer within \\boxed{{}}.\n\nAssistant:",
        "{output}",
        "\n\n\n",
    ),
    "kpmath": (
        "User: Please reason step by step and put your final answer at the end "
        'with "The answer is: ".\n\n{input}\n\nAssistant:',
        "{output}",
    ),
    "jiuzhang": (
        "## Question\n{input}\n\n## Solution\n",
        "{output}",
        "\n\n\n",
    ),
    "jiuzhang_tora": (
        "## Question\n{input}\n\n## Code Solution\n",
        "{output}",
        "\n\n\n",
    ),
    "jiuzhang_nl": (
        "## Question\n{input}\n\n## Natural Language Solution\n",
        "{output}",
        "\n\n\n",
    ),
    "mmiqc": (
        'Please solve the following problem and put your answer at the end with "The answer is: ".\n\n{input}\n\n',
        "{output}",
        "\n\n\n",
    ),
    "abel": (
        "Question:\n{input}\nAnswer:\nLet's think step by step.\n",
        "{output}",
        "\n\n",
    ),
    "shepherd": ("{input}\n", "{output}", "\n\n\n"),
    "qwen-boxed": (
        "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
        "<|im_start|>user\n{input}\nPlease reason step by step, and put your final answer within \\boxed{{}}.<|im_end|>\n"
        "<|im_start|>assistant\n",
        "{output}",
        "\n\n",
    ),
    # ************** Qwen **************
    "qwen25-math-cot": (
        "<|im_start|>system\n" + step_by_step + "<|im_end|>\n"
        "<|im_start|>user\n{input}<|im_end|>\n"
        "<|im_start|>assistant\n",
        "{output}",
        "\n\n",
    ),
    "coarse-to-fine-qwen": (
        "<|im_start|>system\n" + coarse_to_fine + "<|im_end|>\n"
        "<|im_start|>user\n{input}<|im_end|>\n"
        "<|im_start|>assistant\n",
        "{output}",
        "\n\n", 
    ),
    "qwen25-step-by-step-hard": (
        "<|im_start|>system\n" + step_by_step_hard + "<|im_end|>\n"
        "<|im_start|>user\n{input}<|im_end|>\n"
        "<|im_start|>assistant\n",
        "{output}",
        "\n\n",
    ),
    # ************** Mathstral **************
    "mathstral-step-by-step-hard": (
        "<s>[INST] " + step_by_step_hard + "\n\n{input}[/INST]",
        "{output}",
        "\n\n",
    ),
    "mathstral-step-by-step": (
        "<s>[INST] " + step_by_step + "\n\n{input}[/INST]",
        "{output}",
        "\n\n",
    ),
    "mathstral-coarse-to-fine": (
        "<s>[INST] " + coarse_to_fine + "\n\n{input}[/INST]",
        "{output}",
        "\n\n", 
    ),
    # ************** Skywork **************
    "skywork-step-by-step": (
        "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n" + step_by_step + "<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{input}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n",
        "{output}",
        "\n\n",
    ),
    "skywork-coarse-to-fine": (
        "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n" + coarse_to_fine + "<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{input}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n",
        "{output}",
        "\n\n",
    ),
    "skywork-step-by-step-hard": (
        "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n" + step_by_step_hard + "<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{input}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n",
        "{output}",
        "\n\n",
    ),
    # ************** DeepSeek **************
    "deepseek-step-by-step": (
        "<｜begin▁of▁sentence｜>" + step_by_step + "\n\n"
        "User: {input}\n\n"
        "Assistant:",
        "{output}",
        "\n\n",
    ),
    "deepseek-coarse-to-fine": (
        "<｜begin▁of▁sentence｜>" + coarse_to_fine + "\n\n"
        "User: {input}\n\n"
        "Assistant:",
        "{output}",
        "\n\n",
    ),
    "deepseek-step-by-step-hard": (
        "<｜begin▁of▁sentence｜>" + step_by_step_hard + "\n\n"
        "User: {input}\n\n"
        "Assistant:",
        "{output}",
        "\n\n",
    ),
    # ************** PowerInfer **************
    "smallthinker-step-by-step": (
        "<|im_start|>system\n" + step_by_step + "<|im_end|>\n"
        "<|im_start|>user\n{input}<|im_end|>\n"
        "<|im_start|>assistant\n",
        "{output}",
        "\n\n",
    ),
    "smallthinker-coarse-to-fine": (
        "<|im_start|>system\n" + coarse_to_fine + "<|im_end|>\n"
        "<|im_start|>user\n{input}<|im_end|>\n"
        "<|im_start|>assistant\n",
        "{output}",
        "\n\n",
    ),
    "smallthinker-step-by-step-hard": (
        "<|im_start|>system\n" + step_by_step_hard + "<|im_end|>\n"
        "<|im_start|>user\n{input}<|im_end|>\n"
        "<|im_start|>assistant\n",
        "{output}",
        "\n\n",
    ),
    # ************** InternLM **************
    "internlm-math-fs": ("Question:{input}\nAnswer:", "{output}", "\n"),
    "internlm-math-chat": (
        "<|im_start|>user\n{input}<|im_end|>\n" "<|im_start|>assistant\n",
        "{output}",
        "\n\n",
    ),
    "mistral": (
        "[INST] {input}[/INST]",
        "{output}",
        "\n\n",
    ),
    "numina": ("### Problem: {input}\n### Solution:", " {output}", "\n\n"),
}
