__MODEL_LIST__ = {
    # Qwen Models
    "qwen2.5-32b": "Qwen/Qwen2.5-32B-Instruct",
    "qwen2.5-14b": "Qwen/Qwen2.5-14B-Instruct",
    "qwen2.5-7b": "Qwen/Qwen2.5-7B-Instruct",
    "qwen2.5-3b": "Qwen/Qwen2.5-3B-Instruct",
    "qwen2.5-1.5b": "Qwen/Qwen2.5-1.5B-Instruct",
    "qwq": "Qwen/QwQ-32B-Preview",
    
    # Qwen Math Models
    "qwen2.5-math-7b": "Qwen/Qwen2.5-Math-7B-Instruct",
    "qwen2.5-math-1.5b": "Qwen/Qwen2.5-Math-1.5B-Instruct",
    
    # # InternLM Models
    # "internlm2.5-20b": "internlm/internlm2_5-20b-chat",
    # "internlm2.5-7b": "internlm/internlm2_5-7b-chat",
    # "internlm2.5-1.8b": "internlm/internlm2_5-1_8b-chat",
    
    # # InternLM Math Models
    # "internlm2-math-20b": "internlm/internlm2-math-plus-20b",
    # "internlm2-math-7b": "internlm/internlm2-math-plus-7b",
    # "internlm2-math-1.8b": "internlm/internlm2-math-plus-1_8b",
    
    # Mistral Models
    "mistral-small": "mistralai/Mistral-Small-Instruct-2409",
    "ministral-8b": "mistralai/Ministral-8B-Instruct-2410",
    "mistral-nemo": "mistralai/Mistral-Nemo-Instruct-2407",
    
    # Mistral Math Models
    "mathstral-7b": "mistralai/Mathstral-7B-v0.1",
    
    # Meta Llama Models
    "llama3.1-8b": "meta-llama/Llama-3.1-8B-Instruct",
    "llama3.2-3b": "meta-llama/Llama-3.2-3B-Instruct",
    "llama3.2-1b": "meta-llama/Llama-3.2-1B-Instruct",
    
    # phi Models
    "phi-4": "microsoft/phi-4",
    "phi-3-medium": "microsoft/Phi-3-medium-128k-instruct",
    "phi-3-mini": "microsoft/Phi-3-mini-128k-instruct",
    "phi-3-small": "microsoft/Phi-3-small-128k-instruct",
    "phi-3-5-mini": "microsoft/Phi-3.5-mini-instruct",

    # o1-like Models
    "skywork-8b": "Skywork/Skywork-o1-Open-Llama-3.1-8B", # o1 need special setup
    "smallthinker-3b": "PowerInfer/SmallThinker-3B-Preview",
    "sky-t1-32b": "NovaSky-AI/Sky-T1-32B-Preview",
    "drd-qwen-1.5b": "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
    "drd-qwen-7b": "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
    "drd-qwen-14b": "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
    "drd-qwen-32b": "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
    "drd-llama-8b": "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
    
    # Gemma Models don't support system role
    "gemma-27b": "google/gemma-2-27b-it",
    "gemma-9b": "google/gemma-2-9b-it",
    "gemma-2b": "google/gemma-2-2b-it",
    
    # # DeepSeek Models
    # "deepseek-lite": "deepseek-ai/DeepSeek-V2-Lite-Chat",
    # "deepseek-math-7b": "deepseek-ai/deepseek-math-7b-instruct",
    # "deepseek-math-7b-rl": "deepseek-ai/deepseek-math-7b-rl",
}

o1_like_models = {
    "Qwen/QwQ-32B-Preview": "QwQ",
    # "Skywork/Skywork-o1-Open-Llama-3.1-8B": "Skywork-o1-Llama", 
    # "PowerInfer/SmallThinker-3B-Preview": "SmallThinker",
    "NovaSky-AI/Sky-T1-32B-Preview": "Sky-T1",
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B": "DRD-Qwen-1.5B",
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B": "DRD-Qwen-7B",
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B": "DRD-Qwen-14B",
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B": "DRD-Qwen-32B",
    "deepseek-ai/DeepSeek-R1-Distill-Llama-8B": "DRD-Llama-8B",
}

math_models = {
    "Qwen/Qwen2.5-Math-1.5B-Instruct": "Qwen2.5-Math-1.5B",
    "Qwen/Qwen2.5-Math-7B-Instruct": "Qwen2.5-Math-7B",
    "mistralai/Mathstral-7B-v0.1": "Mathstral-7B",
    
}

instruction_models = {
    "Qwen/Qwen2.5-32B-Instruct": "Qwen2.5-32B",
    "Qwen/Qwen2.5-14B-Instruct": "Qwen2.5-14B",
    "Qwen/Qwen2.5-7B-Instruct": "Qwen2.5-7B",
    "Qwen/Qwen2.5-3B-Instruct": "Qwen2.5-3B",
    "Qwen/Qwen2.5-1.5B-Instruct": "Qwen2.5-1.5B",
    "mistralai/Mistral-Small-Instruct-2409": "Mistral-Small",
    "mistralai/Mistral-Nemo-Instruct-2407": "Mistral-Nemo",
    "mistralai/Ministral-8B-Instruct-2410": "Ministral-8B",
    "google/gemma-2-27b-it": "Gemma-27B",
    "google/gemma-2-9b-it": "Gemma-9B",
    "google/gemma-2-2b-it": "Gemma-2B",
    "microsoft/Phi-3-medium-128k-instruct": "Phi-3-Medium",
    "microsoft/Phi-3-small-128k-instruct": "Phi-3-Small",
    "microsoft/Phi-3.5-mini-instruct": "Phi-3.5-Mini",
    "microsoft/Phi-3-mini-128k-instruct": "Phi-3-Mini",
    "microsoft/phi-4": "Phi-4",
    "meta-llama/Llama-3.1-8B-Instruct": "Llama-3.1-8B",
    "meta-llama/Llama-3.2-3B-Instruct": "Llama-3.2-3B",
    "meta-llama/Llama-3.2-1B-Instruct": "Llama-3.2-1B",
}
