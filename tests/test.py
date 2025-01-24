import os 
os.environ["TOKENIZERS_PARALLELISM"] = "false"
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

question = "How many prime numbers less than 100 have a units digit of 3?"
system_message = "please reason step by step"
    
phi3small_format = "<|user|>\n{input}<|end|>\n<|assistant|>"

tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3-small-128k-instruct", trust_remote_code=True)


model = AutoModelForCausalLM.from_pretrained(
    "microsoft/Phi-3-small-128k-instruct",
    trust_remote_code=True,
    device_map="auto",
    torch_dtype="auto"
)

phi3small_format = phi3small_format.format(input=system_message + "\n" + question)
tokenized_input = tokenizer(phi3small_format, return_tensors="pt").to(model.device)

output = model.generate(
    **tokenized_input,
    max_new_tokens=1024,
    temperature=0.0,
    do_sample=False
)
# remove the input tokens from the output
output = tokenizer.decode(output[0][len(tokenized_input.input_ids[0]):], skip_special_tokens=True)
print(output)
