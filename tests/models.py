from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

checkpoint = "mistralai/Mathstral-7B-v0.1"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForCausalLM.from_pretrained(checkpoint, device_map="auto", torch_dtype=torch.bfloat16)

prompt = [{"role": "user", "content": "A robe takes 2 bolts of blue fiber and half that much white fiber.  How many bolts in total does it take? \n Please think step by step."}]
tokenized_prompt = tokenizer.apply_chat_template(prompt, add_generation_prompt=True, return_dict=True, return_tensors="pt").to(model.device)

out = model.generate(**tokenized_prompt, max_new_tokens=512)
tokenizer.decode(out[0])
print(tokenizer.decode(out[0]))