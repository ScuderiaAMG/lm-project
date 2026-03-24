#!/usr/bin/env python3
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# 加载模型
model_dir = "./models/final_model_20260115_044955"
tokenizer = GPT2Tokenizer.from_pretrained(model_dir)
model = GPT2LMHeadModel.from_pretrained(model_dir).cuda()

print("=== GPT-2 Text Generator ===")
print("Enter a prompt (or 'exit' to quit):")

while True:
    prompt = input("\nPrompt: ")
    if prompt.lower() == 'exit':
        break
    
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=100,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id
        )
    
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"\nGenerated text:\n{generated_text}")
