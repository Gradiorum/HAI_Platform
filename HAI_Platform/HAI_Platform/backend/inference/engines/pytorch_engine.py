import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

class PyTorchEngine:
    def __init__(self, model_name, max_tokens=512, temperature=0.7, top_k=50, top_p=0.9):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p

    def generate(self, prompt):
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt")
        outputs = self.model.generate(
            input_ids,
            max_new_tokens=self.max_tokens,
            do_sample=True,
            temperature=self.temperature,
            top_k=self.top_k,
            top_p=self.top_p
        )
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
