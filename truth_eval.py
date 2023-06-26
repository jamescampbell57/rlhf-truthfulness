# %%
import os
import torch
from torch.utils.data import DataLoader
from transformers import LlamaForCausalLM, LlamaTokenizer, AutoModelForCausalLM
from datasets import load_dataset


device = torch.device('cuda')

tokenizer = LlamaTokenizer.from_pretrained("decapoda-research/llama-7b-hf") #float32 requires 30GB of VRAM
model = LlamaForCausalLM.from_pretrained(f"{os.getcwd()}/vicuna-7b-hf")
model.to(device)

dataset = load_dataset("boolq")

loader = DataLoader(dataset, batch_size=1, shuffle=False,)

buffer = {}
for idx, batch in loader:
    batch.to()
    buffer[idx] = 