# %%
import os
import sys
tl_path = f"{os.getcwd()}/TransformerLens"
sys.path.insert(0, tl_path)

import torch
import jaxtyping
from transformer_lens.hook_points import HookPoint
from transformer_lens import HookedTransformer
from transformer_lens import utils, HookedTransformer, HookedTransformerConfig, ActivationCache

from transformers import LlamaForCausalLM, LlamaTokenizer

device = torch.device('cuda')


use_TL = True

tokenizer = LlamaTokenizer.from_pretrained("decapoda-research/llama-7b-hf")

model = LlamaForCausalLM.from_pretrained("decapoda-research/llama-7b-hf")
if use_TL:
    model = HookedTransformer.from_pretrained("llama-7b-hf", hf_model=model, device='cpu')
model.to(torch.float16)
model.to(device)

'''
evaluation_prompts = [
    "Summarize the main themes of the book 'War and Peace'.",
    "Describe the process of photosynthesis in simple terms.",
    "If all bears from Alaska are moved to Texas, what could be the potential ecological implications?",
    "A train leaves Chicago for New York at 60 mph. At the same time, a car starts driving from New York to Chicago at 40 mph. Which one arrives first?",
    "Write a short story about a unicorn who doesn't believe in humans.",
    "What are the causes and effects of climate change?",
    "You have two buckets, one holds 5 gallons, the other 3 gallons. How can you use them to get exactly 4 gallons?",
    "If no cats bark and some pets are cats, is it true that some pets do not bark?",
    "Translate the following phrase to French: 'It's raining cats and dogs'.",
    "Based on trends up to 2021, what might the future of renewable energy look like?"
]
'''

evaluation_prompts = ["What is the capital of New York?"]


responses = []


model.tokenizer = tokenizer
#model.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
# %%
if use_TL:
    for prompt in evaluation_prompts:
        #output = model.generate(prompt, max_new_tokens=30, use_past_kv_cache=False)
        #print(output)
        #responses.append(output)
        text = "Jane: Dude, what's wrong with you?\nJames:"
        max_tokens = 30
        for i in range(max_tokens):
            output = model(text)
            output = output.squeeze()
            tok_id = torch.argmax(output[-1,:], dim=-1)
            print(tok_id.item())
            text = text + tokenizer.decode(tok_id)
        print(text)
else:
    for prompt in evaluation_prompts:
        token_ids = tokenizer(prompt, return_tensors="pt").to(device)
        output = model.generate(**token_ids, max_new_tokens=30)
        output = tokenizer.batch_decode(output)
        print(output)
        responses.append(output)
# %%
with open('output_hf.txt', 'w') as f:
    for item in responses:
        f.write("%s\n" % item)