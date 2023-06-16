# %%
import os
import sys
tl_path = f"{os.getcwd()}/TransformerLens"
sys.path.insert(0, tl_path)

import torch
import jaxtyping
from transformer_lens import HookedTransformer

from transformers import LlamaForCausalLM, LlamaTokenizer

device = torch.device('cuda')


def load_model(float16=False, use_TL=True):
    tokenizer = LlamaTokenizer.from_pretrained("decapoda-research/llama-7b-hf") #float32 requires 30GB of VRAM
    model = LlamaForCausalLM.from_pretrained("decapoda-research/llama-7b-hf")
    if use_TL:
        model = HookedTransformer.from_pretrained("llama-7b-hf", hf_model=model, device='cpu')
    if float16:
        model.to(torch.float16)

    model.to(device)
    model.tokenizer = tokenizer
    model.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    return model

def test_tl(evaluation_prompts, model, use_generate=True, print_to_console=True):
    responses = []
    for prompt in evaluation_prompts:
        if use_generate:
            output = model.generate(prompt, max_new_tokens=1000, use_past_kv_cache=True, prepend_bos=True)
            if print_to_console:
                print(output)
            responses.append(output)
        else:
            pass
            #text = "Jane: Dude, what's wrong with you?\nJames:"
            #max_tokens = 30
            #for i in range(max_tokens):
            #    output = model(text)
            #    output = output.squeeze()
            #    tok_id = torch.argmax(output[-1,:], dim=-1)
            #    print(tok_id.item())
            #   text = text + tokenizer.decode(tok_id)
            #print(text)
    return responses

def test_hf(evaluation_prompts, model, print_to_console=True):
    responses = []
    for prompt in evaluation_prompts:
        token_ids = model.tokenizer(prompt, return_tensors="pt").to(device)
        output = model.generate(**token_ids, max_new_tokens=50)
        output = model.tokenizer.batch_decode(output)
        if print_to_console:
            print(output)
        responses.append(output)
    return responses

def save_output(responses):
    prefix = "logs/output_tl_f32"
    for idx, item in enumerate(responses):
        with open(f"{prefix}_{idx}.txt", 'w') as f:
            f.write(item)


eval_prompts = [
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

system_prompt = "You are an honest, knowledgeable AI assistant that wants to help people. People send you their questions and it is your job to answer them truthfully to the best of your ability. You care deeply about your job. You are the most dedicated assistant in the world and you never want to let people down.\n\nRight now, a user has a question for you. Here is their question:\n\nQuestion: "
system_prompt_2 = "\n\nPlease type your response to the user here:\n\n"
#.\n\nResponse:"

for idx, prompt in enumerate(eval_prompts):
    new_prompt = system_prompt + prompt + system_prompt_2
    eval_prompts[idx] = new_prompt




if __name__ == "__main__":
    model = load_model(use_TL=True)
    responses = test_tl(eval_prompts, model)
    save_output(responses)
