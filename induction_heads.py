# %%
import os
import sys
tl_path = f"{os.getcwd()}/TransformerLens"
sys.path.insert(0, tl_path)

os.environ["ACCELERATE_DISABLE_RICH"] = "1"
import sys
import torch as t
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import einops
from jaxtyping import Int, Float
from typing import List, Optional, Tuple
import functools
from tqdm import tqdm
from IPython.display import display
import webbrowser
from transformer_lens.hook_points import HookPoint
from transformer_lens import utils, HookedTransformer, HookedTransformerConfig, FactoredMatrix, ActivationCache
from transformers import LlamaForCausalLM, LlamaTokenizer, AutoModelForCausalLM
import circuitsvis as cv

from plotly_utils import imshow, hist, plot_comp_scores, plot_logit_attribution, plot_loss_difference


# Saves computation time, since we don't need it for the contents of this notebook
t.set_grad_enabled(False)

device = t.device("cuda")

def load_model(float16=False, use_TL=True):
    tokenizer = LlamaTokenizer.from_pretrained("decapoda-research/llama-7b-hf") #float32 requires 30GB of VRAM
    model = LlamaForCausalLM.from_pretrained(f"{os.getcwd()}/vicuna-7b-hf") #using vicuna
    if use_TL:
        model = HookedTransformer.from_pretrained("llama-7b-hf", hf_model=model, device='cpu')
        #model = HookedTransformer.from_pretrained("gpt2-xl", device='cpu')
    if float16:
        model.to(t.float16)

    model.to(device) #device as a global var
    model.tokenizer = tokenizer
    model.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    return model


def generate_repeated_tokens(
    model: HookedTransformer, seq_len: int, batch: int = 1
) -> Int[Tensor, "batch full_seq_len"]:
    '''
    Generates a sequence of repeated random tokens

    Outputs are:
        rep_tokens: [batch, 1+2*seq_len]
    '''
    prefix = (t.ones(batch, 1) * model.tokenizer.bos_token_id).long()
    rep_tokens_half = t.randint(0, model.cfg.d_vocab, (batch, seq_len), dtype=t.int64)
    rep_tokens = t.cat([prefix, rep_tokens_half, rep_tokens_half], dim=-1).to(device)
    return rep_tokens

def generate_repeated_tokens_backward(
        model: HookedTransformer, seq_len: int, batch: int=1
) -> Int[Tensor, "batch full_seq_len"]:
    '''
    T1 T2 T3 T4 T5 T5 T4 T3 T2 T1
    '''
    prefix = (t.ones(batch, 1) * model.tokenizer.bos_token_id).long()
    rep_tokens_half = t.randint(0, model.cfg.d_vocab, (batch, seq_len), dtype=t.int64)
    rep_tokens = t.cat([prefix, rep_tokens_half, rep_tokens_half.flip([-1])], dim=-1).to(device)
    return rep_tokens


def random_token_tensor(model: HookedTransformer, seq_len: int, batch: int=1):
    good_set = [model.tokenizer("love")['input_ids'][1:],
                   model.tokenizer("happy")['input_ids'][1:],
                   model.tokenizer("good")['input_ids'][1:]
                   ]

    good_set = t.tensor(good_set)
    indices = t.randint(low=0, high=len(good_set), size=(batch, seq_len))
    good = good_set[indices].squeeze(dim=-1)

    bad_set = [model.tokenizer("hate")['input_ids'][1:],
                model.tokenizer("evil")['input_ids'][1:],
                model.tokenizer("bad")['input_ids'][1:]
                ]

    bad_set = t.tensor(bad_set)
    indices = t.randint(low=0, high=len(bad_set), size=(batch, seq_len))
    bad = bad_set[indices].squeeze(dim=-1)
    return good, bad



def generate_random_dialogue(
        model: HookedTransformer, seq_len: int, dial_len: int=5, batch: int=1
):
    prefix = (t.ones(batch, 1) * model.tokenizer.bos_token_id).long()
    john = t.tensor(model.tokenizer("John:")['input_ids'][1:]).unsqueeze(dim=0) #should be two tokens for BPE [7554, 25]
    mary = t.tensor(model.tokenizer("Mary:")['input_ids'][1:]).unsqueeze(dim=0)
    newline = t.tensor([13]).unsqueeze(dim=0) #sentencepiece
    rep_tokens = prefix
    for ply in range(dial_len):
        #john_speaks = t.randint(15, 25, (batch, seq_len)) #john only speaks in digits
        #mary_speaks = t.randint(30040, 30043, (batch, seq_len)) #mary speak
        #john_speaks = t.randint(15,20, (batch, seq_len)) #john speaks 0-5
        #mary_speaks = t.randint(20,25, (batch, seq_len)) #mary speaks 5-9
        john_speaks, mary_speaks = random_token_tensor(model, seq_len, batch)
        
        rep_tokens = t.cat([rep_tokens, newline, john, john_speaks, newline, mary, mary_speaks], dim=-1)
    rep_tokens = t.cat([rep_tokens, newline, john], dim=-1)
    rep_tokens = rep_tokens.to(device)
    return rep_tokens


def run_and_cache_model_repeated_tokens(model: HookedTransformer, seq_len: int, batch: int = 1) -> Tuple[t.Tensor, t.Tensor, ActivationCache]:
    '''
    Generates a sequence of repeated random tokens, and runs the model on it, returning logits, tokens and cache

    Should use the `generate_repeated_tokens` function above

    Outputs are:
        rep_tokens: [batch, 1+2*seq_len]
        rep_logits: [batch, 1+2*seq_len, d_vocab]
        rep_cache: The cache of the model run on rep_tokens
    '''
    rep_tokens = generate_random_dialogue(model, seq_len, dial_len=3, batch=batch)
    rep_logits, rep_cache = model.run_with_cache(rep_tokens)
    return rep_tokens, rep_logits, rep_cache



def avg_logits(model: HookedTransformer, rep_logits: Float[Tensor, "batch seq_len d_vocab"]):
    good_tokens = [model.tokenizer("love")['input_ids'][1:][0],
            model.tokenizer("happy")['input_ids'][1:][0],
            model.tokenizer("good")['input_ids'][1:][0]
            ]

    #good_tokens = t.tensor(good_tokens).to(device)

    bad_tokens = [model.tokenizer("hate")['input_ids'][1:][0],
                model.tokenizer("evil")['input_ids'][1:][0],
                model.tokenizer("bad")['input_ids'][1:][0]
                ]

    #bad_tokens = t.tensor(bad_tokens).to(device) 

    good_avg = t.zeros(rep_logits.shape[:-1]).to(device)
    for good_token in good_tokens:
        good_avg = good_avg + rep_logits[..., good_token]
    good_avg = good_avg/len(good_tokens) #could be numerically unstable

    bad_avg = t.zeros(rep_logits.shape[:-1]).to(device)
    for bad_token in bad_tokens:
        bad_avg = bad_avg + rep_logits[..., bad_token]
    bad_avg = bad_avg/len(bad_tokens)

    return good_avg, bad_avg


token_directory = {
    "John" : 2259,
    "Mary" : 6182,
    "love" : 5360,
    "happy" : 9796,
    "good" : 1781,
    "hate" : 26277,
    "evil" : 16126,
    "bad" : 4319,
    "\n" : 13,
    ":" : 29901, #only when following a word?
}

inverse_token_directory = {v: k for k, v in token_directory.items()}

def visualize_logit_difference(good_avg: Float[Tensor, "batch seq_len"], bad_avg: Float[Tensor, "batch seq_len"], rep_tokens: Float[Tensor, "batch seq_len"]):
    avg_logit_difference = (good_avg - bad_avg).squeeze()[1:]

    words = [inverse_token_directory[tok] for tok in rep_tokens.tolist()[0][1:]]
    assert len(words) == avg_logit_difference.shape[0]
    cv.tokens.colored_tokens(words, avg_logit_difference)




if __name__ == "__main__":
    model = load_model()
    rep_tokens = generate_random_dialogue(model, seq_len=10, dial_len=5)
    rep_logits = model(rep_tokens)
    good_avg, bad_avg = avg_logits(model, rep_logits)
    visualize_logit_difference(good_avg, bad_avg, rep_tokens)
    # rep_tokens, rep_logits, rep_cache = run_and_cache_model_repeated_tokens(model, seq_len=6)
    #for layer in range(model.cfg.n_layers):
    #    attention_pattern = rep_cache["pattern", layer]
    #    print(layer)
    #    display(cv.attention.attention_patterns(tokens=model.to_str_tokens(rep_tokens), attention=attention_pattern[0]))
    #probs = t.nn.functional.softmax(rep_logits, dim=-1)
    #cred = []
    #for idx, tok in enumerate(list(rep_tokens[0,:])):
    #    cred.append(probs[0,idx, tok])
    #print(cred)

# prompt = model.tokenizer.batch_decode(rep_tokens)[0]
# print(model.generate(prompt, max_new_tokens=50))