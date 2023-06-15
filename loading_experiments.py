# %%
import os
import sys
tl_path = f"{os.getcwd()}/TransformerLens"
sys.path.insert(0, tl_path)

import torch
from jaxtyping import Float
from tqdm import tqdm
from IPython.display import display
from transformer_lens.loading_from_pretrained import get_pretrained_model_config
from transformer_lens.hook_points import HookPoint
from transformer_lens import HookedTransformer
from transformer_lens import utils, HookedTransformer, HookedTransformerConfig, ActivationCache

from transformers import LlamaForCausalLM, LlamaTokenizer
from accelerate import init_empty_weights, load_checkpoint_and_dispatch
from huggingface_hub import snapshot_download

device = torch.device('cuda')
# still have to do tokenizer separately
model = LlamaForCausalLM.from_pretrained("decapoda-research/llama-7b-hf")
model = HookedTransformer.from_pretrained("llama-7b-hf", hf_model=model, device='cpu')
model.to(torch.float16)
model.to(device)