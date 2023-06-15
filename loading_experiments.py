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


model = HookedTransformer.from_pretrained("") #device='cpu', move_state_dict_to_device