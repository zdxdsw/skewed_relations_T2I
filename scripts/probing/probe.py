from torch import nn
from typing import Any, Dict, Optional, Tuple, Union, List
from functools import partial

import sys
sys.path.append("../")
from probing_utils import *

class PROBE(nn.Module):
    def __init__(self,
                 hidden_dim_multipliers: List[float],
                 num_classes: int,
                 lm: str,
                 lm_kwargs: dict,
                 lm_howto_select_encoding_positions: str="encode_subj_obj",
    ):
        super().__init__()
        
        LM = __import__(lm)
        self.encode_subj_obj = partial(
            eval(f"LM.{lm_howto_select_encoding_positions}"), 
            **lm_kwargs
        )
        self.text_embed_dim = LM.get_encoded_dim(**lm_kwargs)
        #print(self.text_embed_dim)
        self.tokenized_articles, self.tokenized_relations = \
            LM.tokenize_relations_and_articles(
                RELATION_PHRASES, 
                **lm_kwargs
            )

        hidden_dims = [int(self.text_embed_dim * m) for m in hidden_dim_multipliers]
        
        layers = nn.ModuleList()

        layer_sizes = [self.text_embed_dim] + hidden_dims
        for dim_in, dim_out in zip(layer_sizes[:-1], layer_sizes[1:]):
            layers.append(nn.Linear(dim_in, dim_out)) # default bias=True?
            layers.append(nn.ReLU())
        
        self.layers = nn.Sequential(*layers)
        self.out = nn.Linear(layer_sizes[-1], num_classes)

    def forward(self, texts):
        x = self.encode_subj_obj(
                texts, 
                self.tokenized_articles, 
                self.tokenized_relations
            ) # batch_size, 2, dim
        #print(f"encode_subj_obj.size() = {x.size()}")

        x = x.view(-1, self.text_embed_dim) # batch_size*2, dim
        #print(f"enc output dtype = {x.dtype}")
        x = self.layers(x)
        x = self.out(x)
        return x # batch_size*2, num_classes
            

