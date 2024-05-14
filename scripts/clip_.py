import logging
import torch
import transformers, re
from transformers import AutoTokenizer, CLIPTokenizer, CLIPTokenizerFast 
from transformers import CLIPTextModel, CLIPTextConfig
from beartype import beartype
from typing import List, Union
from helper import *

transformers.logging.set_verbosity_error()

clip = "openai/clip-vit-base-patch32" #"openai/clip-vit-large-patch14" #
CLIP_CONFIGS = CLIPTextConfig().to_dict()

def get_tokenizer(**kwargs):
    global clip
    tokenizer = AutoTokenizer.from_pretrained(clip)
    return tokenizer

def get_model(**kwargs):
    global clip
    model = CLIPTextModel.from_pretrained(clip)
    return model

def get_model_and_tokenizer(**kwargs):

    global CLIP_CONFIGS

    if "model" not in CLIP_CONFIGS:
        CLIP_CONFIGS["model"] = get_model()
    if "tokenizer" not in CLIP_CONFIGS:
        CLIP_CONFIGS["tokenizer"] = get_tokenizer()

    return CLIP_CONFIGS['model'], CLIP_CONFIGS['tokenizer']

def get_encoded_dim(**kwargs):
    global clip
    res = 512 if "base" in clip else 768
    #print(f"get {clip} dim = {res}")
    return res # hardcoded #CLIP_CONFIGS['hidden_size']


# encoding text

@beartype
def encode_text(
    texts: Union[str, List[str]],
    output_device = None,
    **kwargs
):
    if isinstance(texts, str):
        texts = [texts]

    model, tokenizer = get_model_and_tokenizer()

    if torch.cuda.is_available():
        model = model.cuda()

    device = next(model.parameters()).device

    encoded = tokenizer.batch_encode_plus(
        texts,
        return_tensors = "pt",
        padding = True,
        truncation = True
    )

    input_ids = encoded.input_ids.to(device)
    attn_mask = encoded.attention_mask.to(device)

    model.eval()

    with torch.no_grad():
        output = model(input_ids = input_ids, attention_mask = attn_mask)
        encoded_text = output.last_hidden_state.detach()

    attn_mask = attn_mask.bool()
    encoded_text = encoded_text.masked_fill(~attn_mask[..., None], 0.)

    if not exists(output_device):
        return encoded_text

    encoded_text.to(output_device)
    return encoded_text

@beartype
def encode_subj_obj(
    texts: Union[str, List[str]],
    tokenized_articles: List[str], 
    tokenized_relations: List[str],
    token_pos: str,
    **kwargs
):  
    model, tokenizer = get_model_and_tokenizer()

    if torch.cuda.is_available():
        model = model.cuda()

    device = next(model.parameters()).device

    encoded = tokenizer.batch_encode_plus(
        texts,
        return_tensors = "pt",
        padding = True,
        truncation = True
    )

    input_ids = encoded.input_ids
    input_ids_list = input_ids.tolist()
    input_ids = input_ids.to(device)
    attn_mask = encoded.attention_mask.to(device)

    model.eval()

    with torch.no_grad():
        output = model(input_ids = input_ids, attention_mask = attn_mask)
        encoded_text = output.last_hidden_state.detach()

    attn_mask = attn_mask.bool()
    encoded_text = encoded_text.masked_fill(~attn_mask[..., None], 0.) # batch_size, max_len, dim

    if token_pos == 'first':
        subj_tok_positions, obj_tok_positions, _, __ = get_subj_obj_tok_positions(input_ids_list, tokenized_articles, tokenized_relations)
    elif token_pos == 'last':
        _, __, subj_tok_positions, obj_tok_positions = get_subj_obj_tok_positions(input_ids_list, tokenized_articles, tokenized_relations)
    else:
        return encoded_text
    
    subj_encodings = torch.stack([e[p] for e, p in zip(encoded_text, subj_tok_positions)])
    obj_encodings = torch.stack([e[p] for e, p in zip(encoded_text, obj_tok_positions)])
    
    return torch.stack([subj_encodings, obj_encodings], dim=1)

def get_subj_obj_tok_positions(input_ids, tokenized_articles, tokenized_relations):
    subj_first_tok_positions, obj_first_tok_positions, subj_last_tok_positions, obj_last_tok_positions = [], [], [], []
    for input_id in input_ids:

        x = "$" + " ".join([str(i) for i in input_id if i<49407])
        offsets = [0, 0, 0, 0]
        for s in tokenized_articles:
            if s in x:
                x = x.replace(s, "")
                offsets[0] = len(s.split())
                break
        for s in tokenized_relations:
            if s in x:
                x = x.replace(s, ", ")
                offsets[2] = len(s.split())
                break
        offsets[1] = len(x.split(", ")[0].split())
        offsets[3] = len(x.split(", ")[1].split())
        #print(x, offsets)
        subj_first_pos, obj_first_pos, subj_last_pos, obj_last_pos = offsets[0], sum(offsets[:3]), sum(offsets[:2])-1, sum(offsets)-1
        subj_first_tok_positions.append(subj_first_pos)
        obj_first_tok_positions.append(obj_first_pos)
        subj_last_tok_positions.append(subj_last_pos)
        obj_last_tok_positions.append(obj_last_pos)
                                  
    return subj_first_tok_positions, obj_first_tok_positions, subj_last_tok_positions, obj_last_tok_positions

def tokenize_relations_and_articles(RELATION_PHRASES, **kwargs):
    tokenizer = get_tokenizer()
    texts = list(RELATION_PHRASES.values()) + ["a", "an"]
    input_ids = tokenizer.batch_encode_plus(
            texts,
            return_tensors = "pt",
            padding = True,
            truncation = True
        ).input_ids.tolist()
    R = []
    ARTICALS = []
    for y in input_ids[-2:]:
        Y = [str(i) for i in y if i<49406]
        ARTICALS.append("$49406 {} ".format(" ".join(Y)))
        for x in input_ids[:-2]:
            X = [str(i) for i in x if i<49406]
            R.append(" {} ".format(" ".join(X)) + "{} ".format(" ".join(Y)))
    return ARTICALS, R
    

@beartype
def encode_subj_obj2(
    texts: Union[str, List[str]],
    tokenized_articles: List[str], 
    tokenized_relations: List[str],
    token_pos: str,
    **kwargs,
):  
    ### Avg token encodings in the relation phrase, and add it to subj/obj encodings
     
    model, tokenizer = get_model_and_tokenizer()

    if torch.cuda.is_available():
        model = model.cuda()

    device = next(model.parameters()).device

    encoded = tokenizer.batch_encode_plus(
        texts,
        return_tensors = "pt",
        padding = True,
        truncation = True
    )

    input_ids = encoded.input_ids
    input_ids_list = input_ids.tolist()
    input_ids = input_ids.to(device)
    attn_mask = encoded.attention_mask.to(device)

    model.eval()

    with torch.no_grad():
        output = model(input_ids = input_ids, attention_mask = attn_mask)
        encoded_text = output.last_hidden_state.detach() # bs, seq_len, dim

    attn_mask = attn_mask.bool()
    encoded_text = encoded_text.masked_fill(~attn_mask[..., None], 0.) # batch_size, max_len, dim

    subj_first_positions, obj_first_positions, subj_last_positions, obj_last_positions = get_subj_obj_tok_positions(input_ids.tolist(), tokenized_articles, tokenized_relations)
    avg_relation_encodings = torch.stack([
        torch.mean(e[subj_last_pos+2:obj_first_pos-2], dim=0) 
        for e, subj_last_pos, obj_first_pos in zip(encoded_text, subj_last_positions, obj_first_positions)
    ])
    if token_pos == "first":
        subj_encodings = torch.stack([e[p] for e, p in zip(encoded_text, subj_first_positions)]) + avg_relation_encodings
        obj_encodings = torch.stack([e[p] for e, p in zip(encoded_text, obj_first_positions)]) + avg_relation_encodings
    elif token_pos == "last":
        subj_encodings = torch.stack([e[p] for e, p in zip(encoded_text, subj_last_positions)]) + avg_relation_encodings
        obj_encodings = torch.stack([e[p] for e, p in zip(encoded_text, obj_last_positions)]) + avg_relation_encodings
    else:
        return encoded_text
    
    return torch.stack([subj_encodings, obj_encodings], dim=1)