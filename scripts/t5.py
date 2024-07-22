import logging
import torch
import transformers, re
from transformers import T5Tokenizer, T5EncoderModel, T5Config
from beartype import beartype
from typing import List, Union
from helper import *

transformers.logging.set_verbosity_error()


# config

MAX_LENGTH = 256
T5_CONFIGS = {}

# singleton globals

def get_tokenizer(name, dtype, **kwargs):
    if isinstance(dtype, str): dtype = TORCH_DTYPES[dtype]
    tokenizer = T5Tokenizer.from_pretrained(name, torch_dtype=dtype)
    return tokenizer

def get_model(name, dtype, **kwargs):
    if isinstance(dtype, str): dtype = TORCH_DTYPES[dtype]
    model = T5EncoderModel.from_pretrained(name, torch_dtype=dtype)
    return model

def get_model_and_tokenizer(name, dtype, **kwargs):
    global T5_CONFIGS

    if isinstance(dtype, str): dtype = TORCH_DTYPES[dtype]

    if name not in T5_CONFIGS:
        T5_CONFIGS[name] = dict()
    if "model" not in T5_CONFIGS[name]:
        T5_CONFIGS[name]["model"] = get_model(name, dtype)
    if "tokenizer" not in T5_CONFIGS[name]:
        T5_CONFIGS[name]["tokenizer"] = get_tokenizer(name, dtype)

    return T5_CONFIGS[name]['model'], T5_CONFIGS[name]['tokenizer']

def get_encoded_dim(name, **kwargs):
    if name not in T5_CONFIGS:
        # avoids loading the model if we only want to get the dim
        config = T5Config.from_pretrained(name)
        T5_CONFIGS[name] = dict(config=config)
    elif "config" in T5_CONFIGS[name]:
        config = T5_CONFIGS[name]["config"]
    elif "model" in T5_CONFIGS[name]:
        config = T5_CONFIGS[name]["model"].config
    else:
        assert False
    return config.d_model

# encoding text

@beartype
def encode_text(
    texts: Union[str, List[str]],
    name: str,
    dtype: str, 
    output_device = None,
    **kwargs
):
    if isinstance(texts, str):
        texts = [texts]
    if isinstance(dtype, str): dtype = TORCH_DTYPES[dtype]

    t5, tokenizer = get_model_and_tokenizer(name, dtype)

    if torch.cuda.is_available():
        t5 = t5.cuda()

    device = next(t5.parameters()).device
    #print(next(t5.parameters()).dtype)

    encoded = tokenizer.batch_encode_plus(
        texts,
        return_tensors = "pt",
        padding = 'longest',
        max_length = MAX_LENGTH,
        truncation = True
    )

    input_ids = encoded.input_ids.to(device)
    attn_mask = encoded.attention_mask.to(device)

    t5.eval()

    with torch.no_grad():
        output = t5(input_ids = input_ids, attention_mask = attn_mask)
        encoded_text = output.last_hidden_state.detach()

    attn_mask = attn_mask.bool()
    encoded_text = encoded_text.masked_fill(~attn_mask[..., None], 0.)
    #print(encoded_text.dtype)

    if not exists(output_device):
        return encoded_text

    encoded_text.to(output_device)
    return encoded_text

@beartype
def encode_subj_obj(
    texts: Union[str, List[str]],
    tokenized_articles: List[str], 
    tokenized_relations: List[str],
    name: str,
    dtype: str, 
    token_pos: str,
    **kwargs,
):  
    if isinstance(dtype, str): dtype = TORCH_DTYPES[dtype]
    t5, tokenizer = get_model_and_tokenizer(name, dtype)

    if torch.cuda.is_available():
        t5 = t5.cuda()

    device = next(t5.parameters()).device
    #print(next(t5.parameters()).dtype)

    encoded = tokenizer.batch_encode_plus(
        texts,
        return_tensors = "pt",
        padding = 'longest',
        max_length = MAX_LENGTH,
        truncation = True
    )

    input_ids = encoded.input_ids
    input_ids_list = input_ids.tolist()
    input_ids = input_ids.to(device)
    attn_mask = encoded.attention_mask.to(device)

    t5.eval()

    with torch.no_grad():
        output = t5(input_ids = input_ids, attention_mask = attn_mask)
        encoded_text = output.last_hidden_state.detach()

    attn_mask = attn_mask.bool()
    encoded_text = encoded_text.masked_fill(~attn_mask[..., None], 0.) # batch_size, max_len, dim

    if token_pos == 'first':
        subj_tok_positions, obj_tok_positions, _, __ = get_subj_obj_tok_positions(input_ids.tolist(), tokenized_articles, tokenized_relations)
    elif token_pos == 'last':
        _, __, subj_tok_positions, obj_tok_positions = get_subj_obj_tok_positions(input_ids.tolist(), tokenized_articles, tokenized_relations)
    else:
        return encoded_text
        #raise ValueError(f"Unrecognized token_pos ({token_pos}). Please choose from ['first', 'last'].")
    
    subj_encodings = torch.stack([e[p] for e, p in zip(encoded_text, subj_tok_positions)])
    obj_encodings = torch.stack([e[p] for e, p in zip(encoded_text, obj_tok_positions)])
    
    return torch.stack([subj_encodings, obj_encodings], dim=1)

def get_subj_obj_tok_positions(input_ids, tokenized_articles, tokenized_relations):
    subj_first_tok_positions, obj_first_tok_positions, subj_last_tok_positions, obj_last_tok_positions = [], [], [], []
    for input_id in input_ids:
        x = "$" + " ".join([str(i) for i in input_id if i>1 and i!=5])
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

def tokenize_relations_and_articles(RELATION_PHRASES, name, dtype, **kwargs):
    tokenizer = get_tokenizer(name, dtype)
    texts = list(RELATION_PHRASES.values()) + ["a", "an"]
    input_ids = tokenizer.batch_encode_plus(
            texts,
            return_tensors = "pt",
            padding = 'longest',
            max_length = MAX_LENGTH,
            truncation = True
        ).input_ids.tolist()
    R = []
    ARTICALS = []
    for y in input_ids[-2:]:
        Y = [str(i) for i in y if i>1]
        ARTICALS.append("${} ".format(" ".join(Y)))
        for x in input_ids[:-2]:
            X = [str(i) for i in x if i>1]
            R.append(" {} ".format(" ".join(X)) + "{} ".format(" ".join(Y)))
    return ARTICALS, R
    

@beartype
def encode_subj_obj2(
    texts: Union[str, List[str]],
    tokenized_articles: List[str], 
    tokenized_relations: List[str],
    name: str,
    dtype: str, 
    token_pos: str,
    **kwargs,
):  
    """ 
    Where encode_subj_obj2() differs from encode_subj_obj():
        In addition to getting subj/obj insividual token encodings,
            we avg all token encodings in the relation phrase, and add it to subj/obj encodings 
            -- forcing the subj/obj encodings to contain sentence-level information.
    """
     
    if isinstance(dtype, str): dtype = TORCH_DTYPES[dtype]
    t5, tokenizer = get_model_and_tokenizer(name, dtype)

    if torch.cuda.is_available():
        t5 = t5.cuda()

    device = next(t5.parameters()).device
    #print(next(t5.parameters()).dtype)

    encoded = tokenizer.batch_encode_plus(
        texts,
        return_tensors = "pt",
        padding = 'longest',
        max_length = MAX_LENGTH,
        truncation = True
    )

    input_ids = encoded.input_ids
    input_ids_list = input_ids.tolist()
    input_ids = input_ids.to(device)
    attn_mask = encoded.attention_mask.to(device)

    t5.eval()

    with torch.no_grad():
        output = t5(input_ids = input_ids, attention_mask = attn_mask)
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
        #raise ValueError(f"Unrecognized token_pos ({token_pos}). Please choose from ['first', 'last'].")
    
    ### Sanity check
    # for e, sf, of, sl, ol in zip(input_ids, subj_first_positions, obj_first_positions, subj_last_positions, obj_last_positions):
    #     print(e)
    #     print(tokenizer.convert_ids_to_tokens(e))
    #     print(f"subj first token: {tokenizer.convert_ids_to_tokens([e[sf]])}")
    #     print(f"subj las token: {tokenizer.convert_ids_to_tokens([e[sl]])}")
    #     print(f"relation phrase tokens: {tokenizer.convert_ids_to_tokens(e[sl+2:of-2])}")
    #     print(f"obj first token: {tokenizer.convert_ids_to_tokens([e[of]])}")
    #     print(f"obj last token: {tokenizer.convert_ids_to_tokens([e[ol]])}")
    # exit()
    
    return torch.stack([subj_encodings, obj_encodings], dim=1)