import torch
from torch import nn
from transformers.models.bert.modeling_bert import BertConfig, BertEmbeddings
from transformers import BertTokenizer
from functools import partial
from timm.models.layers import DropPath, to_2tuple

import logging
import transformers, re, sys
from beartype import beartype
from typing import List, Union
from helper import *
from torchvision import transforms
from PIL import Image

transformers.logging.set_verbosity_error()
class PatchEmbed(nn.Module):
    """ Image to Patch Embedding """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        B, C, H, W = x.shape
        # FIXME look at relaxing size constraints
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x

class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, mask=None):

        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)

        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)
        attn = (q @ k.transpose(-2, -1)) * self.scale

        if mask is not None:
            mask = mask.bool()
            attn = attn.masked_fill(~mask[:, None, None, :], float("-inf"))

        # For rare cases, the attention weights are inf due to the mix-precision training.
        # We clamp the tensor to the max values of the current data type
        # This is different from MAE training as we don't observe such cases on image-only MAE.
        if torch.isinf(attn).any():
            clamp_value = torch.finfo(attn.dtype).max-1000
            attn = torch.clamp(attn, min=-clamp_value, max=clamp_value)

        attn = attn.softmax(dim=-1)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, mask=None):
        x = x + self.drop_path(self.attn(self.norm1(x), mask=mask))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class MaskedAutoencoderViT(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3,
                 embed_dim=1024, depth=24, num_heads=16,
                 decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
                 mlp_ratio=4., norm_layer=nn.LayerNorm, norm_pix_loss=False):
        super().__init__()
        # --------------------------------------------------------------------------
        # MAE encoder specifics
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim), requires_grad=False)  # fixed sin-cos embedding

        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, qk_scale=None, norm_layer=norm_layer)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)
        # --------------------------------------------------------------------------

        # --------------------------------------------------------------------------
        # MAE decoder specifics
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))

        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, decoder_embed_dim), requires_grad=False)  # fixed sin-cos embedding

        self.decoder_blocks = nn.ModuleList([
            Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, qk_scale=None, norm_layer=norm_layer)
            for i in range(decoder_depth)])

        self.decoder_norm = norm_layer(decoder_embed_dim)
        self.decoder_pred = nn.Linear(decoder_embed_dim, patch_size**2 * in_chans, bias=True) # encoder to decoder
        # --------------------------------------------------------------------------

    def visual_embed(self, x):
        x = self.patch_embed(x)
        x = x + self.pos_embed[:, 1:, :]
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        return x


# set recommended archs
def mae_vit_base_patch16_dec512d8b(**kwargs):
    model = MaskedAutoencoderViT(
        patch_size=16, embed_dim=768, depth=12, num_heads=12,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def mae_vit_large_patch16_dec512d8b(**kwargs):
    model = MaskedAutoencoderViT(
        patch_size=16, embed_dim=1024, depth=24, num_heads=16,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

mae_vit_base_patch16 = mae_vit_base_patch16_dec512d8b  # decoder: 512 dim, 8 blocks
mae_vit_large_patch16 = mae_vit_large_patch16_dec512d8b  # decoder: 512 dim, 8 blocks


class Pooler(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output
    
class VLCTransformer(nn.Module):
    def __init__(self, 
                 hidden_size: int,
                 vlc_version: str,
                 vocab_size: int=30522,
                 num_layers: int=12,
                 num_heads: int=12,
                 mlp_ratio: int=4,
                 max_text_len: int=40,
                 drop_rate: float=0.1,                 
        ):
        super().__init__()

        bert_config = BertConfig(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            num_hidden_layers=num_layers,
            num_attention_heads=num_heads,
            intermediate_size=hidden_size * mlp_ratio,
            max_position_embeddings=max_text_len,
            hidden_dropout_prob=drop_rate,
            attention_probs_dropout_prob=drop_rate,
        )

        self.text_embeddings = BertEmbeddings(bert_config)

        self.token_type_embeddings = nn.Embedding(2, hidden_size)

        if vlc_version == "base":
            self.transformer = mae_vit_base_patch16(img_size=384)
        elif vlc_version == "large":
            self.transformer = mae_vit_large_patch16(img_size=384)

        self.pooler = Pooler(hidden_size)


    def forward_language_encoder(self, input_ids, attention_masks):

        text_embeds = self.text_embeddings(input_ids)\
            + self.token_type_embeddings(torch.zeros_like(attention_masks))

        x = text_embeds
        for i, blk in enumerate(self.transformer.blocks):
            x = blk(x, mask=None)
        encoded_text = self.transformer.norm(x)
        
        return encoded_text


    def forward_vl_encoder(self, input_ids, attention_masks, img):
        text_embeds = self.text_embeddings(input_ids) +\
            self.token_type_embeddings(torch.zeros_like(attention_masks))
        
        image_embeds = self.transformer.visual_embed(img)
        image_embeds = image_embeds +\
            self.token_type_embeddings(torch.ones((image_embeds.shape[0], image_embeds.shape[1]), dtype=torch.long, device=attention_masks.device))
        
        
        x = torch.cat([text_embeds, image_embeds], dim=1)

        for i, blk in enumerate(self.transformer.blocks):
            x = blk(x, mask=None)
        x = self.transformer.norm(x)
        encoded_text, encoded_image = x[:, :text_embeds.shape[1]], x[:, text_embeds.shape[1]:]
        return encoded_text, encoded_image

trs = transforms.Compose([
        transforms.Resize((384, 384), interpolation=Image.BICUBIC), 
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

def preproc_image(path):
    image = Image.open(path).convert("RGB")
    return trs(image).unsqueeze(0)

VLC_CONFIGS ={}

def get_tokenizer(**kwargs):
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", do_lower_case=True)
    tokenizer.pad_token_id = 102
    return tokenizer

def get_model(**kwargs):
    model = VLCTransformer(
        vlc_version = "large",
        hidden_size = 1024
    )
    #path = "<largefiles_dir>/skewed_relations_T2I/scripts/probing/vlc_L16.ckpt" # [Important]: update this to the correct path
    path = "/data/yingshac/clevr_control/scripts/probing/vlc_L16.ckpt"
    state_dict = torch.load(path, map_location="cpu")["state_dict"]
    model.load_state_dict(state_dict, strict=False)
    return model

def get_model_and_tokenizer(**kwargs):

    global VLC_CONFIGS

    if "model" not in VLC_CONFIGS:
        VLC_CONFIGS["model"] = get_model()
    if "tokenizer" not in VLC_CONFIGS:
        VLC_CONFIGS["tokenizer"] = get_tokenizer()

    return VLC_CONFIGS['model'], VLC_CONFIGS['tokenizer']


def get_encoded_dim(**kwargs):
    return 1024 #VLC_CONFIGS['hidden_size']

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
        padding=True,
        truncation=True,
        return_tensors="pt",
        return_special_tokens_mask=True,
    )

    input_ids = encoded.input_ids.to(device)
    attn_mask = encoded.attention_mask.to(device)

    model.eval()

    with torch.no_grad():
        encoded_text = model.forward_language_encoder(
            input_ids,
            attn_mask,
        ).detach()

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
        padding=True,
        truncation=True,
        return_tensors="pt",
        return_special_tokens_mask=True,
    )

    input_ids = encoded.input_ids
    input_ids_list = input_ids.tolist()
    input_ids = input_ids.to(device)
    attn_mask = encoded.attention_mask.to(device)

    model.eval()

    with torch.no_grad():
        encoded_text = model.forward_language_encoder(
            input_ids,
            attn_mask,
        ).detach()

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
    #for subj_pos, obj_pos, t, token_ids in zip(
    #        subj_tok_positions, obj_tok_positions, texts, input_ids.tolist()
    #    ):
    #    print(t,
    #          tokenizer.convert_ids_to_tokens([token_ids[subj_pos]]),
    #          tokenizer.convert_ids_to_tokens([token_ids[obj_pos]])
    #    )
    #exit()
    #print(f"subj_encodings.size() = {obj_encodings.size()}")
    #print(f"obj_encodings.size() = {obj_encodings.size()}")
    
    return torch.stack([subj_encodings, obj_encodings], dim=1)

@beartype
def encode_subj_obj_with_vl_input(
    img_paths: List[str], 
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
        padding=True,
        truncation=True,
        return_tensors="pt",
        return_special_tokens_mask=True,
    )

    input_ids = encoded.input_ids
    input_ids_list = input_ids.tolist()
    input_ids = input_ids.to(device)
    attn_mask = encoded.attention_mask.to(device)

    imgs = torch.cat([preproc_image(path) for path in img_paths], dim=0).to(device)


    model.eval()

    with torch.no_grad():
        encoded_text = model.forward_vl_encoder(
            input_ids,
            attn_mask,
            imgs,
        )[0].detach()

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
    #for subj_pos, obj_pos, t, token_ids in zip(
    #        subj_tok_positions, obj_tok_positions, texts, input_ids.tolist()
    #    ):
    #    print(t,
    #          tokenizer.convert_ids_to_tokens([token_ids[subj_pos]]),
    #          tokenizer.convert_ids_to_tokens([token_ids[obj_pos]])
    #    )
    #exit()
    #print(f"subj_encodings.size() = {obj_encodings.size()}")
    #print(f"obj_encodings.size() = {obj_encodings.size()}")
    
    return torch.stack([subj_encodings, obj_encodings], dim=1)

def get_subj_obj_tok_positions(input_ids, tokenized_articles, tokenized_relations):
    #print(tokenized_articles)
    #print(tokenized_relations)s
    subj_first_tok_positions, obj_first_tok_positions, subj_last_tok_positions, obj_last_tok_positions = [], [], [], []
    for input_id in input_ids:

        x = "$" + " ".join([str(i) for i in input_id if i!=102])
        #print(x)
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
        padding=True,
        truncation=True,
        return_tensors="pt",
        return_special_tokens_mask=True,
    ).input_ids.tolist()
    R = []
    ARTICALS = []
    for y in input_ids[-2:]:
        Y = [str(i) for i in y if i not in [101, 102]]
        ARTICALS.append("$101 {} ".format(" ".join(Y)))
        for x in input_ids[:-2]:
            X = [str(i) for i in x if i not in [101, 102]]
            R.append(" {} ".format(" ".join(X)) + "{} ".format(" ".join(Y)))
    return ARTICALS, R
    
