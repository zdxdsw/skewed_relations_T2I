from typing import Any, Callable, Dict, List, Optional, Union, Tuple
import numpy as np
import torch, PIL, os, sys, math
sys.path.append("../")
from functools import partial

from torch import nn
from tqdm import tqdm
from diffusers.models.unet_2d_blocks import (
    CrossAttnDownBlock2D,
    CrossAttnUpBlock2D,
    UNetMidBlock2DCrossAttn,
    get_down_block,
    get_up_block,
)
from diffusers.models import UNet2DConditionModel
from diffusers.models.transformer_2d import (
    Transformer2DModel,
    Transformer2DModelOutput,
)
from diffusers.models.embeddings import (
    PatchEmbed,
    GaussianFourierProjection,
    Timesteps,
    get_2d_sincos_pos_embed_from_grid,
)
from diffusers.utils import logging, make_image_grid

class PatchEmbed_flexible(PatchEmbed):
    def __init__(
        self,
        height=224,
        width=224,
        patch_size=16,
        in_channels=3,
        embed_dim=768,
        layer_norm=False,
        flatten=True,
        bias=True,
        interpolation_scale=1,
    ):
        super().__init__(
            height=height,
            width=width,
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=embed_dim,
            layer_norm=layer_norm,
            flatten=flatten,
            bias=bias,
            interpolation_scale=interpolation_scale
        )

        ### Overwrite pos_embed
        grid_size = int(height//patch_size), int(width//patch_size)
        grid_h = np.arange(grid_size[0], dtype=np.float32) / interpolation_scale
        grid_w = np.arange(grid_size[1], dtype=np.float32) / interpolation_scale
        grid = np.meshgrid(grid_w, grid_h)  # here w goes first
        grid = np.stack(grid, axis=0)

        grid = grid.reshape([2, 1, grid_size[1], grid_size[0]])
        pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)

        self.register_buffer("pos_embed", torch.from_numpy(pos_embed).float().unsqueeze(0), persistent=False)


    # def forward(self, latent):
    #     height, width = latent.shape[-2] // self.patch_size, latent.shape[-1] // self.patch_size

    #     latent = self.proj(latent)
    #     if self.flatten:
    #         latent = latent.flatten(2).transpose(1, 2)  # BCHW -> BNC
    #     if self.layer_norm:
    #         latent = self.norm(latent)

    #     # Interpolate positional embeddings if needed.
    #     # (For PixArt-Alpha: https://github.com/PixArt-alpha/PixArt-alpha/blob/0f55e922376d8b797edd44d25d0e7464b260dcab/diffusion/model/nets/PixArtMS.py#L162C151-L162C160)
    #     if self.height != height or self.width != width:
    #         raise ValueError(f"Size mismatch: self.height = {self.height} != height ({height}) OR self.width = {self.width} != width ({width})")
    #     else:
    #         pos_embed = self.pos_embed

    #     return (latent + pos_embed).to(latent.dtype)

class Transformer2DModel_with_nonsquare_input(Transformer2DModel):
    def __init__(
        self,
        num_attention_heads: int = 16,
        attention_head_dim: int = 88,
        in_channels: Optional[int] = None,
        out_channels: Optional[int] = None,
        num_layers: int = 1,
        dropout: float = 0.0,
        norm_num_groups: int = 32,
        cross_attention_dim: Optional[int] = None,
        attention_bias: bool = False,
        sample_size: Union[int, Tuple[int]] = None,
        num_vector_embeds: Optional[int] = None,
        patch_size: Optional[int] = None,
        activation_fn: str = "geglu",
        num_embeds_ada_norm: Optional[int] = None,
        use_linear_projection: bool = False,
        only_cross_attention: bool = False,
        double_self_attention: bool = False,
        upcast_attention: bool = False,
        norm_type: str = "layer_norm", 
        #YS: Otherwise this line https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/transformer_2d.py#L431
        #  would throw an error: 'LayerNorm' object has no attribute 'emb' 
        norm_elementwise_affine: bool = True,
        norm_eps: float = 1e-5,
        attention_type: str = "default",
        caption_channels: int = None,
    ):

        super().__init__(
            num_attention_heads=num_attention_heads,
            attention_head_dim=attention_head_dim,
            in_channels=in_channels,
            num_layers=num_layers,
            cross_attention_dim=cross_attention_dim,
            norm_num_groups=norm_num_groups,
            use_linear_projection=use_linear_projection,
            only_cross_attention=only_cross_attention,
            upcast_attention=upcast_attention,
            norm_type=norm_type,
            attention_type=attention_type,
        )

        ### Overwrite PatchEmbed
        self.is_input_continuous = (in_channels is not None) and (patch_size is None)
        self.is_input_vectorized = num_vector_embeds is not None
        self.is_input_patches = in_channels is not None and patch_size is not None
        
        if self.is_input_patches:
            assert sample_size is not None, "Transformer2DModel over patched input must provide sample_size"

            if isinstance(sample_size, int):
                self.height, self.width = sample_size, sample_size
            else:
                self.height, self.width = sample_size

            self.patch_size = patch_size
            inner_dim = num_attention_heads * attention_head_dim

            self.pos_embed = PatchEmbed_flexible(
                height=self.height,
                width=self.width,
                patch_size=self.patch_size,
                in_channels=in_channels,
                embed_dim=inner_dim,
                interpolation_scale=1,
            )

            # delete unnecessary attributes initialized by super
            delattr(self, "proj_in")
            delattr(self, "norm")

            if norm_type != "ada_norm_single":
                self.norm_out = nn.LayerNorm(inner_dim, eps=1e-6)
                self.proj_out = nn.Linear(inner_dim, patch_size * patch_size * self.out_channels)
            else:
                raise NotImplementedError("Current Transformer2DModel_with_nonsquare_input implementation doesn't support is_input_patches && norm_type == \"ada_norm_single\".")

        elif self.is_input_vectorized:
            raise NotImplementedError("Current Transformer2DModel_with_nonsquare_input implementation doesn't support is_input_vectorized.")

        

    ## YS: I have to copy everything just because 
    # I'd like to change the default value of added_cond_kwargs 
    # in the function signature
    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        timestep: Optional[torch.LongTensor] = None,
        added_cond_kwargs: Dict[str, torch.Tensor] = {
            "resolution": None, 
            "aspect_ratio": None,
        },
        class_labels: Optional[torch.LongTensor] = None,
        cross_attention_kwargs: Dict[str, Any] = None,
        attention_mask: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        return_dict: bool = True,
    ):
        
        if attention_mask is not None and attention_mask.ndim == 2:
            
            attention_mask = (1 - attention_mask.to(hidden_states.dtype)) * -10000.0
            attention_mask = attention_mask.unsqueeze(1)

        # convert encoder_attention_mask to a bias the same way we do for attention_mask
        if encoder_attention_mask is not None and encoder_attention_mask.ndim == 2:
            encoder_attention_mask = (1 - encoder_attention_mask.to(hidden_states.dtype)) * -10000.0
            encoder_attention_mask = encoder_attention_mask.unsqueeze(1)

        # Retrieve lora scale.
        lora_scale = cross_attention_kwargs.get("scale", 1.0) if cross_attention_kwargs is not None else 1.0

        # 1. Input
        if self.is_input_continuous:
            batch, _, height, width = hidden_states.shape
            residual = hidden_states

            hidden_states = self.norm(hidden_states)
            if not self.use_linear_projection:
                hidden_states = (
                    self.proj_in(hidden_states, scale=lora_scale)
                    if not USE_PEFT_BACKEND
                    else self.proj_in(hidden_states)
                )
                inner_dim = hidden_states.shape[1]
                hidden_states = hidden_states.permute(0, 2, 3, 1).reshape(batch, height * width, inner_dim)
            else:
                inner_dim = hidden_states.shape[1]
                hidden_states = hidden_states.permute(0, 2, 3, 1).reshape(batch, height * width, inner_dim)
                hidden_states = (
                    self.proj_in(hidden_states, scale=lora_scale)
                    if not USE_PEFT_BACKEND
                    else self.proj_in(hidden_states)
                )

        elif self.is_input_vectorized:
            hidden_states = self.latent_image_embedding(hidden_states)
        elif self.is_input_patches:
            height, width = hidden_states.shape[-2] // self.patch_size, hidden_states.shape[-1] // self.patch_size
            hidden_states = self.pos_embed(hidden_states)

            # if self.adaln_single is not None:
            #     if self.use_additional_conditions and added_cond_kwargs is None:
            #         raise ValueError(
            #             "`added_cond_kwargs` cannot be None when using additional conditions for `adaln_single`."
            #         )
            #     batch_size = hidden_states.shape[0]
            #     timestep, embedded_timestep = self.adaln_single(
            #         timestep, added_cond_kwargs, batch_size=batch_size, hidden_dtype=hidden_states.dtype
            #     )

        # 2. Blocks
        if self.caption_projection is not None:
            batch_size = hidden_states.shape[0]
            encoder_hidden_states = self.caption_projection(encoder_hidden_states)
            encoder_hidden_states = encoder_hidden_states.view(batch_size, -1, hidden_states.shape[-1])

        for block in self.transformer_blocks:
            if self.training and self.gradient_checkpointing:

                def create_custom_forward(module, return_dict=None):
                    def custom_forward(*inputs):
                        if return_dict is not None:
                            return module(*inputs, return_dict=return_dict)
                        else:
                            return module(*inputs)

                    return custom_forward

                ckpt_kwargs: Dict[str, Any] = {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}
                hidden_states = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    hidden_states,
                    attention_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                    timestep,
                    cross_attention_kwargs,
                    class_labels,
                    **ckpt_kwargs,
                )
            else:
                hidden_states = block(
                    hidden_states,
                    attention_mask=attention_mask,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=encoder_attention_mask,
                    timestep=timestep,
                    cross_attention_kwargs=cross_attention_kwargs,
                    class_labels=class_labels,
                )

        # 3. Output
        if self.is_input_continuous:
            if not self.use_linear_projection:
                hidden_states = hidden_states.reshape(batch, height, width, inner_dim).permute(0, 3, 1, 2).contiguous()
                hidden_states = (
                    self.proj_out(hidden_states, scale=lora_scale)
                    if not USE_PEFT_BACKEND
                    else self.proj_out(hidden_states)
                )
            else:
                hidden_states = (
                    self.proj_out(hidden_states, scale=lora_scale)
                    if not USE_PEFT_BACKEND
                    else self.proj_out(hidden_states)
                )
                hidden_states = hidden_states.reshape(batch, height, width, inner_dim).permute(0, 3, 1, 2).contiguous()

            output = hidden_states + residual
        elif self.is_input_vectorized:
            hidden_states = self.norm_out(hidden_states)
            logits = self.out(hidden_states)
            # (batch, self.num_vector_embeds - 1, self.num_latent_pixels)
            logits = logits.permute(0, 2, 1)

            # log(p(x_0))
            output = F.log_softmax(logits.double(), dim=1).float()

        if self.is_input_patches:
            # YS: No conditioning on timestep/class_label in `AdaLayerNorm`.
            hidden_states = self.norm_out(hidden_states)
            hidden_states = self.proj_out(hidden_states)

            # unpatchify
            hidden_states = hidden_states.reshape(
                shape=(-1, height, width, self.patch_size, self.patch_size, self.out_channels)
            )
            hidden_states = torch.einsum("nhwpqc->nchpwq", hidden_states) # bs, c, h, patch_size, w, patch_size
            
            output = hidden_states.reshape(
                shape=(-1, self.out_channels, height * self.patch_size, width * self.patch_size)
            )

        if not return_dict:
            return (output,)

        return Transformer2DModelOutput(sample=output)


class CrossAttnDownBlock2D_with_posemb(CrossAttnDownBlock2D):
    def __init__(
        self,
        feature_map_size: Union[int, Tuple[int]],
        in_channels: int,
        out_channels: int,
        temb_channels: int,
        patch_size: int = 16,
        dropout: float = 0.0,
        num_layers: int = 1,
        transformer_layers_per_block: Union[int, Tuple[int]] = 1,
        resnet_eps: float = 1e-6,
        resnet_time_scale_shift: str = "default",
        resnet_act_fn: str = "swish",
        resnet_groups: int = 32,
        resnet_pre_norm: bool = True,
        num_attention_heads: int = 1,
        cross_attention_dim: int = 1280,
        output_scale_factor: float = 1.0,
        downsample_padding: int = 1,
        add_downsample: bool = True,
        dual_cross_attention: bool = False,
        use_linear_projection: bool = False,
        only_cross_attention: bool = False,
        upcast_attention: bool = False,
        attention_type: str = "default",
    ):
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            temb_channels=temb_channels,
            dropout=dropout,
            num_layers=num_layers,
            transformer_layers_per_block=transformer_layers_per_block,
            resnet_eps=resnet_eps,
            resnet_time_scale_shift=resnet_time_scale_shift,
            resnet_act_fn=resnet_act_fn,
            resnet_groups=resnet_groups,
            resnet_pre_norm=resnet_pre_norm,
            num_attention_heads=num_attention_heads,
            cross_attention_dim=cross_attention_dim,
            output_scale_factor=output_scale_factor,
            downsample_padding=downsample_padding,
            add_downsample=add_downsample,
            dual_cross_attention=dual_cross_attention,
            use_linear_projection=use_linear_projection,
            only_cross_attention=only_cross_attention,
            upcast_attention=upcast_attention,
            attention_type=attention_type,
        )

        ### Overwrite self.attentions
        if isinstance(transformer_layers_per_block, int):
            transformer_layers_per_block = [transformer_layers_per_block] * num_layers

        attentions = []
        for i in range(num_layers):
            in_channels = in_channels if i == 0 else out_channels
            
            assert not dual_cross_attention,\
            f"Current CrossAttnDownBlock2D_with_posemb implementation doesn't support dual_cross_attention."
            
            attentions.append(
                Transformer2DModel_with_nonsquare_input(
                    num_attention_heads,
                    out_channels // num_attention_heads,
                    in_channels=out_channels,
                    num_layers=transformer_layers_per_block[i],
                    cross_attention_dim=cross_attention_dim,
                    sample_size=feature_map_size,
                    patch_size=patch_size,
                    norm_num_groups=resnet_groups,
                    use_linear_projection=use_linear_projection,
                    only_cross_attention=only_cross_attention,
                    upcast_attention=upcast_attention,
                    attention_type=attention_type,
                )
            )

        self.attentions = nn.ModuleList(attentions)


class CrossAttnUpBlock2D_with_posemb(CrossAttnUpBlock2D):
    def __init__(
        self,
        feature_map_size: Union[int, Tuple[int]],
        in_channels: int,
        out_channels: int,
        prev_output_channel: int,
        temb_channels: int,
        patch_size: int = 16,
        resolution_idx: Optional[int] = None,
        dropout: float = 0.0,
        num_layers: int = 1,
        transformer_layers_per_block: Union[int, Tuple[int]] = 1,
        resnet_eps: float = 1e-6,
        resnet_time_scale_shift: str = "default",
        resnet_act_fn: str = "swish",
        resnet_groups: int = 32,
        resnet_pre_norm: bool = True,
        num_attention_heads: int = 1,
        cross_attention_dim: int = 1280,
        output_scale_factor: float = 1.0,
        add_upsample: bool = True,
        dual_cross_attention: bool = False,
        use_linear_projection: bool = False,
        only_cross_attention: bool = False,
        upcast_attention: bool = False,
        attention_type: str = "default",
    ):
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            prev_output_channel=prev_output_channel,
            temb_channels=temb_channels,
            resolution_idx=resolution_idx,
            dropout=dropout,
            num_layers=num_layers,
            transformer_layers_per_block=transformer_layers_per_block,
            resnet_eps=resnet_eps,
            resnet_time_scale_shift=resnet_time_scale_shift,
            resnet_act_fn=resnet_act_fn,
            resnet_groups=resnet_groups,
            resnet_pre_norm=resnet_pre_norm,
            num_attention_heads=num_attention_heads,
            cross_attention_dim=cross_attention_dim,
            output_scale_factor=output_scale_factor,
            add_upsample=add_upsample,
            dual_cross_attention=dual_cross_attention,
            use_linear_projection=use_linear_projection,
            only_cross_attention=only_cross_attention,
            upcast_attention=upcast_attention,
            attention_type=attention_type,
        )

        ### Overwrite self.attentions
        if isinstance(transformer_layers_per_block, int):
            transformer_layers_per_block = [transformer_layers_per_block] * num_layers

        attentions = []

        for i in range(num_layers):
            assert not dual_cross_attention,\
            f"Current CrossAttnUpBlock2D_with_posemb implementation doesn't support dual_cross_attention."
            
            attentions.append(
                Transformer2DModel_with_nonsquare_input(
                    num_attention_heads,
                    out_channels // num_attention_heads,
                    in_channels=out_channels,
                    num_layers=transformer_layers_per_block[i],
                    cross_attention_dim=cross_attention_dim,
                    sample_size=feature_map_size,
                    patch_size=patch_size,
                    norm_num_groups=resnet_groups,
                    use_linear_projection=use_linear_projection,
                    only_cross_attention=only_cross_attention,
                    upcast_attention=upcast_attention,
                    attention_type=attention_type,
                )
            )
            
        self.attentions = nn.ModuleList(attentions)


class UNetMidBlock2DCrossAttn_with_posemb(UNetMidBlock2DCrossAttn):
    def __init__(
        self,
        feature_map_size: Union[int, Tuple[int]],
        in_channels: int,
        temb_channels: int,
        patch_size: int = 16,
        dropout: float = 0.0,
        num_layers: int = 1,
        transformer_layers_per_block: Union[int, Tuple[int]] = 1,
        resnet_eps: float = 1e-6,
        resnet_time_scale_shift: str = "default",
        resnet_act_fn: str = "swish",
        resnet_groups: int = 32,
        resnet_pre_norm: bool = True,
        num_attention_heads: int = 1,
        output_scale_factor: float = 1.0,
        cross_attention_dim: int = 1280,
        dual_cross_attention: bool = False,
        use_linear_projection: bool = False,
        only_cross_attention: bool = False,
        upcast_attention: bool = False,
        attention_type: str = "default",
    ):
        super().__init__(
            in_channels=in_channels,
            temb_channels=temb_channels,
            dropout=dropout,
            num_layers=num_layers,
            transformer_layers_per_block=transformer_layers_per_block,
            resnet_eps=resnet_eps,
            resnet_time_scale_shift=resnet_time_scale_shift,
            resnet_act_fn=resnet_act_fn,
            resnet_groups=resnet_groups,
            resnet_pre_norm=resnet_pre_norm,
            num_attention_heads=num_attention_heads,
            output_scale_factor=output_scale_factor,
            cross_attention_dim=cross_attention_dim,
            dual_cross_attention=dual_cross_attention,
            use_linear_projection=use_linear_projection,
            upcast_attention=upcast_attention,
            attention_type=attention_type,
        )

        ### Overwrite self.attentions
        if isinstance(transformer_layers_per_block, int):
            transformer_layers_per_block = [transformer_layers_per_block] * num_layers

        attentions = []

        for i in range(num_layers):
            assert not dual_cross_attention,\
            f"Current UNetMidBlock2DCrossAttn_with_posemb implementation doesn't support dual_cross_attention."
            
            attentions.append(
                Transformer2DModel_with_nonsquare_input(
                    num_attention_heads,
                    in_channels // num_attention_heads,
                    in_channels=in_channels,
                    num_layers=transformer_layers_per_block[i],
                    cross_attention_dim=cross_attention_dim,
                    sample_size=feature_map_size,
                    patch_size=patch_size,
                    norm_num_groups=resnet_groups,
                    use_linear_projection=use_linear_projection,
                    only_cross_attention=only_cross_attention,
                    upcast_attention=upcast_attention,
                    attention_type=attention_type,
                )
            )

        self.attentions = nn.ModuleList(attentions)


class UNet2DConditionModel_with_posemb(UNet2DConditionModel):
    def __init__(
        self,
        patch_size: int = 16,
        sample_size: Union[int, Tuple[int]] = None,
        in_channels: int = 4,
        out_channels: int = 4,
        center_input_sample: bool = False,
        flip_sin_to_cos: bool = True,
        freq_shift: int = 0,
        down_block_types: Tuple[str] = (
            "CrossAttnDownBlock2D",
            "CrossAttnDownBlock2D",
            "CrossAttnDownBlock2D",
            "DownBlock2D",
        ),
        mid_block_type: Optional[str] = "UNetMidBlock2DCrossAttn",
        up_block_types: Tuple[str] = ("UpBlock2D", "CrossAttnUpBlock2D", "CrossAttnUpBlock2D", "CrossAttnUpBlock2D"),
        only_cross_attention: Union[bool, Tuple[bool]] = False,
        block_out_channels: Tuple[int] = (320, 640, 1280, 1280),
        layers_per_block: Union[int, Tuple[int]] = 2,
        downsample_padding: int = 1,
        mid_block_scale_factor: float = 1,
        dropout: float = 0.0,
        act_fn: str = "silu",
        norm_num_groups: Optional[int] = 32,
        norm_eps: float = 1e-5,
        cross_attention_dim: Union[int, Tuple[int]] = 1280,
        transformer_layers_per_block: Union[int, Tuple[int], Tuple[Tuple]] = 1,
        reverse_transformer_layers_per_block: Optional[Tuple[Tuple[int]]] = None,
        encoder_hid_dim: Optional[int] = None,
        encoder_hid_dim_type: Optional[str] = None,
        attention_head_dim: Union[int, Tuple[int]] = 8,
        num_attention_heads: Optional[Union[int, Tuple[int]]] = None,
        dual_cross_attention: bool = False,
        use_linear_projection: bool = False,
        class_embed_type: Optional[str] = None,
        addition_embed_type: Optional[str] = None,
        addition_time_embed_dim: Optional[int] = None,
        num_class_embeds: Optional[int] = None,
        upcast_attention: bool = False,
        resnet_time_scale_shift: str = "default",
        resnet_skip_time_act: bool = False,
        resnet_out_scale_factor: int = 1.0,
        time_embedding_type: str = "positional",
        time_embedding_dim: Optional[int] = None,
        time_embedding_act_fn: Optional[str] = None,
        timestep_post_act: Optional[str] = None,
        time_cond_proj_dim: Optional[int] = None,
        conv_in_kernel: int = 3,
        conv_out_kernel: int = 3,
        projection_class_embeddings_input_dim: Optional[int] = None,
        attention_type: str = "default",
        class_embeddings_concat: bool = False,
        mid_block_only_cross_attention: Optional[bool] = None,
        cross_attention_norm: Optional[str] = None,
        addition_embed_type_num_heads=64,
    ):
        super().__init__(
            sample_size=sample_size,  # the target image resolution
            in_channels=in_channels,  # the number of input channels, 3 for RGB images
            out_channels=out_channels,  # the number of output channels
            layers_per_block=layers_per_block,  # how many ResNet layers to use per UNet block
            block_out_channels=block_out_channels,  # the number of output channels for each UNet block
            down_block_types=down_block_types,
            up_block_types=up_block_types,
            encoder_hid_dim=encoder_hid_dim,
            encoder_hid_dim_type=encoder_hid_dim_type,
            cross_attention_dim=cross_attention_dim,
            conv_in_kernel=conv_in_kernel,
            conv_out_kernel=conv_out_kernel,
        )
        
        
        self.patch_size = patch_size
        if isinstance(sample_size, int):
            self.height, self.width = sample_size, sample_size
        else:
            self.height, self.width = sample_size
        
        ### Check: dimensions of the smallest feature map are multiples of patch_size
        h, w = self.height, self.width
        for i in range(len(block_out_channels)):
            assert h % self.patch_size == 0 and w % self.patch_size == 0\
            and h / self.patch_size >= 1 and w / self.patch_size >= 1,\
            f"Feature map in downsample block {i} has dimensions ({h}, {w}) which aren't multiples of patch_size ({self.patch_size})."

            h /= 2
            w /= 2

        ### Some necessary variables
        num_attention_heads = num_attention_heads or attention_head_dim

        if isinstance(only_cross_attention, bool):
            only_cross_attention = [only_cross_attention] * len(down_block_types)

        if isinstance(num_attention_heads, int):
            num_attention_heads = (num_attention_heads,) * len(down_block_types)

        if isinstance(attention_head_dim, int):
            attention_head_dim = (attention_head_dim,) * len(down_block_types)

        if isinstance(cross_attention_dim, int):
            cross_attention_dim = (cross_attention_dim,) * len(down_block_types)

        if isinstance(layers_per_block, int):
            layers_per_block = [layers_per_block] * len(down_block_types)
        if isinstance(transformer_layers_per_block, int):
            transformer_layers_per_block = [transformer_layers_per_block] * len(down_block_types)

        if time_embedding_type == "fourier":
            time_embed_dim = time_embedding_dim or block_out_channels[0] * 2
            if time_embed_dim % 2 != 0:
                raise ValueError(f"`time_embed_dim` should be divisible by 2, but is {time_embed_dim}.")
            self.time_proj = GaussianFourierProjection(
                time_embed_dim // 2, set_W_to_weight=False, log=False, flip_sin_to_cos=flip_sin_to_cos
            )
            timestep_input_dim = time_embed_dim
        elif time_embedding_type == "positional":
            time_embed_dim = time_embedding_dim or block_out_channels[0] * 4

            self.time_proj = Timesteps(block_out_channels[0], flip_sin_to_cos, freq_shift)
            timestep_input_dim = block_out_channels[0]
        else:
            raise ValueError(
                f"{time_embedding_type} does not exist. Please make sure to use one of `fourier` or `positional`."
            )
        
        if class_embeddings_concat:
            # The time embeddings are concatenated with the class embeddings. The dimension of the
            # time embeddings passed to the down, middle, and up blocks is twice the dimension of the
            # regular time embeddings
            blocks_time_embed_dim = time_embed_dim * 2
        else:
            blocks_time_embed_dim = time_embed_dim
        
        
        ### Overwrite down_blocks
        self.down_blocks = nn.ModuleList([])
        output_channel = block_out_channels[0]
        h, w = self.height, self.width
        for i, down_block_type in enumerate(down_block_types):
            input_channel = output_channel
            output_channel = block_out_channels[i]
            is_final_block = i == len(block_out_channels) - 1

            if down_block_type == "CrossAttnDownBlock2D":
                down_block = CrossAttnDownBlock2D_with_posemb(
                    feature_map_size=(h, w),
                    patch_size=patch_size,
                    num_layers=layers_per_block[i],
                    transformer_layers_per_block=transformer_layers_per_block[i],
                    in_channels=input_channel,
                    out_channels=output_channel,
                    temb_channels=blocks_time_embed_dim,
                    dropout=dropout,
                    add_downsample=not is_final_block,
                    resnet_eps=norm_eps,
                    resnet_act_fn=act_fn,
                    resnet_groups=norm_num_groups,
                    downsample_padding=downsample_padding,
                    cross_attention_dim=cross_attention_dim[i],
                    num_attention_heads=num_attention_heads[i],
                    dual_cross_attention=dual_cross_attention,
                    use_linear_projection=use_linear_projection,
                    only_cross_attention=only_cross_attention[i],
                    upcast_attention=upcast_attention,
                    resnet_time_scale_shift=resnet_time_scale_shift,
                    attention_type=attention_type,
                )
            else:
                down_block = get_down_block(
                    down_block_type,
                    num_layers=layers_per_block[i],
                    transformer_layers_per_block=transformer_layers_per_block[i],
                    in_channels=input_channel,
                    out_channels=output_channel,
                    temb_channels=blocks_time_embed_dim,
                    add_downsample=not is_final_block,
                    resnet_eps=norm_eps,
                    resnet_act_fn=act_fn,
                    resnet_groups=norm_num_groups,
                    cross_attention_dim=cross_attention_dim[i],
                    num_attention_heads=num_attention_heads[i],
                    downsample_padding=downsample_padding,
                    dual_cross_attention=dual_cross_attention,
                    use_linear_projection=use_linear_projection,
                    only_cross_attention=only_cross_attention[i],
                    upcast_attention=upcast_attention,
                    resnet_time_scale_shift=resnet_time_scale_shift,
                    attention_type=attention_type,
                    resnet_skip_time_act=resnet_skip_time_act,
                    resnet_out_scale_factor=resnet_out_scale_factor,
                    cross_attention_norm=cross_attention_norm,
                    attention_head_dim=attention_head_dim[i] if attention_head_dim[i] is not None else output_channel,
                    dropout=dropout,
                )

            h /= 2
            w /= 2
            self.down_blocks.append(down_block)


        ### Overwrite mid_block
        h *= 2
        w *= 2
        if mid_block_type == "UNetMidBlock2DCrossAttn":
            self.mid_block = UNetMidBlock2DCrossAttn_with_posemb(
                feature_map_size=(h, w),
                in_channels=block_out_channels[-1],
                temb_channels=blocks_time_embed_dim,
                patch_size=patch_size,
                transformer_layers_per_block=transformer_layers_per_block[-1],
                dropout=dropout,
                resnet_eps=norm_eps,
                resnet_act_fn=act_fn,
                output_scale_factor=mid_block_scale_factor,
                resnet_time_scale_shift=resnet_time_scale_shift,
                cross_attention_dim=cross_attention_dim[-1],
                num_attention_heads=num_attention_heads[-1],
                resnet_groups=norm_num_groups,
                dual_cross_attention=dual_cross_attention,
                use_linear_projection=use_linear_projection,
                only_cross_attention=only_cross_attention[-1],
                upcast_attention=upcast_attention,
                attention_type=attention_type,
            )


        ### Overwrite up_blocks
        self.num_upsamplers = 0
        self.up_blocks = nn.ModuleList([])
        reversed_block_out_channels = list(reversed(block_out_channels))
        reversed_num_attention_heads = list(reversed(num_attention_heads))
        reversed_layers_per_block = list(reversed(layers_per_block))
        reversed_cross_attention_dim = list(reversed(cross_attention_dim))
        reversed_transformer_layers_per_block = (
            list(reversed(transformer_layers_per_block))
            if reverse_transformer_layers_per_block is None
            else reverse_transformer_layers_per_block
        )
        only_cross_attention = list(reversed(only_cross_attention))

        output_channel = reversed_block_out_channels[0]
        for i, up_block_type in enumerate(up_block_types):
            is_final_block = i == len(block_out_channels) - 1

            prev_output_channel = output_channel
            output_channel = reversed_block_out_channels[i]
            input_channel = reversed_block_out_channels[min(i + 1, len(block_out_channels) - 1)]

            # add upsample block for all BUT final layer
            if not is_final_block:
                add_upsample = True
                self.num_upsamplers += 1
            else:
                add_upsample = False

            if up_block_type == "CrossAttnUpBlock2D":
                up_block = CrossAttnUpBlock2D_with_posemb(
                    feature_map_size=(h, w),
                    patch_size=patch_size,
                    num_layers=reversed_layers_per_block[i] + 1,
                    transformer_layers_per_block=transformer_layers_per_block[i],
                    in_channels=input_channel,
                    out_channels=output_channel,
                    prev_output_channel=prev_output_channel,
                    temb_channels=blocks_time_embed_dim,
                    resolution_idx=i,
                    dropout=dropout,
                    add_upsample=add_upsample,
                    resnet_eps=norm_eps,
                    resnet_act_fn=act_fn,
                    resnet_groups=norm_num_groups,
                    cross_attention_dim=cross_attention_dim[i],
                    num_attention_heads=num_attention_heads[i],
                    dual_cross_attention=dual_cross_attention,
                    use_linear_projection=use_linear_projection,
                    only_cross_attention=only_cross_attention[i],
                    upcast_attention=upcast_attention,
                    resnet_time_scale_shift=resnet_time_scale_shift,
                    attention_type=attention_type,
                )
            else:
                up_block = get_up_block(
                    up_block_type,
                    num_layers=reversed_layers_per_block[i] + 1,
                    transformer_layers_per_block=reversed_transformer_layers_per_block[i],
                    in_channels=input_channel,
                    out_channels=output_channel,
                    prev_output_channel=prev_output_channel,
                    temb_channels=blocks_time_embed_dim,
                    add_upsample=add_upsample,
                    resnet_eps=norm_eps,
                    resnet_act_fn=act_fn,
                    resolution_idx=i,
                    resnet_groups=norm_num_groups,
                    cross_attention_dim=reversed_cross_attention_dim[i],
                    num_attention_heads=reversed_num_attention_heads[i],
                    dual_cross_attention=dual_cross_attention,
                    use_linear_projection=use_linear_projection,
                    only_cross_attention=only_cross_attention[i],
                    upcast_attention=upcast_attention,
                    resnet_time_scale_shift=resnet_time_scale_shift,
                    attention_type=attention_type,
                    resnet_skip_time_act=resnet_skip_time_act,
                    resnet_out_scale_factor=resnet_out_scale_factor,
                    cross_attention_norm=cross_attention_norm,
                    attention_head_dim=attention_head_dim[i] if attention_head_dim[i] is not None else output_channel,
                    dropout=dropout,
                )
            
            h *= 2
            w *= 2
            self.up_blocks.append(up_block)
            
            prev_output_channel = output_channel

        
        

