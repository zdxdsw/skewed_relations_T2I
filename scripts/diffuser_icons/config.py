from dataclasses import dataclass

@dataclass
class ConditionalTrainingConfig:
    experiment="conditional"
    date="debug"
    nouns_file="../../data/nouns/all_nouns.txt"
    icons_file="../../data/matplotlib/unicode.jsonl"
    split_method="split7" #"custom_split(nouns, icons, '../../data/vl_models_are_bows/custom_splits_bijective/fb_train_triplets.txt')" # 
    max_num_objs=30
    lm="t5"
    layers_per_block = 2
    block_out_channels = (64, 256, 1024)
    down_block_types = (
        "DownBlock2D",
        "CrossAttnDownBlock2D",
        "CrossAttnDownBlock2D",
    )
    #mid_block_type = 'UNetMidBlock2DCrossAttn'
    up_block_types = (
        "CrossAttnUpBlock2D",  # a ResNet upsampling block with spatial self-attention
        "CrossAttnUpBlock2D",
        "UpBlock2D",
    )
    encoder_hid_dim_type = "text_proj" #None #
    cross_attention_dim = 128
    only_cross_attention = False
    dual_cross_attention = False
    image_size = (256, 128)  # the generated image resolution
    draw_icon_font_size = 120 # 120 for icon_size=128; 220 for icon_size=256
    patch_size = 2
    mixed_precision = "fp16"  # `no` for float32, `fp16` for automatic mixed precision
    output_dir = "output_rbt"
    ckpt_dir = "/data/yingshac/clevr_control/scripts/diffuser_icons/output" #"output"
    save_image_steps = 1000
    save_model_epochs = 20 # 1000 for single-obj pretraining
    t5_name = "t5-small" #"google/t5-v1_1-xxl" #"google/t5-efficient-xxl" #
    noise_schedule = "squaredcos_cap_v2"
    learning_rate = 1e-4
    lr_warmup_steps = 1000
    gradient_accumulation_steps = 1
    train_batch_size = 4
    eval_batch_size = 18  # how many images to sample during evaluation
    num_train_timesteps = 100
    num_epochs = 600
    conv_in_kernel = 3 # might change for latent diffusion according to the number of channels in vae's output
    conv_out_kernel = 3
    #trainable_parameters = ["attentions", "encoder_hid_proj"] # ["attn2", "norm3", "encoder_hid_proj"] # ["transformer_blocks.0.norm1", "attn1", "transformer_blocks.0.norm2", "encoder_hid_proj"] # ["attn2", "norm3", "encoder_hid_proj"] #
    #load_from_dir = "0312_194052" 
    init_from_ckpt = "0514_082837/ckpts/4999_30000_unet.pt" #1201_214632 1217_160404 
    vae_weights_dir = "/data/yingshac/clevr_control/from_pretrained/vae/sd2"
    vae_downsample_factor = 8
    latent_channels = 4


@dataclass
class default_ConditionalTrainingConfig:
    vae_weights_dir = None
    init_from_ckpt = None
    trainable_parameters = []


