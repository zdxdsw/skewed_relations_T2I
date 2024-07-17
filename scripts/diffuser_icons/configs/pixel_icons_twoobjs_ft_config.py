from dataclasses import dataclass

@dataclass
class ConditionalTrainingConfig:
    experiment="conditional"
    date="debug"
    nouns_file="../../data/nouns/all_nouns.txt"
    icons_file="../../data/matplotlib/unicode.jsonl"
    split_method="split24" 
    max_num_objs=30 # [30, 40, 50, 60, 70, 80, 90]
    lm="t5"
    layers_per_block = 2
    block_out_channels = (64, 256, 1024)
    down_block_types = (
        "DownBlock2D",
        "CrossAttnDownBlock2D",
        "CrossAttnDownBlock2D",
    )
    up_block_types = (
        "CrossAttnUpBlock2D",  # a ResNet upsampling block with spatial self-attention
        "CrossAttnUpBlock2D",
        "UpBlock2D",
    )
    encoder_hid_dim_type = "text_proj" #None #
    cross_attention_dim = 128
    only_cross_attention = False
    dual_cross_attention = False
    image_size = (64, 32)  # the generated image resolution
    draw_icon_font_size = 28 # 28 for icon_size=32^2; 120 for icon_size=128^2
    patch_size = 2
    mixed_precision = "fp16"  # `no` for float32, `fp16` for automatic mixed precision
    output_dir = "output"
    ckpt_dir ="<largefiles_dir>/skewed_relations_T2I/scripts/diffuser_icons/output" # "output"
    save_image_steps = 1000
    save_model_epochs = 1000 # suggested: 1000 for single-obj pretraining, 20 for two-objs finetuning
    t5_name = "t5-small" #"google/t5-v1_1-xxl" #"google/t5-efficient-xxl" #
    noise_schedule = "squaredcos_cap_v2"
    learning_rate = 1e-4
    lr_warmup_steps = 1000
    gradient_accumulation_steps = 1
    train_batch_size = 4
    eval_batch_size = 20  # how many images to sample during evaluation
    num_train_timesteps = 100
    num_epochs = 600
    conv_in_kernel = 3 # might change for latent diffusion according to the number of channels in vae's output
    conv_out_kernel = 3
    init_from_ckpt = "<path_to_your_singleobj_pretraining_ckpt>.pt", # Relative path to ckpt_dir. e.g. "1213_154255/ckpts/4999_30000_unet.pt" 


@dataclass
class default_ConditionalTrainingConfig:
    vae_weights_dir = None
    init_from_ckpt = None
    trainable_parameters = []


