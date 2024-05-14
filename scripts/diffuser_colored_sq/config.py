from dataclasses import dataclass

@dataclass
class UnConditionalTrainingConfig:
    experiment="unconditional"
    metadata_dir="../../data/matplotlib/colored_single_sq/random_80_20_split/" # "../../data/matplotlib/colored_sq/split1/"
    image_dir ="../../data/matplotlib/colored_single_sq/images/" # "../../data/matplotlib/colored_sq/images/"
    layers_per_block = 2
    block_out_channels = (32, 128, 512)
    down_block_types = (
        "DownBlock2D",
        "AttnDownBlock2D",
        "AttnDownBlock2D",
    )
    #mid_block_type = 'UNetMidBlock2DCrossAttn'
    up_block_types = (
        "AttnUpBlock2D",  # a ResNet upsampling block with spatial self-attention
        "AttnUpBlock2D",
        "UpBlock2D",
    )
    image_size = 32  # the generated image resolution
    noise_schedule = "squaredcos_cap_v2"
    train_batch_size = 8
    eval_batch_size = 8  # how many images to sample during evaluation
    num_train_timesteps = 100
    num_epochs = 300
    gradient_accumulation_steps = 1
    learning_rate = 1e-4
    lr_warmup_steps = 5000
    save_image_steps = 1000
    save_model_epochs = 10
    mixed_precision = "fp16"  # `no` for float32, `fp16` for automatic mixed precision
    output_dir = "output" 
    #load_from_dir = "1027_085618" 


@dataclass
class ConditionalTrainingConfig:
    experiment="conditional"
    metadata_dir="../../data/matplotlib/colored_2sq/split5/" # "../../data/matplotlib/colored_sq/split1/"
    image_dir ="../../data/matplotlib/colored_2sq/images/" # "../../data/matplotlib/colored_sq/images/"
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
    image_size = 32  # the generated image resolution
    mixed_precision = "fp16"  # `no` for float32, `fp16` for automatic mixed precision
    output_dir = "output"
    t5_name = "google/t5-v1_1-xxl" #"t5-small" #"google/t5-efficient-xxl" #
    noise_schedule = "squaredcos_cap_v2"
    gradient_accumulation_steps = 1
    learning_rate = 1e-4
    train_batch_size = 8
    eval_batch_size = 8  # how many images to sample during evaluation
    num_train_timesteps = 100
    num_epochs = 300
    lr_warmup_steps = 1000
    save_image_steps = 1000
    save_model_epochs = 10
    conv_in_kernel = 3
    conv_out_kernel = 3
    #load_from_dir = "1105_154436" 
    #seed = 0


