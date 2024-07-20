from dataclasses import dataclass

@dataclass
class ConditionalTrainingConfig:
    experiment="real_conditional"
    date="debug"
    annotations="../../data/aggregated/whatsup_vlm_b_lr_autofill_remove_sun_rem_pho.json"
    imdir="<largefiles_dir>/skewed_relations_T2I/data/" 
    dataset_class="real_dataset" #"whatsup_singleobj_dataset" for pretraining; "real_dataset" for finetuning
    subsample_method="splitU"
    lm="t5"
    layers_per_block = 2
    block_out_channels = (512, 512, 1024) 
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
    image_size = (128, 256) # the generated image resolution
    patch_size = 2
    mixed_precision = "fp16"  # `no` for float32, `fp16` for automatic mixed precision
    output_dir = "output_withvae"
    ckpt_dir = "<largefiles_dir>/skewed_relations_T2I/scripts/diffuser_real/output" # "output"
    save_image_steps = 2000 
    save_model_epochs = 200
    t5_name = "t5-small" #"google/t5-v1_1-xxl" #"google/t5-efficient-xxl" #
    noise_schedule = "squaredcos_cap_v2"
    learning_rate = 5e-4
    lr_warmup_steps = 1000
    gradient_accumulation_steps = 1
    train_batch_size = 4 # Per gpu batch size
    eval_batch_size = 20  # how many images to sample during evaluation
    num_train_timesteps = 100
    num_epochs = 6000
    conv_in_kernel = 3
    conv_out_kernel = 3
    trainable_parameters = ["attentions", "encoder_hid_proj"]
    init_from_ckpt = "<path_to_your_singleobj_pretraining_ckpt>.pt", # Relative path to ckpt_dir. e.g. "0514_083120/ckpts/49999_100000_unet.pt"   

@dataclass
class default_ConditionalTrainingConfig:
    lm="t5"
    vae_weights_dir = None
    init_from_ckpt = None
    load_from_dir = None
    trainable_parameters = []
    subsample_method = None
    latent_channels = None
    vae_downsample_factor = 1
