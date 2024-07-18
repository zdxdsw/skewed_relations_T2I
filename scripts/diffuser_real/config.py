from dataclasses import dataclass

@dataclass
class ConditionalTrainingConfig:
    experiment="real_conditional"
    date="debug"
    annotations="../../data/aggregated/whatsup_vlm_b_lr_autofill_remove_sun_rem_pho.json" #whatsup_vlm_b_lr.json"  #whatsup_vlm_b_lr_autofill.json" #  "vgr_nocaps_fb_both_complete.json"
    imdir="/data/yingshac/clevr_control/data/" 
    dataset_class="real_dataset" #"whatsup_singleobj_dataset" #
    subsample_method="splitU"
    lm="t5"
    layers_per_block = 2
    block_out_channels = (512, 512, 1024) #(512, 512, 1024, 1536)
    down_block_types = (
        "DownBlock2D",
        "CrossAttnDownBlock2D",
        "CrossAttnDownBlock2D",
        #"DownBlock2D",
        #"DownBlock2D",
    )
    #mid_block_type = 'UNetMidBlock2DCrossAttn'
    up_block_types = (
        #"UpBlock2D",
        #"UpBlock2D",
        "CrossAttnUpBlock2D",  # a ResNet upsampling block with spatial self-attention
        "CrossAttnUpBlock2D",
        "UpBlock2D",
    )
    encoder_hid_dim_type = "text_proj" #None #
    cross_attention_dim = 128
    only_cross_attention = False
    dual_cross_attention = False
    image_size = (128, 256) #(128, 128) #(32,64)  # the generated image resolution
    patch_size = 2
    mixed_precision = "fp16"  # `no` for float32, `fp16` for automatic mixed precision
    output_dir = "output_rbt"
    ckpt_dir = "/data/yingshac/clevr_control/scripts/diffuser_real/output" #"output"
    save_image_steps = 2000 #3000 #
    save_model_epochs = 200 #5000 #
    t5_name = "t5-small" #"google/t5-v1_1-xxl" #"google/t5-efficient-xxl" #
    noise_schedule = "squaredcos_cap_v2"
    learning_rate = 5e-4 #1e-4 #
    lr_warmup_steps = 0#1000
    gradient_accumulation_steps = 1
    train_batch_size = 4 # Per gpu batch size
    eval_batch_size = 18 #20  # how many images to sample during evaluation
    num_train_timesteps = 100
    num_epochs = 6000 #50000 #
    conv_in_kernel = 3
    conv_out_kernel = 3
    trainable_parameters = ["attentions", "encoder_hid_proj"] # ["attn2", "norm3", "encoder_hid_proj"] # ["transformer_blocks.0.norm1", "attn1", "transformer_blocks.0.norm2", "encoder_hid_proj"] # ["attn2", "norm3", "encoder_hid_proj"] #
    load_from_dir = "0516_001146" 
    #init_from_ckpt = "0514_083120/ckpts/49999_100000_unet.pt" #"0304_002415/ckpts/49999_100000_unet.pt" # 1217_160404 
    vae_weights_dir = "/data/yingshac/clevr_control/from_pretrained/vae/sd2"
    vae_downsample_factor = 8
    latent_channels = 4


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
