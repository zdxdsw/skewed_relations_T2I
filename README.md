`master` branch for public viewing

`development` branch contains notebooks for early dataset exploration, debugging, unbatched inference and making plots.

Modify `<largefiles_dir>` if you keep large files in a separate directory.


# Setup

### Python Environment
```
git clone git@github.com:zdxdsw/skewed_relations_T2I.git &&
cd skewed_relations_T2I &&
python3 -m venv venv &&
source venv/bin/activate &&
pip install --upgrade pip &&
pip install -r requirements.txt
```

Toubleshooting: If you're having problems installing torch or import errors, try installing the specific version.
`pip install torch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 --index-url https://download.pytorch.org/whl/cu118`. This requires cuda11.8. If your machine supports multiple cuda versions, you might want to do the following: `export LD_LIBRARY_PATH=/usr/local/cuda-11.8/lib`.


### Accelerate config
`$ accelerate config` # This will automatically generate `~/.cache/huggingface/accelerate/default_config.yaml`.

Example config:
```
compute_environment: LOCAL_MACHINE
debug: false
distributed_type: MULTI_GPU
downcast_bf16: 'no'
gpu_ids: all
machine_rank: 0
main_training_function: main
mixed_precision: fp16
num_machines: 1
num_processes: 4
rdzv_backend: static
same_network: true
tpu_env: []
tpu_use_cluster: false
tpu_use_sudo: false
use_cpu: false
```

# Pixel Diffusion Experiments with Synthetic Images



```
cd skewed_relations_T2I/scripts/diffusion_icons
accelerate launch trainer.py
```


# Pixel Diffusion Experiments with Natural Images

### Download WhatsUp dataset
Images are released by the [WhatsUp official repo](https://github.com/amitakamath/whatsup_vlms?tab=readme-ov-file#downloading-the-data). Download `controlled_clevr.tar.gz` from https://drive.google.com/drive/u/0/folders/164q6X9hrvP-QYpi3ioSnfMuyHpG5oRkZ.

```
cd <largefiles_dir>/skewed_relations_T2I &&
mkdir -p data/whatsup_vlms
```
Move the folder `controlled_clevr` to `<largefiles_dir>/skewed_relations_T2I/data/whatsup_vlms/`.

### Autoeval with ViT

```
cd <largefiles_dir>/skewed_relations_T2I &&
mkdir autoeval
```
Download the finetuned ViT checkpoint from [here](https://drive.google.com/file/d/1wgzwoUmKmdETmD-donHaTrXd8ykQtVQl/view?usp=sharing) (328MB) and move it to `<largefiles_dir>/skewed_relations_T2I/autoeval`.

# Latent Diffusion Experiments

Download pre-trained vae checkpoints from huggingface.
```
cd <largefiles_dir>/skewed_relations_T2I &&
mkdir -p from_pretrained/vae/sd2 &&
cd from_pretrained/vae/sd2 &&
wget https://huggingface.co/stabilityai/stable-diffusion-2-1/resolve/main/vae/config.json &&
wget https://huggingface.co/stabilityai/stable-diffusion-2-1/resolve/main/vae/diffusion_pytorch_model.bin &&
wget https://huggingface.co/stabilityai/stable-diffusion-2-1/resolve/main/vae/diffusion_pytorch_model.fp16.bin &&
wget https://huggingface.co/stabilityai/stable-diffusion-2-1/resolve/main/vae/diffusion_pytorch_model.fp16.safetensors &&
wget https://huggingface.co/stabilityai/stable-diffusion-2-1/resolve/main/vae/diffusion_pytorch_model.safetensors
```


