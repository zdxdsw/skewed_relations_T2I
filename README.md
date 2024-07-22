`master` branch for public viewing

`development` branch contains notebooks for early dataset exploration, debugging, unbatched inference and making plots.

Modify `<largefiles_dir>` if you keep large files in a separate directory.

Calculation of the proposed metrics, COMPLETENESS and BALANCE: [`quantifying_skew.ipynb`](https://github.com/zdxdsw/skewed_relations_T2I/blob/master/notebooks/quantifying_skew.ipynb)

<br>

## Setup

#### 1. Python Environment
```
git clone git@github.com:zdxdsw/skewed_relations_T2I.git &&
cd skewed_relations_T2I &&
python3 -m venv venv &&
source venv/bin/activate &&
pip install --upgrade pip &&
pip install -r requirements.txt
```

Toubleshooting: If you're having ImportError or imcompatibility issues, try installing the specific version.
`pip install torch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 --index-url https://download.pytorch.org/whl/cu118`. This requires cuda11.8. If your machine supports multiple cuda versions, you might want to do the following: `export LD_LIBRARY_PATH=/usr/local/cuda-11.8/lib`.


#### 2. Accelerate config
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

<br>

## Pixel Diffusion Experiments with Synthetic Images

#### 1. Training configs

Config your training hyperparameters in `skewed_relations_T2I/scripts/diffuser_icons/config.py`.

To reproduce results in our paper, copy configs from [skewed_relations_T2I/scripts/diffuser_icons/configs/pixel_icons_singleobj_pt_config.py](https://github.com/zdxdsw/skewed_relations_T2I/blob/master/scripts/diffuser_icons/configs/pixel_icons_singleobj_pt_config.py) and [skewed_relations_T2I/scripts/diffuser_icons/configs/pixel_icons_twoobjs_ft_config.py](https://github.com/zdxdsw/skewed_relations_T2I/blob/master/scripts/diffuser_icons/configs/pixel_icons_twoobjs_ft_config.py)

#### 2. Synthetic dataset

Due to the simplicity of synthetic data, we do not save a copy. Data is constructed on the fly in the dataloader. Please refer to [`dataset.py`](https://github.com/zdxdsw/skewed_relations_T2I/blob/master/scripts/diffuser_icons/dataset.py) for how splits with different degrees of skew are created, and this [summary chart](https://github.com/zdxdsw/skewed_relations_T2I/blob/master/scripts/diffuser_icons/metric_summary.png) for mapping `split_method` to metrics.

#### 3. Training commands
```
cd skewed_relations_T2I/scripts/diffusion_icons
accelerate launch trainer.py
```

#### 4. Testing commands
```
cd skewed_relations_T2I/scripts/diffusion_icons
accelerate launch tester.py --load_from_dir <handle> --load_from_epochs <load_from_epochs> --eval_batch_size <eval_batch_size>
```

`<handle>`: Every experiment will have a unique identifier, created from the timestamp at which it is launched. E.g. 0515_222602 (`%m%d_%H%M%S`)

`<load_from_epochs>`: String seperated by spaces. E.g. "99 199 299 399 499 599"

`<eval_batch_size>`: Per gpu batch size.


By default, `tester.py` will run inference on both training and testing set. To opt out from training (testing) set, set `--num_iter_train 0` (`--num_iter_test 0`).

#### 5. Evaluation script

Fixed filters are created from GTH icons. Then generated images are evaluated via pixel-level pattern matching. Please refer to this [notebook](https://github.com/zdxdsw/skewed_relations_T2I/blob/master/notebooks/evaluate_generated_icons.ipynb).

#### 6. Ablation experiments

To disable image positional embeddings, comment the line `patch_size = 2` in [`config.py`](https://github.com/zdxdsw/skewed_relations_T2I/blob/master/scripts/diffuser_icons/config.py) or set `patch_size = None`. (It needs to re-run both single-obj pretraining and two-objs finetuning.)

To switch language encoder from T5 to CLIP, modify [`config.py`](https://github.com/zdxdsw/skewed_relations_T2I/blob/master/scripts/diffuser_icons/config.py): `lm = "t5"` <--> `lm = "clip_"`

<br>

## Pixel Diffusion Experiments with Natural Images

#### 1. Download WhatsUp dataset
Images are released by the [WhatsUp official repo](https://github.com/amitakamath/whatsup_vlms?tab=readme-ov-file#downloading-the-data). Download `controlled_clevr.tar.gz` from https://drive.google.com/drive/u/0/folders/164q6X9hrvP-QYpi3ioSnfMuyHpG5oRkZ.

```
cd <largefiles_dir>/skewed_relations_T2I &&
mkdir -p data/whatsup_vlms
```
Move the folder `controlled_clevr` to `<largefiles_dir>/skewed_relations_T2I/data/whatsup_vlms/`.

&#x2610; TODO: explain how the annotation file is pre-processed into `skewed_relations_T2I/data/aggregated/whatsup_vlm_b_lr.json` and `skewed_relations_T2I/data/aggregated/whatsup_vlm_b_lr_autofill_remove_sun_rem_pho.json`

#### 2. Training configs

Config your training hyperparameters in `skewed_relations_T2I/scripts/diffuser_real/config.py`. 

To reproduce results in our paper, copy configs from [skewed_relations_T2I/scripts/diffuser_real/configs/pixel_natural_singleobj_pt_config.py](https://github.com/zdxdsw/skewed_relations_T2I/blob/master/scripts/diffuser_real/configs/pixel_natural_singleobj_pt_config.py) and [skewed_relations_T2I/scripts/diffuser_real/configs/pixel_natural_twoobjs_ft_config.py](https://github.com/zdxdsw/skewed_relations_T2I/blob/master/scripts/diffuser_real/configs/pixel_natural_twoobjs_ft_config.py)

#### 3. Drawing subsamples

Instances are converted to the tuple representation $(f_1, r_1, f_2, r_2)$ and subsampled in the tuple representation space. Please refer to [`dataset.py`](https://github.com/zdxdsw/skewed_relations_T2I/blob/master/scripts/diffuser_real/dataset.py) for how subsamples with different degrees of skew are drawn, and this [summary chart](https://github.com/zdxdsw/skewed_relations_T2I/blob/master/scripts/diffuser_real/metric_summary.png) for mapping `subsample_method` to metrics.

#### 4. Training commands
```
cd skewed_relations_T2I/scripts/diffusion_real
accelerate launch trainer.py
```

#### 5. Testing commands
```
cd skewed_relations_T2I/scripts/diffusion_real
accelerate launch tester.py --load_from_dir <handle> --load_from_epochs <load_from_epochs> --eval_batch_size <eval_batch_size>
```

`<handle>`: Every experiment will have a unique identifier, created from the timestamp at which it is launched. E.g. 0515_222602 (`%m%d_%H%M%S`)

`<load_from_epochs>`: String seperated by spaces. E.g. "99 199 299 399 499 599"

`<eval_batch_size>`: Per gpu batch size.


By default, `tester.py` will run inference on both training and testing set. To opt out from training (testing) set, set `--num_iter_train 0` (`--num_iter_test 0`).

#### 6. AutoEval with ViT

```
cd <largefiles_dir>/skewed_relations_T2I &&
mkdir autoeval
```
Download the finetuned ViT checkpoint from [here](https://drive.google.com/file/d/1wgzwoUmKmdETmD-donHaTrXd8ykQtVQl/view?usp=sharing) (328MB) and move it to `<largefiles_dir>/skewed_relations_T2I/autoeval`.

For your reference, we provide [code for finetuning ViT](https://github.com/zdxdsw/skewed_relations_T2I/blob/master/notebooks/finetune_vit.ipynb).

#### 7. Evaluation commands

```
cd skewed_relations_T2I/scripts/diffusion_real
python eval.py --ckpt_handle <handle> --epochs_for_eval <epochs_for_eval> --output_folder <output_folder> # single_gpu job
```
`<handle>`: Every experiment will have a unique identifier, created from the timestamp at which it is launched. E.g. 0515_222602 (`%m%d_%H%M%S`)

`<epochs_for_eval>`: String seperated by spaces. E.g. "1999 3999 5999"

`<output_folder>`: E.g. "output" or "output_withvae"

<br>

## Latent Diffusion Experiments

#### 1. Download pre-trained vae checkpoints from huggingface.

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

#### 2. Training configs


To reproduce results in our paper, copy configs from

- Experiments on synthetic images: [skewed_relations_T2I/scripts/diffuser_icons/configs/vae_icons_singleobj_pt_config.py](https://github.com/zdxdsw/skewed_relations_T2I/blob/master/scripts/diffuser_icons/configs/vae_icons_singleobj_pt_config.py) and [skewed_relations_T2I/scripts/diffuser_icons/configs/vae_icons_twoobjs_ft_config.py](https://github.com/zdxdsw/skewed_relations_T2I/blob/master/scripts/diffuser_icons/configs/vae_icons_twoobjs_ft_config.py)


- Experiments on natural images: [skewed_relations_T2I/scripts/diffuser_real/configs/vae_natural_singleobj_pt_config.py](https://github.com/zdxdsw/skewed_relations_T2I/blob/master/scripts/diffuser_real/configs/vae_natural_singleobj_pt_config.py) and [skewed_relations_T2I/scripts/diffuser_real/configs/vae_natural_twoobjs_ft_config.py](https://github.com/zdxdsw/skewed_relations_T2I/blob/master/scripts/diffuser_real/configs/vae_natural_twoobjs_ft_config.py)

#### 3. Training/Testing/Evaluation commands

Same as previous sections.

<br>

## Credits
@huggingface [Diffusers](https://github.com/huggingface/diffusers/tree/main)

@amitakamath [whatsup_vlms](https://github.com/amitakamath/whatsup_vlms)


## Cite Us &#x1f64f;
```
@article{chang2024skews,
  title={Skews in the Phenomenon Space Hinder Generalization in Text-to-Image Generation},
  author={Chang, Yingshan and Zhang, Yasi and Fang, Zhiyuan and Wu, Yingnian and Bisk, Yonatan and Gao, Feng},
  journal={arXiv preprint arXiv:2403.16394},
  year={2024}
}
```

