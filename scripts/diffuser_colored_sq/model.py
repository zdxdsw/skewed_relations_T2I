from typing import Any, Callable, Dict, List, Optional, Union
import numpy as np
import torch, PIL, os, sys, math
sys.path.append("../")
#from t5 import encode_text
from functools import partial
from torchvision import transforms
from torch import nn
from tqdm import tqdm
from diffusers.models import AutoencoderKL, UNet2DConditionModel, UNet2DModel
from module import UNet2DConditionModel_with_posemb
from diffusers import DDPMScheduler
from diffusers.utils import logging, make_image_grid
from diffusers.utils.torch_utils import randn_tensor
from transformers import T5EncoderModel, T5Tokenizer, T5Config
from training_utils import *

#logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

class T2IDiffusion(nn.Module):
    def __init__(
            self,
            config
    ):
        super().__init__()
        #eval(f"from {config.lm} import encode_text")
        LM = __import__(config.lm)
        encode_text = LM.encode_text
        #self.tokenizer = T5Tokenizer.from_pretrained(config.t5_name, torch_dtype=torch.float16)
        #self.text_encoder = T5EncoderModel.from_pretrained(config.t5_name, torch_dtype=torch.float16)
        
        #if torch.cuda.is_available(): self.text_encoder = self.text_encoder.cuda()
        
        if config.encoder_hid_dim_type is None:
            self.encoder_hid_dim = None
        else:
            if config.lm=='t5':
                self.text_encoder_config = T5Config.from_pretrained(config.t5_name)
                self.encoder_hid_dim = self.text_encoder_config.d_model
            elif config.lm=="clip_":
                self.encoder_hid_dim = LM.get_encoded_dim()
            print(f"auto suggesting encoder_hid_dim = {self.encoder_hid_dim}")

        if "patch_size" in dir(config) and config.patch_size is not None:
            self.unet = UNet2DConditionModel_with_posemb(
                patch_size=config.patch_size,
                sample_size=config.image_size,  # the target image resolution
                in_channels=3,  # the number of input channels, 3 for RGB images
                out_channels=3,  # the number of output channels
                layers_per_block=config.layers_per_block,  # how many ResNet layers to use per UNet block
                block_out_channels=config.block_out_channels,  # the number of output channels for each UNet block
                down_block_types=config.down_block_types,
                up_block_types=config.up_block_types,
                encoder_hid_dim=self.encoder_hid_dim,
                encoder_hid_dim_type=config.encoder_hid_dim_type,
                cross_attention_dim=config.cross_attention_dim,
                conv_in_kernel=config.conv_in_kernel,
                conv_out_kernel=config.conv_out_kernel,
                only_cross_attention=config.only_cross_attention,
                dual_cross_attention=config.dual_cross_attention
            )
        else: 
            assert not config.dual_cross_attention or config.only_cross_attention, \
                f"UNet2DConditionModel's implementation probably needs revision for only_cross_attention=True (received {config.only_cross_attention}) or dual_cross_attention=False (received {config.dual_cross_attention})."
            self.unet = UNet2DConditionModel(
                sample_size=config.image_size,  # the target image resolution
                in_channels=3,  # the number of input channels, 3 for RGB images
                out_channels=3,  # the number of output channels
                layers_per_block=config.layers_per_block,  # how many ResNet layers to use per UNet block
                block_out_channels=config.block_out_channels,  # the number of output channels for each UNet block
                down_block_types=config.down_block_types,
                up_block_types=config.up_block_types,
                encoder_hid_dim=self.encoder_hid_dim,
                encoder_hid_dim_type=config.encoder_hid_dim_type,
                cross_attention_dim=config.cross_attention_dim,
                conv_in_kernel=config.conv_in_kernel,
                conv_out_kernel=config.conv_out_kernel,
            )
        self.noise_scheduler = DDPMScheduler(config.num_train_timesteps, beta_schedule=config.noise_schedule)

        self.encode_text = partial(
            encode_text,
            name = config.t5_name,
            dtype = config.mixed_precision
        )

        #self.postprocess = transforms.Normalize(mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225], std=[1/0.229, 1/0.224, 1/0.225])

    
    def forward(self, clean_images, texts):
        #self.unet.train()
        device = self.unet.device
        prompt_embeds = self.encode_text(texts, device=device)
        noise = torch.randn(clean_images.shape).to(device)
        bs = clean_images.shape[0]

        # Sample a random timestep for each image
        timesteps = torch.randint(
            0, self.noise_scheduler.config.num_train_timesteps, (bs,), device=device
        ).long()

        # forward diffusion 
        noisy_images =self.noise_scheduler.add_noise(
            clean_images.to(device), 
            noise, 
            timesteps
        )
        # predict noise
        noise_pred = self.unet(
            noisy_images, 
            encoder_hidden_states=prompt_embeds, 
            timestep=timesteps, 
            return_dict=False
        )[0]
        return noise_pred, noise

    def inference(
            self, 
            epoch, 
            texts: List[str], 
            output_dir,
            save=True,
            disable_pgbar=False
        ):
        # Sample some images from random noise (this is the backward diffusion process).
        # The default pipeline output type is `List[PIL.Image]`
        #self.unet.eval()
        device = self.unet.device
        prompt_embeds = self.encode_text(texts, device=device)
        if isinstance(self.unet.config.sample_size, int): 
            image_shape = (len(texts), self.unet.config.in_channels, self.unet.config.sample_size, self.unet.config.sample_size)
        else:
            image_shape = (len(texts), self.unet.config.in_channels, *self.unet.config.sample_size)
        image = randn_tensor(image_shape, device=device)

        with torch.no_grad():
            for t in tqdm(self.noise_scheduler.timesteps, disable=disable_pgbar):
                model_output = self.unet(image, encoder_hidden_states=prompt_embeds, timestep=t).sample
                image = self.noise_scheduler.step(model_output, t, image).prev_sample
        
        #_min, _max = image.amin(dim=(1, 2, 3)), image.amax(dim=(1, 2, 3))
        #image = (image - _min[:, :, :, None]) / (_max-_min)[:, :, :, None]
        image = (image / 2 + 0.5).clamp(0, 1) # (image / 2 + 0.5).
        #image = self.postprocess(image).clamp(0, 1)

        image = image.cpu().permute(0, 2, 3, 1).numpy() # (bs, h, w, c)
        num_placeholder = math.ceil(len(texts)/16) * 16 - len(texts)
        if num_placeholder:
            placeholder = np.ones((num_placeholder, *image.shape[1:]))
            image = np.concatenate((image, placeholder))
        image = numpy_to_pil(image)
        # Make a grid out of the images
        num_cols = math.ceil(len(texts)/16)
        image_grid = make_image_grid(image, rows=min(len(texts), 16), cols=num_cols)

        # Save the images
        if output_dir is not None and save:
            test_dir = os.path.join(output_dir, "samples")
            os.makedirs(test_dir, exist_ok=True)
            image_grid.save("{}/{}.png".format(test_dir, epoch))
            with open("{}/{}.txt".format(test_dir, epoch), "w") as f:
                f.write("\n".join(texts))
        else: return image_grid # type: PIL.Image.Image


class Diffusion(nn.Module):
    def __init__(
            self,
            config
    ):
        super().__init__()
        self.unet = UNet2DModel(
            sample_size=config.image_size,  # the target image resolution
            in_channels=3,  # the number of input channels, 3 for RGB images
            out_channels=3,  # the number of output channels
            layers_per_block=config.layers_per_block,  # how many ResNet layers to use per UNet block
            block_out_channels=config.block_out_channels,  # the number of output channels for each UNet block
            down_block_types=config.down_block_types,
            up_block_types=config.up_block_types,
        )
        self.noise_scheduler = DDPMScheduler(config.num_train_timesteps, beta_schedule=config.noise_schedule)

    def forward(self, clean_images):
        device = self.unet.device
        noise = torch.randn(clean_images.shape).to(device)
        bs = clean_images.shape[0]

        # Sample a random timestep for each image
        timesteps = torch.randint(
            0, self.noise_scheduler.config.num_train_timesteps, (bs,), device=device
        ).long()

        # forward diffusion 
        noisy_images =self.noise_scheduler.add_noise(
            clean_images.to(device), 
            noise, 
            timesteps
        )
        # predict noise
        noise_pred = self.unet(
            noisy_images, 
            timestep=timesteps, 
            return_dict=False
        )[0]
        return noise_pred, noise

    def inference(
            self, 
            epoch, 
            eval_batch_size,
            output_dir,
            save=True,
            disable_pgbar=False
        ):
        # Sample some images from random noise (this is the backward diffusion process).
        # The default pipeline output type is `List[PIL.Image]`
        device = self.unet.device
        image_shape = (eval_batch_size, self.unet.config.in_channels, self.unet.config.sample_size, self.unet.config.sample_size)
        image = randn_tensor(image_shape, device=device)

        with torch.no_grad():
            for t in tqdm(self.noise_scheduler.timesteps, disable=disable_pgbar):
                model_output = self.unet(image, timestep=t).sample
                image = self.noise_scheduler.step(model_output, t, image).prev_sample
        
        #_min, _max = image.amin(dim=(1, 2, 3)), image.amax(dim=(1, 2, 3))
        #image = (image - _min[:, :, :, None]) / (_max-_min)[:, :, :, None]
        image = (image / 2 + 0.5).clamp(0, 1) # (image / 2 + 0.5).
        image = image.cpu().permute(0, 2, 3, 1).numpy()
        image = numpy_to_pil(image)
        # Make a grid out of the images
        image_grid = make_image_grid(image, rows=image_shape[0], cols=1)

        # Save the images
        if output_dir is not None and save:
            test_dir = os.path.join(output_dir, "samples")
            os.makedirs(test_dir, exist_ok=True)
            image_grid.save("{}/{}.png".format(test_dir, epoch))
        else: return image_grid # type: PIL.Image.Image

    
from diffusers import AutoencoderKL

class T2ILatentDiffusion(nn.Module):
    def __init__(
            self,
            config
    ):
        super().__init__()

        LM = __import__(config.lm)
        encode_text = LM.encode_text

        self.vae = AutoencoderKL.from_pretrained(config.vae_weights_dir)
        self.vae.requires_grad_(False)
        self.vae_downsample_factor = config.vae_downsample_factor

        if config.encoder_hid_dim_type is None:
            self.encoder_hid_dim = None
        else:
            if config.lm=='t5':
                self.text_encoder_config = T5Config.from_pretrained(config.t5_name)
                self.encoder_hid_dim = self.text_encoder_config.d_model
            elif config.lm=="clip_":
                self.encoder_hid_dim = LM.get_encoded_dim()
            print(f"auto suggesting encoder_hid_dim = {self.encoder_hid_dim}")

        if "patch_size" in dir(config) and config.patch_size is not None:
            self.unet = UNet2DConditionModel_with_posemb(
                patch_size=config.patch_size,
                sample_size=config.image_size,  # the target image resolution
                in_channels=config.latent_channels,  # the number of input channels
                out_channels=config.latent_channels,  # the number of output channels
                layers_per_block=config.layers_per_block,  # how many ResNet layers to use per UNet block
                block_out_channels=config.block_out_channels,  # the number of output channels for each UNet block
                down_block_types=config.down_block_types,
                up_block_types=config.up_block_types,
                encoder_hid_dim=self.encoder_hid_dim,
                encoder_hid_dim_type=config.encoder_hid_dim_type,
                cross_attention_dim=config.cross_attention_dim,
                conv_in_kernel=config.conv_in_kernel,
                conv_out_kernel=config.conv_out_kernel,
                only_cross_attention=config.only_cross_attention,
                dual_cross_attention=config.dual_cross_attention
            )
        else: 
            assert not config.dual_cross_attention or config.only_cross_attention, \
                f"UNet2DConditionModel's implementation probably needs revision for only_cross_attention=True (received {config.only_cross_attention}) or dual_cross_attention=False (received {config.dual_cross_attention})."
            self.unet = UNet2DConditionModel(
                sample_size=config.image_size,  # the target image resolution
                in_channels=config.latent_channels,  # the number of input channels
                out_channels=config.latent_channels,  # the number of output channels
                layers_per_block=config.layers_per_block,  # how many ResNet layers to use per UNet block
                block_out_channels=config.block_out_channels,  # the number of output channels for each UNet block
                down_block_types=config.down_block_types,
                up_block_types=config.up_block_types,
                encoder_hid_dim=self.encoder_hid_dim,
                encoder_hid_dim_type=config.encoder_hid_dim_type,
                cross_attention_dim=config.cross_attention_dim,
                conv_in_kernel=config.conv_in_kernel,
                conv_out_kernel=config.conv_out_kernel,
            )
        self.noise_scheduler = DDPMScheduler(config.num_train_timesteps, beta_schedule=config.noise_schedule)

        self.encode_text = partial(
            encode_text,
            name = config.t5_name,
            dtype = config.mixed_precision
        )

        #self.postprocess = transforms.Normalize(mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225], std=[1/0.229, 1/0.224, 1/0.225])
    
    def forward(self, clean_images, texts):
        #self.unet.train()
        device = self.unet.device
        prompt_embeds = self.encode_text(texts, device=device)

        with torch.no_grad(): 
            clean_images = self.vae.encode(clean_images).latent_dist.sample() * self.vae.config.scaling_factor

        noise = torch.randn(clean_images.shape).to(device)
        bs = clean_images.shape[0]

        # Sample a random timestep for each image
        timesteps = torch.randint(
            0, self.noise_scheduler.config.num_train_timesteps, (bs,), device=device
        ).long()

        # forward diffusion 
        noisy_images =self.noise_scheduler.add_noise(
            clean_images.to(device), 
            noise, 
            timesteps
        )
        # predict noise
        noise_pred = self.unet(
            noisy_images, 
            encoder_hidden_states=prompt_embeds, 
            timestep=timesteps, 
            return_dict=False
        )[0]
        return noise_pred, noise

    def inference(
            self, 
            epoch, 
            texts: List[str], 
            output_dir,
            save=True,
            disable_pgbar=False
        ):
        # Sample some images from random noise (this is the backward diffusion process).
        # The default pipeline output type is `List[PIL.Image]`
        #self.unet.eval()
        device = self.unet.device
        prompt_embeds = self.encode_text(texts, device=device)
        if isinstance(self.unet.config.sample_size, int): 
            latent_shape = (len(texts), 
                            self.unet.config.in_channels, 
                            self.unet.config.sample_size//self.vae_downsample_factor, 
                            self.unet.config.sample_size//self.vae_downsample_factor)
        else:
            latent_shape = (len(texts), 
                            self.unet.config.in_channels, 
                            self.unet.config.sample_size[0]//self.vae_downsample_factor, 
                            self.unet.config.sample_size[1]//self.vae_downsample_factor)
        image = randn_tensor(latent_shape, device=device)

        with torch.no_grad():
            for t in tqdm(self.noise_scheduler.timesteps, disable=disable_pgbar):
                model_output = self.unet(image, encoder_hidden_states=prompt_embeds, timestep=t).sample
                image = self.noise_scheduler.step(model_output, t, image).prev_sample
            
            image = 1 / self.vae.config.scaling_factor * image
            image = self.vae.decode(image).sample #.clamp(0, 1)
        
        
        image = (image / 2 + 0.5).clamp(0, 1)
        #image = self.postprocess(image)
        
        #print("image.shape = ", image.shape)
        image = image.cpu().permute(0, 2, 3, 1).numpy() # (bs, h, w, c)
        num_placeholder = math.ceil(len(texts)/16) * 16 - len(texts)
        if num_placeholder:
            placeholder = np.ones((num_placeholder, *image.shape[1:]))
            image = np.concatenate((image, placeholder))
        image = numpy_to_pil(image)
        # Make a grid out of the images
        num_cols = math.ceil(len(texts)/16)
        image_grid = make_image_grid(image, rows=min(len(texts), 16), cols=num_cols)

        # Save the images
        if output_dir is not None and save:
            test_dir = os.path.join(output_dir, "samples")
            os.makedirs(test_dir, exist_ok=True)
            image_grid.save("{}/{}.png".format(test_dir, epoch))
            with open("{}/{}.txt".format(test_dir, epoch), "w") as f:
                f.write("\n".join(texts))
        else: return image_grid # type: PIL.Image.Image

