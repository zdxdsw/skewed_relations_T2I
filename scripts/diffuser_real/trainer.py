from config import ConditionalTrainingConfig
import torch, sys, os, random, pytz, json
from dataset import *
sys.path.append("../diffuser_colored_sq")
from torch.utils.data import DataLoader
from accelerate import Accelerator, DistributedDataParallelKwargs
from tqdm.auto import tqdm
import torch.nn.functional as F
from diffusers.optimization import get_constant_schedule_with_warmup, get_cosine_schedule_with_warmup
from model import T2IDiffusion
from training_utils import numpy_to_pil, cycle

from datetime import datetime
timezone = pytz.timezone('America/New_York') 
date = datetime.now(timezone).strftime("%m%d_%H%M%S")

config = ConditionalTrainingConfig()
config.date = date

""" Prepare Model """
model = T2IDiffusion(config)
config.encoder_hid_dim = model.encoder_hid_dim

ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
accelerator = Accelerator(
    mixed_precision=config.mixed_precision,
    gradient_accumulation_steps=config.gradient_accumulation_steps,
    log_with="tensorboard",
    kwargs_handlers=[ddp_kwargs],
    project_dir=os.path.join(config.output_dir, config.date),
)
accelerator.wait_for_everyone()
if accelerator.is_main_process:
    accelerator.init_trackers("tensorboard")
    os.makedirs(os.path.join(config.ckpt_dir, config.date, "ckpts"), exist_ok=True)
    # dump config
    C = {k:config.__getattribute__(k) for k in dir(config) if not k.startswith("__")}
    with open(os.path.join(config.output_dir, config.date, "config.json"), "w") as f:
        json.dump(C, f, indent=2)

if "trainable_parameters" in dir(config) and len(config.trainable_parameters):
    for n, p in model.named_parameters():
        if any([keyword in n for keyword in config.trainable_parameters]):
            p.requires_grad = True
        else: p.requires_grad = False

accelerator.print("#Trainable prameters = {}".format(sum(p.numel() for p in model.parameters() if p.requires_grad)))
# create optimizer, lr_scheduler
optimizer = torch.optim.AdamW(model.unet.parameters(), lr=config.learning_rate)

lr_scheduler = get_constant_schedule_with_warmup(
    optimizer=optimizer,
    num_warmup_steps=config.lr_warmup_steps * accelerator.num_processes / config.gradient_accumulation_steps,
)


""" Prepare Data """
annotations = json.load(open(config.annotations, "r"))

if not "subsample_method" in dir(config): config.subsample_method = None
train_data = eval(config.dataset_class)(
    config.imdir, annotations, imsize=config.image_size, 
    subsample_method=f"subsample_whatsup_{config.subsample_method}" if config.subsample_method is not None else None
    )
test_data = eval(config.dataset_class)(
    config.imdir, annotations, imsize=config.image_size, 
    subsample_method=f"subsample_whatsup_{config.subsample_method}" if config.subsample_method is not None else None
    ) # TODO: figure out whether do testing is worthy in the training loop

accelerator.print(f'Number of training examples: {len(train_data)}')
accelerator.print(f'Number of testing examples: {len(test_data)}')

train_dataloader = DataLoader(train_data, shuffle=True, batch_size=config.train_batch_size)
test_dataloader = DataLoader(test_data, shuffle=False, batch_size=config.eval_batch_size)

accelerator.print("Prepare Data: finish\n")

# Prepare everything
model, optimizer, train_dataloader, test_dataloader, lr_scheduler = accelerator.prepare(
    model, optimizer, train_dataloader, test_dataloader, lr_scheduler
)
test_dataiter = cycle(test_dataloader)
global_step = 0
global_epoch = 0

# Resume from ckpt
if "load_from_dir" in dir(config):
    ckpt_dir = os.path.join(config.ckpt_dir, config.load_from_dir, "ckpts")
    load_from_pt = sorted(os.listdir(ckpt_dir), key=lambda x: int(x.split("_")[0]))[-1]

    global_epoch = int(load_from_pt.split("_")[0]) + 1
    global_step = int(load_from_pt.split("_")[1]) + 1

    accelerator.print(f"resume from ckpt: {load_from_pt}\n\tepoch {global_epoch} step {global_step}")
    torch.cuda.set_device(accelerator.device)
    state_dict = torch.load(os.path.join(ckpt_dir, load_from_pt), map_location=accelerator.device)
    unet = accelerator.unwrap_model(model).unet
    unet.load_state_dict(state_dict, strict=False)

if "init_from_ckpt" in dir(config):
    init_from_ckpt = os.path.join(config.ckpt_dir, config.init_from_ckpt)
    accelerator.print(f"init from ckpt: {init_from_ckpt}")
    torch.cuda.set_device(accelerator.device)
    state_dict = torch.load(init_from_ckpt, map_location=accelerator.device)
    unet = accelerator.unwrap_model(model).unet
    unet.load_state_dict(state_dict, strict=False)


# Now you train the model
accelerator.print(f"Start job {config.date}")
progress_bar = tqdm(total=config.num_epochs-global_epoch, disable=not accelerator.is_local_main_process)
for epoch in range(global_epoch, config.num_epochs):
    #progress_bar.set_description(f"Epoch {epoch}/{config.num_epochs}")

    for step, batch in enumerate(train_dataloader):
        
        if global_step % config.save_image_steps == 0: 
            texts = next(test_dataiter)['sentence']
            #print(sentence)
            with accelerator.autocast():
                model_to_eval = accelerator.unwrap_model(model)
                model_to_eval.eval()
                model_to_eval.inference(
                    f"{epoch}_{global_step}", 
                    texts, 
                    os.path.join(config.output_dir, config.date),
                    save = accelerator.is_local_main_process,
                    disable_pgbar = not accelerator.is_local_main_process
                )
                model_to_eval.train()
        
        accelerator.wait_for_everyone()
        with accelerator.autocast() as autocast, torch.backends.cuda.sdp_kernel(enable_flash=False) as disable:
            optimizer.zero_grad()
            noise_pred, noise = model(
                batch['image'],
                batch['sentence']
            )
            loss = F.mse_loss(noise_pred, noise)
            accelerator.backward(loss / config.gradient_accumulation_steps)

            if (global_step+1) % config.gradient_accumulation_steps == 0:
                accelerator.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                lr_scheduler.step()
            

        
        logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0], "step": global_step}
        progress_bar.set_postfix(**logs)
        accelerator.log(logs, step=global_step)
        global_step += 1
        #break
    progress_bar.update(1)
    if accelerator.is_main_process and (epoch+1) % config.save_model_epochs == 0:
        save_path = os.path.join(config.ckpt_dir, config.date, f"ckpts/{epoch}_{global_step}_unet.pt")
        torch.save(accelerator.unwrap_model(model).unet.state_dict(), save_path)
    if epoch == config.num_epochs - 1:
        texts = next(test_dataiter)['sentence']
        with accelerator.autocast():
            model_to_eval = accelerator.unwrap_model(model)
            model_to_eval.eval()
            model_to_eval.inference(
                f"{epoch}_{global_step}", 
                texts, 
                os.path.join(config.output_dir, config.date),
                save = accelerator.is_local_main_process,
                disable_pgbar = not accelerator.is_local_main_process
            )
            model_to_eval.train()
    accelerator.wait_for_everyone()

accelerator.print(f"Finish!!! {config.date}")
