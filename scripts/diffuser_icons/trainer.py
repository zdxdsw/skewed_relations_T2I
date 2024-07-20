from config import ConditionalTrainingConfig
import torch, sys, os, random, pytz, json, argparse, time
sys.path.append("../diffuser_common/")
from dataset import *
from torch.utils.data import DataLoader
from accelerate import Accelerator, DistributedDataParallelKwargs
from tqdm.auto import tqdm, trange
import torch.nn.functional as F
from diffusers.optimization import get_constant_schedule_with_warmup, get_cosine_schedule_with_warmup
from model import T2IDiffusion, T2ILatentDiffusion
from training_utils import *

from datetime import datetime
timezone = pytz.timezone('America/Los_Angeles') 
date = datetime.now(timezone).strftime("%m%d_%H%M%S")

config = ConditionalTrainingConfig()
config.date = date

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


"""------------- Sleep if needed -------------"""
parser = argparse.ArgumentParser()
parser.add_argument('--sleep', type=int)
args = parser.parse_args()

if args.sleep is not None:
    for i in trange(args.sleep, desc="Sleeping"):
        time.sleep(60)


""" Prepare Model """
if "vae_weights_dir" not in dir(config) or config.vae_weights_dir is None:
    model = T2IDiffusion(config) 
    accelerator.print("------------------------ create T2IDiffusion model ------------------------")
else:
    model = T2ILatentDiffusion(config)
    model.vae.requires_grad_(False)
    accelerator.print("------------------------ create T2ILatentDiffusion model ------------------------")
    
config.encoder_hid_dim = model.encoder_hid_dim

if "trainable_parameters" in dir(config) and len(config.trainable_parameters):
    for n, p in model.named_parameters():
        if any([keyword in n for keyword in config.trainable_parameters]):
            p.requires_grad = True
        else: p.requires_grad = False

accelerator.print("#Trainable prameters = {}".format(sum(p.numel() for p in model.parameters() if p.requires_grad)))
# create optimizer, lr_scheduler
optimizer = torch.optim.AdamW(model.unet.parameters(), lr=config.learning_rate)
#lr_scheduler = get_cosine_schedule_with_warmup(
#    optimizer=optimizer,
#    num_warmup_steps=config.lr_warmup_steps * accelerator.num_processes,
#    num_training_steps=len(train_dataloader) * config.num_epochs * accelerator.num_processes,
#)
lr_scheduler = get_constant_schedule_with_warmup(
    optimizer=optimizer,
    num_warmup_steps=config.lr_warmup_steps * accelerator.num_processes / config.gradient_accumulation_steps,
)


""" Prepare Data """
with open(config.nouns_file, "r") as f: nouns = [l.strip() for l in f.readlines()]
with open(config.icons_file, "r", encoding="unicode-escape") as f: 
    icons = [(json.loads(x)[0], json.loads(x)[2]) for x in f.readlines()]
if "max_num_objs" in dir(config): 
    nouns = nouns[:config.max_num_objs]
    icons = icons[:config.max_num_objs]


if "custom" in config.split_method:
    train_pairs, test_pairs = eval(f"create_data_{config.split_method}")
else: 
    draw_icon_font_size = config.draw_icon_font_size if "draw_icon_font_size" in dir(config) else 28 # for backward compatibility
    train_pairs, test_pairs = eval(f"create_data_{config.split_method}")(
        nouns, 
        icons,
        canvas_size=config.image_size,
        icon_size=min(config.image_size),
        fontsize=draw_icon_font_size
    )

train_data = icons_dataset(train_pairs)
test_data = icons_dataset(test_pairs)

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
