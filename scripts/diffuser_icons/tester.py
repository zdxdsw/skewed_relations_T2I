import sys, os, warnings
sys.path.append("../diffuser_colored_sq/")
from training_utils import *
from config import ConditionalTrainingConfig, default_ConditionalTrainingConfig
import torch, random, pytz, json, argparse
from dataset import *
from torch.utils.data import DataLoader
from tqdm.auto import tqdm, trange
from model import T2IDiffusion, T2ILatentDiffusion
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np
import os, sys, random, json
from tqdm import tqdm
from collections import Counter, defaultdict
from accelerate import Accelerator, DistributedDataParallelKwargs
from PIL import Image
np.set_printoptions(precision=4)
from pprint import pprint
import textwrap
from datetime import datetime
timezone = pytz.timezone('America/Los_Angeles') 

parser = argparse.ArgumentParser()
parser.add_argument('--load_from_dir', type=str)
parser.add_argument('--load_from_epochs', type=str)
parser.add_argument('--eval_batch_size', type=int)
parser.add_argument('--num_iter_train', type=int, default=1)
parser.add_argument('--num_iter_test', type=int, default=1)
parser.add_argument('--split_method', type=str)
parser.add_argument('--image_size', type=str)
parser.add_argument('--obj_indices', type=str)
args = parser.parse_args()


ckpt = args.load_from_dir
config = ConditionalTrainingConfig()
default_config = default_ConditionalTrainingConfig()

ckpt_config = json.load(open(os.path.join(config.output_dir, ckpt, "config.json"), "r"))
#for k, v in ckpt_config.items():
    #setattr(config, k, v)

config_keys = dir(config)
for k in config_keys:
    if k.startswith("__"): continue
    if k in ckpt_config: setattr(config, k, ckpt_config[k])
    else:
        setattr(config, k, default_config.__getattribute__(k))
        warnings.warn(f"Cannot find {k} in the resume_from_config. Set to {default_config.__getattribute__(k)} by default.")


if args.eval_batch_size is not None: config.eval_batch_size = args.eval_batch_size
if args.split_method is not None: 
    config.split_method = args.split_method
    config.image_size = eval(args.image_size)
    split_method_prefix = config.split_method+"_"
else:
    split_method_prefix = ""
if not os.path.exists("/data/yingshac"): config.ckpt_dir = config.output_dir
ckpt_dir = os.path.join(config.ckpt_dir, ckpt, "ckpts")


ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
accelerator = Accelerator(
    mixed_precision=config.mixed_precision,
    gradient_accumulation_steps=config.gradient_accumulation_steps,
    log_with="tensorboard",
    kwargs_handlers=[ddp_kwargs],
    project_dir=os.path.join(config.output_dir, config.date),
    even_batches=False,
)
accelerator.wait_for_everyone()

""" Prepare Data """
with open(config.nouns_file, "r") as f: nouns = [l.strip() for l in f.readlines()]
with open(config.icons_file, "r", encoding="unicode-escape") as f: 
    lines = f.readlines()
    icons = [(json.loads(x)[0], json.loads(x)[2]) for x in lines]
    icons_names = [json.loads(x)[1] for x in lines]
nouns_to_names = {n:m for n, m in zip(nouns, icons_names)}

if args.obj_indices is not None:
    s, e = eval(args.obj_indices)
    accelerator.print(f"Use objects #{s}-{e}")
    split_method_prefix += f"{s}_{e}_"
    nouns = nouns[s:e]
    icons = icons[s:e]
elif "max_num_objs" in dir(config): 
    nouns = nouns[:config.max_num_objs]
    icons = icons[:config.max_num_objs]

if "custom" in config.split_method:
    train_pairs, test_pairs = eval(f"create_data_{config.split_method}")
else: train_pairs, test_pairs = eval(f"create_data_{config.split_method}")(nouns, icons)
train_data = icons_dataset(train_pairs)
test_data = icons_dataset(test_pairs)

accelerator.print(f'Number of training examples: {len(train_data)}')
accelerator.print(f'Number of testing examples: {len(test_data)}')
#train_sentences = [x[0] for x in train_pairs]
#test_sentences = [x[0] for x in test_pairs]

train_dataloader = DataLoader(train_data, shuffle=True, batch_size=config.eval_batch_size)
test_dataloader = DataLoader(test_data, shuffle=False, batch_size=config.eval_batch_size)

accelerator.print("Prepare Data: finish\n")


""" Prepare Model """
if "vae_weights_dir" not in dir(config) or config.vae_weights_dir is None:
    model = T2IDiffusion(config) 
    accelerator.print("------------------------ create T2IDiffusion model ------------------------")
else:
    model = T2ILatentDiffusion(config)
    model.vae.requires_grad_(False)
    accelerator.print("------------------------ create T2ILatentDiffusion model ------------------------")

model, train_dataloader, test_dataloader = accelerator.prepare(model, train_dataloader, test_dataloader)

args.load_from_epochs = [int(x) for x in args.load_from_epochs.split()]

for load_from_epoch in args.load_from_epochs:
    load_from_pt = None
    for f in os.listdir(ckpt_dir):
        if int(f.split("_")[0]) == load_from_epoch: 
            load_from_pt = f
            break

    accelerator.print(f"load from ckpt: {load_from_pt}")
    torch.cuda.set_device(accelerator.device)
    state_dict = torch.load(os.path.join(ckpt_dir, load_from_pt), map_location=accelerator.device)#"cuda:0")
    unet = accelerator.unwrap_model(model).unet
    unet.load_state_dict(state_dict, strict=False)

    model = model.half()
    model.cuda()
    model.eval()
    accelerator.wait_for_everyone()

    # Inference and save images, without visualization
    date = datetime.now(timezone).strftime("%m%d_%H%M%S")

    if args.num_iter_train:
        save_infr_dir = os.path.join(config.output_dir, ckpt, f"infr/{split_method_prefix}train_sentences/epoch{load_from_epoch}_{date}")
        os.makedirs(save_infr_dir, exist_ok=True)
        for n in trange(args.num_iter_train):
            #for i in tqdm(range(len(train_sentences)//config.eval_batch_size+1)):
                #texts = train_sentences[i*config.eval_batch_size:(i+1)*config.eval_batch_size]
                #if not len(texts): break
            for step, batch in enumerate(train_dataloader):
                if batch is None: break
                texts = batch['sentence']
                #with torch.autocast(device_type="cuda", dtype=torch.float16):
                with accelerator.autocast():
                    model_to_eval = accelerator.unwrap_model(model)
                    model_to_eval.eval()
                    model_to_eval.inference(f"{step}_{n}_{Accelerator().process_index}", 
                                            texts, 
                                            save_infr_dir, 
                                            disable_pgbar=not accelerator.is_local_main_process)
                
                with open(os.path.join(save_infr_dir, f"samples/{step}_{n}_{Accelerator().process_index}.txt"), "a") as f:
                    f.write("\n===================================\n")
                    for t in texts:
                        x = t.replace(".", "").split()
                        y = []
                        for w in x:
                            if w in nouns: y.append(nouns_to_names[w])
                            else: y.append(w)
                        x = " ".join(y) + "."
                        f.write(x+"\n")

            
    if args.num_iter_test:
        save_infr_dir = os.path.join(config.output_dir, ckpt, f"infr/{split_method_prefix}test_sentences/epoch{load_from_epoch}_{date}")
        os.makedirs(save_infr_dir, exist_ok=True)
        for n in trange(args.num_iter_test):
            # for i in tqdm(range(len(test_sentences)//config.eval_batch_size+1)):
            #     texts = test_sentences[i*config.eval_batch_size:(i+1)*config.eval_batch_size]
            #     if not len(texts): break
            for step, batch in enumerate(test_dataloader):
                if batch is None: break
                texts = batch['sentence']
                #print(texts)
                #with torch.autocast(device_type="cuda", dtype=torch.float16):
                with accelerator.autocast():
                    model_to_eval = accelerator.unwrap_model(model)
                    model_to_eval.eval()
                    model_to_eval.inference(f"{step}_{n}_{Accelerator().process_index}", 
                                            texts, 
                                            save_infr_dir, 
                                            disable_pgbar=not accelerator.is_local_main_process)
                with open(os.path.join(save_infr_dir, f"samples/{step}_{n}_{Accelerator().process_index}.txt"), "a") as f:
                    f.write("\n===================================\n")
                    for t in texts:
                        x = t.replace(".", "").split()
                        y = []
                        for w in x:
                            if w in nouns: y.append(nouns_to_names[w])
                            else: y.append(w)
                        x = " ".join(y) + "."
                        f.write(x+"\n")