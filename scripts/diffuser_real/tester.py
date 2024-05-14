import sys, os
sys.path.append("../diffuser_colored_sq/")
from training_utils import numpy_to_pil, cycle
from config import ConditionalTrainingConfig
import torch, random, pytz, json, argparse
from dataset import *
from torch.utils.data import DataLoader
from tqdm.auto import tqdm, trange
from model import T2IDiffusion
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
timezone = pytz.timezone('America/New_York') 

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
ckpt_config = json.load(open(os.path.join(config.output_dir, ckpt, "config.json"), "r"))
for k, v in ckpt_config.items():
    setattr(config, k, v)
if not "subsample_method" in ckpt_config: config.subsample_method = None
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
annotations = json.load(open(config.annotations, "r"))

train_data = real_dataset(
    config.imdir, annotations, imsize=config.image_size, 
    subsample_method=f"subsample_whatsup_{config.subsample_method}" if config.subsample_method is not None else None
)

test_tuples = get_test_tuples(train_data.data)
test_data = real_dataset(config.imdir, test_tuples, imsize=config.image_size)

accelerator.print(f'Number of training examples: {len(train_data)}')
accelerator.print(f'Number of testing examples: {len(test_data)}')
#train_sentences = [x[0] for x in train_pairs]
#test_sentences = [x[0] for x in test_pairs]

train_dataloader = DataLoader(train_data, shuffle=True, batch_size=config.eval_batch_size)
test_dataloader = DataLoader(test_data, shuffle=False, batch_size=config.eval_batch_size)

accelerator.print("Prepare Data: finish\n")


model = T2IDiffusion(config)
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
