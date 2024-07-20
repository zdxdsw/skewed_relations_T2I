import sys, os
from workspace.clevr_control.scripts.diffuser_colored_sq.training_utils import *
from config import ConditionalTrainingConfig
import torch, random, pytz, json, argparse
from dataset import dataset
from torch.utils.data import DataLoader
from tqdm.auto import tqdm, trange
from model import T2IDiffusion
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np
import os, sys, random, json
from pprint import pprint
from tqdm import tqdm
from collections import Counter, defaultdict
from PIL import Image
np.set_printoptions(precision=4)
from pprint import pprint
import textwrap
from datetime import datetime
timezone = pytz.timezone('America/New_York') 

parser = argparse.ArgumentParser()
parser.add_argument('--load_from_dir', type=str)
parser.add_argument('--load_from_epochs', type=str)
parser.add_argument('--num_iter_train', type=int, default=1)
parser.add_argument('--num_iter_test', type=int, default=1)
args = parser.parse_args()


ckpt = args.load_from_dir
config = ConditionalTrainingConfig()
ckpt_config = json.load(open(os.path.join(config.output_dir, ckpt, "config.json"), "r"))
for k, v in ckpt_config.items():
    setattr(config, k, v)
ckpt_dir = os.path.join(config.output_dir, ckpt, "ckpts")

train_data = dataset(
    metadata_dir=config.metadata_dir, 
    image_dir =config.image_dir, 
    split = "train"
)
train_dataloader = DataLoader(train_data, batch_size=config.eval_batch_size, shuffle=False)

test_data = dataset(
    metadata_dir=config.metadata_dir, 
    image_dir =config.image_dir, 
    split = "test"
)
test_dataloader = DataLoader(test_data, batch_size=config.eval_batch_size, shuffle=False)

print(f"split_dir = {config.metadata_dir}")
train_sentences = set()
for d in train_data: train_sentences.add(d['sentence'])
test_sentences = set()
for d in test_data: test_sentences.add(d['sentence'])
train_sentences, test_sentences = list(train_sentences), list(test_sentences)
print(f"#train = {len(train_sentences)}, #test = {len(test_sentences)}")

model = T2IDiffusion(config)

args.load_from_epochs = [int(x) for x in args.load_from_epochs.split()]

for load_from_epoch in args.load_from_epochs:
    load_from_pt = None
    for f in os.listdir(ckpt_dir):
        if int(f.split("_")[0]) == load_from_epoch: 
            load_from_pt = f
            break
    print(load_from_pt)
    state_dict = torch.load(os.path.join(ckpt_dir, load_from_pt), map_location="cuda:0")
    model.unet.load_state_dict(state_dict)

    model = model.half()
    model.cuda()
    model.eval()

    # Inference and save images, without visualization
    date = datetime.now(timezone).strftime("%m%d_%H%M%S")

    if args.num_iter_train:
        save_infr_dir = os.path.join(config.output_dir, ckpt, f"infr/train_sentences/epoch{load_from_epoch}_{date}")
        os.makedirs(save_infr_dir, exist_ok=True)
        for n in trange(args.num_iter_train):
            for i in tqdm(range(len(train_sentences)//config.eval_batch_size+1)):
                texts = train_sentences[i*config.eval_batch_size:(i+1)*config.eval_batch_size]
                if not len(texts): break
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    model.inference(f"{i}_{n}", texts, save_infr_dir, disable_pgbar=True)
            
    if args.num_iter_test:
        save_infr_dir = os.path.join(config.output_dir, ckpt, f"infr/test_sentences/epoch{load_from_epoch}_{date}")
        os.makedirs(save_infr_dir, exist_ok=True)
        for n in trange(args.num_iter_test):
            for i in tqdm(range(len(test_sentences)//config.eval_batch_size+1)):
                texts = test_sentences[i*config.eval_batch_size:(i+1)*config.eval_batch_size]
                if not len(texts): break
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    model.inference(f"{i}_{n}", texts, save_infr_dir, disable_pgbar=True)