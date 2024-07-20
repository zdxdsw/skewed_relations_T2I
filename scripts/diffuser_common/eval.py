from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np
import os, sys, random, json, argparse, re
from pprint import pprint
from tqdm import tqdm
from collections import Counter, defaultdict
from PIL import Image
np.set_printoptions(precision=4)
from workspace.clevr_control.scripts.diffuser_colored_sq.training_utils import *
import textwrap


parser = argparse.ArgumentParser()
parser.add_argument('--epochs_for_eval', type=str, default="all")
parser.add_argument('--split_for_eval', type=str, default="train test")
parser.add_argument('--error_metric', type=str, default="mse")
parser.add_argument('--ckpt_handle', type=str)
args = parser.parse_args()

color2id = {k:i for i, k in enumerate(COLORS.keys())}
error_metric = eval(args.error_metric)
colors_array = np.array([hex_to_rgb(v) for v in COLORS.values()]) / 255

if not args.epochs_for_eval == "all":
    args.epochs_for_eval = [int(x) for x in args.epochs_for_eval.split()]

for split in args.split_for_eval.split():

    for infr_handle in os.listdir(f"output/{args.ckpt_handle}/infr/{split}_sentences/"):
        count = 0
        two_colors_match = 0
        em = 0
        epoch_for_eval = int(re.findall(r'epoch(\d+)', infr_handle)[0])
        if args.epochs_for_eval == "all" or epoch_for_eval in args.epochs_for_eval:

            samples_dir = f"output/{args.ckpt_handle}/infr/{split}_sentences/{infr_handle}/samples"

            #confusion_matrix = np.zeros((len(COLORS), len(COLORS)*2))
            for f in os.listdir(samples_dir):
                if ".png" in f:
                    with open(os.path.join(samples_dir, f.replace(".png", ".txt")), "r") as txt:
                        input_texts = [l.strip() for l in txt.readlines()]
                    im = Image.open(os.path.join(samples_dir, f))
                    im_array = np.array(im, dtype=np.float64) / 255
                    h, w, d = tuple(im_array.shape)
                    bs = int(h // w)
                    pixels = np.reshape(im_array, (bs, h//bs, w, d))
                    pixels = np.reshape(pixels, (bs, -1, d))
                    
                    for i, m in enumerate(pixels):
                        sentence = input_texts[i]
                        color_pair = tuple([w for w in sentence.strip().replace(".", "").split() if w in COLORS])
                        if "bottom" in sentence: color_pair = color_pair[::-1]
                        gth_top_color, gth_bottom_color = color_pair

                        kmeans = KMeans(n_clusters=3, n_init="auto", random_state=0).fit(m)
                        labels = kmeans.predict(m)
                        image_size = np.sqrt(labels.shape[0])
                        
                        x_coords, y_coords = defaultdict(list), defaultdict(list)
                        for i, l in enumerate(labels):
                            x_coords[l].append(i%image_size)
                            y_coords[l].append(image_size - i//image_size - 1)
                        cluster_centers = {
                            l: (np.mean(x_coords[l]), np.mean(y_coords[l]))
                            for l in labels
                        }
                        cluster_centers = dict(sorted([(k, v) for k, v in cluster_centers.items()], key=lambda x: x[1][1]))
                        bottom_cluster_id, background_cluster_id, top_cluster_id = list(cluster_centers.keys())
                        postproc = kmeans.cluster_centers_
                        
                        
                        error = error_metric([postproc[top_cluster_id]], colors_array)
                        top_color = list(COLORS.keys())[error.argmin()]
                        error = error_metric([postproc[bottom_cluster_id]], colors_array)
                        bottom_color = list(COLORS.keys())[error.argmin()]

                        count += 1
                        if gth_top_color == top_color and gth_bottom_color == bottom_color: 
                            em += 1
                            two_colors_match += 1
                            continue
                        elif gth_top_color == bottom_color and gth_bottom_color == top_color:
                            two_colors_match += 1 
                        # confusion_matrix[color2id[gth_top_color]]\
                        #     [color2id[top_color]+int(top_color == gth_bottom_color)*len(COLORS)] += 1
                        # confusion_matrix[color2id[gth_bottom_color]]\
                        #     [color2id[bottom_color]+int(bottom_color == gth_top_color)*len(COLORS)] += 1
            print(f"{args.ckpt_handle} epoch {epoch_for_eval}, {split} samples")
            print(f"Acc = {em/count:.2f} ({em}/{count})")
            print(f"colors_match = {two_colors_match/count:.2f} ({two_colors_match}/{count})")
