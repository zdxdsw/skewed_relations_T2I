import matplotlib.pyplot as plt
import numpy as np
import os, sys, random, json, argparse, re, torch
from pprint import pprint
from tqdm import tqdm
from collections import Counter, defaultdict
from transformers import ViTForImageClassification, ViTImageProcessor
from PIL import Image
np.set_printoptions(precision=4)

def process_gen_sample(pilimage, gth_caption, whichset):
    W, H = pilimage.size
    if W==H: 
        crop1 = pilimage.crop((W//4, 0, 3*W//4, H//2)) # behind
        crop2 = pilimage.crop((W//4, H//2, 3*W//4, H)) # front
        crop3 = pilimage.crop((0, H//4, W//2, 3*H//4)) # left
        crop4 = pilimage.crop((W//2, H//4, W, 3*H//4)) # right
    elif W == 2*H:
        crop3 = pilimage.crop((0, 0, W//2, H)) # left
        crop4 = pilimage.crop((W//2, 0, W, H)) # right
    elif H == 2*W:
        crop1 = pilimage.crop((0, 0, W, H//2)) # behind
        crop2 = pilimage.crop((0, H//2, W, H)) # front
    else: raise ValueError(f"Invalid image shape: width = {W}, height = {H}. Only support aspect ratio of 1, 0.5, 2")
    
    tmp = gth_caption.split()
    if whichset == "train":
        f1, f2 = tmp[1], tmp[-1]
        r = " ".join(tmp[2:-2])
    elif whichset == "test":
        f1, f2 = tmp[0], tmp[-1]
        r = " ".join(tmp[1:-1])
    else: raise ValueError(f"Invalid whichset: {whichset}")
    T = (f1, f2, r)

    if "left of" in r: labels, crops = [n2i[f1], n2i[f2]], [crop3, crop4]
    elif "right of" in r: labels, crops = [n2i[f2], n2i[f1]], [crop3, crop4]
    elif "in-front of" in r: labels, crops = [n2i[f2], n2i[f1]], [crop1, crop2]
    elif "behind" in r: labels, crops = [n2i[f1], n2i[f2]], [crop1, crop2]
    else: raise ValueError(f"Invalid relation: {r}")

    for i, c in enumerate(crops):
        c.save(f"../../notebooks/data/tmp_crop{i}.png")
    
    return [{
        "image": crop,
        "label": label,
    } for crop, label in zip(crops, labels)], T


parser = argparse.ArgumentParser()
parser.add_argument('--epochs_for_eval', type=str)
parser.add_argument('--split_for_eval', type=str, default="test train")
parser.add_argument('--ckpt_handle', type=str)
parser.add_argument('--output_folder', type=str, default="output")
parser.add_argument('--eval_batch_size', type=int, default=16)
parser.add_argument('--vit_for_autoeval_ckpt_dir', type=str, default="/data/yingshac/clevr_control/autoeval/")
parser.add_argument('--vit_for_autoeval_ckpt_name', type=str, default="vit-base-patch16-224-in21k_0311_211459.pt") #in21k_0303_182910
parser.add_argument('--device', type=int, default=0)
parser.add_argument('--print_errors', action='store_true')
args = parser.parse_args()

config = json.load(open(os.path.join(args.output_folder, args.ckpt_handle, "config.json"), "r"))

#classes = ['empty', 'mug', 'plate', 'book', 'bowl', 'can', 'cap', 'cup', 'remote', 'sunglasses', 
#          'tape', 'candle', 'flower', 'fork', 'headphones', 'scissors', 'spoon', 'knife', 'phone']
classes = ['empty', 'book', 'bowl', 'can', 'cap', 'cup', 'mug', 'plate', 'candle', 'flower', 'fork', 
           'headphones', 'knife', 'scissors', 'spoon', 'tape']
n2i = {n:i for i, n in enumerate(classes)}
device = f"cuda:{args.device}"
ckpt_dir = os.path.join(args.vit_for_autoeval_ckpt_dir, args.vit_for_autoeval_ckpt_name)


""" ---------- Prepare the finetuned ViT model for image classification as the auto eval engine ---------- """
print("Prepare ViTImageProcessor")
model_name_or_path = 'google/vit-base-patch16-224-in21k'
processor = ViTImageProcessor.from_pretrained(model_name_or_path)

print("Prepare metric")
from datasets import load_metric
metric = load_metric("accuracy")

model = ViTForImageClassification.from_pretrained(
    model_name_or_path,
    num_labels=len(classes),
    id2label={str(i): c for i, c in enumerate(classes)},
    label2id={c: str(i) for i, c in enumerate(classes)}
)

print("Load model")
state_dict = torch.load(ckpt_dir, map_location=device)
model.load_state_dict(state_dict)
model.to(device)
model.eval()

def get_acc(output, gth, verbose=False):
    output, gth = output.detach().cpu(), gth.detach().cpu()
    pred = np.argmax(output, axis=1)
    if verbose:
        print("pred = ", pred)
        print("gth  = ", gth)
    acc = metric.compute(predictions=pred, references=gth)['accuracy']
    p1, p2 = pred[0].item(), pred[1].item()
    g1, g2 = gth[0].item(), gth[1].item()
    if acc == 0.5:
        if p1 == 0 or p2 == 0: detail = "oneBlank"
        elif p1 == p2: detail = "duplication"
        else: detail = "oneWrong"
    elif acc == 0:
        if p1 == g2 and p2 == g1: detail = "flipOrder"
        elif p1 == p2 == 0: detail = "twoBlank"
        elif p1 * p2 == 0: detail = "oneBlankoneWrong"
        elif p1 == p2: detail = "twoWrongWithDuplication"
        else: detail = "twoWrong"
    else: detail = "correct"
    confusion = []
    if not p1==g1: confusion.append((g1, p1))
    if not p2==g2: confusion.append((g2, p2))
    return acc, detail, confusion

def collate_fn(batch):
    inputs = {}
    inputs['pixel_values'] = processor([x['image'] for x in batch], return_tensors='pt')['pixel_values']
    inputs['label'] = torch.LongTensor([x['label'] for x in batch])
    return inputs


def eval_epoch(model, eval_batch_size, whichset, pilimages, gth_captions, device):
    ACC = []
    outer_batch = []
    dataiter = iter(zip(pilimages, gth_captions))
    DETAIL = Counter()
    CONFUSION = np.zeros((len(classes), len(classes)))
    while True:
        try:
            image, text = next(dataiter)
            batch, T = process_gen_sample(image, text, whichset)
            outer_batch.extend(batch)
            if len(outer_batch) == 2*eval_batch_size:
                input_batch = collate_fn(outer_batch)
                labels = input_batch['label'].to(device)
                outputs = model(input_batch['pixel_values'].to(device))['logits']

                for b in range(eval_batch_size):
                    acc, detail, confusion = get_acc(outputs[2*b:2*b+2], labels[2*b:2*b+2], False)
                    ACC.append(acc)
                    DETAIL[detail] += 1
                    for i, j in confusion: CONFUSION[i][j] += 1
                
                outer_batch = []
        except StopIteration:
            if len(outer_batch):
                input_batch = collate_fn(outer_batch)
                labels = input_batch['label'].to(device)
                outputs = model(input_batch['pixel_values'].to(device))['logits']
                for b in range(len(outer_batch)//2):
                    acc, detail, confusion = get_acc(outputs[2*b:2*b+2], labels[2*b:2*b+2], False)
                    ACC.append(acc)
                    DETAIL[detail] += 1
                    for i, j in confusion: CONFUSION[i][j] += 1
            break

    return ACC, DETAIL, CONFUSION


""" ---------------------------------------- Run Evaluation! ---------------------------------------- """
ckpt_handle = args.ckpt_handle
for split_for_eval in args.split_for_eval.split():
    for target_epoch in [int(x) for x in args.epochs_for_eval.split()]:
        image_height, image_width = config['image_size']

        pilimages, gth_captions = [], []
        for infr_handle in os.listdir(f"{args.output_folder}/{ckpt_handle}/infr/{split_for_eval}_sentences"):
            epoch_for_eval = int(re.findall(r'epoch(\d+)', infr_handle)[0])
            if epoch_for_eval == target_epoch:
                samples_dir = f"{args.output_folder}/{ckpt_handle}/infr/{split_for_eval}_sentences/{infr_handle}/samples"
                for f in os.listdir(samples_dir):
                    if ".png" in f:
                        im = Image.open(os.path.join(samples_dir, f)).convert("RGB")
                        im_array = np.array(im, dtype=np.float64)
                        h, w, d = tuple(im_array.shape)

                        num_rols, num_cols = int(h // image_height), int(w // image_width)
                        bs = num_rols * num_cols
                        pixels = np.reshape(im_array, (num_rols, image_height, w, d))
                        pixels = np.reshape(pixels, (num_rols, image_height, num_cols, image_width, d))
                        pixels = np.transpose(pixels, (0, 2, 1, 3, 4))#(1, 3, 2, 4, 0)) #
                        pixels = np.reshape(pixels, (bs, image_height, image_width, d))

                        with open(os.path.join(samples_dir, f.replace(".png", ".txt")), "r") as txt:
                            input_texts = [x.strip() for x in txt.readlines()]
                        
                        pilimages.extend([Image.fromarray(p.astype('uint8'), 'RGB') for p in pixels][:len(input_texts)])
                        gth_captions.extend(input_texts)
                
        # max([np.sum(np.asarray(im)/(255*3*64*64)) for im in images]) # check no placeholder images sneaked in
        ACC, DETAIL, CONFUSION = eval_epoch(model, args.eval_batch_size, split_for_eval, pilimages, gth_captions, device)
        em, count = sum([a==1 for a in ACC]), len(ACC)
        error_analysis = {k: f"{round(DETAIL[k]*100/count, 1)}%" for k in sorted(list(DETAIL.keys()))}

        print(f"Epoch {target_epoch} {split_for_eval} acc = {em/count:.4f} ({em}/{count})")
        if args.print_errors: 
            print(error_analysis, "\n")
        if target_epoch - (target_epoch // 100)*100 >= 99: print()
    
    if args.print_errors:
        for i, c in enumerate(classes):
            print("{} has been mistakenly generated as: {}".format(c, [f"{classes[j]}: {CONFUSION[i][j]}" for j in range(len(classes)) if CONFUSION[i][j]>0]))
            np.save(open("confusion_matrices/{}/{}_{}.npy".format(
                split_for_eval,
                args.ckpt_handle,
                "_".join(args.epochs_for_eval.split()),
            ), "wb"), CONFUSION)