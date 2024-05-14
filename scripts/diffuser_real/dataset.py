import os, json, random, sys
sys.path.append("../formalism")
from entropy import Transpose
from torchvision import transforms
from torch.utils.data import Dataset
from PIL import Image
from typing import Any, Dict, Optional, Tuple, Union, List
from tqdm import tqdm, trange
from collections import defaultdict, Counter
#from textblob import Word
import numpy as np

def get_test_tuples(train_data):
    relations = list(set([d[-1][-1] for d in train_data]))
    nouns = set([d[-1][0] for d in train_data] + [d[-1][1] for d in train_data])
    #nouns = ["mug", "plate", "book", "bowl", "can", "cap", "cup", "remote", "sunglasses", "tape", "candle", "flower", "fork", "headphones", "scissors", "spoon", "knife", "phone"]
    tuples = [tuple(d[-1]) for d in train_data]
    test_tuples = []
    for subj in nouns:
        for obj in nouns:
            if subj == obj: continue
            for r in relations:
                if not (subj, obj, r) in tuples: test_tuples.append((subj, obj, r))
    return test_tuples

class real_dataset(Dataset):
    def __init__(self,
                 imdir: str,
                 data: List,
                 imsize = (64, 64),
                 subsample_method = None,
                 ):
        super().__init__()
        self.imdir = imdir
        self.data = data
        self.preprocess = transforms.Compose(
            [   
                transforms.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
                transforms.Resize(imsize),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
        occurrences = [a[-1][0] for a in data] + [a[-1][1] for a in data]
        c = Counter(occurrences)
        self.classes = sorted(c.keys(), key=lambda x: (-c[x], x))
        
        if subsample_method is not None:
            R = set([a[-1][-1] for a in self.data])
            if R == set(["left of", "right of"]): 
                allowed_train_tuples = eval(subsample_method)(self.classes, "lr")
                self.data = [d for d in self.data if tuple(d[-1]) in allowed_train_tuples]
            elif R == set(["in-front of", "behind"]):
                allowed_train_tuples = eval(subsample_method)(self.classes, "fb")
                self.data = [d for d in self.data if tuple(d[-1]) in allowed_train_tuples]
            else: raise ValueError(f"customized subsample method is only allowed for two symmetric relations. Received R = {R}")

        
    def __len__(self): return len(self.data)
    
    def __getitem__(self, i): 
        if isinstance(self.data[i], list):
            text, image_path, tuples = self.data[i]
            image = Image.open(os.path.join(self.imdir, image_path))
            width, height = image.size
            new_dimension = min(image.size)
            left = (width - new_dimension)/2
            top = (height - new_dimension)/2
            right = (width + new_dimension)/2
            bottom = (height + new_dimension)/2
            image = image.crop((left, top, right, bottom))
            #print(image.size)
            W, H = image.size
            assert W==H

            r = tuples[-1]
            if r in ["left of", "right of"]: image = image.crop((0, H//4, W, 3*H//4))
            elif r in ["in-front of", "behind"]: image = image.crop((W//4, 0, 3*W//4, H))
            
            return {
                'image': self.preprocess(image),
                'sentence': text
            }
        else:
            subj, obj, r = self.data[i]
            text = " ".join([subj, r, obj])
            return {'sentence': text}


def subsample_whatsup_splitA(nouns, relations, transpose=False):
    # Incomplete, worse than splitB, larger coverage than split
    if relations == "lr": relations = ['left of', 'right of']
    elif relations == "fb": relations = ["in-front of", "behind"]
    else: raise ValueError(f"Invalid relations: {relations}")

    train_tuples = [] # (f1, f2, r)
    for i in nouns:
        for j in nouns:
            if i==j: continue
            if i in ["scissors", "spoon", "knife"]: continue
            if j in ["tape", "candle", "flower", "fork"]: continue
            train_tuples.append((i, j, relations[0])) 
            train_tuples.append((i, j, relations[1])) 

    if transpose: train_tuples = Transpose(train_tuples, apply_to_relations=["right of"])
    return train_tuples


def subsample_whatsup_splitB(nouns, relations, transpose=False):
    # Incomplete
    if relations == "lr": relations = ['left of', 'right of']
    elif relations == "fb": relations = ["in-front of", "behind"]
    else: raise ValueError(f"Invalid relations: {relations}")

    train_tuples = [] # (f1, f2, r)
    for i in nouns:
        for j in nouns:
            if i==j: continue
            if i in ["book", "bowl"]: continue
            if j in ["mug", "plate"]: continue
            train_tuples.append((i, j, relations[0])) 
            train_tuples.append((i, j, relations[1])) 

    if transpose: train_tuples = Transpose(train_tuples, apply_to_relations=["right of"])
    return train_tuples


def subsample_whatsup_splitC(nouns, relations, transpose=False):
    # Complete but UnBalanced, same coverage as splitA & B
    if relations == "lr": relations = ['left of', 'right of']
    elif relations == "fb": relations = ["in-front of", "behind"]
    else: raise ValueError(f"Invalid relations: {relations}")

    train_tuples = [] # (f1, f2, r)
    for i in nouns:
        for j in nouns:
            if i==j: continue
            if i in ["mug", "plate", "book"]:
                train_tuples.append((i, j, relations[0])) 
                train_tuples.append((i, j, relations[1])) 
            elif j in ["bowl", "can", "cap", "cup"]:
                train_tuples.append((i, j, relations[0])) 
                train_tuples.append((i, j, relations[1])) 
            elif i in ["tape", "candle", "flower", "fork"] and j in ["mug", "plate", "book"]:
                train_tuples.append((i, j, relations[0])) 
                train_tuples.append((i, j, relations[1])) 

    if transpose: train_tuples = Transpose(train_tuples, apply_to_relations=["right of"])
    return train_tuples


def subsample_whatsup_splitD(nouns, relations, transpose=False):
    # Complete and Balanced, with moderate coverage, mirror split22
    if relations == "lr": relations = ['left of', 'right of']
    elif relations == "fb": relations = ["in-front of", "behind"]
    else: raise ValueError(f"Invalid relations: {relations}")

    train_tuples = [] # (f1, f2, r)
    margin = len(nouns) // 3
    for idx1, i in enumerate(nouns):
        for idx2, j in enumerate(nouns):
            if i==j: continue
            if 0 < np.abs(idx1 - idx2) <= margin or np.abs(idx1 - idx2) >= len(nouns) - margin:
                train_tuples.append((i, j, relations[0])) 
                train_tuples.append((i, j, relations[1])) 
                
    if transpose: train_tuples = Transpose(train_tuples, apply_to_relations=["right of"])
    return train_tuples


def subsample_whatsup_splitE(nouns, relations, transpose=False):
    # Complete but UnBalanced, more severe imbalance than splitC
    if relations == "lr": relations = ['left of', 'right of']
    elif relations == "fb": relations = ["in-front of", "behind"]
    else: raise ValueError(f"Invalid relations: {relations}")

    train_tuples = [] # (f1, f2, r)
    for idx1, i in enumerate(nouns):
        for idx2, j in enumerate(nouns):
            if i==j: continue
            if len(nouns)-3 <= idx1+idx2 <= len(nouns):
                train_tuples.append((i, j, relations[0])) 
                train_tuples.append((i, j, relations[1])) 
            elif idx1+idx2 > len(nouns) and idx1 > idx2:
                train_tuples.append((i, j, relations[0])) 
                train_tuples.append((i, j, relations[1])) 
            elif idx1+idx2 < len(nouns)-3 and idx1 < idx2:
                train_tuples.append((i, j, relations[0])) 
                train_tuples.append((i, j, relations[1])) 
                
    if transpose: train_tuples = Transpose(train_tuples, apply_to_relations=["right of"])
    return train_tuples


def subsample_whatsup_splitF(nouns, relations, transpose=False):
    # Mirror split8, complete and balanced
    if relations == "lr": relations = ['left of', 'right of']
    elif relations == "fb": relations = ["in-front of", "behind"]
    else: raise ValueError(f"Invalid relations: {relations}")

    train_tuples = [] # (f1, f2, r)
    for idx1, i in enumerate(nouns):
        for idx2, j in enumerate(nouns):
            if i==j: continue
            if idx1+idx2 >= len(nouns): continue
            train_tuples.append((i, j, relations[0])) 
            train_tuples.append((i, j, relations[1])) 

    if transpose: train_tuples = Transpose(train_tuples, apply_to_relations=["right of"])
    return train_tuples


def subsample_whatsup_splitH(nouns, relations, transpose=False):
    # Complete but UnBalanced, same coverage as splitA & B
    if relations == "lr": relations = ['left of', 'right of']
    elif relations == "fb": relations = ["in-front of", "behind"]
    else: raise ValueError(f"Invalid relations: {relations}")

    train_tuples = [] # (f1, f2, r)
    for i in nouns:
        for j in nouns:
            if i==j: continue
            if i in ["bowl", "book"]:
                train_tuples.append((i, j, relations[0])) 
                train_tuples.append((i, j, relations[1])) 
            elif j in ["plate", "mug"]:
                train_tuples.append((i, j, relations[0])) 
                train_tuples.append((i, j, relations[1])) 
            elif i in ["book", "bowl", "can", "cap", "cup", "mug", "plate"] and j in ["candle", "flower", "fork", "headphones", "knife", "scissors", "spoon", "tape"]:
                train_tuples.append((i, j, relations[0])) 
                train_tuples.append((i, j, relations[1])) 
    
    if transpose: train_tuples = Transpose(train_tuples, apply_to_relations=["right of"])
    return train_tuples


# Experiments before 0305 7pm used this version
# def subsample_whatsup_splitG(nouns, relations, transpose=False):
#     # Incomplete, mirror split13
#     if relations == "lr": relations = ['left of', 'right of']
#     elif relations == "fb": relations = ["in-front of", "behind"]
#     else: raise ValueError(f"Invalid relations: {relations}")

#     train_tuples = [] # (f1, f2, r)
#     for i in nouns:
#         for j in nouns:
#             if i==j: continue
#             if j in ["candle", "flower", "fork", "headphones", "knife", "scissors", "spoon", "tape", ]: continue
#             train_tuples.append((i, j, relations[0])) 
#             train_tuples.append((i, j, relations[1])) 

#     if transpose: train_tuples = Transpose(train_tuples, apply_to_relations=["right of"])
#     return train_tuples

def subsample_whatsup_splitG(nouns, relations, transpose=False):
    # Incomplete, mirror split13
    if relations == "lr": relations = ['left of', 'right of']
    elif relations == "fb": relations = ["in-front of", "behind"]
    else: raise ValueError(f"Invalid relations: {relations}")

    train_tuples = [] # (f1, f2, r)
    for i in nouns:
        for j in nouns:
            if i==j: continue
            if i in ["candle", "flower", "fork", "headphones", "knife", "scissors", "spoon", "tape", ]: continue
            train_tuples.append((i, j, relations[0])) 
            train_tuples.append((i, j, relations[1])) 

    if transpose: train_tuples = Transpose(train_tuples, apply_to_relations=["right of"])
    return train_tuples


def subsample_whatsup_splitI(nouns, relations, transpose=False):
    # Complete and Balanced, mirror split23
    # Incomplete, mirror split9
    if relations == "lr": relations = ['left of', 'right of']
    elif relations == "fb": relations = ["in-front of", "behind"]
    else: raise ValueError(f"Invalid relations: {relations}")

    train_tuples = [] # (f1, f2, r)
    margin = len(nouns) // 2
    half_margin = len(nouns) // 4
    for idx1, i in enumerate(nouns):
        for idx2, j in enumerate(nouns):
            if i==j: continue
            #print(idx1-idx2, idx2-idx1, len(nouns))
            if 0 < idx1 - idx2 <= half_margin or len(nouns)-half_margin <= idx1 - idx2 < len(nouns): 
                train_label = 0
            elif half_margin < idx1 - idx2 <= margin or len(nouns)-margin <= idx1 - idx2 < len(nouns)-half_margin:
                train_label = 1
            elif 0 < idx2 - idx1 <= half_margin or len(nouns)-half_margin <= idx2 - idx1 < len(nouns):
                train_label = 1
            elif half_margin < idx2 - idx1 <= margin or len(nouns)-margin <= idx2 - idx1 < len(nouns)-half_margin:
                train_label = 0
            else: continue
            #print(train_label)
            train_tuples.append((i, j, relations[train_label]))
            
    train_tuples = Transpose(train_tuples, apply_to_relations=["right of"])
    return train_tuples


def subsample_whatsup_splitJ(nouns, relations, transpose=False):
    # Complete and Balanced, mirror split9
    if relations == "lr": relations = ['left of', 'right of']
    elif relations == "fb": relations = ["in-front of", "behind"]
    else: raise ValueError(f"Invalid relations: {relations}")

    half = len(nouns) // 2
    train_tuples = [] # (f1, f2, r)
    for idx1, i in enumerate(nouns):
        for idx2, j in enumerate(nouns):
            if i==j: continue
            if 0 <= idx1 < half and half <= idx2 < len(nouns): 
                train_tuples.append((i, j, relations[0])) 
                train_tuples.append((i, j, relations[1])) 
            elif half <= idx1 < len(nouns) and 0 <= idx2 < half: 
                train_tuples.append((i, j, relations[0])) 
                train_tuples.append((i, j, relations[1])) 

    if transpose: train_tuples = Transpose(train_tuples, apply_to_relations=["right of"])
    return train_tuples


def subsample_whatsup_splitK(nouns, relations, transpose=False):
    if relations == "lr": relations = ['left of', 'right of']
    elif relations == "fb": relations = ["in-front of", "behind"]
    else: raise ValueError(f"Invalid relations: {relations}")

    half = len(nouns) // 2
    train_tuples = [] # (f1, f2, r)
    for idx1, i in enumerate(nouns):
        for idx2, j in enumerate(nouns):
            if i==j: continue
            if -2 <= idx1 - idx2 <= 2 or np.abs(idx1 - idx2) == len(nouns)-1: 
                train_tuples.append((i, j, relations[0])) 
                train_tuples.append((i, j, relations[1])) 
                continue
            elif np.abs(idx1 - idx2) == len(nouns)-2:
                train_label = 1
            elif np.abs(idx1 - idx2) %2: # odd
                train_label = 0
            else:
                train_label = 1
            train_tuples.append((i, j, relations[train_label])) 

    train_tuples = Transpose(train_tuples, apply_to_relations=["right of"])
    return train_tuples


def subsample_whatsup_splitU(nouns, relations, transpose=False):
    return subsample_whatsup_splitA(nouns, relations, True)

def subsample_whatsup_splitV(nouns, relations, transpose=False):
    return subsample_whatsup_splitB(nouns, relations, True)

def subsample_whatsup_splitW(nouns, relations, transpose=False):
    return subsample_whatsup_splitC(nouns, relations, True)

def subsample_whatsup_splitX(nouns, relations, transpose=False):
    return subsample_whatsup_splitE(nouns, relations, True)

def subsample_whatsup_splitZ(nouns, relations, transpose=False):
    return subsample_whatsup_splitG(nouns, relations, True)

def subsample_whatsup_splitT(nouns, relations, transpose=False):
    return subsample_whatsup_splitH(nouns, relations, True)

"""
def subsample_whatsup_splitA(nouns, relations, transpose=False):
    # Complete but UnBalanced, same coverage as splitB, larger coverage than splitC
    if relations == "lr": relations = ['left of', 'right of']
    elif relations == "fb": relations = ["in-front of", "behind"]
    else: raise ValueError(f"Invalid relations: {relations}")

    train_tuples = [] # (f1, f2, r)
    for i in nouns:
        for j in nouns:
            if i==j: continue
            if i in ["mug", "plate", "book", "bowl", "can", "phone"] or j in ["cap", "cup", "remote", "sunglasses", "tape"]:
                train_tuples.append((i, j, relations[0])) 
                train_tuples.append((i, j, relations[1])) 
            
            # -------------------- This extra chunk can be removed --------------------
            if i in ["cap", "cup"] and not j in ["mug", "plate", "book", "bowl", "can"]:
                train_tuples.append((i, j, relations[0])) 
                train_tuples.append((i, j, relations[1])) 
            # --------------------------------------------------------------------------

    if transpose: train_tuples = Transpose(train_tuples, apply_to_relations=["right of"])
    return train_tuples


def subsample_whatsup_splitB(nouns, relations, transpose=False):
    # Incomplete, worse than splitC, larger coverage than splitC
    # missing cols ["tape", "candle", "flower", "fork"], missing rows ["headphones", "scissors", "spoon", "knife"]
    if relations == "lr": relations = ['left of', 'right of']
    elif relations == "fb": relations = ["in-front of", "behind"]
    else: raise ValueError(f"Invalid relations: {relations}")

    train_tuples = [] # (f1, f2, r)
    for i in nouns:
        for j in nouns:
            if i==j: continue
            if i in ["tape", "candle", "flower", "fork"]: continue
            if j in ["headphones", "scissors", "spoon", "knife"]: continue
            train_tuples.append((i, j, relations[0])) 
            train_tuples.append((i, j, relations[1])) 

    if transpose: train_tuples = Transpose(train_tuples, apply_to_relations=["right of"])
    return train_tuples


def subsample_whatsup_splitC(nouns, relations, transpose=False):
    # Incomplete
    # missing cols ["candle", "flower", "fork"], missing rows ["can", "cap", "cup"]
    if relations == "lr": relations = ['left of', 'right of']
    elif relations == "fb": relations = ["in-front of", "behind"]
    else: raise ValueError(f"Invalid relations: {relations}")

    train_tuples = [] # (f1, f2, r)
    for i in nouns:
        for j in nouns:
            if i==j: continue
            if i in ["candle", "tape", "sunglasses"]: continue
            if j in ["can", "cap", "cup"]: continue
            train_tuples.append((i, j, relations[0])) 
            train_tuples.append((i, j, relations[1])) 

    if transpose: train_tuples = Transpose(train_tuples, apply_to_relations=["right of"])
    return train_tuples


def subsample_whatsup_splitD(nouns, relations, transpose=False):
    # Complete and Balanced, with moderate coverage
    if relations == "lr": relations = ['left of', 'right of']
    elif relations == "fb": relations = ["in-front of", "behind"]
    else: raise ValueError(f"Invalid relations: {relations}")

    train_tuples = [] # (f1, f2, r)
    margin = len(nouns) // 3
    for idx1, i in enumerate(nouns):
        for idx2, j in enumerate(nouns):
            if i==j: continue
            if 0 < np.abs(idx1 - idx2) <= margin or np.abs(idx1 - idx2) >= len(nouns) - margin:
                train_tuples.append((i, j, relations[0])) 
                train_tuples.append((i, j, relations[1])) 
                
    if transpose: train_tuples = Transpose(train_tuples, apply_to_relations=["right of"])
    return train_tuples
"""


class whatsup_singleobj_dataset(Dataset):
    def __init__(self,
                 imdir: str,
                 annotations: List,
                 imsize = (32, 32),
                 ):
        super().__init__()
        self.imdir = imdir
        self.annotations = annotations
        self.preprocess = transforms.Compose(
            [   
                transforms.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
                transforms.Resize(imsize),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

        #print("Data Preprocessing")
        self.concept2pilimages = defaultdict(list)
        for a in tqdm(self.annotations):
            gth_caption, image_path, T = a
            image = Image.open(os.path.join(self.imdir, image_path))

            width, height = image.size
            new_dimension = min(image.size)
            left = (width - new_dimension)/2
            top = (height - new_dimension)/2
            right = (width + new_dimension)/2
            bottom = (height + new_dimension)/2
            pilimage = image.crop((left, top, right, bottom))

            W, H = pilimage.size
            crop1 = pilimage.crop((W//4, 0, 3*W//4, H//2)) # behind
            crop2 = pilimage.crop((W//4, H//2, 3*W//4, H)) # front
            crop3 = pilimage.crop((0, H//4, W//2, 3*H//4)) # left
            crop4 = pilimage.crop((W//2, H//4, W, 3*H//4)) # right
            f1, f2, r = T

            if "left of" in r: 
                self.concept2pilimages[f1].append(crop3)
                self.concept2pilimages[f2].append(crop4)
            elif "right of" in r: 
                self.concept2pilimages[f2].append(crop3)
                self.concept2pilimages[f1].append(crop4)
            elif "in-front of" in r: 
                self.concept2pilimages[f2].append(crop1)
                self.concept2pilimages[f1].append(crop2)
            elif "behind" in r: 
                self.concept2pilimages[f1].append(crop1)
                self.concept2pilimages[f2].append(crop2)
            else: raise ValueError(f"Invalid relation: {r}")

        self.classes = sorted(self.concept2pilimages.keys(), key=lambda x: (-len(self.concept2pilimages[x]), x))

        #print("Finish Preprocessing")
        for k in self.classes:
            print(f"concept {k} has {len(self.concept2pilimages[k])} crops")
        print()
        
    def __len__(self): return len(self.classes)
    
    def __getitem__(self, i): 
        f = self.classes[i]
        image = random.choice(self.concept2pilimages[f])
        text = f"an image of a {f}"
        return {
            'image': self.preprocess(image),
            'sentence': text
        }
    


from datasets import load_dataset

class butterfly_dataset(Dataset):
    def __init__(self,
                 imsize = (64, 64),
                 ):
        super().__init__()
        self.preprocess = transforms.Compose(
            [   
                transforms.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
                transforms.Resize(imsize),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )
        dataset_name = "huggan/smithsonian_butterflies_subset"
        self.data = load_dataset(dataset_name, split="train")
    
    def __len__(self): return len(self.data)
    
    def __getitem__(self, i): 
        image = self.data[i]['image']
        return {
            'image': self.preprocess(image),
            'sentence': "an image of a butterfly",
        }
    

class flickr30k_concept_warmup(Dataset):
    def __init__(self,
                 triplets_of_interest_file: str,
                 flickr_imdir: str,
                 flickr_annotation_file: str="../../data/flickr30k/karpathy/dataset_flickr30k.json",
                 imsize = (64, 64)
                 ):
        super().__init__()
        self.imdir = flickr_imdir
        self.preprocess = transforms.Compose(
            [   
                transforms.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
                transforms.Resize(imsize),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
        self.annotations = json.load(open(flickr_annotation_file, "r"))['images']

        ### get crux_nouns from triplets_of_interest
        # J = json.load(open(triplets_of_interest_file, "r"))
        # crux_nouns = sorted(list(set([
        #     a[-1][0] for a in J
        # ])))
        # print(f"{len(J)} instances have {len(crux_nouns)} unique nouns.")

        ### get flickr30k examples containing crux_nouns
        self.examples = []
        self.noun2exampleid = defaultdict(list)

        for a in tqdm(self.annotations):
            for s in a['sentences']:
                text = " ".join(s['tokens'])
                tokens_of_interest = [1]
                # for w in s['tokens']:
                #     singular_w = Word(w).lemmatize()
                #     if singular_w == False: singular_w = w
                #     if singular_w in crux_nouns:
                #         tokens_of_interest.append(w)
                #         self.noun2exampleid[singular_w].append(len(self.examples))
                if len(tokens_of_interest):
                    self.examples.append([
                        text,
                        "flickr30k/images/" + a['filename'],
                        tokens_of_interest
                    ])
        print(f"Found {len(self.examples)} that can help warmup {len(self.noun2exampleid)} nouns.")

    def __len__(self): return len(self.examples)
    
    def __getitem__(self, i): 
        text, image_path, _ = self.examples[i]
        image = Image.open(os.path.join(self.imdir, image_path))
        return {
            'image': self.preprocess(image),
            'sentence': "a real image" # text
        }

