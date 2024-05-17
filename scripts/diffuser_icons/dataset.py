from PIL import Image, ImageDraw, ImageFont
import os, torch
import numpy as np
from torchvision import transforms
from torch.utils.data import Dataset
from typing import Any, Dict, Optional, Tuple, Union, List

RELATIONS = ["{0} is on top of {1}.", "{1} is at the bottom of {0}."]
FONT_DIR = "fonts/" #/home/yingshan/clevr_control/scripts/diffuser_icons/

class icons_dataset(Dataset):
    def __init__(self,
                 data: List,
                 ):
        super().__init__()
        self.data = data
        self.preprocess = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )
    
    def __len__(self): return len(self.data)
    
    def __getitem__(self, i): 
        text, image = self.data[i]
        return {
            'image': self.preprocess(image),
            'sentence': text
        }
        
def draw_icon(unicode1, unicode2=None, canvas_size=32, icon_size=32, fontsize=28):
    if isinstance(canvas_size, int): H, W = canvas_size, canvas_size
    else: H, W = canvas_size
    # if unicode2 is not None: 
    #     W *= 2
    #     H *= 2

    im = Image.new("RGB", (W, H), (255,255,255))
    draw = ImageDraw.Draw(im)
    
    unicode_text, font = unicode1
    #print(font)
    unicode_font = ImageFont.truetype(os.path.join(FONT_DIR, f"{font}.ttf"), fontsize)
    _, _, w, h = draw.textbbox((0, 0), unicode_text, font=unicode_font)
    draw.text(((W-w)/2,(icon_size-h)/2), unicode_text, font=unicode_font, fill="black")
    
    if unicode2 is not None:
        unicode_text2, font2 = unicode2
        unicode_font2 = ImageFont.truetype(os.path.join(FONT_DIR, f"{font2}.ttf"), fontsize)
        _, _, w, h = draw.textbbox((0, 0), unicode_text2, font=unicode_font2)
        draw.text(((W-w)/2,(icon_size-h)/2+icon_size), unicode_text2, font=unicode_font2, fill="black")

    return im

def create_data_single_obj(nouns, icons, canvas_size=32, icon_size=32, fontsize=28):
    train_pairs = []
    for i, _A in enumerate(nouns):
        if _A[0] in 'aeiuo': A = "an "+_A
        else: A = "a "+_A
        train_pairs.append((
            f"an image of {A}.",
            draw_icon(icons[i], canvas_size=canvas_size, icon_size=icon_size, fontsize=fontsize),
        ))
    test_pairs = train_pairs
    return train_pairs, test_pairs

def create_data_single_obj_2_positions(nouns, icons, canvas_size=64, icon_size=32, fontsize=28):
    train_pairs = []
    for i, _A in enumerate(nouns):
        if _A[0] in 'aeiuo': A = "an "+_A
        else: A = "a "+_A
        train_pairs.append((
            f"an image of {A}.",
            draw_icon(icons[i], ("", "Symbola"), canvas_size=canvas_size, icon_size=icon_size, fontsize=fontsize),
        ))
        train_pairs.append((
            f"an image of {A}.",
            draw_icon(("", "Symbola"), icons[i], canvas_size=canvas_size, icon_size=icon_size, fontsize=fontsize),
        ))
    test_pairs = [train_pairs[i] for i in range(len(train_pairs)) if i%2]
    return train_pairs, test_pairs

def create_data_all(nouns, icons, canvas_size=(64, 32), icon_size=32, fontsize=28):
    train_pairs = []
    for i, _A in enumerate(nouns):
        for j, _B in enumerate(nouns):
            if i==j: continue
            unicode1, unicode2 = icons[i], icons[j]

            if _A[0] in 'aeiuo': A = "an "+_A
            else: A = "a "+_A
            if _B[0] in 'aeiuo': B = "an "+_B
            else: B = "a "+_B

            for r in RELATIONS: 
                train_pairs.append((
                    r.format(A, B),
                    draw_icon(unicode1, unicode2, canvas_size=canvas_size, icon_size=icon_size, fontsize=fontsize),
                ))   
    test_pairs = train_pairs
    return train_pairs, test_pairs

def create_data_split2(nouns, icons, canvas_size=(64, 32), icon_size=32, fontsize=28):
    train_pairs, test_pairs = [], []
    for i, _A in enumerate(nouns):
        for j, _B in enumerate(nouns):
            unicode1, unicode2 = icons[i], icons[j]
            if i < j:
                if _A[0] in 'aeiuo': A = "an "+_A
                else: A = "a "+_A
                if _B[0] in 'aeiuo': B = "an "+_B
                else: B = "a "+_B
                for r in RELATIONS: 
                    train_pairs.append((
                        r.format(A, B),
                        draw_icon(unicode1, unicode2, canvas_size=canvas_size, icon_size=icon_size, fontsize=fontsize),
                    ))   
            elif i > j:
                if _A[0] in 'aeiuo': A = "an "+_A
                else: A = "a "+_A
                if _B[0] in 'aeiuo': B = "an "+_B
                else: B = "a "+_B
                for r in RELATIONS: 
                    test_pairs.append((
                        r.format(A, B),
                        draw_icon(unicode1, unicode2, canvas_size=canvas_size, icon_size=icon_size, fontsize=fontsize),
                    ))   
    return train_pairs, test_pairs


def create_data_split3(nouns, icons, canvas_size=(64, 32), icon_size=32, fontsize=28):
    train_pairs, test_pairs = [], [] 
    half = len(nouns) // 2
    for i, _A in enumerate(nouns):
        for j, _B in enumerate(nouns):
            if i==j: continue
            unicode1, unicode2 = icons[i], icons[j]
            if 1 <= i - j < half or -len(nouns)+1 <= i - j <= -half:
                if _A[0] in 'aeiuo': A = "an "+_A
                else: A = "a "+_A
                if _B[0] in 'aeiuo': B = "an "+_B
                else: B = "a "+_B
                for r in RELATIONS: 
                    train_pairs.append((
                        r.format(A, B),
                        draw_icon(unicode1, unicode2, canvas_size=canvas_size, icon_size=icon_size, fontsize=fontsize),
                    ))   
            else:
                if _A[0] in 'aeiuo': A = "an "+_A
                else: A = "a "+_A
                if _B[0] in 'aeiuo': B = "an "+_B
                else: B = "a "+_B
                for r in RELATIONS: 
                    test_pairs.append((
                        r.format(A, B),
                        draw_icon(unicode1, unicode2, canvas_size=canvas_size, icon_size=icon_size, fontsize=fontsize),
                    ))   
    return train_pairs, test_pairs


def create_data_split4(nouns, icons, canvas_size=(64, 32), icon_size=32, fontsize=28):
    train_pairs, test_pairs = [], [] 
    half = len(nouns) // 2
    for i, _A in enumerate(nouns):
        for j, _B in enumerate(nouns):
            if i==j: continue
            unicode1, unicode2 = icons[i], icons[j]
            if i < half and j < half:
                if _A[0] in 'aeiuo': A = "an "+_A
                else: A = "a "+_A
                if _B[0] in 'aeiuo': B = "an "+_B
                else: B = "a "+_B
                train_pairs.append((
                    RELATIONS[0].format(A, B),
                    draw_icon(unicode1, unicode2, canvas_size=canvas_size, icon_size=icon_size, fontsize=fontsize),
                )) 
                test_pairs.append((
                    RELATIONS[1].format(A, B),
                    draw_icon(unicode1, unicode2, canvas_size=canvas_size, icon_size=icon_size, fontsize=fontsize),
                ))   
            elif i >= half and j >= half:
                if _A[0] in 'aeiuo': A = "an "+_A
                else: A = "a "+_A
                if _B[0] in 'aeiuo': B = "an "+_B
                else: B = "a "+_B
                test_pairs.append((
                    RELATIONS[0].format(A, B),
                    draw_icon(unicode1, unicode2, canvas_size=canvas_size, icon_size=icon_size, fontsize=fontsize),
                )) 
                train_pairs.append((
                    RELATIONS[1].format(A, B),
                    draw_icon(unicode1, unicode2, canvas_size=canvas_size, icon_size=icon_size, fontsize=fontsize),
                ))   
            else:
                if _A[0] in 'aeiuo': A = "an "+_A
                else: A = "a "+_A
                if _B[0] in 'aeiuo': B = "an "+_B
                else: B = "a "+_B
                test_pairs.append((
                    RELATIONS[0].format(A, B),
                    draw_icon(unicode1, unicode2, canvas_size=canvas_size, icon_size=icon_size, fontsize=fontsize),
                )) 
                test_pairs.append((
                    RELATIONS[1].format(A, B),
                    draw_icon(unicode1, unicode2, canvas_size=canvas_size, icon_size=icon_size, fontsize=fontsize),
                )) 
    return train_pairs, test_pairs


def create_data_split6(nouns, icons, canvas_size=(64, 32), icon_size=32, fontsize=28):
    train_pairs, test_pairs = [], []
    for i, _A in enumerate(nouns):
        for j, _B in enumerate(nouns):
            if i == j: continue
            unicode1, unicode2 = icons[i], icons[j]
            if _A[0] in 'aeiuo': A = "an "+_A
            else: A = "a "+_A
            if _B[0] in 'aeiuo': B = "an "+_B
            else: B = "a "+_B

            if i+j < len(nouns)-1:
                train_pairs.append((
                    RELATIONS[0].format(A, B),
                    draw_icon(unicode1, unicode2, canvas_size=canvas_size, icon_size=icon_size, fontsize=fontsize),
                )) 
                test_pairs.append((
                    RELATIONS[1].format(A, B),
                    draw_icon(unicode1, unicode2, canvas_size=canvas_size, icon_size=icon_size, fontsize=fontsize),
                ))  
            elif i+j > len(nouns)-1:
                train_pairs.append((
                    RELATIONS[1].format(A, B),
                    draw_icon(unicode1, unicode2, canvas_size=canvas_size, icon_size=icon_size, fontsize=fontsize),
                )) 
                test_pairs.append((
                    RELATIONS[0].format(A, B),
                    draw_icon(unicode1, unicode2, canvas_size=canvas_size, icon_size=icon_size, fontsize=fontsize),
                ))  
    return train_pairs, test_pairs


def create_data_split5(nouns, icons, canvas_size=(64, 32), icon_size=32, fontsize=28):
    train_pairs, test_pairs = [], []
    half = len(nouns) // 2
    for i, _A in enumerate(nouns):
        for j, _B in enumerate(nouns):
            if i==j: continue
            unicode1, unicode2 = icons[i], icons[j]
            if _A[0] in 'aeiuo': A = "an "+_A
            else: A = "a "+_A
            if _B[0] in 'aeiuo': B = "an "+_B
            else: B = "a "+_B

            if 0 <= i < half and half <= j < len(nouns):
                train_label = 0
            elif  0 <= j < half and half <= i < len(nouns):
                train_label = 1
            else:
                test_pairs.append((
                    RELATIONS[0].format(A, B),
                    draw_icon(unicode1, unicode2, canvas_size=canvas_size, icon_size=icon_size, fontsize=fontsize),
                )) 
                test_pairs.append((
                    RELATIONS[1].format(A, B),
                    draw_icon(unicode1, unicode2, canvas_size=canvas_size, icon_size=icon_size, fontsize=fontsize),
                )) 
                continue

            test_label = 1-train_label  
            train_pairs.append((
                RELATIONS[train_label].format(A, B),
                draw_icon(unicode1, unicode2, canvas_size=canvas_size, icon_size=icon_size, fontsize=fontsize),
            ))
            test_pairs.append((
                RELATIONS[test_label].format(A, B),
                draw_icon(unicode1, unicode2, canvas_size=canvas_size, icon_size=icon_size, fontsize=fontsize),
            ))
            
    return train_pairs, test_pairs


def create_data_split7(nouns, icons, canvas_size=(64, 32), icon_size=32, fontsize=28):
    train_texts, test_texts = [], []
    for i, _A in enumerate(nouns):
        for j, _B in enumerate(nouns):
            if i==j: continue
            unicode1, unicode2 = icons[i], icons[j]
            
            if _A[0] in 'aeiuo': A = "an "+_A
            else: A = "a "+_A
            if _B[0] in 'aeiuo': B = "an "+_B
            else: B = "a "+_B

            train_label = int(i>j)
            
            test_label = 1-train_label  
            train_texts.append((
                RELATIONS[train_label].format(A, B),
                draw_icon(unicode1, unicode2, canvas_size=canvas_size, icon_size=icon_size, fontsize=fontsize),
            ))
            test_texts.append((
                RELATIONS[test_label].format(A, B),
                draw_icon(unicode1, unicode2, canvas_size=canvas_size, icon_size=icon_size, fontsize=fontsize),
            ))
            
    return train_texts, test_texts


def create_data_split9(nouns, icons, canvas_size=(64, 32), icon_size=32, fontsize=28):
    train_texts, test_texts = [], []
    half = len(nouns) // 2
    for i, _A in enumerate(nouns):
        for j, _B in enumerate(nouns):
            if i==j: continue
            unicode1, unicode2 = icons[i], icons[j]
            if _A[0] in 'aeiuo': A = "an "+_A
            else: A = "a "+_A
            if _B[0] in 'aeiuo': B = "an "+_B
            else: B = "a "+_B

            if 0 <= i < half and half <= j < len(nouns): 
                for r in RELATIONS: 
                    train_texts.append((
                        r.format(A, B),
                        draw_icon(unicode1, unicode2, canvas_size=canvas_size, icon_size=icon_size, fontsize=fontsize),
                    )) 
            elif half <= i < len(nouns) and 0 <= j < half: 
                for r in RELATIONS: 
                    train_texts.append((
                        r.format(A, B),
                        draw_icon(unicode1, unicode2, canvas_size=canvas_size, icon_size=icon_size, fontsize=fontsize),
                    )) 
            else:
                for r in RELATIONS: 
                    test_texts.append((
                        r.format(A, B),
                        draw_icon(unicode1, unicode2, canvas_size=canvas_size, icon_size=icon_size, fontsize=fontsize),
                    )) 
            
    return train_texts, test_texts


def create_data_split8(nouns, icons, canvas_size=(64, 32), icon_size=32, fontsize=28):
    train_texts, test_texts = [], []
    half = len(nouns) // 2
    for i, _A in enumerate(nouns):
        for j, _B in enumerate(nouns):
            if i==j: continue

            unicode1, unicode2 = icons[i], icons[j]
            if _A[0] in 'aeiuo': A = "an "+_A
            else: A = "a "+_A
            if _B[0] in 'aeiuo': B = "an "+_B
            else: B = "a "+_B

            if i+j >= len(nouns) - 1:
                for r in RELATIONS: 
                    train_texts.append((
                        r.format(A, B),
                        draw_icon(unicode1, unicode2, canvas_size=canvas_size, icon_size=icon_size, fontsize=fontsize),
                    )) 
            else:
                for r in RELATIONS: 
                    test_texts.append((
                        r.format(A, B),
                        draw_icon(unicode1, unicode2, canvas_size=canvas_size, icon_size=icon_size, fontsize=fontsize),
                    ))
            
    return train_texts, test_texts


def create_data_split12(nouns, icons, canvas_size=(64, 32), icon_size=32, fontsize=28):
    train_texts, test_texts = [], []
    half = len(nouns) // 2
    for i, _A in enumerate(nouns):
        for j, _B in enumerate(nouns):
            if i==j: continue

            unicode1, unicode2 = icons[i], icons[j]
            if _A[0] in 'aeiuo': A = "an "+_A
            else: A = "a "+_A
            if _B[0] in 'aeiuo': B = "an "+_B
            else: B = "a "+_B
            if 0 <= i < half and half <= j < len(nouns): 
                for r in RELATIONS: 
                    train_texts.append((
                        r.format(A, B),
                        draw_icon(unicode1, unicode2, canvas_size=canvas_size, icon_size=icon_size, fontsize=fontsize),
                    )) 
            else:
                for r in RELATIONS: 
                    test_texts.append((
                        r.format(A, B),
                        draw_icon(unicode1, unicode2, canvas_size=canvas_size, icon_size=icon_size, fontsize=fontsize),
                    ))   
    return train_texts, test_texts

def create_data_split13(nouns, icons, canvas_size=(64, 32), icon_size=32, fontsize=28):
    train_texts, test_texts = [], []
    half = len(nouns) // 2
    for i, _A in enumerate(nouns):
        for j, _B in enumerate(nouns):
            if i==j: continue

            unicode1, unicode2 = icons[i], icons[j]
            if _A[0] in 'aeiuo': A = "an "+_A
            else: A = "a "+_A
            if _B[0] in 'aeiuo': B = "an "+_B
            else: B = "a "+_B
            if half <= j < len(nouns): 
                for r in RELATIONS: 
                    train_texts.append((
                        r.format(A, B),
                        draw_icon(unicode1, unicode2, canvas_size=canvas_size, icon_size=icon_size, fontsize=fontsize),
                    )) 
            else:
                for r in RELATIONS: 
                    test_texts.append((
                        r.format(A, B),
                        draw_icon(unicode1, unicode2, canvas_size=canvas_size, icon_size=icon_size, fontsize=fontsize),
                    ))   
    return train_texts, test_texts


def create_data_split14(nouns, icons, canvas_size=(64, 32), icon_size=32, fontsize=28):
    train_texts, test_texts = [], []
    cut = len(nouns) // 4
    for i, _A in enumerate(nouns):
        for j, _B in enumerate(nouns):
            if i==j: continue

            unicode1, unicode2 = icons[i], icons[j]
            if _A[0] in 'aeiuo': A = "an "+_A
            else: A = "a "+_A
            if _B[0] in 'aeiuo': B = "an "+_B
            else: B = "a "+_B

            if j < cut or i >= len(nouns) - cut: 
                for r in RELATIONS: 
                    train_texts.append((
                        r.format(A, B),
                        draw_icon(unicode1, unicode2, canvas_size=canvas_size, icon_size=icon_size, fontsize=fontsize),
                    )) 
            else:
                for r in RELATIONS: 
                    test_texts.append((
                        r.format(A, B),
                        draw_icon(unicode1, unicode2, canvas_size=canvas_size, icon_size=icon_size, fontsize=fontsize),
                    )) 
            
    return train_texts, test_texts


def create_data_split15(nouns, icons, canvas_size=(64, 32), icon_size=32, fontsize=28):
    train_texts, test_texts = [], []
    cut = len(nouns) // 3
    for i, _A in enumerate(nouns):
        for j, _B in enumerate(nouns):
            if i==j: continue

            unicode1, unicode2 = icons[i], icons[j]
            if _A[0] in 'aeiuo': A = "an "+_A
            else: A = "a "+_A
            if _B[0] in 'aeiuo': B = "an "+_B
            else: B = "a "+_B

            if j < cut or i >= len(nouns) - cut: 
                for r in RELATIONS: 
                    train_texts.append((
                        r.format(A, B),
                        draw_icon(unicode1, unicode2, canvas_size=canvas_size, icon_size=icon_size, fontsize=fontsize),
                    )) 
            else:
                for r in RELATIONS: 
                    test_texts.append((
                        r.format(A, B),
                        draw_icon(unicode1, unicode2, canvas_size=canvas_size, icon_size=icon_size, fontsize=fontsize),
                    )) 
            
    return train_texts, test_texts


def create_data_split19(nouns, icons, canvas_size=(64, 32), icon_size=32, fontsize=28):
    train_texts, test_texts = [], []
    cut = 2*len(nouns) // 5 
    for i, _A in enumerate(nouns):
        for j, _B in enumerate(nouns):
            if i==j: continue

            unicode1, unicode2 = icons[i], icons[j]
            if _A[0] in 'aeiuo': A = "an "+_A
            else: A = "a "+_A
            if _B[0] in 'aeiuo': B = "an "+_B
            else: B = "a "+_B
            
            if i < cut:
                if j < cut:
                    for r in RELATIONS: 
                        test_texts.append((
                            r.format(A, B),
                            draw_icon(unicode1, unicode2, canvas_size=canvas_size, icon_size=icon_size, fontsize=fontsize),
                        ))
                    continue
                else:
                    train_label = 0
            elif cut <= i < len(nouns) - cut:
                if j < cut: 
                    train_label = 1
                elif cut <= j < len(nouns) - cut: 
                    for r in RELATIONS: 
                        train_texts.append((
                            r.format(A, B),
                            draw_icon(unicode1, unicode2, canvas_size=canvas_size, icon_size=icon_size, fontsize=fontsize),
                        )) 
                    continue
                else:
                    train_label = 0
            else:
                if j < len(nouns) - cut:
                    train_label = 1
                else:
                    for r in RELATIONS: 
                        test_texts.append((
                            r.format(A, B),
                            draw_icon(unicode1, unicode2, canvas_size=canvas_size, icon_size=icon_size, fontsize=fontsize),
                        ))
                    continue

            test_label = 1 - train_label
            train_texts.append((
                RELATIONS[train_label].format(A, B),
                draw_icon(unicode1, unicode2, canvas_size=canvas_size, icon_size=icon_size, fontsize=fontsize),
            ))
            test_texts.append((
                RELATIONS[test_label].format(A, B),
                draw_icon(unicode1, unicode2, canvas_size=canvas_size, icon_size=icon_size, fontsize=fontsize),
            ))
            
    return train_texts, test_texts


def create_data_split16(nouns, icons, canvas_size=(64, 32), icon_size=32, fontsize=28):
    train_texts, test_texts = [], []
    cut = len(nouns) // 3
    for i, _A in enumerate(nouns):
        for j, _B in enumerate(nouns):
            if i==j: continue
            
            unicode1, unicode2 = icons[i], icons[j]
            if _A[0] in 'aeiuo': A = "an "+_A
            else: A = "a "+_A
            if _B[0] in 'aeiuo': B = "an "+_B
            else: B = "a "+_B

            if j < cut or i >= len(nouns) - cut: 
                for r in RELATIONS: 
                    test_texts.append((
                        r.format(A, B),
                        draw_icon(unicode1, unicode2, canvas_size=canvas_size, icon_size=icon_size, fontsize=fontsize),
                    ))
            else:
                for r in RELATIONS: 
                    train_texts.append((
                        r.format(A, B),
                        draw_icon(unicode1, unicode2, canvas_size=canvas_size, icon_size=icon_size, fontsize=fontsize),
                    ))
            
    return train_texts, test_texts


def create_data_split17(nouns, icons, canvas_size=(64, 32), icon_size=32, fontsize=28):
    train_texts, test_texts = [], []
    cut = len(nouns) // 4
    for i, _A in enumerate(nouns):
        for j, _B in enumerate(nouns):
            if i==j: continue
            
            unicode1, unicode2 = icons[i], icons[j]
            if _A[0] in 'aeiuo': A = "an "+_A
            else: A = "a "+_A
            if _B[0] in 'aeiuo': B = "an "+_B
            else: B = "a "+_B

            if j < cut or i >= len(nouns) - cut: 
                for r in RELATIONS: 
                    test_texts.append((
                        r.format(A, B),
                        draw_icon(unicode1, unicode2, canvas_size=canvas_size, icon_size=icon_size, fontsize=fontsize),
                    ))
            else:
                for r in RELATIONS: 
                    train_texts.append((
                        r.format(A, B),
                        draw_icon(unicode1, unicode2, canvas_size=canvas_size, icon_size=icon_size, fontsize=fontsize),
                    ))
            
    return train_texts, test_texts

def create_data_split20(nouns, icons, canvas_size=(64, 32), icon_size=32, fontsize=28):
    train_texts, test_texts = [], []
    cut = len(nouns) // 6
    for i, _A in enumerate(nouns):
        for j, _B in enumerate(nouns):
            if i==j: continue
            
            unicode1, unicode2 = icons[i], icons[j]
            if _A[0] in 'aeiuo': A = "an "+_A
            else: A = "a "+_A
            if _B[0] in 'aeiuo': B = "an "+_B
            else: B = "a "+_B

            if j < cut or i >= len(nouns) - cut: 
                for r in RELATIONS: 
                    test_texts.append((
                        r.format(A, B),
                        draw_icon(unicode1, unicode2, canvas_size=canvas_size, icon_size=icon_size, fontsize=fontsize),
                    ))
            else:
                for r in RELATIONS: 
                    train_texts.append((
                        r.format(A, B),
                        draw_icon(unicode1, unicode2, canvas_size=canvas_size, icon_size=icon_size, fontsize=fontsize),
                    ))
            
    return train_texts, test_texts


def create_data_split21(nouns, icons, canvas_size=(64, 32), icon_size=32, fontsize=28):
    train_texts, test_texts = [], []
    half = len(nouns) // 2
    cut = len(nouns) // 20
    for i, _A in enumerate(nouns):
        for j, _B in enumerate(nouns):
            if i==j: continue

            unicode1, unicode2 = icons[i], icons[j]
            if _A[0] in 'aeiuo': A = "an "+_A
            else: A = "a "+_A
            if _B[0] in 'aeiuo': B = "an "+_B
            else: B = "a "+_B

            if 0 <= i < half and half <= j < len(nouns):
                train_label = 0
            elif  0 <= j < half and half <= i < len(nouns):
                train_label = 1
            elif i < cut and i < j:
                train_label = 0
            elif j < cut and i > j:
                train_label = 1
            elif i >= len(nouns) - cut and i > j:
                train_label = 1
            elif j >= len(nouns) - cut and i < j:
                train_label = 0
            else:
                for r in RELATIONS: 
                    test_texts.append((
                        r.format(A, B),
                        draw_icon(unicode1, unicode2, canvas_size=canvas_size, icon_size=icon_size, fontsize=fontsize),
                    )) 
                continue

            test_label = 1-train_label  
            train_texts.append((
                RELATIONS[train_label].format(A, B),
                draw_icon(unicode1, unicode2, canvas_size=canvas_size, icon_size=icon_size, fontsize=fontsize),
            ))
            test_texts.append((
                RELATIONS[test_label].format(A, B),
                draw_icon(unicode1, unicode2, canvas_size=canvas_size, icon_size=icon_size, fontsize=fontsize),
            ))
            
    return train_texts, test_texts


def create_data_split22(nouns, icons, canvas_size=(64, 32), icon_size=32, fontsize=28):
    train_texts, test_texts = [], []
    margin = len(nouns) // 6
    for i, _A in enumerate(nouns):
        for j, _B in enumerate(nouns):
            if i==j: continue
            
            unicode1, unicode2 = icons[i], icons[j]
            if _A[0] in 'aeiuo': A = "an "+_A
            else: A = "a "+_A
            if _B[0] in 'aeiuo': B = "an "+_B
            else: B = "a "+_B

            if 0 < np.abs(i - j) <= margin or np.abs(i - j) >= len(nouns) - margin:
                for r in RELATIONS: 
                    train_texts.append((
                        r.format(A, B),
                        draw_icon(unicode1, unicode2, canvas_size=canvas_size, icon_size=icon_size, fontsize=fontsize),
                    )) 
            else:
                for r in RELATIONS: 
                    test_texts.append((
                        r.format(A, B),
                        draw_icon(unicode1, unicode2, canvas_size=canvas_size, icon_size=icon_size, fontsize=fontsize),
                    )) 
            
    return train_texts, test_texts

def create_data_split24(nouns, icons, canvas_size=(64, 32), icon_size=32, fontsize=28):
    train_texts, test_texts = [], []
    margin = len(nouns) // 10
    for i, _A in enumerate(nouns):
        for j, _B in enumerate(nouns):
            if i==j: continue
            
            unicode1, unicode2 = icons[i], icons[j]
            if _A[0] in 'aeiuo': A = "an "+_A
            else: A = "a "+_A
            if _B[0] in 'aeiuo': B = "an "+_B
            else: B = "a "+_B

            if 0 < np.abs(i - j) <= margin or np.abs(i - j) >= len(nouns) - margin:
                for r in RELATIONS: 
                    train_texts.append((
                        r.format(A, B),
                        draw_icon(unicode1, unicode2, canvas_size=canvas_size, icon_size=icon_size, fontsize=fontsize),
                    )) 
            else:
                for r in RELATIONS: 
                    test_texts.append((
                        r.format(A, B),
                        draw_icon(unicode1, unicode2, canvas_size=canvas_size, icon_size=icon_size, fontsize=fontsize),
                    )) 
            
    return train_texts, test_texts


def create_data_split23(nouns, icons, canvas_size=(64, 32), icon_size=32, fontsize=28):
    train_texts, test_texts = [], []
    margin = len(nouns) // 3
    half_margin = len(nouns) // 6
    for i, _A in enumerate(nouns):
        for j, _B in enumerate(nouns):
            if i==j: continue

            unicode1, unicode2 = icons[i], icons[j]
            if _A[0] in 'aeiuo': A = "an "+_A
            else: A = "a "+_A
            if _B[0] in 'aeiuo': B = "an "+_B
            else: B = "a "+_B

            if 0 < i - j <= half_margin or len(nouns)-half_margin <= i - j < len(nouns): 
                train_label = 0
            elif half_margin < i - j <= margin or len(nouns)-margin <= i - j < len(nouns)-half_margin:
                train_label = 1
            elif 0 < j - i <= half_margin or len(nouns)-half_margin <= j - i < len(nouns):
                train_label = 1
            elif half_margin < j - i <= margin or len(nouns)-margin <= j - i < len(nouns)-half_margin:
                train_label = 0
            else:
                for r in RELATIONS: 
                    test_texts.append((
                        r.format(A, B),
                        draw_icon(unicode1, unicode2, canvas_size=canvas_size, icon_size=icon_size, fontsize=fontsize),
                    )) 
                continue
            test_label = 1-train_label  
            train_texts.append((
                RELATIONS[train_label].format(A, B),
                draw_icon(unicode1, unicode2, canvas_size=canvas_size, icon_size=icon_size, fontsize=fontsize),
            ))
            test_texts.append((
                RELATIONS[test_label].format(A, B),
                draw_icon(unicode1, unicode2, canvas_size=canvas_size, icon_size=icon_size, fontsize=fontsize),
            ))
            
    return train_texts, test_texts


def create_data_custom_split(nouns, icons, train_triplets_file, canvas_size=(64, 32), icon_size=32, fontsize=28):
    with open(train_triplets_file, "r") as f:
        lines = f.readlines()
    train_triplets = []
    for l in lines:
        if len(l.strip()) == 0: continue
        for t in l.strip().replace("[(", "").replace(")]", "").split("), ("):
            train_triplets.append([int(x) for x in t.split(", ")])
    
    ## train_triplets may contain non-contiguous concept ids
    unique_O1 = set([t[0] for t in train_triplets])
    unique_O2 = set([t[1] for t in train_triplets])
    assert unique_O1 == unique_O2
    print(f"num_objs = {len(unique_O1)}")
    non_contiguous_concept_ids = sorted(list(unique_O1))
    to_contigous = {x:i for i, x in enumerate(non_contiguous_concept_ids)}

    D = {(i, j, r): False for i in range(len(unique_O1)) for j in range(len(unique_O1)) for r in range(len(RELATIONS)) if not i==j}
    # :False means in testing set, :True meanings in training set
    train_texts, test_texts = [], []

    for t in train_triplets:
        i, j = to_contigous[t[0]], to_contigous[t[1]]
        unicode1, unicode2 = icons[i], icons[j]
        _A, _B = nouns[i], nouns[j]
        if _A[0] in 'aeiuo': A = "an "+_A
        else: A = "a "+_A
        if _B[0] in 'aeiuo': B = "an "+_B
        else: B = "a "+_B

        train_texts.append((
                        RELATIONS[t[-1]].format(A, B),
                        draw_icon(unicode1, unicode2, canvas_size=canvas_size, icon_size=icon_size, fontsize=fontsize),
                    )) 
        D[(i, j, t[-1])] = True
    
    for (i, j, r) in D:
        if D[(i, j, r)] or i==j: continue
        unicode1, unicode2 = icons[i], icons[j]
        _A, _B = nouns[i], nouns[j]
        if _A[0] in 'aeiuo': A = "an "+_A
        else: A = "a "+_A
        if _B[0] in 'aeiuo': B = "an "+_B
        else: B = "a "+_B

        test_texts.append((
                        RELATIONS[r].format(A, B),
                        draw_icon(unicode1, unicode2, canvas_size=canvas_size, icon_size=icon_size, fontsize=fontsize),
                    )) 

    return train_texts, test_texts